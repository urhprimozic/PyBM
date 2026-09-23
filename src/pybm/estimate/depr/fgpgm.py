"""
FGPGM (Fast Gaussian Process Gradient Matching) - Wenk, Gotovos, Bauer, Gorbach, Krause, Buhmann,
"Fast Gaussian process based gradient matching for parameter identification in systems of nonlinear
ODEs", AISTATS 2019. See `notes/mcmc-gradient-matching.md` for the full derivation this module
implements - this docstring only covers what's needed to read the code.

Unlike `pybm.estimate.gradient_matching` (a point estimate via `scipy.optimize.least_squares`),
FGPGM is Bayesian: it samples (via Metropolis-Hastings MCMC) from the joint posterior

    p(x, c | y, phi, gamma, sigma)
        ~  p(c)                              [prior on constants - here: uniform within .range]
         * N(x | 0, K)                        [GP prior on the latent state]
         * N(y | x, sigma^2 I)                 [data fit]
         * N(F(x,c) | D x, A + gamma I)        [gradient-matching term]

where `x` is the LATENT state trajectory at a fixed collocation grid (not fixed at the GP posterior
mean like `gradient_matching`/`integral_matching` - it is free to move if that better reconciles
the GP prior, the data, and the ODE). `K` = GP prior covariance, `D`/`A` = the GP-implied conditional
mean/covariance of the derivative given the state (closed form, same construction as
`gradient_matching`'s `_kernel_deriv_wrt_first`, extended to the FULL T x T joint covariance here
instead of just the pointwise/diagonal `var_deriv`).

Two deliberate deviations from the paper, both for practical reasons at this codebase's data scale:

1. **Block proposals, not single-site.** Algorithm 1 in the paper proposes ONE state or parameter
   component at a time (a Metropolis-within-Gibbs sweep over every element of `x` and `c`) - fine
   for their own systems (T~15-20 observation points). On real Bled-scale data (T in the thousands),
   a literal single-site sweep would need thousands of individual proposals PER MCMC iteration, each
   needing (at least a partial) re-evaluation of the target - computationally infeasible here. This
   module instead proposes the WHOLE state vector (per variable) and the WHOLE constant vector each
   as one block - still a valid Metropolis-Hastings scheme (the acceptance rule (1.1) in
   `notes/mcmc-gradient-matching.md` doesn't care about the proposal's shape), just coarser mixing
   per iteration than single-site would give.
2. **Modest collocation grid, not every data point.** `t`/`y` passed in do not have to be the raw
   full data grid - see `max_points` below. GP-implied `A`/`K` are `T x T` and get Cholesky-factored
   ONCE (`O(T^3)`) plus one `O(T^2)` solve per MCMC iteration for each of the two quadratic-form
   terms - infeasible at T~thousands within a reasonable MCMC iteration budget. `max_points`
   subsamples (same convention as `gradient_matching.fit_gps`'s `max_gp_points`) to keep this
   tractable; the paper's own systems never needed this (T was already small).

`gamma` (ODE/GP mismatch tolerance) is NOT inferred - the paper treats it as a fixed hyperparameter,
manually swept over a handful of values and picked by fit quality (Wenk et al. tried 8 log-spaced
values between 1 and 1e-4). See `sweep_gamma` below for the equivalent here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from pybm.estimate.depr.gradient_matching import (
    _const_bounds,
    _fit_gp_1d,
    _kernel_deriv_wrt_first,
    _qualified_var_names,
    _rbf_kernel,
    _series_as_numpy,
    estimate_gradient_matching,
)
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _make_rhs


def _rbf_kernel_deriv2(t_a: np.ndarray, t_b: np.ndarray, lengthscale: float, signal_var: float) -> np.ndarray:
    """
    `cov(f'(t_a), f'(t_b)) = d^2/(dt_a dt_b) k(t_a,t_b)` - see `notes/gradient-matching.md` §1.5 for
    the derivation (there evaluated only at `t_a=t_b`, the diagonal/prior-variance special case;
    here the FULL matrix, needed for the joint `T x T` derivative covariance `A`).
    """
    diff = t_a[:, None] - t_b[None, :]
    k = signal_var * np.exp(-0.5 * (diff / lengthscale) ** 2)
    return k / lengthscale**2 * (1.0 - (diff / lengthscale) ** 2)


def _fit_fgpgm_hyperparams(t_obs: np.ndarray, y_obs: np.ndarray, max_iter: int = 200) -> "tuple[float, float, float]":
    """
    Step 1 of Algorithm 1 (Wenk et al.): standardize (z-score) the data, then max-likelihood fit
    (reusing `gradient_matching._fit_gp_1d` unchanged). Standardizing first is the paper's own fix
    for numerically ill-conditioned marginal-likelihood optimization when different states/variables
    live on very different scales - `gradient_matching.fit_gps` does NOT do this (a difference
    worth trying there too, per the chat discussion this module follows from).

    Hyperparameters are converted back to REAL (destandardized) units before returning: lengthscale
    is a time-axis quantity, unaffected by y-axis standardization; `signal_var`/`noise_var` scale by
    `std(y_obs)**2`. Everything downstream of this function (the latent state `x` sampled by MCMC,
    the ODE right-hand side `F(x,c)`) operates in REAL units throughout - standardization is only an
    internal aid for this one optimization.
    """
    mu = float(y_obs.mean())
    sigma = max(float(y_obs.std()), 1e-8)
    y_std = (y_obs - mu) / sigma
    gp_std = _fit_gp_1d(t_obs, y_std, max_points=len(t_obs), max_iter=max_iter)
    return gp_std.lengthscale, gp_std.signal_var * sigma**2, gp_std.noise_var * sigma**2


@dataclass
class _FGPGMPrecomp:
    """Everything about one state variable's GP that stays fixed for the whole MCMC run - computed
    ONCE from `t`/hyperparameters, reused at every iteration (only `x`/`c` change during sampling)."""

    t: np.ndarray  # (T,) collocation grid
    y: np.ndarray  # (T,) observed data AT that grid
    noise_var: float  # observation noise variance (real units) - the y-likelihood term's own scale
    K_chol: Any  # cho_factor(K + jitter*I) - K = GP prior covariance, Cov(x,x)
    D: np.ndarray  # (T,T): E[xdot | x] = D @ x
    A_chol: Any  # cho_factor(A + jitter*I) - A = Cov(xdot,xdot | x), the gradient-matching term's own scale


def _precompute_fgpgm(
    t: np.ndarray, y: np.ndarray, lengthscale: float, signal_var: float, noise_var: float,
    gamma: float, jitter: float = 1e-6,
) -> _FGPGMPrecomp:
    """
    Builds `K`, `D`, `A+gamma*I` (see module docstring) from the fitted GP hyperparameters - the
    standard multivariate-Gaussian conditioning (Schur complement) applied to the joint GP over
    `(x, xdot)` at the SAME `T` collocation points:
        E[xdot|x] = Cov(xdot,x) Cov(x,x)^-1 x =: D x
        Cov(xdot,xdot|x) = Cov(xdot,xdot) - Cov(xdot,x) Cov(x,x)^-1 Cov(x,xdot) =: A
    `Cov(xdot,x)(i,j) = d/dt_i k(t_i,t_j)` is `_kernel_deriv_wrt_first` (same function
    `gradient_matching.GradientMatchingResult` already uses for the posterior derivative MEAN);
    `Cov(xdot,xdot)` is `_rbf_kernel_deriv2` above. No observation noise enters `K` here - the
    relationship between the LATENT `x` and its derivative is a noiseless GP identity; `sigma^2`
    (observation noise) only enters separately, as the `y`-likelihood term's own scale. `gamma`
    (ODE/GP mismatch tolerance, module docstring) is folded into `A` HERE, once, since it is fixed
    for the whole MCMC run - `A_chol` below is actually `cho_factor(A + gamma*I + jitter*I)`.
    """
    n = len(t)
    eye = np.eye(n)
    K = _rbf_kernel(t, t, lengthscale, signal_var)
    K_chol = cho_factor(K + jitter * eye, lower=True)

    Cprime = _kernel_deriv_wrt_first(t, t, lengthscale, signal_var)  # Cov(xdot_i, x_j)
    Cpp = _rbf_kernel_deriv2(t, t, lengthscale, signal_var)  # Cov(xdot_i, xdot_j)
    D = cho_solve(K_chol, Cprime.T).T  # D = Cprime @ K^-1, solved instead of inverting explicitly
    A = Cpp - D @ Cprime.T
    A_chol = cho_factor(A + (gamma + jitter) * eye, lower=True)

    return _FGPGMPrecomp(t=t, y=y, noise_var=noise_var, K_chol=K_chol, D=D, A_chol=A_chol)


@dataclass
class FGPGMResult(ParamEstimationResults):
    model: InducedModel
    t: np.ndarray  # (T,) shared collocation grid every state was sampled on
    consts: np.ndarray  # (n_consts,) POSTERIOR MEAN (Algorithm 1's own final "return the mean of S")
    const_by_name: "dict[str, float]"
    const_samples: np.ndarray  # (n_kept, n_consts) post-burn-in MCMC samples, for uncertainty
    x_samples: "dict[str, np.ndarray]"  # var qualified name -> (n_kept, T) post-burn-in state samples
    accept_rate_theta: float
    accept_rate_x: float
    gamma: float


def estimate_fgpgm(
    model: InducedModel,
    t_eval,
    gamma: float = 1e-3,
    max_points: int = 150,
    n_mcmc: int = 2000,
    n_burnin: int = 1000,
    thin: int = 5,
    sigma_p: "float | None" = None,
    sigma_s: "float | None" = None,
    adapt_every: int = 50,
    target_accept: float = 0.234,
    max_gp_iter: int = 200,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    seed: "int | None" = None,
    verbose: int = 0,
) -> FGPGMResult:
    """
    FGPGM estimate of `model`'s constants - see the module docstring for the method, and for the two
    deliberate deviations from Algorithm 1 (block proposals, subsampled collocation grid) needed to
    run this at Bled's data scale.

    Parameters
    ----------
    model : InducedModel
        Must have `engine == "torch"`, fully resolved, every equation in differentiable torch ops -
        same precondition as `estimate_gradient_matching`.
    t_eval : array-like
        Time points defining the collocation grid (before `max_points` subsampling).
    gamma : float, optional
        ODE/GP mismatch tolerance (see module docstring) - a fixed hyperparameter, not inferred.
        Default `1e-3`. See `sweep_gamma` to search over this instead of guessing.
    max_points : int, optional
        Collocation grid is subsampled (evenly, same convention as `fit_gps`'s `max_gp_points`) to
        at most this many points - keeps the `O(T^3)` one-time factorization and `O(T^2)`-per-MCMC-
        iteration cost tractable. Default `150` (Wenk et al.'s own systems used T~15-20; this is
        already generous for a well-mixing chain at reasonable cost).
    n_mcmc, n_burnin : int, optional
        Post-burn-in samples to draw / burn-in length. Default 2000/1000 (modest - this is a
        demonstration-scale MCMC, not a production-grade one; see `notes/mcmc-gradient-matching.md`
        §3.3 for the real cost this trades off against).
    thin : int, optional
        Keep every `thin`-th post-burn-in sample (reduces autocorrelation in the returned samples,
        and how much history the caller has to carry around). Default 5.
    sigma_p, sigma_s : float, optional
        Random-walk proposal std-devs for the constant block / state block. Default: a fraction of
        each constant's own `.range` width, and a fraction of each state's own GP `signal_var`
        respectively - `None` triggers this data-driven default.
    adapt_every, target_accept : optional
        During burn-in only, rescale `sigma_p`/`sigma_s` every `adapt_every` steps to push the
        running acceptance rate toward `target_accept` (Gelman 1997's asymptotically-optimal
        `0.234` for random-walk Metropolis - see `notes/mcmc-gradient-matching.md` §1.4). Frozen
        after burn-in ends (a chain whose proposal keeps changing during the "real" sampling phase
        no longer has a fixed, valid stationary distribution).
    max_gp_iter : int, optional
        Passed to the GP hyperparameter fit (step 1) - see its own docs.
    device, dtype : optional
        Where/at what precision the ODE right-hand side is evaluated (the MCMC bookkeeping itself is
        plain numpy - the state/parameter dimensionality here is nowhere near large enough for torch
        to be worth the overhead, only `F(x,c)` itself needs torch for batched evaluation).
    seed : int, optional
        RNG seed.
    verbose : int, optional
        `0` silent; `>=1` prints acceptance rates periodically; `>=2` prints every `adapt_every` steps.

    Returns
    -------
    FGPGMResult
        `consts`: posterior MEAN (Algorithm 1's own point-estimate convention). `const_samples`/
        `x_samples`: the kept post-burn-in draws, for anyone who wants the actual posterior instead
        of just its mean (credible intervals, propagating uncertainty through a forward simulation -
        see `notes/mcmc-gradient-matching.md` §3.2 point 3).
    """
    if model.engine != "torch":
        raise ValueError(f"estimate_fgpgm requires an InducedModel built with engine='torch', got engine={model.engine!r}.")

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    rng = np.random.default_rng(seed)

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    qualified = _qualified_var_names(model)
    var_names = [qualified[id(v)] for v in state_vars]
    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    if len(t_eval) > max_points:
        idx = np.linspace(0, len(t_eval) - 1, max_points).astype(int)
        t_grid = t_eval[idx]
    else:
        t_grid = t_eval

    # Step 1 (Algorithm 1): standardize + max-likelihood fit, ONE state at a time, no ODE involved yet.
    precomp: "dict[str, _FGPGMPrecomp]" = {}
    for var, name in zip(state_vars, var_names):
        if var.data is None:
            raise ValueError(f"Variable {name} has no data; FGPGM needs an observed trajectory.")
        t_obs, y_obs = _series_as_numpy(var.data.t, var.data.x)
        lengthscale, signal_var, noise_var = _fit_fgpgm_hyperparams(t_obs, y_obs, max_iter=max_gp_iter)
        y_grid = np.interp(t_grid, t_obs, y_obs)  # nearest observed values at the (sub)grid
        precomp[name] = _precompute_fgpgm(t_grid, y_grid, lengthscale, signal_var, noise_var, gamma=gamma)
        if verbose >= 2:
            print(f"[fgpgm] {name}: lengthscale={lengthscale:.4g} signal_var={signal_var:.4g} noise_var={noise_var:.4g}")

    lo, hi = _const_bounds(model)
    x = {name: precomp[name].y.copy() for name in var_names}  # start each state at its own observed data

    # Warm-start theta (and its proposal step size) from a quick gradient_matching fit, rather than a
    # crude bounds-midpoint guess + isotropic step. Two problems this fixes, found by direct testing
    # on the predator/prey benchmark: (1) a poor starting theta wastes burn-in just walking toward the
    # right region; (2) theta's OWN acceptance rate stayed pinned near 0 even after aggressive
    # isotropic-step adaptation (checked directly: ~0.01-0.03 after 2000 burn-in steps) - because
    # theta's effect on the log-target is SUMMED over all T collocation points at once, its effective
    # local curvature is far sharper than a naive "theta has only p dimensions" isotropic guess
    # (Roberts, Gelman & Gilks 1997's `2.38/sqrt(d)` rule) assumes. `estimate_gradient_matching`'s own
    # converged Jacobian already gives exactly this curvature (same `G=J^T J` construction as
    # `_identifiability_report`) - reuse it directly: per-parameter proposal std from
    # `sqrt(diag(G^+))` (Moore-Penrose pseudo-inverse - `G` can be near-singular along
    # poorly-identified directions, see `notes/integral-matching.md` §4).
    gm_result = estimate_gradient_matching(model, t_eval, device=device, dtype=dtype)
    theta = np.clip(gm_result.consts, lo, hi)
    if sigma_p is None:
        jac = gm_result.least_squares_result.jac
        U, s, Vt = np.linalg.svd(jac, full_matrices=False)
        s_safe = np.where(s > 1e-8 * (s[0] if len(s) else 1.0), s, np.inf)  # near-zero -> huge step, not NaN
        G_pinv_diag = np.einsum("ij,j,ij->i", Vt.T, 1.0 / s_safe**2, Vt.T)  # diag(V S^-2 V^T)
        sigma_p = (2.38 / np.sqrt(len(theta))) * np.sqrt(np.clip(G_pinv_diag, 0, 1e6))
    else:
        sigma_p = np.full(len(theta), sigma_p, dtype=float)
    if sigma_s is None:
        sigma_s = {
            name: (2.38 / np.sqrt(len(t_grid))) * np.sqrt(precomp[name].noise_var + 1e-8) for name in var_names
        }
    else:
        sigma_s = {name: sigma_s for name in var_names}

    t_grid_t = torch.as_tensor(t_grid, dtype=dtype, device=device)

    def _eval_rhs(theta_np: np.ndarray, x_dict: "dict[str, np.ndarray]") -> "dict[str, np.ndarray]":
        x_stack = torch.stack(
            [torch.as_tensor(x_dict[name], dtype=dtype, device=device) for name in var_names], dim=1
        )  # (T, n_vars)
        const_ctx = torch.as_tensor(theta_np, dtype=dtype, device=device).unsqueeze(0).expand(len(t_grid), -1)
        f_vals = rhs(t_grid_t, x_stack, const_ctx).detach().cpu().numpy()  # (T, n_vars)
        return {name: f_vals[:, i] for i, name in enumerate(var_names)}

    def _log_target(theta_np: np.ndarray, x_dict: "dict[str, np.ndarray]") -> float:
        if np.any(theta_np < lo) or np.any(theta_np > hi):
            return -np.inf
        try:
            xdot = _eval_rhs(theta_np, x_dict)
        except Exception:
            return -np.inf
        total = 0.0
        for name in var_names:
            pc = precomp[name]
            xk = x_dict[name]
            if not np.all(np.isfinite(xk)):
                return -np.inf
            total += -0.5 * xk @ cho_solve(pc.K_chol, xk)
            total += -0.5 * np.sum((pc.y - xk) ** 2) / pc.noise_var
            resid = xdot[name] - pc.D @ xk
            if not np.all(np.isfinite(resid)):
                return -np.inf
            total += -0.5 * resid @ cho_solve(pc.A_chol, resid)
        return float(total)

    log_pi = _log_target(theta, x)
    n_theta = len(theta)
    n_total = n_burnin + n_mcmc
    kept_theta = []
    kept_x: "dict[str, list]" = {name: [] for name in var_names}
    acc_theta = acc_x = 0
    acc_theta_window = acc_x_window = 0

    for step in range(n_total):
        # --- block 1: propose the whole constant vector at once ---
        theta_new = theta + rng.normal(scale=sigma_p, size=n_theta)
        log_pi_new = _log_target(theta_new, x)
        if np.log(rng.uniform()) < log_pi_new - log_pi:
            theta, log_pi = theta_new, log_pi_new
            acc_theta += 1
            acc_theta_window += 1

        # --- block 2: propose each state's whole trajectory vector at once ---
        x_new = {name: x[name] + rng.normal(scale=sigma_s[name], size=len(t_grid)) for name in var_names}
        log_pi_new = _log_target(theta, x_new)
        if np.log(rng.uniform()) < log_pi_new - log_pi:
            x, log_pi = x_new, log_pi_new
            acc_x += 1
            acc_x_window += 1

        if step < n_burnin and (step + 1) % adapt_every == 0:
            rate_theta = acc_theta_window / adapt_every
            rate_x = acc_x_window / adapt_every
            sigma_p = sigma_p * np.exp(1.5 * (rate_theta - target_accept))
            sigma_s = {name: sigma_s[name] * np.exp(1.5 * (rate_x - target_accept)) for name in var_names}
            if verbose >= 2:
                print(f"[fgpgm] burn-in step {step+1}: accept theta={rate_theta:.2f} x={rate_x:.2f}")
            acc_theta_window = acc_x_window = 0

        if step >= n_burnin and (step - n_burnin) % thin == 0:
            kept_theta.append(theta.copy())
            for name in var_names:
                kept_x[name].append(x[name].copy())

    if verbose >= 1:
        print(f"[fgpgm] overall accept: theta={acc_theta/n_total:.3f} x={acc_x/n_total:.3f} kept={len(kept_theta)} samples")

    const_samples = np.stack(kept_theta)  # (n_kept, n_consts)
    consts_mean = const_samples.mean(axis=0)
    const_by_name = {name: float(consts_mean[c.index_in_ctx]) for name, c in model.consts.items()}
    x_samples = {name: np.stack(kept_x[name]) for name in var_names}

    return FGPGMResult(
        model=model,
        t=t_grid,
        consts=consts_mean,
        const_by_name=const_by_name,
        const_samples=const_samples,
        x_samples=x_samples,
        accept_rate_theta=acc_theta / n_total,
        accept_rate_x=acc_x / n_total,
        gamma=gamma,
    )


def sweep_gamma(
    model: InducedModel,
    t_eval,
    gammas: "list[float]" = (1.0, 0.1, 0.01, 1e-3, 1e-4),
    **kwargs,
) -> "dict[float, FGPGMResult]":
    """
    Runs `estimate_fgpgm` once per value in `gammas`, returns `{gamma: result}` - the equivalent of
    Wenk et al.'s own "tried 8 log-spaced values between 1 and 1e-4, picked by observation fit"
    (§5, sampling setup). Picking the "best" one is left to the caller (e.g. compare each result's
    constants' own forward-simulated fit against held-out data - the same `_test_mse`-style check
    used throughout `pybm.benchamark`) rather than baked in here, since "best" depends on what the
    caller is actually optimizing for.
    """
    return {gamma: estimate_fgpgm(model, t_eval, gamma=gamma, **kwargs) for gamma in gammas}
