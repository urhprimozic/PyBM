"""
Two-step gradient matching: a cheap initial guess for ODE constants.

Method
------
1. Fit an independent Gaussian process (squared-exponential / RBF kernel)
   to each observed endogenous variable's data, choosing hyperparameters
   by maximizing the GP marginal likelihood.
2. Because the derivative of a GP is itself a GP (for a differentiable
   kernel), read off the posterior mean of the state AND its derivative
   in closed form -- no finite differences on noisy data, no ODE solve.
3. Fit the model's constants `c` by plain least squares between this
   GP-implied derivative and the ODE right-hand side `f(t, x_hat(t), c)`
   evaluated at the GP-implied state. Still no ODE integration.

This is the classical "two-step" estimator (Varah, 1982), using a GP as
the smoother the way Calderhead, Girolami & Lawrence set it up:

    Calderhead, B., Girolami, M., & Lawrence, N. D. (2008).
    "Accelerating Bayesian Inference over Nonlinear Differential
    Equations with Gaussian Processes." NeurIPS 21.

The closed-form cross-covariances between a GP and its derivative
(used in `_rbf_kernel` / `_kernel_deriv_wrt_first` below) are from:

    Solak, E., Murray-Smith, R., Leithead, W. E., Leith, D. J., &
    Rasmussen, C. E. (2003). "Derivative observations in Gaussian
    process models of dynamic systems." NeurIPS 16.

Unlike Calderhead et al. (2008) -- and its "adaptive" fix, Dondelinger,
Filippone, Rogers & Husmeier, "ODE Parameter Inference using Adaptive
Gradient Matching with Gaussian Processes" (AISTATS 2013), which jointly
samples the GP hyperparameters and ODE parameters via MCMC so the ODE can
feed back into the smoothing -- this module is deliberately the plain,
un-adaptive two-step point estimate. It does not weight residuals by the
GP's posterior uncertainty, and it is not meant to be a final,
statistically calibrated fit: only a fast initializer for
`pybm.estimate.multishooting_torch.estimate_torch` (or any other proper
estimator in this package). On sparse/noisy data (see Dondelinger et al.,
Section 5) the plain two-step estimate can be well off -- that is
expected and fine here, since it only needs to point the real optimizer
in roughly the right direction.

Example
-------
>>> result = estimate_gradient_matching(model, t_eval)
>>> init_params = result.init_params(n_subintervals=10)
>>> fit = estimate_torch(model, t_eval, method="weighted_sum",
...                       n_subintervals=10, n_candidates=1,
...                       init_params=init_params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import least_squares, minimize as scipy_minimize
from torch.func import jacfwd

from pybm.estimate.multishooting_torch import uniform_sub_indices
from pybm.model import _make_rhs
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var

type Array = np.ndarray | torch.Tensor  # type alias for convenience

# ---------------------------------------------------------------------------
# RBF kernel and its derivative cross-covariances (Solak et al., 2003)
# ---------------------------------------------------------------------------


def _rbf_kernel(t_a: Array , t_b: Array, lengthscale: float, signal_var: float) -> Array:
    """Returns cov(f(t_a), f(t_b)) = k(t_a, t_b) for the RBF kernel."""
    diff = t_a[:, None] - t_b[None, :]
    return signal_var * np.exp(-0.5 * (diff / lengthscale) ** 2)



def _kernel_deriv_wrt_first(t_a: Array, t_b: Array, lengthscale: float, signal_var: float) -> Array:
    """cov(f'(t_a), f(t_b)) = d/dt_a k(t_a, t_b)."""
    diff = t_a[:, None] - t_b[None, :]
    k = signal_var * np.exp(-0.5 * (diff / lengthscale) ** 2)
    return -k * diff / lengthscale**2


# ---------------------------------------------------------------------------
# Per-variable GP fit
# ---------------------------------------------------------------------------


@dataclass
class _FittedGP:
    t_obs: np.ndarray
    y_obs: np.ndarray
    lengthscale: float
    signal_var: float
    noise_var: float
    alpha: np.ndarray  # (K + noise_var*I)^-1 y_obs, precomputed
    log_marginal_likelihood: float
    L: "np.ndarray | None" = None  # lower Cholesky of (K + noise_var*I) - see the *_var methods below

    def mean(self, t_star: np.ndarray) -> np.ndarray:
        k_star = _rbf_kernel(t_star, self.t_obs, self.lengthscale, self.signal_var)
        return k_star @ self.alpha

    def mean_deriv(self, t_star: np.ndarray) -> np.ndarray:
        k_star_deriv = _kernel_deriv_wrt_first(t_star, self.t_obs, self.lengthscale, self.signal_var)
        return k_star_deriv @ self.alpha

    def _v(self, k_star: np.ndarray) -> np.ndarray:
        """Solves `L @ v = k_star^T` - the "explained by training data" projection shared by every
        posterior-(co)variance query below. `k_star` is `(m, n_obs)`; returns `(n_obs, m)`."""
        if self.L is None:
            raise ValueError(
                "This _FittedGP has no stored Cholesky factor (L=None) - only GPs fit by the "
                "current _fit_gp_1d support variance queries; an older pickled result predates this."
            )
        return solve_triangular(self.L, k_star.T, lower=True)

    def var(self, t_star: np.ndarray) -> np.ndarray:
        """
        Posterior variance of `x̂(t*)` itself: `k(t*,t*) - k(t*)^T (K+σ_n²I)^-1 k(t*)`, computed as
        `signal_var - ||v(t*)||²` via the stored Cholesky (`v = L^-1 k(t*)^T`) rather than forming
        the full posterior covariance matrix - see `notes/integral-matching.md`'s identifiability
        section for the same "read the fit's own uncertainty off its Cholesky factor" idea applied
        to constants instead of the interpolant. Floored at `noise_var`, not an arbitrary small
        constant: the posterior CAN legitimately approach exactly 0 right at a densely-observed
        point (correct - the GP really is that sure about the smoothed MEAN there), but a residual
        weight of `1/sqrt(var)` computed from that would blow up numerically and let a handful of
        near-zero-variance points dominate a `weight_by_uncertainty=True` fit (checked directly:
        unweighted `own_fit_cost`~500-9500 on real Bled data, with only a `1e-12` floor it jumped to
        `~1e7-1e8` and `test_mse` got WORSE, not better) - `noise_var` is the natural stopping point
        because you can never really be more certain about a NEW residual than the data's own
        measurement noise allows, regardless of how much smoothing the GP's mean benefits from.
        """
        k_star = _rbf_kernel(t_star, self.t_obs, self.lengthscale, self.signal_var)  # (m, n_obs)
        v = self._v(k_star)  # (n_obs, m)
        return np.maximum(self.signal_var - np.sum(v**2, axis=0), self.noise_var)

    def var_deriv(self, t_star: np.ndarray) -> np.ndarray:
        """
        Posterior variance of `x̂'(t*)` - same construction as `var`, but with the derivative cross-
        kernel (`_kernel_deriv_wrt_first`, already used by `mean_deriv`) in place of `k(t*,·)`, and
        the RBF prior derivative variance `signal_var/lengthscale²` (Rasmussen & Williams §9.4, the
        `t=t'` value of `∂²k/∂t∂t'`) in place of `k(t*,t*)`. This is exactly the quantity that
        justifies gradient matching's own noise-floor claim (`notes/integral-matching.md` §6.3,
        `‖e'(t)‖ ~ ε/ℓ`) - a LARGE `var_deriv` at some `t*` is the GP itself telling you its
        derivative estimate there is untrustworthy (e.g. too little nearby data, or right at the
        edge of the observed span), independent of how confident `var` is about the function value
        there. Floored at `noise_var/lengthscale²` - the dimensionally-matched derivative-scale
        analogue of `var`'s own `noise_var` floor (same reasoning: see `var`'s docstring).
        """
        k_star_deriv = _kernel_deriv_wrt_first(t_star, self.t_obs, self.lengthscale, self.signal_var)  # (m, n_obs)
        v = self._v(k_star_deriv)  # (n_obs, m)
        prior_deriv_var = self.signal_var / self.lengthscale**2
        return np.maximum(prior_deriv_var - np.sum(v**2, axis=0), self.noise_var / self.lengthscale**2)

    def window_var(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """
        Posterior variance of the ACCUMULATED CHANGE `x̂(ends[i]) - x̂(starts[i])`, one value per
        paired `(starts[i], ends[i])` - the quantity integral matching's own residual target
        (`x̂(t_{i+1})-x̂(t_i)`, see `pybm.estimate.integral_matching`'s Eq. 2.1) actually needs a
        weight for, not the pointwise `var` at either endpoint alone: the two endpoints are
        CORRELATED under the same GP posterior (`Cov[x̂(a),x̂(b)] != 0` whenever `|a-b|` isn't much
        larger than `lengthscale`), so `Var[x̂(b)-x̂(a)] = Var[x̂(b)] + Var[x̂(a)] - 2·Cov[x̂(a),x̂(b)]`
        is NOT just `var(a) + var(b)` - ignoring the covariance term overstates the true uncertainty
        of a NARROW window (where `a` and `b` are nearly the same random variable, so their
        difference is much better-determined than either endpoint alone) and would push
        `stride="auto"`-scale windows toward being weighted as if they were far noisier than they
        actually are.

        Derivation (posterior covariance `Cov_post(t,t') = k(t,t') - v(t)^T v(t')` with `v(t) =
        L^-1 k(t,·)^T`, `k(t,t) = signal_var` constant for a stationary kernel):
            Var[x̂(b)-x̂(a)] = 2·signal_var - 2·k(a,b) - ‖v(a)-v(b)‖²

        Floored at `noise_var` - same reasoning as `var`'s own floor (see its docstring); a window
        collapsing to `Δ→0` makes this quantity legitimately approach 0 too, and needs the same
        numerical stop.
        """
        k_ab = self.signal_var * np.exp(-0.5 * ((starts - ends) / self.lengthscale) ** 2)  # (N,), paired diag
        v_a = self._v(_rbf_kernel(starts, self.t_obs, self.lengthscale, self.signal_var))  # (n_obs, N)
        v_b = self._v(_rbf_kernel(ends, self.t_obs, self.lengthscale, self.signal_var))  # (n_obs, N)
        var = 2 * self.signal_var - 2 * k_ab - np.sum((v_a - v_b) ** 2, axis=0)
        return np.maximum(var, self.noise_var)


def _qualified_var_names(model: InducedModel) -> "dict[int, str]":
    """Maps `id(var) -> model.vars`' dotted key for it (e.g. "phyto.conc", "ortp.conc").

    `var.name` alone is only the LOCAL template attribute name ("conc") and collides whenever a
    library reuses the same attribute name across multiple entities - exactly what happens in the
    Bled model, where `ortp.conc`, `no.conc`, `silica.conc`, `daph.conc` and `phyto.conc` all have
    `var.name == "conc"`. The `gps` dict below is keyed by this qualified name instead, so each
    entity's own observed data stays matched to its own GP.
    """
    return {id(var): name for name, var in model.vars.items()}


def _series_as_numpy(t: Any, x: Any) -> tuple[np.ndarray, np.ndarray]:
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(t, dtype=float), np.asarray(x, dtype=float)


def _fit_gp_1d(t_obs: np.ndarray, y_obs: np.ndarray, max_points: int, max_iter: int, jitter: float = 1e-8) -> _FittedGP:
    if len(t_obs) > max_points:
        # GP fitting is O(n^3); subsample evenly for the (repeated) marginal
        # likelihood evaluations. The posterior mean/derivative queries
        # later are cheap (O(n_train * n_query)) and use this same subset.
        idx = np.linspace(0, len(t_obs) - 1, max_points).astype(int)
        t_obs, y_obs = t_obs[idx], y_obs[idx]

    t_span = max(t_obs.max() - t_obs.min(), 1e-6)
    y_std = max(y_obs.std(), 1e-6)
    log0 = np.log([t_span / 10.0, y_std, 0.05 * y_std + 1e-6])

    def neg_log_marginal_likelihood(log_params: np.ndarray) -> float:
        lengthscale, signal_std, noise_std = np.exp(log_params)
        K = _rbf_kernel(t_obs, t_obs, lengthscale, signal_std**2)
        K[np.diag_indices_from(K)] += noise_std**2 + jitter
        try:
            L = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve(L, y_obs)
        nll = 0.5 * y_obs @ alpha + np.sum(np.log(np.diag(L[0]))) + 0.5 * len(t_obs) * np.log(2 * np.pi)
        return float(nll)

    res = scipy_minimize(
        neg_log_marginal_likelihood,
        log0,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4},
    )
    lengthscale, signal_std, noise_std = np.exp(res.x)
    signal_var, noise_var = float(signal_std**2), float(noise_std**2)

    K = _rbf_kernel(t_obs, t_obs, lengthscale, signal_var)
    K[np.diag_indices_from(K)] += noise_var + jitter
    L_factor = cho_factor(K, lower=True)
    alpha = cho_solve(L_factor, y_obs)

    return _FittedGP(
        t_obs=t_obs,
        y_obs=y_obs,
        lengthscale=float(lengthscale),
        signal_var=signal_var,
        noise_var=noise_var,
        alpha=alpha,
        log_marginal_likelihood=-float(res.fun),
        L=np.tril(L_factor[0]),  # cho_factor may leave garbage above the diagonal - zero it out
    )


# ---------------------------------------------------------------------------
# Constant fit: least squares against the GP-implied derivative, no ODE solve
# ---------------------------------------------------------------------------


def _initial_const_guess(model: InducedModel) -> np.ndarray:
    """
    Starting guess for the constant fit: `initial_value` where known, otherwise the midpoint of
    `range` (falling back to 0.0 only when neither is available). A blanket 0.0 default (as
    `int_scipy.get_initial_const_ctx` uses) can land a constant that appears as a divisor in some
    equation (e.g. a `refTemp`) at exactly zero, which makes the very first residual evaluation
    NaN/inf - `least_squares` refuses to even start from a point like that.
    """
    c0 = np.zeros(len(model.consts), dtype=float)
    for const in model.consts.values():
        if const.initial_value is not None:
            c0[const.index_in_ctx] = const.initial_value
        elif const.range is not None and np.isfinite(const.range[0]) and np.isfinite(const.range[1]):
            c0[const.index_in_ctx] = 0.5 * (const.range[0] + const.range[1])
        else:
            c0[const.index_in_ctx] = 0.0
    return c0


def _const_bounds(model: InducedModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(model.consts), -np.inf)
    hi = np.full(len(model.consts), np.inf)
    for const in model.consts.values():
        if const.range is not None:
            lo[const.index_in_ctx] = const.range[0]
            hi[const.index_in_ctx] = const.range[1]
    return lo, hi


def _identifiability_report(jac: np.ndarray, const_names: "list[str]", threshold: float = 1e-3) -> dict:
    """
    Practical-identifiability diagnostic (see `notes/integral-matching.md` §4): the Gauss-Newton
    Hessian `G = J_r^T J_r` at the fitted solution (`J_r` = the residual Jacobian scipy's
    `least_squares` already computed to converge - `result.jac` - no extra model evaluations) plays
    the role of a Fisher information matrix for this residual. A near-zero eigenvalue of `G` means
    the standard error along that eigenvector is huge (Eq. 4.4: `|c-c0| ~ sqrt(noise/lambda)`) - a
    combination of constants this fit's OWN residual cannot pin down, independent of how small the
    residual itself got. Computed via SVD of `jac` directly (`G`'s eigenvalues are `singular_values
    ** 2`, its eigenvectors are `jac`'s right singular vectors) rather than forming `G` explicitly,
    for numerical stability - `jac` is often close to rank-deficient exactly in the cases this
    report exists to catch, and squaring first would double `G`'s already-poor condition number.

    Parameters
    ----------
    jac : np.ndarray
        Shape `(n_residuals, n_consts)` - `least_squares_result.jac` from either
        `GradientMatchingResult` or `IntegralMatchingResult`.
    const_names : list[str]
        Qualified constant names, in `const.index_in_ctx` order (same order `jac`'s columns use).
    threshold : float, optional
        A singular value below `threshold * max(singular values)` is flagged as poorly identified.
        Default `1e-3`, i.e. more than ~1000x worse-determined than the best-identified direction -
        a practical, not universal, cutoff; inspect `singular_values`/`condition_number` directly
        for a stricter or looser call.

    Returns
    -------
    dict
        `condition_number` (`s_max/s_min`, `inf` if `s_min` is exactly 0), `singular_values`
        (descending), `flagged` - a list of `{"singular_value", "relative_scale", "combination"}`
        dicts, one per poorly-identified direction, `"combination"` a `{const_name: weight}` dict
        (that direction's right singular vector, sign-normalized so the largest-magnitude weight is
        positive, restricted to weights with `|weight| > 0.1` and sorted by `|weight|` descending) -
        e.g. `{"daph.maxFiltrationRate": 0.71, "feedsOn.phytoLim472.halfSaturation": -0.70}` reads as
        "this fit cannot separate these two constants, only a ~1:1 combination of them".
    """
    U, s, Vt = np.linalg.svd(jac, full_matrices=False)
    s_max = float(s[0]) if len(s) else 0.0
    condition_number = float(s_max / s[-1]) if len(s) and s[-1] > 0 else float("inf")

    flagged = []
    for sigma, v in zip(s, Vt):
        if s_max > 0 and sigma < threshold * s_max:
            dominant_idx = int(np.argmax(np.abs(v)))
            v = v if v[dominant_idx] >= 0 else -v
            combo = {
                const_names[i]: float(v[i])
                for i in np.argsort(-np.abs(v))
                if abs(v[i]) > 0.1
            }
            flagged.append({
                "singular_value": float(sigma),
                "relative_scale": float(sigma / s_max) if s_max > 0 else 0.0,
                "combination": combo,
            })
    return {
        "condition_number": condition_number,
        "singular_values": s.tolist(),
        "flagged": flagged,
    }


def _fit_constants(
    model: InducedModel,
    vars_: list[Var],
    state_at_collocation: np.ndarray,  # (n_vars, n_col)
    deriv_at_collocation: np.ndarray,  # (n_vars, n_col)
    collocation_times: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
    deriv_var_at_collocation: "np.ndarray | None" = None,  # (n_vars, n_col), see weight_by_uncertainty
) -> Any:
    # `vars_` here is the same state-only (differential) list `estimate_gradient_matching` built
    # via `_split_endo_vars` - algebraic/frozen variables still need to be handed to `_make_rhs`
    # so any state var's equation that reads one of them (e.g. growth reading a temperature-
    # limitation factor) sees a correct, settled value instead of nothing.
    _, algebraic_vars, frozen_values = model.split_endo_vars()
    rhs = _make_rhs(vars_, algebraic_vars, frozen_values)

    y_col = torch.as_tensor(state_at_collocation.T, dtype=dtype, device=device)  # (n_col, n_vars)
    target = torch.as_tensor(deriv_at_collocation.T, dtype=dtype, device=device)  # (n_col, n_vars)
    t_col = torch.as_tensor(collocation_times, dtype=dtype, device=device)  # (n_col,)
    n_col = y_col.shape[0]

    # `weight[t,v] = 1/sqrt(Var[x̂'(t)])` - a collocation point where the GP itself is unsure about
    # the derivative (sparse/edge-of-data region - see `_FittedGP.var_deriv`) contributes LESS to
    # the fit, the same "trust the interpolant less where it's less trustworthy" idea integral
    # matching's `weight_by_uncertainty` uses (`notes/integral-matching.md` never derives this for
    # gradient matching explicitly, but the construction is the direct analogue: `var_deriv` plays
    # the role `window_var` plays there). `None` (default) reproduces the original unweighted fit
    # exactly - this is opt-in, not a silent behavior change.
    weight = None
    if deriv_var_at_collocation is not None:
        weight = torch.as_tensor(
            1.0 / np.sqrt(np.maximum(deriv_var_at_collocation.T, 1e-12)), dtype=dtype, device=device
        )  # (n_col, n_vars)

    def residual_fn(c: torch.Tensor) -> torch.Tensor:
        const_ctx = c.unsqueeze(0).expand(n_col, -1)  # same constants at every collocation point
        pred = rhs(t_col, y_col, const_ctx)
        residual = pred - target
        if weight is not None:
            residual = residual * weight
        return residual.reshape(-1)

    def fun(c_np: np.ndarray) -> np.ndarray:
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return residual_fn(c).detach().cpu().numpy()

    def jac(c_np: np.ndarray) -> np.ndarray:
        # n_consts is small and n_col*n_vars can be large, so forward-mode
        # AD (cost scales with #inputs, not #outputs) is the right tool.
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return jacfwd(residual_fn)(c).detach().cpu().numpy()

    c0 = _initial_const_guess(model)
    lo, hi = _const_bounds(model)
    c0 = np.clip(c0, lo, hi)

    return least_squares(
        fun, x0=c0, jac=jac, bounds=(lo, hi), method="trf", ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class GradientMatchingResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray  # (n_consts,), ordered by const.index_in_ctx
    const_by_name: dict[str, float]
    collocation_times: np.ndarray
    state_at_collocation: np.ndarray  # (n_vars, n_col), GP posterior mean
    deriv_at_collocation: np.ndarray  # (n_vars, n_col), GP posterior mean derivative
    gps: dict[str, _FittedGP]
    least_squares_result: Any

    def init_params(
        self,
        n_subintervals: int = 1,
        n_candidates: int = 1,
        jitter: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Build a `[consts..., s_0, ..., s_{K-1}]` tensor in the layout
        `estimate_torch` expects, using this fit's constants and the GP
        posterior mean (evaluated at each subinterval's start time) as
        the shooting-node seeds.
        """
        device = device or torch.device("cpu")
        vars_, _, _ = self.model.split_endo_vars()  # state vars only - see estimate_torch's init_params layout
        qualified = _qualified_var_names(self.model)

        sub_indices = uniform_sub_indices(self.t_eval, n_subintervals)
        seed_times = self.t_eval[sub_indices[:-1]]

        seeds = np.zeros((n_subintervals, len(vars_)), dtype=float)
        for i, var in enumerate(vars_):
            seeds[:, i] = self.gps[qualified[id(var)]].mean(seed_times)

        flat = np.concatenate([self.consts, seeds.reshape(-1)])
        params = torch.as_tensor(flat, dtype=dtype, device=device).unsqueeze(0).repeat(n_candidates, 1)

        if jitter and n_candidates > 1:
            rng = np.random.default_rng(seed)
            noise = rng.normal(scale=jitter, size=(n_candidates, params.shape[1]))
            noise[0, :] = 0.0
            params = params + torch.as_tensor(noise, dtype=dtype, device=device)

        return params

    def identifiability_report(self, threshold: float = 1e-3) -> dict:
        """See `pybm.estimate.gradient_matching._identifiability_report` - uses this fit's own
        `least_squares_result.jac` (already computed, no extra evaluations)."""
        const_names: "list[str]" = [""] * len(self.model.consts)
        for name, c in self.model.consts.items():
            const_names[c.index_in_ctx] = name
        return _identifiability_report(self.least_squares_result.jac, const_names, threshold=threshold)


def fit_gps(
    vars_: "dict[str, Var]", max_gp_points: int = 300, max_gp_iter: int = 200, verbose: int = 0
) -> "dict[str, _FittedGP]":
    """
    Fits one GP per variable in `vars_`, keyed by the SAME dotted name given here (e.g.
    `model.vars`' key, like "phyto.conc") - step 1 of gradient matching, split out on its own
    because it depends ONLY on a variable's own observed data (`var.data`), never on a model's
    equations/structure. That makes it safe (and, across a structural search over many induced
    models sharing the same underlying data, a large amount of redundant work to skip) to fit ONCE
    up front and reuse across every candidate model - see `estimate_gradient_matching`'s `gps`
    argument and `pybm.estimate.estimate.estimate_model`, which does exactly that.

    Takes a `dict`, not a bare `list[Var]`, specifically so the caller supplies a key that's
    actually unique per variable - `var.name` alone is only the local template attribute name
    ("conc"), which multiple different entities can share (see `_qualified_var_names`); keying by
    that would silently let one entity's GP clobber another's.
    """
    gps: dict[str, _FittedGP] = {}
    for name, var in vars_.items():
        if var.data is None:
            raise ValueError(f"Variable {name} has no data; gradient matching needs an observed trajectory.")
        t_obs, y_obs = _series_as_numpy(var.data.t, var.data.x)
        gps[name] = _fit_gp_1d(t_obs, y_obs, max_points=max_gp_points, max_iter=max_gp_iter)
        if verbose>= 2:
            gp = gps[name]
            print(
                f"[gradient_matching] {name}: lengthscale={gp.lengthscale:.4g} "
                f"signal_std={gp.signal_var**0.5:.4g} noise_std={gp.noise_var**0.5:.4g} "
                f"log_marginal_likelihood={gp.log_marginal_likelihood:.4g}"
            )
    return gps


def estimate_gradient_matching(
    model: InducedModel,
    t_eval,
    collocation_times=None,
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    gps: "dict[str, _FittedGP] | None" = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: int = 0,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
    weight_by_uncertainty: bool = False,
) -> GradientMatchingResult:
    """
    Two-step gradient-matching initial guess for a model's constants.

    Parameters
    ----------
    model : InducedModel
        Must have `engine == "torch"` with every `Var.ode` written in
        differentiable torch ops (same precondition as `estimate_torch`).
        Exogenous variables' data must already be `TimeSeries.to_torch()`-ed.
    t_eval : array-like
        Not fit against directly -- only used as the default collocation
        grid and to lay out `init_params`' shooting-node seed times.
    collocation_times : array-like, optional
        Where the GP-implied state/derivative are evaluated for the
        constant fit. Defaults to `t_eval`.
    max_gp_points : int
        GP hyperparameter fitting is O(n^3); observed series longer than
        this are subsampled evenly before fitting (posterior queries
        afterwards stay cheap and use the full collocation grid). Ignored for any variable already
        covered by `gps`.
    gps : dict[str, _FittedGP], optional
        Precomputed GPs (see `fit_gps`), keyed by variable name - step 1 (the GP fit) is skipped
        for any state variable already present here, and only fit fresh for the rest. Pass this
        when calling `estimate_gradient_matching` many times over structurally different models
        that share the same underlying observed data (e.g. a structural search over many induced
        models from the same `incomplete_model`) - step 1 doesn't depend on the model's equations
        at all, only on each variable's own data, so redoing it per model is pure waste.
    device : torch.device, optional
        Where step 3's constant fit runs (`torch.device("cuda")` to use a GPU) - `model` is
        switched onto this same device before anything else, so exogenous variables' data ends up
        there too (see `InducedModel.switch_engine`); without that, a state/const tensor on
        `device` and an exogenous variable's data left on CPU would collide the moment an equation
        reads both (e.g. `env.temperature / refTemp`). Step 1 (the GP fit itself, plain numpy/
        scipy - `_fit_gp_1d`) always runs on CPU regardless of `device` - it's a small (at most
        `max_gp_points` x `max_gp_points`) Cholesky factorization per variable, not worth a GPU
        round-trip, and rewriting it in torch is its own separate undertaking.
    ftol, xtol, gtol, max_nfev : optional
        Passed straight through to `scipy.optimize.least_squares` (step 3, the constant fit) -
        loosen these (larger tol, smaller `max_nfev`) to stop earlier at a rougher answer, which
        is often fine here since this whole function is only a fast initializer for a proper
        estimator anyway (see the module docstring).
    weight_by_uncertainty : bool, optional
        Default `False` (reproduces the original unweighted fit exactly - opt-in, not a behavior
        change to existing callers). When `True`, each collocation point's residual is divided by
        the GP's own posterior standard deviation of the DERIVATIVE there (`_FittedGP.var_deriv`) -
        a point where the GP itself is unsure about `x̂'(t)` (sparse data nearby, or near the edge
        of the observed span) contributes less to the fit, instead of being trusted equally with a
        well-supported point. This is a genuine change to what the fit is doing (heteroscedastic /
        generalized least squares instead of ordinary least squares), not just a diagnostic - if in
        doubt, compare `test_mse` with and without on your own data before switching a real pipeline
        over.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_gradient_matching requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)  # exogenous data must live on `device` too
    t_eval = np.asarray(t_eval, dtype=float)
    collocation_times = np.asarray(t_eval if collocation_times is None else collocation_times, dtype=float)

    vars_, _, _ = model.split_endo_vars()  # only differential ("state") vars have a trajectory to GP-fit
    qualified = _qualified_var_names(model)
    gps = dict(gps) if gps is not None else {}
    missing = {qualified[id(var)]: var for var in vars_ if qualified[id(var)] not in gps}
    if missing:
        gps.update(fit_gps(missing, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose))

    state_at_collocation = np.stack([gps[qualified[id(var)]].mean(collocation_times) for var in vars_])
    deriv_at_collocation = np.stack([gps[qualified[id(var)]].mean_deriv(collocation_times) for var in vars_])
    deriv_var_at_collocation = None
    if weight_by_uncertainty:
        deriv_var_at_collocation = np.stack(
            [gps[qualified[id(var)]].var_deriv(collocation_times) for var in vars_]
        )

    result = _fit_constants(
        model, vars_, state_at_collocation, deriv_at_collocation, collocation_times, device, dtype,
        ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
        deriv_var_at_collocation=deriv_var_at_collocation,
    )
    if verbose >= 2:
        print(f"[gradient_matching] constant fit: cost={result.cost:.4g} success={result.success}")

    const_by_name = {name: float(result.x[c.index_in_ctx]) for name, c in model.consts.items()}

    return GradientMatchingResult(
        model=model,
        t_eval=t_eval,
        consts=result.x,
        const_by_name=const_by_name,
        collocation_times=collocation_times,
        state_at_collocation=state_at_collocation,
        deriv_at_collocation=deriv_at_collocation,
        gps=gps,
        least_squares_result=result,
    )
