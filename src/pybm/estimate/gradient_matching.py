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
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize as scipy_minimize
from torch.func import jacfwd

from pybm.estimate.int_scipy import get_initial_const_ctx
from pybm.estimate.multishooting_torch import _make_rhs, uniform_sub_indices
from pybm.model import Model, Var

# ---------------------------------------------------------------------------
# RBF kernel and its derivative cross-covariances (Solak et al., 2003)
# ---------------------------------------------------------------------------


def _rbf_kernel(t_a: np.ndarray, t_b: np.ndarray, lengthscale: float, signal_var: float) -> np.ndarray:
    """cov(f(t_a), f(t_b))."""
    diff = t_a[:, None] - t_b[None, :]
    return signal_var * np.exp(-0.5 * (diff / lengthscale) ** 2)


def _kernel_deriv_wrt_first(t_a: np.ndarray, t_b: np.ndarray, lengthscale: float, signal_var: float) -> np.ndarray:
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

    def mean(self, t_star: np.ndarray) -> np.ndarray:
        k_star = _rbf_kernel(t_star, self.t_obs, self.lengthscale, self.signal_var)
        return k_star @ self.alpha

    def mean_deriv(self, t_star: np.ndarray) -> np.ndarray:
        k_star_deriv = _kernel_deriv_wrt_first(t_star, self.t_obs, self.lengthscale, self.signal_var)
        return k_star_deriv @ self.alpha


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
    L = cho_factor(K, lower=True)
    alpha = cho_solve(L, y_obs)

    return _FittedGP(
        t_obs=t_obs,
        y_obs=y_obs,
        lengthscale=float(lengthscale),
        signal_var=signal_var,
        noise_var=noise_var,
        alpha=alpha,
        log_marginal_likelihood=-float(res.fun),
    )


# ---------------------------------------------------------------------------
# Constant fit: least squares against the GP-implied derivative, no ODE solve
# ---------------------------------------------------------------------------


def _const_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(model.const_index), -np.inf)
    hi = np.full(len(model.const_index), np.inf)
    for const in model.consts.values():
        if const.range is not None:
            lo[const.index_in_ctx] = const.range[0]
            hi[const.index_in_ctx] = const.range[1]
    return lo, hi


def _fit_constants(
    model: Model,
    vars_: list[Var],
    state_at_collocation: np.ndarray,  # (n_vars, n_col)
    deriv_at_collocation: np.ndarray,  # (n_vars, n_col)
    collocation_times: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    rhs = _make_rhs(vars_)

    y_col = torch.as_tensor(state_at_collocation.T, dtype=dtype, device=device)  # (n_col, n_vars)
    target = torch.as_tensor(deriv_at_collocation.T, dtype=dtype, device=device)  # (n_col, n_vars)
    t_col = torch.as_tensor(collocation_times, dtype=dtype, device=device)  # (n_col,)
    n_col = y_col.shape[0]

    def residual_fn(c: torch.Tensor) -> torch.Tensor:
        const_ctx = c.unsqueeze(0).expand(n_col, -1)  # same constants at every collocation point
        pred = rhs(t_col, y_col, const_ctx)
        return (pred - target).reshape(-1)

    def fun(c_np: np.ndarray) -> np.ndarray:
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return residual_fn(c).detach().cpu().numpy()

    def jac(c_np: np.ndarray) -> np.ndarray:
        # n_consts is small and n_col*n_vars can be large, so forward-mode
        # AD (cost scales with #inputs, not #outputs) is the right tool.
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return jacfwd(residual_fn)(c).detach().cpu().numpy()

    c0 = get_initial_const_ctx(model)
    lo, hi = _const_bounds(model)
    c0 = np.clip(c0, lo, hi)

    return least_squares(fun, x0=c0, jac=jac, bounds=(lo, hi), method="trf")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class GradientMatchingResult:
    model: Model
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
        vars_ = self.model.get_endo_variables()

        sub_indices = uniform_sub_indices(self.t_eval, n_subintervals)
        seed_times = self.t_eval[sub_indices[:-1]]

        seeds = np.zeros((n_subintervals, len(vars_)), dtype=float)
        for i, var in enumerate(vars_):
            seeds[:, i] = self.gps[var.name].mean(seed_times)

        flat = np.concatenate([self.consts, seeds.reshape(-1)])
        params = torch.as_tensor(flat, dtype=dtype, device=device).unsqueeze(0).repeat(n_candidates, 1)

        if jitter and n_candidates > 1:
            rng = np.random.default_rng(seed)
            noise = rng.normal(scale=jitter, size=(n_candidates, params.shape[1]))
            noise[0, :] = 0.0
            params = params + torch.as_tensor(noise, dtype=dtype, device=device)

        return params


def estimate_gradient_matching(
    model: Model,
    t_eval,
    collocation_times=None,
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: int = 0,
) -> GradientMatchingResult:
    """
    Two-step gradient-matching initial guess for a model's constants.

    Parameters
    ----------
    model : Model
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
        afterwards stay cheap and use the full collocation grid).
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_gradient_matching requires a Model built with engine='torch', got engine={model.engine!r}."
        )

    device = device or torch.device("cpu")
    t_eval = np.asarray(t_eval, dtype=float)
    collocation_times = np.asarray(t_eval if collocation_times is None else collocation_times, dtype=float)

    vars_ = model.get_endo_variables()
    gps: dict[str, _FittedGP] = {}
    for var in vars_:
        if var.data is None:
            raise ValueError(f"Variable {var.name} has no data; gradient matching needs an observed trajectory.")
        t_obs, y_obs = _series_as_numpy(var.data.t, var.data.x)
        gps[var.name] = _fit_gp_1d(t_obs, y_obs, max_points=max_gp_points, max_iter=max_gp_iter)
        if verbose:
            gp = gps[var.name]
            print(
                f"[gradient_matching] {var.name}: lengthscale={gp.lengthscale:.4g} "
                f"signal_std={gp.signal_var**0.5:.4g} noise_std={gp.noise_var**0.5:.4g} "
                f"log_marginal_likelihood={gp.log_marginal_likelihood:.4g}"
            )

    state_at_collocation = np.stack([gps[var.name].mean(collocation_times) for var in vars_])
    deriv_at_collocation = np.stack([gps[var.name].mean_deriv(collocation_times) for var in vars_])

    result = _fit_constants(
        model, vars_, state_at_collocation, deriv_at_collocation, collocation_times, device, dtype
    )
    if verbose:
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
