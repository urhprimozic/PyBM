"""
Function-approximation integral matching: the same loss as `pybm.estimate.integral_matching`,
```
loss(c) = Σ_i || [x̂(t_{i+1}) - x̂(t_i)] - ∫_{t_i}^{t_{i+1}} f(τ, x̂(τ), c) dτ ||²
```
computed a different way - one that decouples cost from the NUMBER of windows.

Why: where the cost comes from, and how this avoids it
----------------------------------------------------------
`integral_matching.py` evaluates `f` fresh at a few quadrature nodes PER WINDOW. That means the
computational graph `jacfwd` differentiates (to get the Jacobian w.r.t. `c`) grows with
`N_windows × quadrature_order` - and cost scales with it, since `jacfwd`'s own cost is (roughly)
`n_consts` forward passes through that graph. Since using MANY, densely overlapping windows is
exactly what the "integral matching" design calls for (see `integral_matching.py`'s module
docstring - it tightens both the anti-overfitting argument and the Theorem-3.1-style bound to the
true trajectory loss), this per-window quadrature cost is the thing to remove.

The trick: `f`'s own values, evaluated at a FIXED, shared set of sample points `t_sample` (fixed
across every candidate `c` - only the sampled VALUES change), are treated as noiseless
observations of a smooth function of `t` and approximated by an RBF-kernel GP too - not because
`f` is noisy (it isn't: it's a deterministic function of `c`, so this GP's `noise_var` is set to
~0, just a numerical jitter), but because a GP is a cheap way to get a closed-form INTEGRAL of a
sampled function over ANY window (the same "derivative of a GP is a GP" fact used in
`gradient_matching.py`, applied to integration instead - see `_rbf_integral_kernel`).

Since `t_sample` and the kernel hyperparameters are fixed across every candidate `c`:
- The Cholesky factor of `K(t_sample, t_sample)` is computed ONCE, reused by every candidate -
  only a cheap triangular solve (`α_c = K⁻¹y_c`) is needed per candidate, not a fresh O(m³) fit.
- The `(N_windows, m_samples)` cross-covariance matrix `K_∫` between "integral of `f` over window
  i" and "`f` at sample j" depends only on window boundaries, sample locations and the kernel - NOT
  on `c` - so it too is computed ONCE, before the search over candidates even starts.

Per candidate, the whole computation collapses to:
    y_c = f(x̂(t_sample), t_sample, c)     # m evaluations of f - the only step depending on c
    α_c = cho_solve(L, y_c)                # cheap, L is the precomputed, reused factor
    all window integrals = K_∫ @ α_c       # one matrix-vector product - EVERY window at once
`jacfwd` only ever differentiates through the `m`-sized `y_c` computation - the rest is a fixed
linear map, essentially free to differentiate through. Cost is therefore `O(m)`, independent of
how many (or how densely overlapping) windows `N_windows` you use.

Whether this is worth the extra complexity over `integral_matching.py`'s plain per-window
quadrature depends on whether `f`-evaluation-inside-the-optimizer's-Jacobian is actually the
bottleneck for a given model - see the two modules' docstrings together before choosing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import erf
from torch.func import jacfwd

from pybm.estimate.depr.gradient_matching import (
    _FittedGP,
    _const_bounds,
    _initial_const_guess,
    _qualified_var_names,
    _rbf_kernel,
    fit_gps,
)
from pybm.estimate.integral_matching import _build_windows
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _make_rhs


def _rbf_integral_kernel(a: np.ndarray, b: np.ndarray, t_sample: np.ndarray, lengthscale: float) -> np.ndarray:
    """
    `Cov(∫_a^b f(s)ds, f(t))` for an RBF-kernel GP with unit signal variance, closed form via the
    error function.

    Derivation: substituting `u=(s-t)/ℓ` turns `∫_a^b exp(-0.5((s-t)/ℓ)²)ds` into a Gaussian-CDF
    difference, `ℓ√(π/2)[erf((b-t)/(ℓ√2)) - erf((a-t)/(ℓ√2))]`.

    Signal variance is fixed to `1` here (see `_fit_gp_surrogate_setup`'s docstring for why it
    cancels out of the final prediction and so never needs to be a free parameter).

    Parameters
    ----------
    a, b : np.ndarray
        Shape `(N,)`: window start/end times.
    t_sample : np.ndarray
        Shape `(m,)`: the fixed sample points.
    lengthscale : float
        RBF kernel lengthscale.

    Returns
    -------
    np.ndarray
        Shape `(N, m)`.
    """
    scale = lengthscale * np.sqrt(2.0)
    z_b = (b[:, None] - t_sample[None, :]) / scale  # (N, m)
    z_a = (a[:, None] - t_sample[None, :]) / scale
    return lengthscale * np.sqrt(np.pi / 2.0) * (erf(z_b) - erf(z_a))


def _default_lengthscale(t_sample: np.ndarray) -> float:
    """
    A simple, defensible default lengthscale for `f`'s own surrogate GP: twice the median spacing
    between sample points - wide enough that neighboring samples meaningfully inform each other
    (smooth interpolation, not a disconnected nearest-neighbor lookup), narrow enough to still
    track genuine local variation in `f` across the sampled domain.

    Repeated timestamps (e.g. several noisy replicate observations sharing the same nominal time
    grid) are handled by ignoring zero-width gaps rather than letting them drag the median to `0`
    - a `0` lengthscale would degenerate the kernel to an identity matrix and, worse, divide by
    zero in `_rbf_integral_kernel`.

    Parameters
    ----------
    t_sample : np.ndarray
        Shape `(m,)`, increasing.

    Returns
    -------
    float
    """
    spacing = np.diff(np.sort(t_sample))
    spacing = spacing[spacing > 0]
    return float(2.0 * np.median(spacing)) if len(spacing) else 1.0


def _fit_gp_surrogate_setup(
    t_sample: np.ndarray, windows: np.ndarray, lengthscale: float, jitter: float = 1e-8
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Precomputes the two pieces of `f`'s surrogate GP that are shared by every candidate `c`: the
    Cholesky factor of its (fixed) kernel matrix, and the cross-covariance between every window's
    integral and every sample point.

    Signal variance is fixed to `1` throughout this module rather than exposed as a parameter:
    the predictive map `y ↦ K_∫ K⁻¹ y` is invariant to it. Writing `K=σ²R + jitter·I` with `R` the
    unit-variance correlation matrix - if `jitter` is also read as a fraction OF `σ²` (as it is
    here, applied to a unit-variance `R`) - `σ²` factors out of both `K` and `K_∫=σ²R_∫`, cancelling
    in `K_∫ K⁻¹ = σ²R_∫·(σ²R+jitter·I)⁻¹ = R_∫(R+jitter·I)⁻¹`. So fixing it to `1` loses nothing.

    Parameters
    ----------
    t_sample : np.ndarray
        Shape `(m,)`, the fixed sample points `f` will be evaluated at for every candidate.
    windows : np.ndarray
        Shape `(N, 2)`, integration windows (see `integral_matching._build_windows`).
    lengthscale : float
        RBF kernel lengthscale for `f`'s own surrogate GP (see `_default_lengthscale`).
    jitter : float, optional
        Added to the kernel matrix's diagonal for numerical stability (there is no real
        observation noise to justify a larger value - `f`'s samples are exact, given `c`).

    Returns
    -------
    L : np.ndarray
        Shape `(m, m)`, lower-triangular Cholesky factor of `K(t_sample, t_sample) + jitter·I`.
    K_int : np.ndarray
        Shape `(N, m)`, `Cov(∫_{window_i} f, f(t_sample_j))` for every window/sample pair.
    """
    K = _rbf_kernel(t_sample, t_sample, lengthscale, signal_var=1.0)
    K[np.diag_indices_from(K)] += jitter
    L = np.linalg.cholesky(K)
    K_int = _rbf_integral_kernel(windows[:, 0], windows[:, 1], t_sample, lengthscale)
    return L, K_int


def _fit_constants_func_approx(
    model: InducedModel,
    state_vars: "list[Var]",
    var_names: "list[str]",
    windows: np.ndarray,
    t_sample: np.ndarray,
    gps: "dict[str, _FittedGP]",
    lengthscale: float,
    jitter: float,
    device: torch.device,
    dtype: torch.dtype,
    ftol: float,
    xtol: float,
    gtol: float,
    max_nfev: "int | None",
) -> Any:
    """
    Fits `model`'s constants by least squares against the integral-matching residual, computing
    every window's integral via `f`'s surrogate GP (see the module docstring) instead of per-window
    quadrature.

    Parameters
    ----------
    model : InducedModel
        The (fully resolved) model whose constants are being fit.
    state_vars : list[Var]
        The differential ("state") variables being matched - `model`'s own, via `_split_endo_vars`.
    var_names : list[str]
        `state_vars`' own qualified names (see `integral_matching._fit_constants_integral`).
    windows : np.ndarray
        Shape `(N, 2)`, integration windows.
    t_sample : np.ndarray
        Shape `(m,)`, the fixed points `f` is evaluated at for every candidate.
    gps : dict[str, _FittedGP]
        Fitted GPs of the OBSERVED DATA (one per state variable) - the fixed interpolant `x̂`
        everything here reads from. Not to be confused with `f`'s own surrogate GP (unnamed,
        built fresh per fit from `lengthscale`/`jitter` - see `_fit_gp_surrogate_setup`).
    lengthscale, jitter : float
        `f`'s surrogate GP hyperparameters - see `_fit_gp_surrogate_setup`.
    device, dtype : torch.device, torch.dtype
        Where/at what precision the constant fit runs.
    ftol, xtol, gtol, max_nfev : optional
        Passed straight through to `scipy.optimize.least_squares`.

    Returns
    -------
    Any
        The raw `scipy.optimize.OptimizeResult` - `.x` holds the fitted constants.
    """
    _, algebraic_vars, frozen_values = model.split_endo_vars()
    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    # Target (left-hand side): exact via the DATA's own GP mean at window endpoints - identical to
    # integral_matching.py's target, see its docstring for why this needs no approximation.
    x_start = torch.stack(
        [torch.as_tensor(gps[name].mean(windows[:, 0]), dtype=dtype, device=device) for name in var_names], dim=1
    )
    x_end = torch.stack(
        [torch.as_tensor(gps[name].mean(windows[:, 1]), dtype=dtype, device=device) for name in var_names], dim=1
    )
    target = x_end - x_start  # (N, n_vars)

    # x̂ at the shared sample points - fixed, independent of `c`.
    x_sample = torch.stack(
        [torch.as_tensor(gps[name].mean(t_sample), dtype=dtype, device=device) for name in var_names], dim=1
    )  # (m, n_vars)
    t_sample_t = torch.as_tensor(t_sample, dtype=dtype, device=device)  # (m,)

    # f's surrogate GP: fixed, independent of `c` - computed once, reused by every residual_fn call.
    L_np, K_int_np = _fit_gp_surrogate_setup(t_sample, windows, lengthscale, jitter)
    L = torch.as_tensor(L_np, dtype=dtype, device=device)
    K_int = torch.as_tensor(K_int_np, dtype=dtype, device=device)  # (N, m)

    def residual_fn(c: torch.Tensor) -> torch.Tensor:
        const_ctx = c.unsqueeze(0).expand(t_sample_t.shape[0], -1)  # (m, n_consts)
        y_c = rhs(t_sample_t, x_sample, const_ctx)  # (m, n_vars): f at the shared samples
        alpha_c = torch.cholesky_solve(y_c, L)  # (m, n_vars), reuses the fixed factor L
        integral = K_int @ alpha_c  # (N, n_vars): every window's integral, one matmul
        return (integral - target).reshape(-1)

    def fun(c_np: np.ndarray) -> np.ndarray:
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return residual_fn(c).detach().cpu().numpy()

    def jac(c_np: np.ndarray) -> np.ndarray:
        c = torch.as_tensor(c_np, dtype=dtype, device=device)
        return jacfwd(residual_fn)(c).detach().cpu().numpy()

    c0 = _initial_const_guess(model)
    lo, hi = _const_bounds(model)
    c0 = np.clip(c0, lo, hi)

    return least_squares(
        fun, x0=c0, jac=jac, bounds=(lo, hi), method="trf", ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev
    )


@dataclass
class FuncApproxIntegralMatchingResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray  # (n_consts,), ordered by const.index_in_ctx
    const_by_name: "dict[str, float]"
    windows: np.ndarray  # (N, 2)
    t_sample: np.ndarray  # (m,)
    lengthscale: float
    gps: "dict[str, _FittedGP]"
    least_squares_result: Any


def estimate_func_approx_integral_matching(
    model: InducedModel,
    t_eval,
    stride: int = 1,
    t_sample: "np.ndarray | None" = None,
    lengthscale: "float | None" = None,
    jitter: float = 1e-8,
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    gps: "dict[str, _FittedGP] | None" = None,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    verbose: int = 0,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
) -> FuncApproxIntegralMatchingResult:
    """
    Function-approximation integral-matching estimate of `model`'s constants - same loss as
    `pybm.estimate.integral_matching.estimate_integral_matching`, computed via a shared surrogate
    GP of `f`'s own values instead of per-window quadrature, so cost is independent of how many
    windows are used - see the module docstring.

    Parameters
    ----------
    model : InducedModel
        Must have `engine == "torch"`, fully resolved (no `Choose` left), every `Var.ode`/
        `.algebraic` written in differentiable torch ops.
    t_eval : array-like
        Time points defining the integration windows (via `stride`).
    stride : int, optional
        Window width in units of `t_eval` indices - see `integral_matching._build_windows`.
        Default `1` (densest) - this method's whole point is that density is (nearly) free here.
    t_sample : array-like, optional
        The fixed points `f` is evaluated at for every candidate, shared across all windows.
        Defaults to `t_eval` itself.
    lengthscale : float, optional
        `f`'s own surrogate GP's lengthscale. Defaults to `_default_lengthscale(t_sample)` (twice
        the median sample spacing) - NOT related to the DATA's own GP lengthscale (`gps`), since
        `f`'s smoothness in `t` is a different quantity than the state's own.
    jitter : float, optional
        Numerical-stability nugget on `f`'s surrogate GP's kernel diagonal - see
        `_fit_gp_surrogate_setup`. Not an observation-noise variance (there is none: `f` is
        deterministic given `c`).
    max_gp_points, max_gp_iter : optional
        Passed to `fit_gps` for the DATA's own GPs (`x̂`) - see its own docs.
    gps : dict[str, _FittedGP], optional
        Precomputed DATA GPs (see `fit_gps`) - pass this across many calls sharing the same
        underlying data, same role as in `estimate_integral_matching`.
    device, dtype : optional
        Where/at what precision the fit runs. Default: CPU, `float64`.
    verbose : int, optional
        `0` silent; `>=2` prints the final residual cost.
    ftol, xtol, gtol, max_nfev : optional
        Passed to `scipy.optimize.least_squares`.

    Returns
    -------
    FuncApproxIntegralMatchingResult
        `consts`/`const_by_name`: the fitted constants. `windows`, `t_sample`, `lengthscale`: what
        was actually used. `gps`: the DATA's own fitted (or reused) GPs. `least_squares_result`:
        the raw `scipy.optimize.OptimizeResult`.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_func_approx_integral_matching requires an InducedModel built with "
            f"engine='torch', got engine={model.engine!r}."
        )

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)

    state_vars, _, _ = model.split_endo_vars()
    qualified = _qualified_var_names(model)
    var_names = [qualified[id(var)] for var in state_vars]

    gps = dict(gps) if gps is not None else {}
    missing = {name: var for name, var in zip(var_names, state_vars) if name not in gps}
    if missing:
        gps.update(fit_gps(missing, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose))

    windows = _build_windows(t_eval, stride=stride)
    t_sample = np.asarray(t_eval if t_sample is None else t_sample, dtype=float)
    lengthscale = _default_lengthscale(t_sample) if lengthscale is None else float(lengthscale)

    result = _fit_constants_func_approx(
        model, state_vars, var_names, windows, t_sample, gps, lengthscale, jitter,
        device, dtype, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
    )
    if verbose >= 2:
        print(f"[func_approx_im] constant fit: cost={result.cost:.4g} success={result.success}")

    const_by_name = {name: float(result.x[c.index_in_ctx]) for name, c in model.consts.items()}

    return FuncApproxIntegralMatchingResult(
        model=model,
        t_eval=t_eval,
        consts=result.x,
        const_by_name=const_by_name,
        windows=windows,
        t_sample=t_sample,
        lengthscale=lengthscale,
        gps=gps,
        least_squares_result=result,
    )
