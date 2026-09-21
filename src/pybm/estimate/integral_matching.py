"""
Integral matching: an alternative to `pybm.estimate.gradient_matching`'s pointwise derivative
residual, for the same "fit constants without ever integrating the ODE" family of estimators.

Why match the INTEGRAL of the derivative instead of the derivative itself
---------------------------------------------------------------------------
Gradient matching compares the model's right-hand side `f` to a GP's derivative estimate at a
handful of collocation points: `f(t*, x̂(t*), c) ≈ x̂'(t*)`. This has two weaknesses:
1. Differentiating a (smoothed) interpolant amplifies noise more than reading its own value does.
2. Each collocation point is an independent, local check - a structurally wrong but flexible `f`
   (more free parameters than the true dynamics need) can fit a handful of isolated points almost
   exactly without matching the real trajectory at all. We saw this empirically on the real Bled
   model: a tiny gradient-matching residual, but a forward-simulated trajectory that diverges
   badly from the data.

Integral matching instead compares the ACCUMULATED change over small windows:
    x̂(t_{i+1}) - x̂(t_i)  ≈  ∫_{t_i}^{t_{i+1}} f(τ, x̂(τ), c) dτ
Both sides are still computed purely from the fixed GP interpolant `x̂` (never a self-consistent,
possibly-unstable simulation) - so this keeps gradient matching's whole appeal (cheap, numerically
safe, no risk of a bad `c` making an ODE solver blow up) while using far more information per
window than a single pointwise derivative check, and without ever needing a derivative estimate at
all (the left-hand side is exact via the fundamental theorem of calculus, not approximated).

This is "Integral Matching" in the sense of Varah (1982) and, in a more recent form specific to
this closed-form-via-collocation derivation, an anonymous ICLR 2025 submission, "ODE Parameter
Identification: An Integral Matching Approach" (openreview.net/pdf?id=X3IcgZEUEi) - see its
Theorem 3.1 for the result this module's design leans on: the gap between this surrogate loss and
the TRUE (real-simulation) loss is bounded by a term that shrinks as the window size shrinks. That
licenses using MANY windows rather than a handful of collocation points - directly targeting the
sparse-collocation overfitting problem found empirically. It also licenses putting the squared norm
OUTSIDE the integral (`‖∫r‖²`, matches the paper's own loss) rather than inside (`∫‖r‖²`, immune to
within-window cancellation but more expensive) - PROVIDED the quadrature computing that integral
stays accurate, which is a property of `n_panels` (see `_simpson_nodes_weights`), not of the window
width `stride` on its own: a wide window with too few quadrature panels is a genuinely different
(bad) approximation, not just "a wide window", and was originally mistaken for evidence that wide
windows hurt before this module scaled `n_panels` with `stride`. With `n_panels` kept adequate,
empirically (see `pybm.benchamark.benchmark_2`) WIDER windows outperform `stride=1` on real data
with enough free constants to have a real overfitting risk - `stride=1` degenerates toward gradient
matching's own weakness (see the derivation: as `h→0`, this loss's residual is `h` times gradient
matching's pointwise one), while on a simple/low-dimensional model wider windows show no benefit at
all (no overfitting risk to protect against in the first place) - "wide is safer" is not a universal
rule, but "narrow is not obviously safer" either, contrary to this module's original, more
cautious framing.

Why `x̂` (the GP interpolant) on BOTH sides, never raw data directly
-----------------------------------------------------------------------
The right-hand side `f` is nonlinear in general. Plugging a raw, noisy data point into it directly
would bias the fit (Jensen-type "errors-in-variables" bias: E[f(x+noise, c)] != f(x, c)) - not just
add variance. The GP posterior mean `x̂` is a LINEAR smoother of the raw data (`k(t*,X)·α`), so it
absorbs the noise before anything nonlinear ever sees it. This is the same principle
`pybm.estimate.gradient_matching` already relies on, applied consistently to both sides of this
module's residual.

See `pybm.estimate.func_approx_im` for a second implementation of the same loss, one that also
approximates `f` itself with a GP so that every window's integral becomes a single shared linear
readout instead of a fresh quadrature per window - worth it once the number of windows is large
enough that quadrature cost starts to dominate; not needed for a first, simpler implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.optimize import least_squares
from torch.func import jacfwd

from pybm.estimate.depr.gradient_matching import (
    _FittedGP,
    _identifiability_report,
    _qualified_var_names,
    fit_gps,
)
from pybm.estimate.gradient_matching import _const_bounds, _initial_const_guess
from pybm.estimate.multishooting_torch import uniform_sub_indices
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _make_rhs


def _build_windows(t_eval: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Builds integration windows covering the whole horizon, from consecutive (or `stride`-apart)
    points of `t_eval`.

    Parameters
    ----------
    t_eval : np.ndarray
        Time points, strictly increasing.
    stride : int, optional
        Window width in units of `t_eval` indices. `1` (default) uses every consecutive pair of
        points - the densest, most informative choice (see the module docstring for why density
        matters both for avoiding within-window cancellation and for the Theorem-3.1-style bound
        to the true trajectory loss). A larger stride gives fewer, wider windows - cheaper per
        candidate, but a looser bound.

    Returns
    -------
    np.ndarray
        Shape `(N, 2)`: `(start, end)` time pairs, one row per window.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    idx = np.arange(0, len(t_eval) - 1, stride)
    starts = t_eval[idx]
    ends = t_eval[np.minimum(idx + stride, len(t_eval) - 1)]
    return np.stack([starts, ends], axis=1)


def _auto_stride(t_eval: np.ndarray, gps: "dict[str, _FittedGP]", var_names: "list[str]") -> int:
    """
    Picks a window width by the rule derived in `notes/integral-matching.md` §5.4: `stride` (in
    `t_eval`-index units) such that the resulting window width in TIME units is on the order of the
    fitted GP's own correlation lengthscale `ℓ`. Below `ℓ`, neighboring window endpoints
    `x̂(t_i)`/`x̂(t_{i-1})` are nearly the same random variable under the GP posterior - the noise
    floor per window stays ~constant instead of shrinking (Eq. 5.2, `Δ≪ℓ` branch), so finer
    windows just add more equations without adding information. Above `ℓ`, the noise floor per
    window keeps shrinking (Eq. 5.2, `Δ≫ℓ` branch) but Grönwall bias grows (§5.2) - `ℓ` itself is
    the point past which that trade stops obviously favoring "wider". Averages `ℓ` across every
    state variable's own GP when there is more than one (they all share the same window grid, see
    `_fit_constants_integral`) and converts to an index count via `t_eval`'s own median spacing
    (`t_eval` need not be uniformly spaced).
    """
    ell = float(np.mean([gps[name].lengthscale for name in var_names]))
    median_dt = float(np.median(np.diff(t_eval)))
    if median_dt <= 0 or not np.isfinite(ell):
        return 1
    return max(1, round(ell / median_dt))


def _simpson_nodes_weights(windows: np.ndarray, n_panels: int = 1) -> "tuple[np.ndarray, np.ndarray]":
    """
    Composite Simpson's rule quadrature nodes and weights for each window, ready to dot-product
    against `f` evaluated at `nodes` to get each window's own `∫f dt` estimate directly.

    A single Simpson panel (`n_panels=1`, three nodes: start/mid/end) is only accurate while the
    window stays narrow - widening a window (larger `stride` in `_build_windows`) without adding
    more panels makes the quadrature itself increasingly wrong (the integrand is no longer
    well-approximated by one quadratic), which then confounds any comparison of "wider window" vs.
    "narrower window" with "worse quadrature" vs. "better quadrature" - two different effects. To
    keep them apart, `n_panels` splits each window into that many equal-width SUB-panels (each
    still a proper 3-point Simpson panel internally, sharing endpoints with its neighbors) - pass
    `n_panels` roughly proportional to the window width (e.g. equal to `stride`) to hold the
    PER-PANEL width - and therefore quadrature accuracy - roughly constant regardless of how wide
    the overall window is.

    True adaptive quadrature (e.g. `scipy.integrate.quad`, refining node count based on an error
    estimate) is not used here on purpose: it would give each window - and each optimizer trial
    point `c`, since the "right" number of nodes can depend on `c` too - a different number of
    evaluations, which breaks the fixed-shape, batched computation `jacfwd` needs to differentiate
    efficiently (see `pybm.estimate.integral_matching`'s module docstring on why forward-mode AD is
    used). A FIXED, width-scaled panel count is the practical way to keep quadrature accuracy from
    degrading with window width while staying inside that constraint.

    Parameters
    ----------
    windows : np.ndarray
        Shape `(N, 2)`, `(start, end)` per window (see `_build_windows`). All windows are assumed
        the same width (as `_build_windows` produces, apart from a possibly-shorter last one) -
        `n_panels` is a single scalar shared by every window, not chosen per window.
    n_panels : int, optional
        Number of equal-width sub-panels per window. `1` (default) is a single 3-point Simpson
        panel - the original, width-unaware behavior.

    Returns
    -------
    nodes : np.ndarray
        Shape `(N, 2*n_panels+1)`: the evaluation times per window.
    weights : np.ndarray
        Shape `(N, 2*n_panels+1)`: composite Simpson weights, pattern `(1,4,2,4,2,...,4,1) * h/6`
        where `h` is the SUB-panel width (`(end-start)/n_panels`).
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1, got {n_panels}.")
    starts, ends = windows[:, 0], windows[:, 1]
    h = (ends - starts) / n_panels  # (N,), sub-panel width

    n_nodes = 2 * n_panels + 1
    offsets = np.arange(n_nodes) * 0.5  # in units of h: 0, 0.5, 1, 1.5, ..., n_panels
    nodes = starts[:, None] + offsets[None, :] * h[:, None]  # (N, n_nodes)

    pattern = np.ones(n_nodes)
    pattern[1:-1:2] = 4.0  # odd-indexed nodes: sub-panel midpoints
    pattern[2:-1:2] = 2.0  # even-indexed interior nodes: shared sub-panel boundaries
    weights = pattern[None, :] * (h[:, None] / 6.0)  # (N, n_nodes)
    return nodes, weights


def _guard_rhs(f_vals: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Softly saturates `F`'s raw output before it enters the integral. A wild trial `c` during the
    search (e.g. a division by a near-zero half-saturation constant) can make `F` overflow to
    `inf`/`nan` well before the trust-region step that produced it gets rejected - once that
    happens the residual is `nan`, its gradient is `nan`, and `scipy.optimize.least_squares` has no
    usable signal to back away with (it doesn't crash outright, but that trial point is a dead end
    it can't learn anything from). `nan_to_num` turns an already-`nan`/`inf` evaluation into a
    finite `±max_val` first; `tanh` then squashes it (and any merely-huge-but-finite value) smoothly
    toward `±max_val`. `d/dx[max_val·tanh(x/max_val)] = 1-tanh(x/max_val)² > 0` everywhere, so the
    residual's gradient w.r.t. `c` never vanishes outright for a large-but-finite blowup - it only
    shrinks - which is what actually steers `trf` back toward the feasible region on the NEXT step,
    rather than stalling. For values that were already `nan` going in, `nan_to_num`'s own gradient
    is zero at that point (there's no real direction to push from a `0/0`) - the guard's honest
    contribution there is "stop the `nan` from propagating and corrupting every other residual in
    the same batched evaluation", not "actively repel this exact trial point".

    Parameters
    ----------
    f_vals : torch.Tensor
        Raw `F` output at every quadrature node.
    max_val : float
        Saturation scale - see `estimate_integral_matching`'s `max_rhs` parameter.
    """
    f_vals = torch.nan_to_num(f_vals, nan=max_val, posinf=max_val, neginf=-max_val)
    return max_val * torch.tanh(f_vals / max_val)


def _fit_constants_integral(
    model: InducedModel,
    state_vars: "list[Var]",
    var_names: "list[str]",
    windows: np.ndarray,
    gps: "dict[str, _FittedGP]",
    n_panels: int,
    device: torch.device,
    dtype: torch.dtype,
    ftol: float,
    xtol: float,
    gtol: float,
    max_nfev: "int | None",
    max_rhs: float = 1e4,
    weight_by_uncertainty: bool = False,
) -> Any:
    """
    Fits `model`'s constants by least squares against the integral-matching residual on `windows`,
    using the same `scipy.optimize.least_squares` + forward-mode-autodiff-Jacobian machinery as
    `pybm.estimate.gradient_matching._fit_constants`.

    Parameters
    ----------
    model : InducedModel
        The (fully resolved) model whose constants are being fit.
    state_vars : list[Var]
        The differential ("state") variables being matched - `model`'s own, via `_split_endo_vars`.
    var_names : list[str]
        `state_vars`' own qualified names (e.g. "phyto.conc"), same order, same length - the key
        each variable's GP is stored under in `gps` (see `_qualified_var_names`: `var.name` alone
        is only the local template attribute name and can collide across entities).
    windows : np.ndarray
        Shape `(N, 2)`, integration windows (see `_build_windows`).
    gps : dict[str, _FittedGP]
        Fitted GPs, one per state variable's own observed data (see `fit_gps`), keyed by qualified
        name - the fixed interpolant `x̂` both sides of the residual read from; never refit here.
    n_panels : int
        Composite-Simpson sub-panels per window - see `_simpson_nodes_weights`.
    device, dtype : torch.device, torch.dtype
        Where/at what precision the constant fit runs.
    ftol, xtol, gtol, max_nfev : optional
        Passed straight through to `scipy.optimize.least_squares` - see its own docs.
    max_rhs : float, optional
        Saturation scale for `_guard_rhs` - see its own docstring for why a wild trial `c` needs
        this instead of being left to produce `nan`/`inf`. Default `1e4`, comfortably above any
        physically sane `F` value on this model's data (state values ~O(1-10)) but small enough to
        actually catch a genuine blowup before it turns into `nan`.
    weight_by_uncertainty : bool, optional
        Default `False` (reproduces the original unweighted fit exactly). When `True`, each
        window's residual is divided by the GP's own posterior standard deviation of the
        ACCUMULATED CHANGE over that window (`_FittedGP.window_var`, `notes/integral-matching.md`
        §2.1's `x̂(t_{i+1})-x̂(t_i)` target) - a window where the GP is less sure about that
        difference (sparse data, or near the edge of the observed span) contributes less. See
        `_FittedGP.window_var`'s own docstring for why this uses the windowed covariance, not just
        the two endpoints' own `var()` added together (they're correlated, so that would overstate
        a narrow window's true uncertainty).

    Returns
    -------
    Any
        The raw `scipy.optimize.OptimizeResult` (same convention as
        `gradient_matching._fit_constants`) - `.x` holds the fitted constants.
    """
    _, algebraic_vars, frozen_values = model.split_endo_vars()
    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    n_windows = windows.shape[0]
    n_vars = len(state_vars)
    nodes, weights = _simpson_nodes_weights(windows, n_panels=n_panels)  # (N, n_nodes), (N, n_nodes)
    n_nodes = nodes.shape[1]

    # Target (left-hand side): x̂(t_{i+1}) - x̂(t_i), exact via the GP's own posterior mean at the
    # two endpoints - no quadrature needed here, this is not an approximation (see module docstring).
    x_start = torch.stack(
        [torch.as_tensor(gps[name].mean(windows[:, 0]), dtype=dtype, device=device) for name in var_names],
        dim=1,
    )  # (N, n_vars)
    x_end = torch.stack(
        [torch.as_tensor(gps[name].mean(windows[:, 1]), dtype=dtype, device=device) for name in var_names],
        dim=1,
    )
    target = x_end - x_start  # (N, n_vars)

    # `weight[i,v] = 1/sqrt(Var[x̂_v(t_{i+1})-x̂_v(t_i)])` - see `_FittedGP.window_var` and
    # `estimate_integral_matching`'s `weight_by_uncertainty` docstring. `None` (default) reproduces
    # the original unweighted fit exactly.
    weight = None
    if weight_by_uncertainty:
        window_var = np.stack(
            [gps[name].window_var(windows[:, 0], windows[:, 1]) for name in var_names], axis=1
        )  # (N, n_vars)
        weight = torch.as_tensor(1.0 / np.sqrt(np.maximum(window_var, 1e-12)), dtype=dtype, device=device)

    # x̂ at every quadrature node (flattened across all windows) - fixed, independent of `c`,
    # computed once, reused by every `residual_fn` call the optimizer makes.
    nodes_flat = nodes.reshape(-1)  # (N*n_nodes,)
    x_nodes = torch.stack(
        [torch.as_tensor(gps[name].mean(nodes_flat), dtype=dtype, device=device) for name in var_names],
        dim=1,
    )  # (N*n_nodes, n_vars)
    t_nodes = torch.as_tensor(nodes_flat, dtype=dtype, device=device)  # (N*n_nodes,)
    weights_t = torch.as_tensor(weights, dtype=dtype, device=device)  # (N, n_nodes)

    def residual_fn(c: torch.Tensor) -> torch.Tensor:
        const_ctx = c.unsqueeze(0).expand(nodes_flat.shape[0], -1)  # (N*n_nodes, n_consts)
        f_vals = _guard_rhs(rhs(t_nodes, x_nodes, const_ctx), max_rhs).reshape(n_windows, n_nodes, n_vars)
        integral = (weights_t.unsqueeze(-1) * f_vals).sum(dim=1)  # (N, n_vars): ∫f dt per window
        residual = integral - target
        if weight is not None:
            residual = residual * weight
        return residual.reshape(-1)

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
class IntegralMatchingResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray  # (n_consts,), ordered by const.index_in_ctx
    const_by_name: "dict[str, float]"
    windows: np.ndarray  # (N, 2)
    gps: "dict[str, _FittedGP]"
    least_squares_result: Any

    def init_params(
        self,
        n_subintervals: int = 1,
        n_candidates: int = 1,
        jitter: float = 0.0,
        seed: "int | None" = None,
        device: "torch.device | None" = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Builds a `[consts..., s_0, ..., s_{K-1}]` tensor in the layout `estimate_torch` expects,
        using this fit's constants and the GP posterior mean (evaluated at each subinterval's
        start time) as the shooting-node seeds - same role and same interface as
        `pybm.estimate.gradient_matching.GradientMatchingResult.init_params`, for warm-starting
        multishooting refinement from an integral-matching fit instead of a gradient-matching one.

        Parameters
        ----------
        n_subintervals : int, optional
            Number of shooting segments (K) - must match the `n_subintervals` passed to
            `estimate_torch`. Default 1 (single-shooting).
        n_candidates : int, optional
            How many copies of this same starting point to stack (B) - combine with `jitter` for
            a cheap multistart around one already-good guess. Default 1.
        jitter : float, optional
            Std-dev of Gaussian noise added to every candidate but the first (kept exact) - only
            applied when `n_candidates > 1`.
        seed : int, optional
            RNG seed for `jitter`.
        device : torch.device, optional
            Where the returned tensor lives. Default: CPU.
        dtype : torch.dtype, optional
            Default `torch.float64`.

        Returns
        -------
        torch.Tensor
            Shape `(n_candidates, n_consts + n_subintervals * n_state_vars)`.
        """
        device = device or torch.device("cpu")
        state_vars, _, _ = self.model.split_endo_vars()
        qualified = _qualified_var_names(self.model)

        sub_indices = uniform_sub_indices(self.t_eval, n_subintervals)
        seed_times = self.t_eval[sub_indices[:-1]]

        seeds = np.zeros((n_subintervals, len(state_vars)), dtype=float)
        for i, var in enumerate(state_vars):
            seeds[:, i] = self.gps[qualified[id(var)]].mean(seed_times)

        consts = np.asarray(
            self.consts.detach().cpu().numpy() if torch.is_tensor(self.consts) else self.consts, dtype=float
        )
        flat = np.concatenate([consts, seeds.reshape(-1)])
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


def estimate_integral_matching(
    model: InducedModel,
    t_eval,
    stride: "int | str" = "auto",
    n_panels: "int | None" = None,
    max_gp_points: int = 500,
    max_gp_iter: int = 200,
    gps: "dict[str, _FittedGP] | None" = None,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    verbose: int = 0,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
    max_rhs: float = 1e4,
    weight_by_uncertainty: bool = False,
) -> IntegralMatchingResult:
    """
    Integral-matching estimate of `model`'s constants - see the module docstring for the method
    and why it's designed this way.

    Parameters
    ----------
    model : InducedModel
        Must have `engine == "torch"`, fully resolved (no `Choose` left), every `Var.ode`/
        `.algebraic` written in differentiable torch ops (same precondition as
        `estimate_gradient_matching`).
    t_eval : array-like
        Time points defining the integration windows (via `stride`) - does not need to match every
        observed data point; `fit_gps` reads each state variable's own `.data` directly regardless.
    stride : int or "auto", optional
        Window width in units of `t_eval` indices - see `_build_windows`. Default `"auto"`: picks
        `stride` so the window width in TIME units matches the fitted GP's own correlation
        lengthscale (`notes/integral-matching.md` §5.4 - see `_auto_stride`), the width past which
        widening further trades a shrinking per-window noise floor for growing Grönwall bias
        without an a-priori "obviously better" direction. On the real Bled data this resolves to
        `stride≈18` (ℓ≈17.9 with `t_eval` in ~1-day steps) - close to, but NOT a clean improvement
        over, `stride=10` (the empirical minimum `pybm.benchamark.benchmark_2.run_stride_sweep`
        found on one fold): checked across a full 8-fold CV, `"auto"` did about the same or worse on
        6/8 folds (once ~8x worse), better on 2 - see `_integral_matching_method`'s docstring for the
        numbers. Kept as the default because it replaces a free, unprincipled sweep-tuned number with
        one that has a stated reason, and both land in the same sensible neighborhood - not because
        it's been shown to generalize better. Pass an explicit `int` to override, or to reproduce a
        specific earlier result exactly.
    n_panels : int, optional
        Composite-Simpson sub-panels per window (see `_simpson_nodes_weights`) - widening a window
        (larger `stride`) without also raising `n_panels` makes the quadrature itself increasingly
        inaccurate, confounding "wider window" with "worse quadrature". Defaults to `stride` itself,
        which holds the per-panel width - and so quadrature accuracy - roughly constant at the
        original `t_eval` spacing regardless of `stride`. Pass a larger value to trade more
        evaluations for tighter quadrature at a fixed `stride`.
    max_gp_points, max_gp_iter : optional
        Passed to `fit_gps` - see its own docs. Ignored for any variable already covered by `gps`.
        `max_gp_points` default raised from `fit_gps`'s own `300` to `500` here: GP hyperparameter
        fitting is O(n³) in points, so this isn't free (checked directly on the pooled Bled
        training data, ~2033 rows: `300`→0.3s, `500`→14s, `800`→49s) - but `300` points, evenly
        subsampled from ~2033, meaningfully UNDER-resolves the actual noise/lengthscale here
        (fitted lengthscale 17.9 at 300 points vs 11.3 at 800; noise_std 0.019 vs 0.0035) - `500`
        is the point on that curve still cheap enough to run inside an 8-fold CV loop without
        dominating total runtime.
    gps : dict[str, _FittedGP], optional
        Precomputed GPs (see `fit_gps`) - pass this across many calls that share the same
        underlying data (e.g. a structural search over many candidate models) to skip refitting
        step 1 every time, the same role it plays in `estimate_gradient_matching`.
    device, dtype : optional
        Where/at what precision the fit runs. Default: CPU, `float64`.
    verbose : int, optional
        `0` silent; `>=2` prints the final residual cost.
    ftol, xtol, gtol, max_nfev : optional
        Passed to `scipy.optimize.least_squares` - see its own docs.
    max_rhs : float, optional
        Passed to `_guard_rhs` (see its own docstring) - softly saturates `F`'s raw output during
        the search so a wild trial `c` produces a large-but-finite, still-differentiable residual
        instead of `nan`/`inf`. Default `1e4`.
    weight_by_uncertainty : bool, optional
        Default `False` - see `_fit_constants_integral`'s own docstring for what this does
        (`_FittedGP.window_var`-based generalized least squares) and why it's opt-in.

    Returns
    -------
    IntegralMatchingResult
        `consts`/`const_by_name`: the fitted constants. `windows`: the integration windows used.
        `gps`: the fitted (or reused) per-state-variable GPs. `least_squares_result`: the raw
        `scipy.optimize.OptimizeResult`.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_integral_matching requires an InducedModel built with engine='torch', got engine={model.engine!r}."
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

    if stride == "auto":
        stride = _auto_stride(t_eval, gps, var_names)
        if verbose >= 2:
            print(f"[integral_matching] auto stride: {stride} (ℓ≈{np.mean([gps[n].lengthscale for n in var_names]):.4g})")
    windows = _build_windows(t_eval, stride=stride)
    n_panels = stride if n_panels is None else n_panels
    result = _fit_constants_integral(
        model, state_vars, var_names, windows, gps, n_panels, device, dtype,
        ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, max_rhs=max_rhs,
        weight_by_uncertainty=weight_by_uncertainty,
    )
    if verbose >= 2:
        print(f"[integral_matching] constant fit: cost={result.cost:.4g} success={result.success}")

    const_by_name = {name: float(result.x[c.index_in_ctx]) for name, c in model.consts.items()}

    return IntegralMatchingResult(
        model=model,
        t_eval=t_eval,
        consts=result.x,
        const_by_name=const_by_name,
        windows=windows,
        gps=gps,
        least_squares_result=result,
    )
