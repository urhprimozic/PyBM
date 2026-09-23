"""
Growing-horizon integral matching (`im_gh`) - the same predictor-corrector continuation
(Davidenko's equation) as `pybm.estimate.growing_horizon`, but applied to a CUMULATIVE
integral-matching residual instead of an actually-integrated ODE trajectory. See the chat this
followed for the full derivation; this docstring only records the result.

Loss (single span, origin `t_i0`, end `t_i1`)
-----------------------------------------------
For a FIXED data interpolant `x_hat` (never re-simulated - same GP-based `x_hat` as
`pybm.estimate.integral_matching`, fit once from data and never touched again):

    r(t;theta,t_i0) = x_hat(t) - x_hat(t_i0) - integral_{t_i0}^{t} F(x_hat(tau), tau, theta) dtau
    l_t(theta,t_i0) = penalty(r(t;theta,t_i0))                (mean over state dims, see `_penalty`)
    L_{i0,i1}(theta) = mean over t in {t_{i0+1},...,t_i1} of l_t(theta,t_i0)
    o(i0,i1)         = argmin_theta L_{i0,i1}(theta)

This is deliberately NOT the same loss as `estimate_integral_matching`, which sums many
INDEPENDENT, non-overlapping LOCAL windows (each one "resets" at its own start point, so error
never accumulates across the whole horizon - that's what keeps its landscape well-conditioned).
Anchoring every `r(t;theta)` in a span at the SAME origin `t_i0` means - confirmed directly with
the user - the loss surface over LONG spans can still be non-convex/occasionally multi-modal (plus
genuine early-window parameter non-identifiability, e.g. a rate constant that only becomes visible
once a state variable has moved away from a near-zero initial value). Four `mode`s below all use
the SAME predictor-corrector continuation machinery to track a good `theta` without ever cold-
fitting a long span directly - they differ only in WHICH span(s) get visited over the course of a
fit, motivated by three separate, explicitly discussed proposals (see the chat this followed).

Davidenko's equation, and why NO span-shape assumption is needed for the predictor
--------------------------------------------------------------------------------------
Differentiating `grad_theta L(o) = 0` through a continuation parameter gives the general
structural equation `H . do + d[grad_theta L] = 0` regardless of what that parameter is. But this
module never actually integrates that as a continuous ODE (unlike `growing_horizon.py`'s
`outer_solver="ode"`) - every span here lives on `t_eval`'s own grid, so continuation is always a
FINITE secant step between one span and the next, old-span-optimum to new-span-optimum. That step
needs no derivative-of-anything-in-a-continuation-parameter at all: writing `theta_cur = o(span_
old)`, by DEFINITION `grad_theta L_old(theta_cur) = 0`, so for ANY other span (grown, shrunk,
shifted, or entirely disjoint):

    grad_theta L_new(theta_cur) = grad_theta L_new(theta_cur) - grad_theta L_old(theta_cur)

is exactly "how much the gradient changed", and one Newton step using the OLD Hessian as a cheap
stand-in for the new one gives the predictor used everywhere below:

    theta_pred = theta_cur - H_old^{-1} . grad_theta L_new(theta_cur)

This is standard continuation/quasi-Newton reasoning, not specific to growing a span - it is EXACTLY
as valid whether `span_new` nests `span_old` (growing), is a positional shift of it (sliding), or
is a disjoint span somewhere else on the grid (an earlier draft of this module's docstring derived
a much more complicated "extra Leibniz term" assuming a literal continuous-`t0` ODE was needed for
sliding - that was solving a harder problem than the one this module actually has, since spans here
are always discrete secant steps, never a continuously-integrated ODE).

Modes
-----
- `"growing"` (default): the ORIGINAL, most-tested behavior. One span `(0, i1)`, `i0` fixed at the
  horizon start, `i1` grows geometrically (in TIME units) until `lambda_min(H) < pd_margin_rel *
  lambda_max(H)` (a bifurcation/non-identifiability signal - see `hessian_mode`) or the full
  horizon is covered.
- `"grow_then_slide"`: grow exactly as `"growing"`, but using a LOOSER, EARLIER-triggering
  threshold `switch_margin_rel` (> `pd_margin_rel`) as a pre-emptive "this span is getting risky"
  signal. Once triggered (before an actual stall), FREEZE the span width at whatever it reached,
  and continue by SLIDING `(i0,i1)` forward together (fixed width) instead of growing further -
  motivated by: a span that stays fixed-width can never accumulate arbitrarily much compounding
  quadrature/nonlinearity error, however far the SLIDING position moves, unlike an ever-growing
  single-origin span. Falls back to plain `"growing"` behavior if the horizon ends before the
  switch threshold ever triggers.
- `"fixed_slide"`: skip growing entirely - establish `o(0, window_width)` once (same GM warm start
  as the other modes), then immediately slide with that FIXED width for the rest of the horizon.
  Simpler than `"grow_then_slide"` (no adaptive discovery of a safe width), but needs a reasonable
  `window_width` supplied (or falls back to `init_points`'s own window).
- `"multi_origin"`: run `"growing"`'s own logic INDEPENDENTLY from `n_starts` different origins
  spread across the horizon (same code, different `i0`), each growing to its own natural stopping
  point. Then POOL every origin's own final span into one combined loss and take ONE more Newton/
  Gauss-Newton correction step on the pooled `theta` (warm-started from the per-origin `theta`s'
  mean). Motivated by REGION-SPECIFIC non-identifiability (a constant that a run from `t_eval[0]`
  can never see might be plainly visible from a later origin) - directly targets the mechanism
  `"growing"`'s own PT failure was traced to (see `benchmark_3.IM_GH_KWARGS`'s comment), rather than
  the compounding-error mechanism `"grow_then_slide"`/`"fixed_slide"` target. Honest caveat: pooled
  spans usually OVERLAP (they share `t_eval` data), so the pooled fit is not a textbook independent-
  samples combination - treated here as a practical approximation, not a rigorous GLS pooling.

Simplifications relative to `growing_horizon.py`, all a direct consequence of never integrating an
ODE:
- `theta` alone is ever optimized - no seed/initial-condition free variable (`x_hat` at any origin
  comes straight from the GP, not a fit parameter), so no continuity term, no segments.
- Every grid point's local panel integral is INDEPENDENT of every other and of `theta`'s effect on
  any OTHER panel - so `_residuals_full` computes the running cumulative integral for the WHOLE
  horizon (from a fixed reference index 0) in ONE batched RHS evaluation plus one `cumsum`; a span
  `(i0,i1)` anchored at ANY origin is then just `cum[i1]-cum[i0]` and `x_hat[i1]-x_hat[i0]` - a
  difference of two already-computed entries, never a fresh integration.
- The exact double-backprop Hessian (`torch.autograd.grad` with `create_graph=True`) is safe for
  ANY span here - unlike `multishooting_adaptive._exact_block_hessian`, there is no torchode
  adjoint anywhere in this module (no ODE solve at all), so its documented K=2/batched-solve NaN
  limitation simply does not apply.

`norm`/`hessian_mode`: same "l2"/"l1" and "exact"/"gauss_newton" choices as before, reusing
`multishooting_adaptive._penalty` and its noise-scale-calibrated `l1_eps` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from scipy.stats import norm as _normal_dist
from torch.func import jacfwd, jacrev, vmap

from pybm.estimate.depr.gradient_matching import (
    _FittedGP,
    _const_bounds,
    _qualified_var_names,
    estimate_gradient_matching,
    fit_gps,
)
from pybm.estimate.depr.integral_matching import _build_windows, _guard_rhs, _simpson_nodes_weights
from pybm.estimate.depr.multishooting_adaptive import _L1_SMOOTH_EPS, _penalty
from pybm.estimate.depr.multishooting_torch import _homotopy_weight
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, _get_data_tensor, _make_rhs

Span = "tuple[int, int]"  # (i0, i1): half-open-on-the-left index pair into t_eval, i0 < i1


@dataclass
class IMGrowingHorizonResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray
    const_by_name: "dict[str, float]"
    status: str  # "completed" | "stalled"
    history: "list[dict]"
    mode: str
    final_spans: "list[Span]"  # the span(s) the returned `theta` was actually fit on
    windows: np.ndarray  # (n_grid-1, 2), see `_build_windows` - kept for reuse/inspection
    gps: "dict[str, _FittedGP]" = field(default_factory=dict)


def estimate_im_growing_horizon(
    model: InducedModel,
    t_eval,
    mode: "Literal['growing', 'grow_then_slide', 'fixed_slide', 'multi_origin']" = "growing",
    init_points: int = 6,
    growth_factor: float = 1.6,
    corrector: "Literal['newton', 'adam']" = "newton",
    corrector_steps: int = 10,
    corrector_lr: float = 1e-2,
    pd_margin_rel: float = 1e-5,
    pd_check: bool = True,
    switch_margin_rel: float = 1e-3,
    window_width: "int | None" = None,
    max_window_length: "float | None" = None,
    slide_step: int = 1,
    n_starts: int = 3,
    max_rounds: int = 30,
    polish: bool = False,
    polish_max_iter: int = 1000,
    polish_tol: float = 1e-3,
    polish_lr: float = 1e-2,
    norm: "Literal['l2', 'l1']" = "l2",
    l1_eps: "float | None" = None,
    hessian_mode: "Literal['exact', 'gauss_newton']" = "exact",
    precondition: bool = False,
    precondition_floor: float = 1e-3,
    epsilon_tol: float = 1e-3,
    gronwall_p: float = 0.05,
    n_panels: int = 1,
    max_rhs: float = 1e4,
    weight_by_uncertainty: bool = True,
    max_gp_points: int = 500,
    max_gp_iter: int = 200,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    torch_threads: "int | None" = None,
    verbose: int = 0,
) -> IMGrowingHorizonResult:
    """
    See module docstring for the method and the four `mode`s. Warm start on every span (theta
    only - no seed/x0, see module docstring) via `estimate_gradient_matching` restricted to that
    span's own `t_eval` slice, same convention as `growing_horizon.py`.

    Parameters
    ----------
    mode : "growing" | "grow_then_slide" | "fixed_slide" | "multi_origin", optional
        See module docstring. Default `"growing"` reproduces this module's original behavior
        exactly.
    init_points : int, optional
        Number of `t_eval` points in the very first (warm-start) span, for every mode except
        `"fixed_slide"` (which uses `window_width` instead, defaulting to this same span if not
        given) - shared across all `"multi_origin"` starts too (every origin's own first span has
        this many points).
    growth_factor : float, optional
        Geometric growth of the TIME step between successful GROWING rounds (in TIME units, not
        index count - see `growing_horizon.py`'s own reasoning, unchanged here). Ignored once
        `"grow_then_slide"` has switched to sliding, and by `"fixed_slide"` entirely.
    corrector : "newton" | "adam", optional
        How each round's corrector snaps back onto `o(span)`. "newton" (default): damped Newton
        with backtracking (see `correct_newton_spans`) - needs very few steps once warm-started
        well, but (confirmed on Bled) can fail to converge at all within `corrector_steps` once the
        window is long/poorly conditioned, in which case it's cut off mid-divergence rather than
        settling anywhere. "adam": plain `torch.optim.Adam` steps on the same windowed loss, no
        Hessian for the step itself (still computed afterward for the `pd_check` diagnostic) -
        typically needs far more `corrector_steps` (pass a much larger value), but degrades more
        gracefully than a cut-off Newton iteration.
    corrector_steps : int, optional
        Max steps per round - damped-Newton iterations for `corrector="newton"` (with early stop on
        small gradient norm and backtracking on the loss value), or plain Adam steps for
        `corrector="adam"`.
    corrector_lr : float, optional
        Learning rate for `corrector="adam"` - ignored for `"newton"`.
    max_window_length : float, optional
        Hard cap, in `t_eval` TIME units, on how wide a single span may grow (`T - t0`). `None`
        (default): unbounded (grows until `pd_check`'s own stop condition or the full horizon) - see
        the chat this followed for why an explicit, user-supplied cap is one of two ways to control
        the compounding-error problem a long single-origin span has (the other being the
        `epsilon_tol`-based diagnostic bound below, which is not yet enforced automatically).
    polish, polish_max_iter, polish_tol, polish_lr : optional
        `polish=True` (default `False`): after growing stops (for ANY reason - `max_window_length`,
        the full horizon, or a `pd_check` stall) with `corrector="newton"`, run one FINAL Adam
        refinement on the last window's own loss - up to `polish_max_iter` steps (default `1000`),
        stopping early once `||grad|| < polish_tol` (default `1e-3`), at learning rate `polish_lr`
        (default `1e-2`). Motivated directly by the Bled finding that `corrector_steps`-limited
        Newton can end a round with a large, non-stationary gradient (cut off mid-divergence, not
        at a real optimum) - this gives the fit one more chance to actually converge before being
        reported. Ignored (a no-op) for `corrector="adam"` (which already ran its own budget).
    pd_margin_rel : float, optional
        A span is accepted only if `lambda_min(H) >= pd_margin_rel * lambda_max(H)` - the real
        stall/bifurcation-or-non-identifiability threshold, used to end growing (`"growing"`), end
        sliding (`"grow_then_slide"`/`"fixed_slide"`), and to grow each independent start
        (`"multi_origin"`) as well as to judge the final pooled fit there. Ignored entirely when
        `pd_check=False`.
    pd_check : bool, optional
        Default `True`. When `False`, SKIPS the extra Hessian evaluation `correct_spans` otherwise
        does purely to report `lambda_min`/`lambda_max` (the Newton/GN step's OWN Hessian, needed
        for the step itself, is still computed regardless - this only removes the one-off,
        diagnostic-only evaluation after the corrector has already converged). `lambda_min`/
        `lambda_max` become `nan` and every stall/switch check based on them becomes a no-op (a
        span is never rejected for being non-PD) - every mode then just runs to the full horizon.
        Motivated directly by measured profiling (see the chat this followed): on `im_gh` this
        extra Hessian is cheap (no ODE solve, ~4ms/call) and skipping it saves ~nothing; the SAME
        option exists on `growing_horizon.py` where it is NOT free (~2s/call there, since its own
        Hessian requires an actual ODE integration + double backprop) - added here too for a
        consistent, apples-to-apples comparison between the two. Turn this off when you don't
        intend to use the stall diagnostic and structural non-identifiability is expected/accepted
        (silently landing on an arbitrary point of a degenerate parameter combination, same as
        `gradient_matching`/`homotopy_observer`/Adam-based methods already do - see the chat).
    switch_margin_rel : float, optional
        `"grow_then_slide"`-only. A LOOSER threshold than `pd_margin_rel` (must be `>=
        pd_margin_rel` to have any effect) that triggers the switch from growing to sliding BEFORE
        an actual stall - i.e. while the span is merely "getting risky", not yet failing. Default
        `1e-3`, two orders of magnitude looser than `pd_margin_rel`'s own default `1e-5`.
    window_width : int, optional
        `"fixed_slide"`-only. Fixed span width in GRID-INDEX units (not time). `None` (default):
        use `init_points - 1` (the same width the other modes' own first span has).
    slide_step : int, optional
        `"grow_then_slide"`/`"fixed_slide"`-only. How many grid points the span advances per
        sliding round. `1` (default): finest-grained, most robust; raise for a coarser/faster slide
        on a long `t_eval`.
    n_starts : int, optional
        `"multi_origin"`-only. Number of independent growing origins, spread evenly across the
        span of valid starting indices (every origin needs room for its own `init_points`-wide
        first span). An origin whose own first span already fails `pd_margin_rel` is skipped (not
        counted as a failure of the whole fit) - printed if `verbose`.
    norm, l1_eps, hessian_mode : optional
        Same as before - see `_penalty`/this module's `hessian_mode` docstring in the previous
        version (unchanged): "l1" is `multishooting_adaptive._penalty`'s smoothed (pseudo-Huber)
        absolute value; `hessian_mode="gauss_newton"` uses a provably PSD Hessian approximation
        (`H_GN = (1/n) J^T diag(g''(rho)) J`), so a stall/non-PD signal under it can only reflect a
        genuinely rank-deficient Jacobian, never spurious negative curvature from the residual's
        own nonlinearity.
    precondition : bool, optional
        Diagonal (Jacobi-style) preconditioning: the corrector's free variable becomes `phi =
        theta/s` (`s` a FIXED, theta-independent per-constant scale, estimated once from the very
        first GM warm start: `s_i = max(|theta0_i|, precondition_floor)`), so a constant with a
        naturally larger/smaller magnitude gets a correspondingly larger/smaller step. Motivated by
        a DIRECT comparison across every method in this codebase's own benchmark suite (see the
        chat this followed): every method that behaves reasonably on Protein Transduction (whose 6
        constants span very different natural magnitudes) has SOME per-constant scale-awareness
        built in - `scipy.optimize.least_squares`'s `bounds=...,method="trf"` (`gradient_matching`/
        `integral_matching`), `torch.optim.Adam`'s own per-parameter adaptive step
        (`homotopy_observer`/`ms_weighted_sum`), or fgpgm's per-constant MCMC proposal std - while
        every method using a RAW, unscaled Newton/Hessian step (`ms_constraints` without bounds,
        `multishooting_adaptive`, `growing_horizon.py`, and this module before this option existed)
        struggles or fails outright there. NOTE: full NEWTON (using the EXACT Hessian) is itself
        provably invariant to any linear reparameterization like this - preconditioning changes
        nothing there. What it DOES change: the isotropic damping `reg*I` (added in `phi`-units,
        where the natural curvature spread is intended to be more even, instead of in raw `theta`-
        units, where one direction's curvature can be orders of magnitude larger purely from scale)
        and the `pd_margin_rel`/`switch_margin_rel` ratio checks (less likely to be swamped by a
        pure scale artifact instead of genuine ill-conditioning) - and, for `hessian_mode=
        "gauss_newton"`, the Gauss-Newton approximation itself is NOT reparameterization-invariant
        (it drops the residual-curvature term, which does not transform as cleanly), so
        preconditioning can matter there too. Default `False` reproduces this module's original
        behavior exactly (`s` all-ones).
    precondition_floor : float, optional
        Floor for `s_i` above (avoids dividing by ~0 for a constant whose GM warm start lands very
        close to zero). Default `1e-3`.
    epsilon_tol, gronwall_p : float, optional
        DIAGNOSTIC ONLY (not yet enforced - see the chat this followed): every growth round logs
        `history[...]["delta_max"]`, a Gronwall-style upper bound on how long the CURRENT window
        could safely grow before its own accumulated interpolation error exceeds `epsilon_tol`
        (default `1e-3`):

            delta_max = epsilon_tol / (L_F * z * sigma_max)

        `L_F`: a running (not certified) estimate of `sup||dF/dx||`, the largest operator norm of
        `F`'s own Jacobian w.r.t. the state seen at any quadrature node visited so far (updated,
        never decreased, every round). `sigma_max`: the largest GP posterior std (`_FittedGP.var`)
        over the CURRENT window's own nodes - a proxy for the interpolant's own worst-case error.
        `z`: the two-sided normal quantile for confidence `1-gronwall_p` (`gronwall_p=0.05` default
        -> `z≈1.96`, i.e. `sigma_max` is scaled to (approximately) a 95% pointwise band, NOT a
        rigorous simultaneous band over the whole window - a known, accepted approximation for this
        kind of engineering bound, not a formal coverage guarantee). Compare `history[...]["T"] -
        history[...]["t0"]` (the window's ACTUAL width) against `delta_max` to see whether growth
        has already run past what this bound calls safe - exactly what happened on Bled (see the
        chat this followed). This bound only controls interpolation-error accumulation - it says
        nothing about the SEPARATE early-vs-late panel-reuse bias also discussed there.

        `history[...]["E_hat"]` is a SEPARATE, TIGHTER estimate of the same accumulated-error
        quantity: `delta_max` inverts a worst-case bound (`sup||dF/dx|| * width * sup(error)` -
        replacing every local value with the single worst one over the whole window), whereas
        `E_hat` is the actual quadrature-weighted integral

            E_hat = integral_{t0}^{T} ||dF/dx(tau)|| * z * sigma_GP(tau) dtau

        computed via the SAME Simpson nodes/weights already used for the panel integrals - i.e. it
        never substitutes a smaller local value for a larger one just because they share a window
        (see `_integrated_error_estimate`). Compare `E_hat` directly against `epsilon_tol`, rather
        than converting to an implied max length.

    Returns
    -------
    IMGrowingHorizonResult
        `status`: `"completed"` if the full horizon ended up covered (by the single span for
        `"growing"`/`"grow_then_slide"`/`"fixed_slide"`, or - for `"multi_origin"` - if the pooled
        fit's own Hessian passes `pd_margin_rel`), `"stalled"` otherwise. `final_spans`: the span(s)
        the returned `theta` was actually fit on. `history`: one dict per round: `{"t0", "T",
        "lambda_min", "loss", "grad_norm", "step_norm"}` (`"multi_origin"` additionally logs one
        `{"origin_t0", "T", "status", "n_rounds"}` summary dict per origin before the final pooled
        entry).
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_im_growing_horizon requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )
    if switch_margin_rel < pd_margin_rel:
        raise ValueError(f"switch_margin_rel ({switch_margin_rel}) must be >= pd_margin_rel ({pd_margin_rel}).")
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    n_grid = len(t_eval)
    if init_points < 2:
        raise ValueError(f"init_points must be >= 2, got {init_points}.")
    max_i1 = n_grid - 1

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    n_vars = len(state_vars)
    n_consts = len(model.consts)
    qualified = _qualified_var_names(model)
    var_names = [qualified[id(v)] for v in state_vars]

    state_var_names = {name: v for name, v in zip(var_names, state_vars)}
    gps = fit_gps(state_var_names, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)  # (n_vars, n_grid), (name order == state_vars order)

    if norm == "l1" and l1_eps is None:
        gp_mean = torch.stack(
            [torch.as_tensor(gps[name].mean(t_eval), dtype=dtype, device=device) for name in var_names], dim=0,
        )
        l1_eps = max(float((data - gp_mean).abs().median()), 1e-8)
        if verbose:
            print(f"[im_growing_horizon] norm='l1': auto-calibrated l1_eps={l1_eps:.4g} from data noise scale")
    elif l1_eps is None:
        l1_eps = _L1_SMOOTH_EPS

    # --- fixed, theta-independent tensors, computed ONCE ---
    windows = _build_windows(t_eval, stride=1)  # (n_grid-1, 2): consecutive (t_i, t_{i+1}) panels
    nodes, panel_weights = _simpson_nodes_weights(windows, n_panels=n_panels)  # (n_grid-1, n_nodes) each
    n_panel_windows, n_nodes = nodes.shape
    nodes_flat = nodes.reshape(-1)
    x_nodes = torch.stack(
        [torch.as_tensor(gps[name].mean(nodes_flat), dtype=dtype, device=device) for name in var_names], dim=1,
    )  # (n_panel_windows*n_nodes, n_vars)
    t_nodes = torch.as_tensor(nodes_flat, dtype=dtype, device=device)
    panel_weights_t = torch.as_tensor(panel_weights, dtype=dtype, device=device)  # (n_panel_windows, n_nodes)

    x_all = torch.stack(
        [torch.as_tensor(gps[name].mean(t_eval), dtype=dtype, device=device) for name in var_names], dim=1,
    )  # (n_grid, n_vars), x_hat at every t_eval point

    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    _gronwall_z = float(_normal_dist.ppf(1 - gronwall_p / 2))

    def _error_diagnostics(theta_t: torch.Tensor, i1: int) -> "tuple[float, float, float]":
        """`(sup||dF/dx||, sup sigma_GP, E_hat)` over the CURRENT window's own quadrature nodes
        (panels `0..i1-1` - a plain prefix slice of the already-built `t_nodes`/`x_nodes`/
        `panel_weights_t`, since panel `j` occupies `nodes_flat[j*n_nodes:(j+1)*n_nodes]`) - see
        `epsilon_tol`'s own docstring for what these feed into. `dF/dx` via `vmap(jacrev(...))` per
        node (batched, cheap - `n_vars` is small); NOT the same Jacobian `hessian_mode=
        "gauss_newton"` computes (that one is w.r.t. `theta`, over the pooled residual; this one is
        w.r.t. the STATE, per node).

        `E_hat` is the TIGHTER, quadrature-weighted integral estimate (see `epsilon_tol`'s
        docstring) - reuses the exact same per-node `||dF/dx||`/`sigma_GP` values as `L_F`/
        `sigma_max` (computed once, not twice), just combined by a weighted SUM (the same Simpson
        weights `_residuals_full` uses for the panel integrals) instead of a `max`."""
        n_used = i1 * n_nodes
        t_sub = t_nodes[:n_used]
        x_sub = x_nodes[:n_used]
        const_ctx_sub = theta_t.unsqueeze(0).expand(n_used, -1)

        def f_single(x_i, t_i, c_i):
            return rhs(t_i.unsqueeze(0), x_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

        J = vmap(jacrev(f_single, argnums=0))(x_sub, t_sub, const_ctx_sub)  # (n_used, n_vars, n_vars)
        Fx_norm = torch.linalg.matrix_norm(J, ord=2)  # (n_used,)
        L_F = float(Fx_norm.max())

        t_sub_np = t_sub.detach().cpu().numpy()
        sigma_np = np.zeros(n_used)
        for name in var_names:
            sigma_np = np.maximum(sigma_np, np.sqrt(np.maximum(gps[name].var(t_sub_np), 0.0)))
        sigma_max = float(sigma_np.max())

        sigma_t = torch.as_tensor(sigma_np, dtype=dtype, device=device)
        weights_sub = panel_weights_t[:i1].reshape(-1)  # (n_used,), same node order as t_sub/x_sub
        E_hat = float((Fx_norm * _gronwall_z * sigma_t * weights_sub).sum())

        return L_F, sigma_max, E_hat

    # ------------------------------------------------------------------ core span machinery ---

    def _residuals_full(theta_t: torch.Tensor) -> torch.Tensor:
        """`(n_grid, n_vars)` cumulative integral `cum[i] = integral_{t_eval[0]}^{t_eval[i]} F dtau`
        (`cum[0]=0`) - ONE batched RHS evaluation + `cumsum` over the WHOLE horizon, independent of
        which span(s) will actually be read from it (see module docstring)."""
        const_ctx = theta_t.unsqueeze(0).expand(t_nodes.shape[0], -1)
        f_vals = _guard_rhs(rhs(t_nodes, x_nodes, const_ctx), max_rhs).reshape(n_panel_windows, n_nodes, n_vars)
        panel_I = (panel_weights_t.unsqueeze(-1) * f_vals).sum(dim=1)  # (n_panel_windows, n_vars)
        cum_I = torch.cumsum(panel_I, dim=0)  # (n_panel_windows, n_vars)
        zeros = torch.zeros(1, n_vars, dtype=dtype, device=device)
        return torch.cat([zeros, cum_I], dim=0)  # (n_grid, n_vars)

    def _span_residual(cum_I_ext: torch.Tensor, i0: int, i1: int) -> torch.Tensor:
        """`(i1-i0, n_vars)` RAW (unweighted) residual `r(t_i;theta,t_{i0})` for `i=i0+1..i1` - a
        difference of two already-computed `_residuals_full` entries, never a fresh integration."""
        return (x_all[i0 + 1 : i1 + 1] - x_all[i0]) - (cum_I_ext[i0 + 1 : i1 + 1] - cum_I_ext[i0])

    def _span_weight(i0: int, i1: int) -> "torch.Tensor | None":
        """`(i1-i0, n_vars)` uncertainty weight for span `(i0,i1)`, or `None` - theta-INDEPENDENT
        (pure GP quantity), computed once per span definition and reused across every `theta` the
        corrector tries on it."""
        if not weight_by_uncertainty:
            return None
        wv = np.stack(
            [gps[name].window_var(np.full(i1 - i0, t_eval[i0]), t_eval[i0 + 1 : i1 + 1]) for name in var_names],
            axis=1,
        )
        return torch.as_tensor(1.0 / np.sqrt(np.maximum(wv, 1e-12)), dtype=dtype, device=device)

    def _residuals_flat_spans(theta_t: torch.Tensor, spans: "list[Span]", w: "list[torch.Tensor | None]") -> torch.Tensor:
        cum_I_ext = _residuals_full(theta_t)
        parts = []
        for (i0, i1), wi in zip(spans, w):
            r = _span_residual(cum_I_ext, i0, i1)
            if wi is not None:
                r = r * wi
            parts.append(r.reshape(-1))
        return torch.cat(parts)

    def _loss_spans(theta_t: torch.Tensor, spans: "list[Span]", w: "list[torch.Tensor | None]") -> torch.Tensor:
        """Pooled mean loss over one or several `(i0,i1)` spans - a POOLED mean (not an unweighted
        sum of per-span means) so each span contributes proportionally to how many residual entries
        it has, like combining regression datasets of different sizes (see module docstring's
        `"multi_origin"` caveat about the resulting overlap/non-independence)."""
        return _penalty(_residuals_flat_spans(theta_t, spans, w), norm, l1_eps).mean()

    def _loss_and_grad_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[float, np.ndarray]":
        """`phi` is the corrector's own free variable - `theta = s * phi` (elementwise, `s` the
        fixed diagonal preconditioner - see `precondition`). Differentiating w.r.t. `phi` (not
        `theta`) folds the chain-rule factor `s` into the gradient/Hessian automatically via
        autograd; `s` all-ones (the `precondition=False` default) makes `phi` numerically identical
        to `theta` and this exactly reproduces the unpreconditioned corrector."""
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device).requires_grad_(True)
        loss = _loss_spans(s_t * phi_t, spans, w)
        (grad,) = torch.autograd.grad(loss, phi_t)
        return float(loss.item()), grad.detach().cpu().numpy()

    def _loss_only_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> float:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        with torch.no_grad():
            return float(_loss_spans(s_t * phi_t, spans, w).item())

    def _exact_hessian_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device).requires_grad_(True)
        loss = _loss_spans(s_t * phi_t, spans, w)
        (grad,) = torch.autograd.grad(loss, phi_t, create_graph=True)
        H = torch.zeros(n_consts, n_consts, dtype=dtype, device=device)
        for i in range(n_consts):
            (row,) = torch.autograd.grad(grad[i], phi_t, retain_graph=True)
            H[i] = row
        H = 0.5 * (H + H.T)
        return H.detach().cpu().numpy()

    def _gauss_newton_hessian_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        J = jacfwd(lambda p: _residuals_flat_spans(s_t * p, spans, w))(phi_t)  # (sum(i1-i0)*n_vars, n_consts)
        rho = _residuals_flat_spans(s_t * phi_t, spans, w)
        gpp = torch.full_like(rho, 2.0) if norm == "l2" else l1_eps**2 / (rho**2 + l1_eps**2) ** 1.5
        H = (J.T * gpp.unsqueeze(0)) @ J / rho.numel()
        return H.detach().cpu().numpy()

    def _hessian_spans(theta: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        return (
            _exact_hessian_spans(theta, spans, w)
            if hessian_mode == "exact"
            else _gauss_newton_hessian_spans(theta, spans, w)
        )

    def correct_newton_spans(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, float]":
        """Damped Newton with backtracking - see the previous version's own `correct_newton`
        docstring for why `reg` uses a larger-than-textbook relative damping (`1e-4`). Operates on
        `phi` (the corrector's own free variable, `theta = s*phi` - see `precondition`) throughout,
        transparently to this function - it only ever calls `_hessian_spans`/`_loss_and_grad_spans`/
        `_loss_only_spans`, which do the `s`-scaling internally.

        Every point this loop ever evaluates is clipped to `[lo_phi,hi_phi]` (the user-supplied
        `Const.range` bounds, converted to `phi`-space) - see the bounds' own definition above for
        why this matters, especially with `pd_check=False`."""
        theta_c = np.clip(theta0, lo_phi, hi_phi)
        grad_norm = step_norm = float("nan")
        for _ in range(corrector_steps):
            H = _hessian_spans(theta_c, spans, w)
            reg = 1e-4 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            loss0, g = _loss_and_grad_spans(theta_c, spans, w)
            grad_norm = float(np.linalg.norm(g))
            delta = np.linalg.solve(H + reg * np.eye(len(H)), g)
            step_norm = float(np.linalg.norm(delta))
            if grad_norm < 1e-8 * max(np.linalg.norm(theta_c), 1.0):
                break
            step = 1.0
            theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            for _ in range(10):
                loss_try = _loss_only_spans(theta_try, spans, w)
                if loss_try < loss0:
                    break
                step *= 0.5
                theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            theta_c = theta_try
        return theta_c, grad_norm, step_norm

    def correct_adam_spans(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, float]":
        """Plain `torch.optim.Adam` corrector - no Hessian for the STEP itself (still computed
        afterward by `correct_spans`, for the `pd_check` diagnostic). Same `[lo_phi,hi_phi]`
        clipping as `correct_newton_spans`, applied after every step. Returns `(theta, grad_norm,
        step_norm=nan)` - there is no single well-defined "next step" the way `H^-1 grad` is for
        Newton, so `step_norm` (used elsewhere as a numerical-error proxy) has no analogue here."""
        phi_t = torch.as_tensor(theta0, dtype=dtype, device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([phi_t], lr=corrector_lr)
        lo_t = torch.as_tensor(lo_phi, dtype=dtype, device=device)
        hi_t = torch.as_tensor(hi_phi, dtype=dtype, device=device)
        grad_norm = float("nan")
        for _ in range(corrector_steps):
            optimizer.zero_grad()
            loss = _loss_spans(s_t * phi_t, spans, w)
            loss.backward()
            grad_norm = float(phi_t.grad.norm().item())
            optimizer.step()
            with torch.no_grad():
                phi_t.clamp_(lo_t, hi_t)
        return phi_t.detach().cpu().numpy(), grad_norm, float("nan")

    def _adam_polish(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, int]":
        """One-off final Adam refinement after growing stops - see `polish`'s own docstring for
        why. Checks `||grad|| < polish_tol` BEFORE each step (not after), so it never takes a step
        it didn't need. Returns `(theta, grad_norm, n_iter)`."""
        phi_t = torch.as_tensor(theta0, dtype=dtype, device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([phi_t], lr=polish_lr)
        lo_t = torch.as_tensor(lo_phi, dtype=dtype, device=device)
        hi_t = torch.as_tensor(hi_phi, dtype=dtype, device=device)
        grad_norm = float("nan")
        n_iter = 0
        for i in range(polish_max_iter):
            optimizer.zero_grad()
            loss = _loss_spans(s_t * phi_t, spans, w)
            loss.backward()
            grad_norm = float(phi_t.grad.norm().item())
            n_iter = i
            if grad_norm < polish_tol:
                break
            optimizer.step()
            with torch.no_grad():
                phi_t.clamp_(lo_t, hi_t)
        return phi_t.detach().cpu().numpy(), grad_norm, n_iter

    def correct_spans(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, float, float, float]":
        if corrector == "newton":
            theta_c, grad_norm, step_norm = correct_newton_spans(theta0, spans, w)
        else:
            theta_c, grad_norm, step_norm = correct_adam_spans(theta0, spans, w)
        if not pd_check:
            # Skip the Hessian entirely - it's otherwise ONLY used to report lambda_min/lambda_max
            # for the stall/switch checks below, all of which compare `lambda_min < threshold` and
            # so become no-ops (nan is never < anything) - see `pd_check`'s own docstring.
            return theta_c, float("nan"), float("nan"), grad_norm, step_norm
        H_final = _hessian_spans(theta_c, spans, w)
        eig = np.linalg.eigvalsh(H_final)
        return theta_c, float(eig[0]), float(eig[-1]), grad_norm, step_norm

    def _warm_start(i0: int, i1: int) -> np.ndarray:
        gm = estimate_gradient_matching(
            model, t_eval, collocation_times=t_eval[i0 : i1 + 1], gps=gps,
            weight_by_uncertainty=weight_by_uncertainty, device=device, dtype=dtype, verbose=verbose,
        )
        return np.asarray(gm.consts, dtype=float)

    def _establish(i0: int, i1: int) -> "tuple[np.ndarray, float, float, bool]":
        """GM warm start + Newton-correct on span `(i0,i1)` - establishes a genuine `o(span)`
        before any growing/sliding starts (the GM warm start optimizes a DIFFERENT loss, so isn't
        yet a real optimum of THIS one). Returns `(phi, lambda_min, lambda_max, ok)` - `phi`, NOT
        `theta` (see `precondition`): `_warm_start` returns raw `theta` units (GM knows nothing
        about `s`), converted to `phi0 = theta0/s` here before entering the (now `phi`-
        parameterized) corrector. `s` all-ones (default) makes this identical to `theta`."""
        theta0 = _warm_start(i0, i1)
        phi0 = theta0 / s
        w = [_span_weight(i0, i1)]
        phi, lmin, lmax, _, _ = correct_spans(phi0, [(i0, i1)], w)
        ok = True if not pd_check else lmin >= pd_margin_rel * max(lmax, 1e-12)
        return phi, lmin, lmax, ok

    def _grow(
        theta0: np.ndarray, i0: int, i1_start: int, stall_rel: float, dT_init: float, prefix: str = "",
    ) -> "tuple[np.ndarray, str, list[dict], int, float, float]":
        """Grow `i1` forward from `i1_start` (`i0` FIXED) until `lambda_min < stall_rel*lambda_max`
        or `max_i1` is reached. `theta0` must already be a genuine optimum of `(i0,i1_start)` (see
        `_establish`). Returns `(theta, status, history, i1_final, lambda_min, lambda_max)` -
        `status="stalled"` under a LOOSE `stall_rel` (e.g. `switch_margin_rel`) means "time to
        switch strategy", not necessarily a real failure - see `mode="grow_then_slide"`."""
        i1 = i1_start
        span: "list[Span]" = [(i0, i1)]
        w = [_span_weight(i0, i1)]
        theta = theta0
        T_cur = t_eval[i1]
        dT = dT_init
        history: "list[dict]" = []
        status = "completed"
        lmin = lmax = float("nan")
        L_F_running = 0.0  # see `epsilon_tol`'s own docstring - a running, not certified, max
        for round_idx in range(max_rounds):
            if i1 >= max_i1:
                break
            T_new = min(T_cur + dT, t_eval[max_i1])
            capped = False
            if max_window_length is not None:
                T_cap = t_eval[i0] + max_window_length
                if T_new > T_cap:
                    T_new, capped = T_cap, True
            i1_new = min(int(np.searchsorted(t_eval, T_new, side="left")), max_i1)
            if i1_new <= i1:
                if capped:
                    if verbose:
                        print(f"{prefix}round {round_idx:3d} reached max_window_length={max_window_length} - stopped growing")
                    break
                i1_new = i1 + 1  # dT smaller than one grid step - force minimal progress
            span_new: "list[Span]" = [(i0, i1_new)]
            w_new = [_span_weight(i0, i1_new)]

            # --- predictor: secant step, old Hessian + new gradient (see module docstring) ---
            # Damped the SAME way as the corrector's own solve (`1e-4`, see `correct_newton_spans`)
            # - with `pd_check=False`, growth no longer stops the moment `H_old` gets close to
            # singular, so an UNDAMPED solve here can hit an exactly-singular matrix and raise
            # (confirmed directly: `LinAlgError: Singular matrix` on Protein Transduction).
            H_old = _hessian_spans(theta, span, w)
            reg = 1e-4 * max(np.linalg.eigvalsh(H_old)[-1], 1.0)
            _, g_new = _loss_and_grad_spans(theta, span_new, w_new)
            theta_pred = theta - np.linalg.solve(H_old + reg * np.eye(len(H_old)), g_new)

            # --- corrector: a few Newton steps to snap back onto o(span_new) ---
            theta, lmin, lmax, gn, sn = correct_spans(theta_pred, span_new, w_new)
            span, w = span_new, w_new
            i1 = i1_new
            T_cur = t_eval[i1]

            if lmin < stall_rel * max(lmax, 1e-12):
                status = "stalled"
                if verbose:
                    print(f"{prefix}round {round_idx:3d} T=[{t_eval[i0]:.4g},{t_eval[i1]:.4g}] lambda_min={lmin:.4g} - stopped")
                break

            loss_new, _ = _loss_and_grad_spans(theta, span, w)

            # Gronwall-style diagnostics (see `epsilon_tol`'s own docstring) - NOT used to gate
            # anything yet, only logged for inspection. `delta_max`: sup-based bound, inverted to an
            # implied max width. `E_hat`: the tighter, quadrature-weighted integral of the same
            # per-node quantities - compare directly against `epsilon_tol`.
            L_F_est, sigma_est, E_hat = _error_diagnostics(torch.as_tensor(s * theta, dtype=dtype, device=device), i1)
            L_F_running = max(L_F_running, L_F_est)
            delta_max = epsilon_tol / max(L_F_running * _gronwall_z * sigma_est, 1e-300)

            history.append({
                "t0": float(t_eval[i0]), "T": float(t_eval[i1]), "lambda_min": lmin, "loss": loss_new,
                "grad_norm": gn, "step_norm": sn, "L_F": L_F_running, "sigma_max": sigma_est,
                "delta_max": delta_max, "E_hat": E_hat,
            })
            if verbose:
                print(
                    f"{prefix}round {round_idx:3d} T=[{t_eval[i0]:.4g},{t_eval[i1]:.4g}] lambda_min={lmin:.4g} "
                    f"loss={loss_new:.4g} delta_max={delta_max:.4g} E_hat={E_hat:.4g} grad_norm={gn:.4g} "
                    f"(width={t_eval[i1]-t_eval[i0]:.4g})"
                )
            dT *= growth_factor
        return theta, status, history, i1, lmin, lmax

    def _slide(
        theta0: np.ndarray, i0_start: int, i1_start: int, stall_rel: float, prefix: str = "",
    ) -> "tuple[np.ndarray, str, list[dict], int, int, float, float]":
        """Slide `(i0,i1)` forward together (FIXED width) by `slide_step` grid points per round -
        same secant predictor / damped-Newton corrector as `_grow` (its derivation never assumed
        the new span nests the old one - see module docstring). `theta0` must already be a genuine
        optimum of `(i0_start,i1_start)`."""
        w_width = i1_start - i0_start
        i0, i1 = i0_start, i1_start
        span: "list[Span]" = [(i0, i1)]
        w = [_span_weight(i0, i1)]
        theta = theta0
        history: "list[dict]" = []
        status = "completed"
        lmin = lmax = float("nan")
        for round_idx in range(max_rounds):
            if i1 >= max_i1:
                break
            i0_new = min(i0 + slide_step, max_i1 - w_width)
            i1_new = i0_new + w_width
            if i0_new <= i0:
                break  # no room left to slide further
            span_new: "list[Span]" = [(i0_new, i1_new)]
            w_new = [_span_weight(i0_new, i1_new)]

            # Damped the same way as `_grow`'s own predictor - see its comment for why (undamped
            # can hit an exactly-singular `H_old` once `pd_check=False`).
            H_old = _hessian_spans(theta, span, w)
            reg = 1e-4 * max(np.linalg.eigvalsh(H_old)[-1], 1.0)
            _, g_new = _loss_and_grad_spans(theta, span_new, w_new)
            theta_pred = theta - np.linalg.solve(H_old + reg * np.eye(len(H_old)), g_new)

            theta, lmin, lmax, gn, sn = correct_spans(theta_pred, span_new, w_new)
            span, w = span_new, w_new
            i0, i1 = i0_new, i1_new

            if lmin < stall_rel * max(lmax, 1e-12):
                status = "stalled"
                if verbose:
                    print(f"{prefix}round {round_idx:3d} T=[{t_eval[i0]:.4g},{t_eval[i1]:.4g}] lambda_min={lmin:.4g} - stalled")
                break

            loss_new, _ = _loss_and_grad_spans(theta, span, w)
            history.append({
                "t0": float(t_eval[i0]), "T": float(t_eval[i1]), "lambda_min": lmin, "loss": loss_new,
                "grad_norm": gn, "step_norm": sn,
            })
            if verbose:
                print(f"{prefix}round {round_idx:3d} T=[{t_eval[i0]:.4g},{t_eval[i1]:.4g}] lambda_min={lmin:.4g} loss={loss_new:.4g}")
        return theta, status, history, i0, i1, lmin, lmax

    # ------------------------------------------------------------------------------- dispatch ---

    idx0 = min(init_points - 1, max_i1)
    history: "list[dict]" = []
    final_spans: "list[Span]"

    # Diagonal preconditioner `s` - see `precondition`'s own docstring. Estimated ONCE, from a GM
    # warm start on the very first (origin-0) span, and shared across every mode/origin so pooled
    # fits (`"multi_origin"`) stay in one consistent `phi`-parameterization throughout.
    if precondition:
        theta0_ref = _warm_start(0, idx0)
        s = np.maximum(np.abs(theta0_ref), precondition_floor)
        if verbose:
            print(f"[im_growing_horizon] precondition=True: s={s}")
    else:
        s = np.ones(n_consts)
    s_t = torch.as_tensor(s, dtype=dtype, device=device)

    # User-supplied constant bounds (`Const.range`, same ones `gradient_matching`/
    # `integral_matching` already use via `_const_bounds`) - unbounded (+-inf) where not given.
    # Converted once to `phi`-space (`s` is always positive, so division preserves ordering) and
    # enforced by clipping every trial point in `correct_newton_spans` - see its own docstring for
    # why this matters once `pd_check=False` removes the PD-margin safety net: without a bound, a
    # genuinely unidentified direction (see the chat's own c1*c2 toy example) lets the corrector run
    # off to an arbitrarily extreme value along it instead of landing somewhere reasonable.
    lo, hi = _const_bounds(model)
    lo_phi = lo / s
    hi_phi = hi / s

    if mode == "growing":
        theta, lmin, lmax, ok = _establish(0, idx0)
        if not ok:
            status, final_spans = "stalled", [(0, idx0)]
            if verbose:
                print(f"[im_growing_horizon] initial window already non-convex (lambda_min={lmin:.4g}, lambda_max={lmax:.4g}) - stalled immediately")
        else:
            dT0 = max(t_eval[idx0] - t_eval[0], 1e-9)
            theta, status, history, i1_final, lmin, lmax = _grow(theta, 0, idx0, pd_margin_rel, dT0)
            final_spans = [(0, i1_final)]
            if polish and corrector == "newton":
                final_w = [_span_weight(0, i1_final)]
                theta, polish_grad_norm, polish_n_iter = _adam_polish(theta, final_spans, final_w)
                loss_polished, _ = _loss_and_grad_spans(theta, final_spans, final_w)
                history.append({
                    "polish": True, "n_iter": polish_n_iter, "grad_norm": polish_grad_norm, "loss": loss_polished,
                })
                if verbose:
                    print(f"[im_growing_horizon] polish: {polish_n_iter} Adam steps, grad_norm={polish_grad_norm:.4g} loss={loss_polished:.4g}")

    elif mode == "grow_then_slide":
        theta, lmin, lmax, ok = _establish(0, idx0)
        if not ok:
            status, final_spans = "stalled", [(0, idx0)]
            if verbose:
                print(f"[im_growing_horizon] initial window already non-convex (lambda_min={lmin:.4g}, lambda_max={lmax:.4g}) - stalled immediately")
        else:
            dT0 = max(t_eval[idx0] - t_eval[0], 1e-9)
            theta, grow_status, grow_hist, i1_switch, lmin, lmax = _grow(
                theta, 0, idx0, switch_margin_rel, dT0, prefix="[grow] ",
            )
            history = list(grow_hist)
            if grow_status == "completed" or i1_switch >= max_i1:
                status, final_spans = "completed" if i1_switch >= max_i1 else "stalled", [(0, i1_switch)]
            else:
                if verbose:
                    print(f"[im_growing_horizon] switching to sliding at width {i1_switch}-0={i1_switch} grid points")
                theta, status, slide_hist, i0_final, i1_final, lmin, lmax = _slide(
                    theta, 0, i1_switch, pd_margin_rel, prefix="[slide] ",
                )
                history.extend(slide_hist)
                final_spans = [(i0_final, i1_final)]

    elif mode == "fixed_slide":
        w0 = idx0 if window_width is None else min(window_width, max_i1)
        theta, lmin, lmax, ok = _establish(0, w0)
        if not ok:
            status, final_spans = "stalled", [(0, w0)]
            if verbose:
                print(f"[im_growing_horizon] initial window already non-convex (lambda_min={lmin:.4g}, lambda_max={lmax:.4g}) - stalled immediately")
        elif w0 >= max_i1:
            status, final_spans = "completed", [(0, w0)]
        else:
            theta, status, history, i0_final, i1_final, lmin, lmax = _slide(theta, 0, w0, pd_margin_rel, prefix="[slide] ")
            final_spans = [(i0_final, i1_final)]

    elif mode == "multi_origin":
        max_origin = max(max_i1 - idx0, 0)
        origins = sorted(set(int(round(x)) for x in np.linspace(0, max_origin, n_starts)))
        spans: "list[Span]" = []
        thetas: "list[np.ndarray]" = []
        for i0_j in origins:
            i1_j0 = min(i0_j + idx0, max_i1)
            theta_j, lmin_j, lmax_j, ok_j = _establish(i0_j, i1_j0)
            if not ok_j:
                if verbose:
                    print(f"[multi_origin] origin t0={t_eval[i0_j]:.4g}: initial window non-convex (lambda_min={lmin_j:.4g}, lambda_max={lmax_j:.4g}) - skipped")
                continue
            dT0_j = max(t_eval[i1_j0] - t_eval[i0_j], 1e-9)
            theta_j, status_j, hist_j, i1_j, lmin_j, lmax_j = _grow(
                theta_j, i0_j, i1_j0, pd_margin_rel, dT0_j, prefix=f"[origin t0={t_eval[i0_j]:.4g}] ",
            )
            spans.append((i0_j, i1_j))
            thetas.append(theta_j)
            history.append({
                "origin_t0": float(t_eval[i0_j]), "T": float(t_eval[i1_j]), "status": status_j, "n_rounds": len(hist_j),
            })

        if not spans:
            # Degenerate fallback: not even one origin could establish a genuine optimum. `theta`
            # here must still be `phi` (raw/s) - the final `theta = s*theta` conversion below is
            # applied unconditionally, regardless of which branch set `theta`.
            theta = _warm_start(0, idx0) / s
            status, final_spans = "stalled", [(0, idx0)]
            if verbose:
                print("[multi_origin] every origin failed to establish - falling back to a plain GM warm start")
        else:
            theta0_pooled = np.mean(np.stack(thetas), axis=0)
            weights_pooled = [_span_weight(*sp) for sp in spans]
            theta, lmin, lmax, gn, sn = correct_spans(theta0_pooled, spans, weights_pooled)
            loss_pooled, _ = _loss_and_grad_spans(theta, spans, weights_pooled)
            status = "completed" if (not pd_check or lmin >= pd_margin_rel * max(lmax, 1e-12)) else "stalled"
            history.append({
                "pooled": True, "n_spans": len(spans), "lambda_min": lmin, "loss": loss_pooled,
                "grad_norm": gn, "step_norm": sn,
            })
            if verbose:
                print(f"[multi_origin] pooled {len(spans)} span(s), lambda_min={lmin:.4g} loss={loss_pooled:.4g} status={status}")
            final_spans = spans
    else:
        raise ValueError(f"Unknown mode {mode!r}.")

    theta = s * theta  # phi -> real theta units (identity when precondition=False, s all-ones)
    const_by_name = {name: float(theta[c.index_in_ctx]) for name, c in model.consts.items()}
    return IMGrowingHorizonResult(
        model=model,
        t_eval=t_eval,
        consts=theta,
        const_by_name=const_by_name,
        status=status,
        history=history,
        mode=mode,
        final_spans=final_spans,
        windows=windows,
        gps=gps,
    )
