"""
Adaptive-window multishooting: grow the fitted time window instead of fixing `n_subintervals` up
front, using a per-round Hessian-positive-definiteness check as the "is this window still nice"
certificate.

Method (agreed with the user across a long design discussion - see the chat, not reproduced here)
--------------------------------------------------------------------------------------------------
1. Warm-start `theta` (and, in `mode="multishooting"`, the first segment's own seed `s_0`) from
   `estimate_gradient_matching`, fit against only a SHORT initial window - short enough that the
   loss surface there is close to convex ("strong overfit" anchor).
2. Fit that window (a small weighted-sum multishooting/single-shooting optimization, reusing
   `pybm.estimate.multishooting_torch`'s low-level solve/residual machinery directly - NOT
   `estimate_torch` itself, since this module needs a residual weighted by GP posterior
   uncertainty, which `estimate_torch` does not support; see `_round_fit`).
3. Check the Hessian of the NEWEST block's own loss (in `mode="multishooting"`: just the newest
   segment's own seed plus the shared `theta`, re-propagating only the ONE preceding segment
   needed to evaluate its continuity term - cost independent of how many segments have
   accumulated; in `mode="single_shooting"` there is only ever one, ever-growing segment, so this
   collapses to the whole `(theta, x0)` pair and its cost grows with the window, unavoidably).
4. If the Hessian stays positive definite with a safety margin, grow the window (geometrically) and
   repeat, warm-started from the current fit. If not, bisect the growth step and retry from the
   same boundary. If the growth step bisects below a floor, stop (`status="stalled"`) rather than
   push into a region this method has no certificate for.

What this does and does NOT prove
----------------------------------
Tracking a Hessian-positive-definite path is a real, established idea - homotopy/continuation
methods (Allgower & Georg, *Numerical Continuation Methods*, 1990), graduated non-convexity (Blake
& Zisserman 1987), formalized by the parametric Implicit Function Theorem (Fiacco, *Introduction to
Sensitivity and Stability Analysis in NLP*, 1983): as long as the Hessian of the loss w.r.t. the
free parameters stays positive definite along the path, that path is a UNIQUE, smoothly-varying
branch of local minimizers - it cannot silently jump to some unrelated critical point. That is a
genuine "no arbitrary jump" certificate, strictly more than what a fixed-`n_subintervals`
`estimate_torch` call gives you (which has no memory of where its solution came from).

It does NOT prove convergence to the GLOBAL minimum / true parameters. This module only checks the
Hessian ALONG the ONE path it is tracking - it has no way to know whether some other, unreached
basin would score better. A true global guarantee would need either (a) genuine convexity of the
loss for every window length up to the full span (defeats the point of this method), or (b) a
topological-degree argument certifying nonsingularity over the WHOLE reachable parameter region,
not just the tracked point (as hard as the original global-optimization problem). Treat this as a
principled, certifiable-per-step HEURISTIC for choosing multishooting segment lengths (replacing
the ad hoc "start short, lengthen uniformly" of `uniform_sub_indices`), not a global-optimality
proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import torch

from pybm.estimate.depr.gradient_matching import (
    _FittedGP,
    _qualified_var_names,
    estimate_gradient_matching,
    fit_gps,
)
from pybm.estimate.depr.multishooting_torch import (
    _SubintervalGrid,
    _build_subinterval_grid,
    _homotopy_weight,
    _make_solver,
    _solve_segments,
)
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _get_data_tensor, _make_rhs


# ---------------------------------------------------------------------------
# GP-derived seeds and uncertainty weights (state, not derivative: this module fits SIMULATED
# trajectories against raw data, unlike gradient_matching, which fits derivatives)
# ---------------------------------------------------------------------------


def _seed_at(gps: "dict[str, _FittedGP]", qualified: "dict[int, str]", vars_: "list[Var]", t: float, device, dtype) -> torch.Tensor:
    """GP posterior mean of every state var at a single time `t`, in `vars_` order."""
    vals = np.array([gps[qualified[id(v)]].mean(np.array([t]))[0] for v in vars_], dtype=float)
    return torch.as_tensor(vals, dtype=dtype, device=device)


def _state_weights(gps: "dict[str, _FittedGP]", qualified: "dict[int, str]", vars_: "list[Var]", t_eval: np.ndarray, device, dtype) -> torch.Tensor:
    """
    `(T, n_vars)` weight `1/sqrt(Var[x̂(t)])` from each variable's own GP posterior - the state-value
    analogue of `gradient_matching._fit_constants`'s `deriv_var_at_collocation` weighting. Uses
    `_FittedGP.var` (the STATE's own posterior variance), not `var_deriv`: this module compares a
    genuinely simulated trajectory against the data, so the relevant uncertainty is about the state
    value itself, not (as in plain gradient matching) about a GP-implied derivative.
    """
    cols = [1.0 / np.sqrt(np.maximum(gps[qualified[id(v)]].var(t_eval), 1e-12)) for v in vars_]
    return torch.as_tensor(np.stack(cols, axis=1), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Weighted residuals (like multishooting_torch._residuals, plus an optional uncertainty weight -
# not added to that module directly to avoid touching its existing, unweighted behavior)
# ---------------------------------------------------------------------------


def _weighted_residuals(
    ys: torch.Tensor,  # (1, K, max_len, n_vars)
    initials: torch.Tensor,  # (1, K, n_vars)
    grid: _SubintervalGrid,
    data: torch.Tensor,  # (n_vars, T) - T == grid.sub_indices[-1] + 1, i.e. THIS window's own data slice
    weight: "torch.Tensor | None",  # (T, n_vars) or None
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    K = ys.shape[1]
    n_vars = ys.shape[-1]
    T = data.shape[1]
    device, dtype = ys.device, ys.dtype

    stitched = torch.zeros(1, T, n_vars, device=device, dtype=dtype)
    for i in range(K):
        L = grid.lengths[i]
        start = int(grid.sub_indices[i])
        seg = ys[:, i, :L, :]
        if i == 0:
            stitched[:, start : start + L, :] = seg
        else:
            stitched[:, start + 1 : start + L, :] = seg[:, 1:, :]

    diff = stitched - data.T.unsqueeze(0)
    if weight is not None:
        diff = diff * weight.unsqueeze(0)
    traj_res = (diff**2).mean(dim=(1, 2))

    if K > 1:
        L0 = grid.lengths[0]
        pred_end = ys[:, 0, L0 - 1, :]
        seed_next = initials[:, 1, :]
        cont_res = ((pred_end - seed_next) ** 2).mean(dim=-1)
    else:
        cont_res = torch.zeros(1, device=device, dtype=dtype)

    return traj_res, cont_res, stitched


# ---------------------------------------------------------------------------
# Per-round fit: a small weighted-sum optimization over the CURRENT window only (a prefix of
# t_eval), warm-started from the previous round's params. Deliberately separate from
# `estimate_torch` (see module docstring) rather than a modification of it.
# ---------------------------------------------------------------------------


def _round_fit(
    state_vars: "list[Var]",
    solver,
    t_window: np.ndarray,
    sub_indices: np.ndarray,
    data_window: torch.Tensor,
    weight_window: "torch.Tensor | None",
    init_params: torch.Tensor,  # (1, n_consts + K*n_vars)
    n_consts: int,
    n_vars: int,
    device,
    dtype,
    max_iter: int,
    lr: float,
    gtol: float,
    B_start: float,
    B_end: float,
    homotopy_schedule: Literal["linear", "exp"],
) -> "tuple[torch.Tensor, float, float]":
    K = len(sub_indices) - 1
    grid = _build_subinterval_grid(t_window, None, device, dtype, sub_indices=sub_indices)
    params = init_params.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=lr)

    def compute(Bw: float):
        const_ctx = params[:, :n_consts]
        initials = params[:, n_consts:].reshape(1, K, n_vars)
        ys = _solve_segments(state_vars, const_ctx, initials, grid, solver)
        traj_res, cont_res, _ = _weighted_residuals(ys, initials, grid, data_window, weight_window)
        return (traj_res + Bw * cont_res).sum(), traj_res, cont_res

    prev_loss: "float | None" = None
    for it in range(max_iter):
        frac = it / max(max_iter - 1, 1)
        Bw = _homotopy_weight(B_start, B_end, frac, homotopy_schedule)
        optimizer.zero_grad()
        loss, traj_res, cont_res = compute(Bw)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if frac >= 1.0 and prev_loss is not None:
            rel = abs(prev_loss - loss_value) / max(abs(prev_loss), 1e-12)
            if rel < gtol:
                break
        prev_loss = loss_value

    with torch.no_grad():
        _, traj_res, cont_res = compute(B_end)
    return params.detach(), float(traj_res.item()), float(cont_res.item())


# ---------------------------------------------------------------------------
# Newest-block Hessian: only the newest segment's own seed + the shared theta (plus, if it exists,
# a re-propagation of the ONE preceding segment needed for the continuity term) - cost independent
# of how many segments have accumulated so far (mode="multishooting"). In mode="single_shooting"
# there is only ever one segment, so this is the whole (theta, x0) pair and its cost grows with the
# window - see module docstring.
# ---------------------------------------------------------------------------


_L1_SMOOTH_EPS = 1e-4  # default/fallback - see _penalty's own docstring; growing_horizon.py
# overrides this with a noise-scale-calibrated value instead of relying on the default (see there).


def _penalty(diff: torch.Tensor, norm: Literal["l2", "l1"], eps: float = _L1_SMOOTH_EPS) -> torch.Tensor:
    """Per-element penalty applied to a residual `diff` before `.mean()`-reducing it to a scalar
    loss. `"l2"`: `diff**2`, as everywhere in this module until now. `"l1"`: a SMOOTHED absolute
    value (pseudo-Huber, `sqrt(diff**2 + eps**2) - eps`), NOT raw `diff.abs()` - callers here
    (`_exact_block_hessian`, used by `growing_horizon.py`'s Newton corrector) need a genuine,
    non-degenerate Hessian; raw L1's second derivative is zero almost everywhere (a Dirac delta at
    the origin), which would make Newton's step direction undefined. The smoothing is negligible
    for `|diff| >> eps` (matches Bock, Kostina & Schloder 2007's l1 formulation there) while keeping
    curvature well-defined near zero residual.

    `eps` MUST be calibrated to the typical scale of `diff` at the optimum (i.e. to the data noise
    level), not left at some arbitrary fixed constant - verified directly (see the chat this
    followed): on Lotka-Volterra (noise_std=0.1, so converged residuals are themselves ~0.1),
    the old fixed `eps=1e-4` is 1000x smaller than that scale, so pseudo-Huber's second derivative
    (`eps**2 / (diff**2+eps**2)**1.5`, which is ~`eps**2/|diff|**3` once `|diff| >> eps`) collapses
    to ~1e-5 almost everywhere - an effectively-singular Hessian for `growing_horizon.py`'s
    Newton corrector and its PD-margin stall check, causing a false-positive "bifurcation" signal
    and premature stopping. On Protein Transduction (noise_std=0.001, comparable to the old fixed
    eps) this problem does not arise, which is why the old fixed value looked fine there."""
    if norm == "l2":
        return diff**2
    return torch.sqrt(diff**2 + eps**2) - eps


def _block_loss(
    z_t: torch.Tensor,
    n_consts: int,
    n_vars: int,
    has_prev: bool,
    prev_seed: "torch.Tensor | None",  # (n_vars,), FIXED (detached) seed of the preceding segment
    state_vars: "list[Var]",
    solver,
    t_local: np.ndarray,  # this block's own time slice (1 or 2 segments), rebased so it starts at t_local[0]
    local_sub_indices: np.ndarray,
    data_local: torch.Tensor,  # (n_vars, len(t_local)) sliced to match t_local
    weight_local: "torch.Tensor | None",
    device,
    dtype,
    norm: Literal["l2", "l1"] = "l2",
    l1_eps: float = _L1_SMOOTH_EPS,
) -> torch.Tensor:
    """The newest block's own scalar loss as a function of `z_t = [theta, s_new]` (or
    `[theta, prev_seed, s_new]` when a preceding segment must be re-propagated) - a plain torch
    computation, reused both by `_block_grad` (one first-order backward, safe everywhere torchode's
    adjoint is used at all) and `_exact_block_hessian` (nested backward, only safe for `has_prev=
    False` - see that function's docstring for why). `norm`/`l1_eps` pick the per-element penalty -
    see `_penalty`."""
    theta = z_t[:n_consts]
    s_new = z_t[n_consts : n_consts + n_vars]

    if has_prev:
        assert prev_seed is not None
        initials = torch.stack([prev_seed, s_new], dim=0).unsqueeze(0)  # (1, 2, n_vars)
    else:
        initials = s_new.reshape(1, 1, n_vars)

    grid = _build_subinterval_grid(t_local, None, device, dtype, sub_indices=local_sub_indices)
    const_ctx = theta.unsqueeze(0)
    ys = _solve_segments(state_vars, const_ctx, initials, grid, solver)

    if has_prev:
        # only the NEW segment's own trajectory error counts (see module docstring point 3) - the
        # preceding segment is re-propagated only to get its end point for the continuity term.
        L_prev = grid.lengths[0]
        seg_new = ys[:, 1, : grid.lengths[1], :]
        data_new = data_local[:, grid.sub_indices[0] + L_prev - 1 :].T.unsqueeze(0)
        # data_new is offset by one (the boundary point is shared/duplicated, matching
        # _weighted_residuals' own convention of dropping a segment's duplicated first point)
        diff = seg_new[:, 1:, :] - data_new[:, 1 : seg_new.shape[1], :]
        w = weight_local
        if w is not None:
            w_new = w[grid.sub_indices[0] + L_prev :, :].unsqueeze(0)
            diff = diff * w_new[:, : diff.shape[1], :]
        traj_res = _penalty(diff, norm, l1_eps).mean()
        pred_end = ys[:, 0, L_prev - 1, :]
        cont_res = _penalty(pred_end - s_new, norm, l1_eps).mean()
        return traj_res + cont_res
    else:
        L = grid.lengths[0]
        seg = ys[:, 0, :L, :]
        diff = seg - data_local.T.unsqueeze(0)
        if weight_local is not None:
            diff = diff * weight_local.unsqueeze(0)
        return _penalty(diff, norm, l1_eps).mean()


def _block_grad(
    z: np.ndarray,
    n_consts: int,
    n_vars: int,
    has_prev: bool,
    prev_seed: "torch.Tensor | None",
    state_vars: "list[Var]",
    solver,
    t_local: np.ndarray,
    local_sub_indices: np.ndarray,
    data_local: torch.Tensor,
    weight_local: "torch.Tensor | None",
    device,
    dtype,
    norm: Literal["l2", "l1"] = "l2",
    l1_eps: float = _L1_SMOOTH_EPS,
) -> np.ndarray:
    """One first-order backward through `_block_loss` - safe everywhere torchode's `AutoDiffAdjoint`
    is used (see `_exact_block_hessian` for what is and isn't safe for a SECOND backward)."""
    z_t = torch.as_tensor(z, dtype=dtype, device=device).requires_grad_(True)
    loss = _block_loss(
        z_t, n_consts, n_vars, has_prev, prev_seed, state_vars, solver,
        t_local, local_sub_indices, data_local, weight_local, device, dtype, norm, l1_eps,
    )
    (grad,) = torch.autograd.grad(loss, z_t)
    return grad.detach().cpu().numpy()


def _finite_diff_hessian(grad_fn, z0: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Central-difference Hessian of `grad_fn`'s underlying loss, via `2*len(z0)` extra gradient
    evaluations. Fallback for the ONE case `_exact_block_hessian` cannot handle (`has_prev=True`,
    i.e. a batched 2-segment torchode solve - see that function's docstring for the direct test that
    found this). Per-dimension relative step (`eps * max(|z0_i|, 1)`) since `z` mixes `theta` and
    state values that can live on very different scales. This is a diagnostic-precision Hessian
    (good enough to judge the SIGN of its smallest eigenvalue), not a high-accuracy one.
    """
    n = len(z0)
    H = np.zeros((n, n))
    for i in range(n):
        h = eps * max(abs(z0[i]), 1.0)
        zp, zm = z0.copy(), z0.copy()
        zp[i] += h
        zm[i] -= h
        H[:, i] = (grad_fn(zp) - grad_fn(zm)) / (2 * h)
    return 0.5 * (H + H.T)


def _exact_block_hessian(
    z0: np.ndarray,
    n_consts: int,
    n_vars: int,
    state_vars: "list[Var]",
    solver,
    t_local: np.ndarray,
    local_sub_indices: np.ndarray,
    data_local: torch.Tensor,
    weight_local: "torch.Tensor | None",
    device,
    dtype,
    norm: Literal["l2", "l1"] = "l2",
    l1_eps: float = _L1_SMOOTH_EPS,
) -> np.ndarray:
    """
    Exact Hessian of `_block_loss` via nested `torch.autograd.grad` (first pass with
    `create_graph=True`, then one more backward per gradient component). ONLY called with
    `has_prev=False` (a single-segment, K=1 torchode solve) - there is no `has_prev`/`prev_seed`
    parameter here on purpose, so a future caller cannot accidentally point this at the K=2 case.

    This is NOT a documentation assumption - it was verified directly (see the chat this followed):
    double backprop through torchode's `AutoDiffAdjoint` gives a CORRECT result for a single-segment
    solve (checked against the analytic Hessian of a toy exponential-decay ODE, matching to 4
    decimal places), but returns NaN on every `theta`-involving entry for a BATCHED two-segment
    solve (the `has_prev=True` case `_block_loss` also builds) - `_finite_diff_hessian` is the
    fallback for that case specifically, not a blanket "torchode never supports this" choice.

    `norm="l1"` uses `_block_loss`'s SMOOTHED L1 (see `_penalty`) precisely so this Hessian stays
    well-defined - raw L1 has none.
    """
    z_t = torch.as_tensor(z0, dtype=dtype, device=device).requires_grad_(True)
    loss = _block_loss(
        z_t, n_consts, n_vars, False, None, state_vars, solver,
        t_local, local_sub_indices, data_local, weight_local, device, dtype, norm, l1_eps,
    )
    (grad,) = torch.autograd.grad(loss, z_t, create_graph=True)
    n = len(z0)
    H = torch.zeros(n, n, dtype=dtype, device=device)
    for i in range(n):
        (row,) = torch.autograd.grad(grad[i], z_t, retain_graph=True)
        H[i] = row
    H = 0.5 * (H + H.T)
    return H.detach().cpu().numpy()


def _newest_block_hessian(
    params: torch.Tensor,  # (1, n_consts + K*n_vars), CURRENT fit (theta + all segment seeds so far)
    sub_indices: np.ndarray,  # (K+1,) global boundary indices into t_eval
    n_consts: int,
    n_vars: int,
    state_vars: "list[Var]",
    solver,
    t_eval: np.ndarray,
    data: torch.Tensor,  # (n_vars, T) full data
    weight: "torch.Tensor | None",  # (T, n_vars) full weight, or None
    device,
    dtype,
    hessian_eps: float,
) -> "tuple[float, float, np.ndarray]":
    """Returns `(lambda_min, lambda_max, H)` of the newest block's own loss Hessian - see the module
    docstring and `_block_grad` for exactly what "newest block" means."""
    K = len(sub_indices) - 1
    theta = params[0, :n_consts].detach().cpu().numpy()
    s_new_idx = n_consts + (K - 1) * n_vars
    s_new = params[0, s_new_idx : s_new_idx + n_vars].detach().cpu().numpy()

    has_prev = K > 1
    if has_prev:
        b0, b1, b2 = int(sub_indices[-3]), int(sub_indices[-2]), int(sub_indices[-1])
        t_local = t_eval[b0 : b2 + 1]
        local_sub_indices = np.array([0, b1 - b0, b2 - b0])
        data_local = data[:, b0 : b2 + 1]
        weight_local = weight[b0 : b2 + 1, :] if weight is not None else None
        s_prev_idx = n_consts + (K - 2) * n_vars
        prev_seed = params[0, s_prev_idx : s_prev_idx + n_vars].detach().to(device=device, dtype=dtype)
        z0 = np.concatenate([theta, s_new])
    else:
        b0, b1 = int(sub_indices[0]), int(sub_indices[1])
        t_local = t_eval[b0 : b1 + 1]
        local_sub_indices = np.array([0, b1 - b0])
        data_local = data[:, b0 : b1 + 1]
        weight_local = weight[b0 : b1 + 1, :] if weight is not None else None
        prev_seed = None
        z0 = np.concatenate([theta, s_new])

    if has_prev:
        # K=2, batched torchode solve - exact double backprop returns NaN here (verified directly,
        # see _exact_block_hessian's docstring), so this is the one case that still needs the FD
        # fallback.
        def grad_fn(z: np.ndarray) -> np.ndarray:
            return _block_grad(
                z, n_consts, n_vars, has_prev, prev_seed, state_vars, solver,
                t_local, local_sub_indices, data_local, weight_local, device, dtype,
            )

        H = _finite_diff_hessian(grad_fn, z0, eps=hessian_eps)
    else:
        # K=1 - exact, via double backprop through torchode's adjoint (verified correct - see
        # _exact_block_hessian's docstring). No `hessian_eps`/FD noise in this path at all.
        H = _exact_block_hessian(
            z0, n_consts, n_vars, state_vars, solver,
            t_local, local_sub_indices, data_local, weight_local, device, dtype,
        )

    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0]), float(eigvals[-1]), H


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class MultishootingAdaptiveResult(ParamEstimationResults):
    model: InducedModel
    mode: Literal["multishooting", "single_shooting"]
    t_eval: np.ndarray
    sub_indices: np.ndarray
    consts: np.ndarray
    const_by_name: "dict[str, float]"
    params: torch.Tensor
    traj_res: float
    cont_res: float
    status: Literal["completed", "stalled"]
    history: "list[dict]" = field(default_factory=list)


def estimate_multishooting_adaptive(
    model: InducedModel,
    t_eval,
    mode: Literal["multishooting", "single_shooting"] = "multishooting",
    weight_by_uncertainty: bool = True,
    init_points: int = 4,
    growth_factor: float = 1.6,
    min_growth_frac: float = 1e-3,
    pd_margin_rel: float = 1e-3,
    max_bisections: int = 6,
    max_rounds: int = 60,
    hessian_eps: float = 1e-3,
    round_max_iter: int = 300,
    round_lr: float = 1e-2,
    round_gtol: float = 1e-6,
    B_start: float = 0.0,
    B_end: float = 1e3,
    homotopy_schedule: Literal["linear", "exp"] = "exp",
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    solver_atol: float = 1e-8,
    solver_rtol: float = 1e-6,
    solver_max_steps: "int | None" = 2000,
    solver_dt_min: "float | None" = None,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    torch_threads: "int | None" = None,
    verbose: int = 0,
) -> MultishootingAdaptiveResult:
    """
    Grow the fitted window from a short, near-convex anchor to the full `t_eval` span, certifying
    each growth step via the newest block's own Hessian positive-definiteness (see module
    docstring). `mode="multishooting"` adds a fresh, GP-seeded segment at each growth step (genuine
    multishooting - a new free initial value per segment, tied to its neighbor by a continuity
    penalty); `mode="single_shooting"` keeps a single, ever-extending segment whose own initial
    value `x0` is a free parameter, warm-started from the GP mean at `t_eval[0]` and never re-seeded.

    Parameters
    ----------
    weight_by_uncertainty : bool, optional
        Default `True`: each collocation point's trajectory residual is divided by the GP's own
        posterior standard deviation of the STATE there (`_FittedGP.var`) - a point the GP itself is
        unsure about (sparse/edge-of-data) contributes less. Pass `False` for plain unweighted MSE.
    init_points : int, optional
        Number of `t_eval` points in the very first (warm-start) window - kept SHORT on purpose, so
        this window's own loss is close to convex ("overfit" anchor - see module docstring).
    growth_factor : float, optional
        Geometric growth of the TIME step between successful rounds (not an index-count step - see
        `estimate_multishooting_adaptive`'s own growth loop for why: a fixed index-count step is a
        poor proxy for "how much harder did the problem get" on a non-uniformly sampled grid).
    min_growth_frac : float, optional
        If the growth step, after `max_bisections` halvings, would still add less than
        `min_growth_frac * (t_eval[-1] - t_eval[0])` of TIME, the run stops (`status="stalled"`)
        instead of forcing a step this method has no PD certificate for.
    pd_margin_rel : float, optional
        A round is accepted only if `lambda_min(H) >= pd_margin_rel * lambda_max(H)` - a safety
        margin, not just `lambda_min > 0` (see module docstring point 3 in the design review this
        followed: pushing exactly to the singularity boundary makes the NEXT round's own fit
        ill-conditioned even though it's technically still PD).
    torch_threads : int, optional
        If given, calls `torch.set_num_threads` before doing any work - use this (plus the usual
        `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` env vars, which must be set
        before numpy/torch are imported and so cannot be set from inside this function) to bound
        memory/CPU use for a long adaptive run.

    Returns
    -------
    MultishootingAdaptiveResult
        `status`: `"completed"` if the window reached the full `t_eval` span, `"stalled"` if growth
        had to stop early (see `min_growth_frac`). `history`: one dict per attempted round -
        `{"t_boundary", "lambda_min", "lambda_max", "dt_step", "accepted", "bisections"}` - the
        `lambda_min` trend across rounds is the practical "Z(t)" diagnostic the design discussion
        this followed was after: a declining trend over several rounds is an early warning of an
        approaching bifurcation, before growth actually stalls.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_multishooting_adaptive requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    T = len(t_eval)
    if init_points < 2:
        raise ValueError(f"init_points must be >= 2, got {init_points}.")

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    n_vars = len(state_vars)
    n_consts = len(model.consts)
    qualified = _qualified_var_names(model)

    # Only state vars need a GP (see estimate_gradient_matching's own `missing` computation, mirrored
    # here) - `model.vars` also holds algebraic/exogenous entries, which need not carry `.data`.
    state_var_names = {qualified[id(v)]: v for v in state_vars}
    gps = fit_gps(state_var_names, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)
    weight = _state_weights(gps, qualified, state_vars, t_eval, device, dtype) if weight_by_uncertainty else None

    solver = _make_solver(
        state_vars, algebraic_vars, frozen_values, solver_atol, solver_rtol,
        max_steps=solver_max_steps, dt_min=solver_dt_min,
    )

    idx0 = min(init_points - 1, T - 1)
    gm0 = estimate_gradient_matching(
        model, t_eval, collocation_times=t_eval[: idx0 + 1], gps=gps,
        weight_by_uncertainty=weight_by_uncertainty, device=device, dtype=dtype, verbose=verbose,
    )
    theta0 = torch.as_tensor(gm0.consts, dtype=dtype, device=device)
    s0 = _seed_at(gps, qualified, state_vars, t_eval[0], device, dtype)
    sub_indices = np.array([0, idx0])
    params = torch.cat([theta0, s0]).unsqueeze(0)  # (1, n_consts + n_vars)

    history: "list[dict]" = []
    # Growth is in TIME units, not index count: an index-count step (the original design) is a poor
    # proxy for "how much harder did the problem get" on a non-uniformly sampled grid - e.g. PT's own
    # t = [0,1,2,4,5,7,10,15,20,30,40,50,60,80,100] means a 3-index step from t=4 lands at t=10, more
    # than doubling the covered SPAN for the same nominal step size. `dt_step` starts at the actual
    # time span of the initial window, so growth is scaled consistently with what's already covered,
    # regardless of how the grid itself is spaced.
    dt_step = max(t_eval[idx0] - t_eval[0], 1e-9)
    min_dt = min_growth_frac * (t_eval[-1] - t_eval[0])
    status: Literal["completed", "stalled"] = "completed"
    traj_res = cont_res = float("nan")

    for round_idx in range(max_rounds):
        cur_end = int(sub_indices[-1])
        if cur_end >= T - 1:
            break

        bisections = 0
        accepted = False
        while True:
            target_t = min(t_eval[cur_end] + dt_step, t_eval[-1])
            target = int(np.searchsorted(t_eval, target_t, side="left"))
            target = min(max(target, cur_end + 1), T - 1)  # always advance at least one grid point
            if mode == "multishooting":
                new_seed = _seed_at(gps, qualified, state_vars, t_eval[cur_end], device, dtype)
                trial_sub_indices = np.append(sub_indices, target)
                trial_params = torch.cat([params, new_seed.unsqueeze(0)], dim=1)
            else:
                trial_sub_indices = sub_indices.copy()
                trial_sub_indices[-1] = target
                trial_params = params

            t_window = t_eval[: target + 1]
            data_window = data[:, : target + 1]
            weight_window = weight[: target + 1, :] if weight is not None else None

            fit_params, traj_res, cont_res = _round_fit(
                state_vars, solver, t_window, trial_sub_indices, data_window, weight_window,
                trial_params, n_consts, n_vars, device, dtype,
                round_max_iter, round_lr, round_gtol, B_start, B_end, homotopy_schedule,
            )
            lambda_min, lambda_max, _ = _newest_block_hessian(
                fit_params, trial_sub_indices, n_consts, n_vars, state_vars, solver,
                t_eval, data, weight, device, dtype, hessian_eps,
            )
            ok = lambda_min >= pd_margin_rel * max(lambda_max, 1e-12)

            history.append({
                "round": round_idx, "t_boundary": float(t_eval[target]), "dt_step": dt_step,
                "lambda_min": lambda_min, "lambda_max": lambda_max, "accepted": ok,
                "bisections": bisections, "traj_res": traj_res, "cont_res": cont_res,
            })
            if verbose:
                print(
                    f"[ms_adaptive round {round_idx:3d}] t={t_eval[target]:.4g} dt_step={dt_step:.4g} "
                    f"lambda_min={lambda_min:.4g} lambda_max={lambda_max:.4g} accepted={ok}"
                )

            if ok:
                sub_indices, params = trial_sub_indices, fit_params
                dt_step = dt_step * growth_factor
                accepted = True
                break

            bisections += 1
            dt_step = dt_step / 2.0
            if bisections > max_bisections or dt_step < min_dt:
                status = "stalled"
                break

        if not accepted:
            break

    const_by_name = {name: float(params[0, c.index_in_ctx]) for name, c in model.consts.items()}
    return MultishootingAdaptiveResult(
        model=model,
        mode=mode,
        t_eval=t_eval,
        sub_indices=sub_indices,
        consts=params[0, :n_consts].detach().cpu().numpy(),
        const_by_name=const_by_name,
        params=params,
        traj_res=traj_res,
        cont_res=cont_res,
        status=status,
        history=history,
    )
