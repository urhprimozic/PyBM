"""
Multishooting ODE estimation with torchode and torch.autograd, batched over multiple candidates.
- ode solved with torchode (Dopri5 + autodiff adjoint)
- gradients computed via torch.autograd.grad (exact, not finite-difference)
- multistart candidates batched together for speed (no Python loop over candidates)


--------------------------------------------------------------------------
IMPORTANT PRECONDITION (per your model convention, engine="torch")
--------------------------------------------------------------------------
This module assumes:
  - `model.engine == "torch"`, and
  - every `Var.ode` / `Var.algebraic` callable you write is expressed with
    differentiable torch operations only
    - all math must be torch ops (no numpy, no scipy, no math module, etc.)
    - use pybm.conditional.RelaxedIfElse instead of classical if-else
--------------------------------------------------------------------------

    NOTE on batching here: `scipy.optimize.minimize` operates on a single
    flat vector -- there is no native way to run one *constrained*
    trust-region optimization jointly over several candidates. So
    "batching across candidates" for this method means: each candidate
    gets its own independent `trust-constr` run (classic multistart), and
    within each run the `n_subintervals` shooting segments of that single
    candidate are solved in one vectorized torchode call. This still gets
    you the real, meaningful speedup (avoiding `n_subintervals` sequential
    `solve_ivp` calls) plus exact gradients. If you want the B candidates
    to run concurrently rather than in a Python loop, see the
    `executor` hook in `_estimate_constraints`.

"weighted_sum"
    Pure torch/torchode, no scipy. Loss:

        loss = A * trajectory_error + B * continuity_penalty

    optimized with a HOMOTOPY / continuation schedule: `B` (the
    continuity weight) starts small -- so the `n_subintervals` shooting
    segments are free to disagree, which makes the early optimization
    landscape much easier (it's just `n_subintervals` independent,
    short, well-conditioned local fits) -- and is annealed up to a large
    final value across iterations, gradually forcing the segments to
    stitch into one continuous trajectory. `A` can be annealed too
    (kept constant by default). All `n_candidates` are optimized jointly
    in one mega-batched torchode call; since none of the loss terms mix
    across candidates, their gradients are independent, so this is
    mathematically equivalent to `n_candidates` parallel single-candidate
    optimizations -- just done as one vectorized computation. Once the
    schedule has fully annealed (A_end, B_end reached), the loop applies
    the same `gtol` used by "constraints"' trust-constr as a plateau
    check and can stop before `max_iter`.

    Optimizer is selectable: "adam" (robust, good default, especially
    while B is still small / the landscape is easy) or "lbfgs" (fast local
    convergence once segments are nearly stitched). You can also run
    "adam" first and feed its result back in as `init_params` for a
    final "lbfgs" polish -- see `estimate_torch`'s docstring for an
    example.

Both methods share `max_iter`/`gtol` (one "how long to optimize" and one
"how converged is converged" knob instead of each method inventing its
own), and both accept an optional `sub_indices` override instead of a
plain `n_subintervals` count -- see `uniform_sub_indices`. Neither of
these builds an adaptive interval scheme; they just make room for one:
a future adaptive scheme only needs to produce its own `sub_indices`
array (e.g. by looking at `_residuals`' per-segment continuity
breakdown, `cont_res_per_segment`, to see which segments are worst
stitched) and hand it to `estimate_torch` -- nothing else in this module
needs to change.

For reconstructing/inspecting a fit's trajectory or scoring a candidate
`consts` against the data, see `pybm.estimate.analysis_torch` -- that's
all post-fit analysis, not part of fitting itself, so it lives separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torchode as to
from scipy.optimize import NonlinearConstraint, minimize

from pybm.model import Choose, Const, Context, InducedModel, Var


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _split_endo_vars(model: InducedModel) -> "tuple[list[Var], list[Var], dict[int, float]]":
    """
    Splits `model`'s endogenous variables into differential ("state", actually integrated by
    torchode), algebraic (re-derived at every RHS evaluation - see `_settle_algebraic`) and
    "frozen" (neither an ODE nor an algebraic equation - held at their initial value for the
    whole trajectory) - mirrors `pybm.simulate.predict.simulate`'s same three-way split, needed
    here for the same reason: a real ProBMoT model can legitimately declare a variable that no
    instantiated process ever writes an equation for (e.g. the chosen structural variant doesn't
    need it).

    Also validates that every equation has actually been resolved (`model.induce()` was called) -
    a stray `Choose` this late means it wasn't.

    Returns `(state_vars, algebraic_vars, frozen_values)`, where `frozen_values` maps a frozen
    variable's `index_in_ctx` to the constant it should be held at.
    """
    all_vars = model.get_endo_variables()
    state_vars = [var for var in all_vars if var.ode is not None]
    algebraic_vars = [var for var in all_vars if var.ode is None and var.algebraic is not None]
    frozen_vars = [var for var in all_vars if var.ode is None and var.algebraic is None]
    for var in all_vars:
        eq = var.ode if var.ode is not None else var.algebraic
        if isinstance(eq, Choose):
            raise ValueError(
                f"Variable {var.name} still has an unresolved Choose() equation. Call "
                "model.induce() to pick a concrete model before estimating."
            )
    frozen_values = {
        var.index_in_ctx: (var.initial if var.initial is not None else 0.0) for var in frozen_vars
    }
    return state_vars, algebraic_vars, frozen_values


def _settle_algebraic(
    algebraic_vars: "list[Var]", t: torch.Tensor, var_slots: "dict[int, torch.Tensor]", const_ctx: torch.Tensor
) -> "dict[int, torch.Tensor]":
    """
    torch/autograd-friendly analogue of `pybm.simulate.predict._settle_algebraic`: fills in
    `algebraic_vars`' values via repeated fixed-point passes, since one algebraic variable can
    depend on another (e.g. `growthRate` depends on `tempGrowthLim`/`nutrientLim`/`lightLim`,
    themselves algebraic) and `Var`/`Process` don't expose a dependency graph to sort by.

    Unlike `predict.py`'s version, this always runs the full, fixed `len(algebraic_vars) + 1`
    passes (the same proven-sufficient bound for any acyclic dependency graph over that many
    variables) instead of stopping early once nothing changes: this runs inside a batched,
    autograd-tracked forward pass (and, for "weighted_sum", under a plain Python loop across
    optimizer iterations too), where a data-dependent stopping condition would need a `.item()`
    call - breaking the graph - and would make different rows of the batch take different numbers
    of passes, which torch has no clean way to express. A fixed pass count sidesteps both.

    `var_slots` is a dict `index_in_ctx -> (N,) tensor` (see `_make_rhs`) - updated out-of-place
    each pass (a fresh dict, not a mutated one) for the same reason `_make_rhs` builds its
    `derivatives` out-of-place: staying friendly to autograd / torch.compile.
    """
    for _ in range(len(algebraic_vars) + 1):
        ctx: Context = {"vars": var_slots, "consts": const_ctx}
        var_slots = dict(var_slots)
        for var in algebraic_vars:
            var_slots[var.index_in_ctx] = var.algebraic(t, ctx)
    return var_slots


def _make_rhs(
    state_vars: "list[Var]", algebraic_vars: "list[Var]", frozen_values: "dict[int, float]"
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Builds the right-hand side f(t, y, const_ctx) that torchode integrates - only `state_vars`
    are actually integrated (`y`'s columns, in `state_vars` order); `algebraic_vars` are re-
    derived from scratch at every call (`_settle_algebraic`) and `frozen_values` are held
    constant, so that any state var's equation reading one of them (e.g. growth reading a
    temperature-limitation factor) still sees a correct, current value.

    Shapes (N = mega-batch size = n_candidates * n_subintervals, flattened):
        t         : (N,)
        y         : (N, len(state_vars))
        const_ctx : (N, n_consts)
    Returns:
        dy/dt     : (N, len(state_vars))

    `ctx["vars"]` is a dict `index_in_ctx -> (N,) tensor` (see `_settle_algebraic`) rather than a
    dense array - `Var.__call__` only ever does `ctx["vars"][self.index_in_ctx]`, which a dict
    supports identically to an array/list, and a dict sidesteps having to know/pre-allocate the
    model's full endogenous-variable count here. `const_ctx` is still transposed to (n_consts, N)
    so `Const.__call__`'s `ctx["consts"][index]` gets back the right (N,) shape.
    """

    def rhs(t: torch.Tensor, y: torch.Tensor, const_ctx_args: torch.Tensor) -> torch.Tensor:
        const_ctx = const_ctx_args.T  # (n_consts, N) -> const_ctx[idx] has shape (N,)
        batch_size = y.shape[0]

        var_slots: "dict[int, torch.Tensor]" = {}
        for i, var in enumerate(state_vars):
            var_slots[var.index_in_ctx] = y[:, i]
        for index, value in frozen_values.items():
            var_slots[index] = torch.full((batch_size,), value, dtype=y.dtype, device=y.device)
        # zero-valued placeholders for every algebraic slot, refined by _settle_algebraic below -
        # without these, a first-pass read of an algebraic var that hasn't been computed yet (e.g.
        # growthRate reading tempGrowthLim, both algebraic) would find no entry at all instead of
        # a harmless 0.0 (see predict.py's dense, zero-initialized var_ctx for the same idea).
        for var in algebraic_vars:
            var_slots[var.index_in_ctx] = torch.zeros(batch_size, dtype=y.dtype, device=y.device)
        var_slots = _settle_algebraic(algebraic_vars, t, var_slots, const_ctx)

        ctx: Context = {"vars": var_slots, "consts": const_ctx}
        # Build derivatives out-of-place (via stack, not in-place index_put)
        # so we stay friendly to autograd / torch.compile.
        derivatives: list[torch.Tensor] = [
            torch.zeros(batch_size, dtype=y.dtype, device=y.device) for _ in state_vars
        ]
        for i, var in enumerate(state_vars):
            derivatives[i] = var.ode(t, ctx)
        return torch.stack(derivatives, dim=-1)  # (N, len(state_vars))

    return rhs


# A trial point that makes the ODE blow up doesn't come back as NaN from
# torchode -- it comes back with a bad `Solution.status` and whatever
# garbage values it had integrated up to that point. `_solve_segments`
# stamps every point of such a (candidate, segment) with this value, the
# same role `dummy_value` plays in `int_scipy.py`'s scipy-based
# multishooting: something clearly, finitely worse than any real fit, so
# the optimizer is pushed away from that region instead of seeing NaN
# (which would corrupt every downstream gradient) or plausible-looking
# partial garbage (which the loss might not even flag as bad).
_DIVERGED_VALUE = 1e6


def _make_solver(
    state_vars: list[Var],
    algebraic_vars: list[Var],
    frozen_values: "dict[int, float]",
    atol: float,
    rtol: float,
    max_steps: Optional[int] = 2000,
    dt_min: Optional[float] = None,
) -> to.AutoDiffAdjoint:
    """
    `max_steps` bounds how long a single (forward or adjoint-backward)
    integration is allowed to run. Without it, an unstable trial point
    during optimizer search (e.g. trust-constr probing around a good
    guess) can make the ODE blow up, and the adaptive step controller
    will keep shrinking its step size chasing that instability -- taking
    an enormous number of steps rather than raising, which looks like the
    optimizer being "stuck": the outer iteration count doesn't move
    because it's still waiting on a single objective/constraint call.
    `int_torch.py`'s single-shooting `simulate` guards against the exact
    same failure mode the same way.

    `dt_min`, if given, complements `max_steps`: instead of (or in
    addition to) capping the *total* step count, it fails a trajectory
    the moment the adaptive controller would need a smaller step than
    this to keep local error under control -- usually the first sign of
    the same blow-up, so it can be caught even earlier. There's no
    universally sane default for this (unlike `max_steps`, it's in the
    same time units as your `t_eval`, so what's "too small a step" is
    model-specific) -- pass one that makes sense for your model's
    timescale if you want this extra guard.

    Either way, torchode has no native per-step event/early-termination
    hook the way `scipy.integrate.solve_ivp(events=...)` does (which is
    what `int_scipy.py`'s `max_offset` bail-out relies on) -- a failed
    trajectory here is caught via `Solution.status` after the (bounded)
    solve returns, in `_solve_segments`, not interrupted mid-flight.
    """
    term = to.ODETerm(_make_rhs(state_vars, algebraic_vars, frozen_values), with_args=True)  # type: ignore[arg-type]
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term, dt_min=dt_min)
    return to.AutoDiffAdjoint(step_method, step_size_controller, max_steps=max_steps)  # type: ignore[arg-type]


def uniform_sub_indices(t_eval: np.ndarray, n_subintervals: int) -> np.ndarray:
    """
    K+1 boundary indices into `t_eval`, splitting it into `n_subintervals`
    ~equal segments (by index count) -- the same convention used
    throughout this module and by `pybm.estimate.gradient_matching`'s
    `init_params`. It's the single place this split is computed, so a
    future adaptive scheme (variable segment lengths, refined where
    `cont_res_per_segment` below is largest) can build its own
    `sub_indices` array and pass it directly to `_build_subinterval_grid`
    / `_sample_initial_params` / `estimate_torch` instead.
    """
    return np.linspace(0, len(t_eval) - 1, n_subintervals + 1, dtype=int)


@dataclass
class _SubintervalGrid:
    """Precomputed, shared-across-candidates time layout for the shooting segments."""

    sub_indices: np.ndarray  # (K+1,) boundary indices into t_eval
    lengths: list[int]  # (K,) true (unpadded) number of points per segment
    max_len: int  # padded length used for the torchode batch call
    t_grid: torch.Tensor  # (K, max_len) padded per-segment time points


def _build_subinterval_grid(
    t_eval: np.ndarray,
    n_subintervals: Optional[int],
    device,
    dtype,
    sub_indices: Optional[np.ndarray] = None,
) -> _SubintervalGrid:
    """
    Splits t_eval into segments and stacks their time points into a single
    padded (K, max_len) tensor so all K segments can be solved in one
    torchode batch call.

    `sub_indices`, if given, overrides the uniform `n_subintervals` split
    (see `uniform_sub_indices`) with explicit segment-boundary indices.

    Segments can have slightly different lengths (integer division
    remainder). Shorter segments are padded by repeating their own last
    time point -- torchode just takes zero-duration steps there, and the
    resulting (repeated) y-values are simply never read back (see
    `lengths` / `_stitch_trajectory`).
    """
    if sub_indices is None:
        assert n_subintervals is not None
        sub_indices = uniform_sub_indices(t_eval, n_subintervals)
    else:
        sub_indices = np.asarray(sub_indices, dtype=int)
    n_subintervals = len(sub_indices) - 1

    segments = [t_eval[sub_indices[i] : sub_indices[i + 1] + 1] for i in range(n_subintervals)]
    lengths = [len(seg) for seg in segments]
    max_len = max(lengths)

    t_grid = torch.zeros(n_subintervals, max_len, dtype=dtype, device=device)
    for i, seg in enumerate(segments):
        seg_t = torch.as_tensor(seg, dtype=dtype, device=device)
        t_grid[i, : len(seg)] = seg_t
        if len(seg) < max_len:
            t_grid[i, len(seg) :] = seg_t[-1]

    return _SubintervalGrid(sub_indices=sub_indices, lengths=lengths, max_len=max_len, t_grid=t_grid)


def _get_data_tensor(vars_: list[Var], t_eval: np.ndarray, device, dtype) -> torch.Tensor:
    """Returns shape (n_vars, T). One-off setup cost, not part of the autograd graph."""
    T = len(t_eval)
    data = torch.zeros(len(vars_), T, dtype=dtype, device=device)
    for i, var in enumerate(vars_):
        if var.data is None:
            raise ValueError(f"Variable {var.name} does not have data defined.")
        for j, t in enumerate(t_eval):
            value = var.data(t)
            if value is None:
                raise ValueError(f"Variable {var.name} returned no data at time {t}.")
            data[i, j] = float(value)
    return data


def _prepare_problem(
    model: InducedModel,
    t_eval: np.ndarray,
    n_subintervals: int,
    device,
    dtype,
    solver_atol: float,
    solver_rtol: float,
    sub_indices: Optional[np.ndarray] = None,
    solver_max_steps: Optional[int] = 2000,
    solver_dt_min: Optional[float] = None,
):
    """
    Shared setup for both estimate_torch methods: vars, grid, data, solver. Only differential
    ("state") variables are returned as `vars_`/integrated/compared against data - algebraic and
    frozen variables are handled internally by the solver's RHS (see `_make_rhs`,
    `_split_endo_vars`), never exposed as free/fitted quantities here.
    """
    state_vars, algebraic_vars, frozen_values = _split_endo_vars(model)
    n_vars = len(state_vars)
    n_consts = len(model.consts)

    grid = _build_subinterval_grid(t_eval, n_subintervals, device, dtype, sub_indices=sub_indices)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)
    solver = _make_solver(
        state_vars, algebraic_vars, frozen_values, solver_atol, solver_rtol,
        max_steps=solver_max_steps, dt_min=solver_dt_min,
    )

    return state_vars, n_vars, n_consts, grid, data, solver


def _solve_segments(
    vars_: list[Var],
    const_ctx: torch.Tensor,  # (B, n_consts)
    initials: torch.Tensor,  # (B, K, n_vars)
    grid: _SubintervalGrid,
    solver: to.AutoDiffAdjoint,
) -> torch.Tensor:
    """
    Solves ALL (candidate, segment) shooting problems in one vectorized
    torchode call. Returns ys of shape (B, K, max_len, n_vars) (padded
    tail entries repeat the last true state -- see `grid.lengths`).
    """
    B, K, n_vars = initials.shape
    n_consts = const_ctx.shape[1]
    max_len = grid.max_len

    # Flatten (B, K) -> mega-batch N = B*K, ordering n = b*K + i.
    y0_flat = initials.reshape(B * K, n_vars)
    t_flat = grid.t_grid.unsqueeze(0).expand(B, K, max_len).reshape(B * K, max_len)
    # consts don't vary across segments of the same candidate -> repeat_interleave
    const_flat = const_ctx.repeat_interleave(K, dim=0)  # (B*K, n_consts)

    problem = to.InitialValueProblem(y0=y0_flat, t_eval=t_flat)  # type: ignore[arg-type]
    sol = solver.solve(problem, args=const_flat)
    ys = sol.ys.reshape(B, K, max_len, n_vars)

    # `sol.status` (0 == torchode's Status.SUCCESS) is the authoritative
    # per-(candidate, segment) signal that this trial point blew up, hit
    # `max_steps`, or (with `dt_min` set) needed too small a step -- see
    # `_make_solver`. A failed row isn't necessarily NaN (torchode just
    # stops and leaves whatever it had), so checking status directly is
    # more reliable than only scanning for NaN/inf afterwards. Stamp the
    # whole failed trajectory with `_DIVERGED_VALUE`: same intent as
    # `int_scipy.py`'s `max_offset`/`dummy_value` bail-out, applied
    # post-hoc since torchode has no native mid-integration event hook.
    failed = (sol.status != 0).reshape(B, K)
    ys = torch.where(failed[:, :, None, None], torch.full_like(ys, _DIVERGED_VALUE), ys)

    # Belt-and-suspenders: a row torchode still reports SUCCESS on can in
    # principle contain a NaN/inf (e.g. from the RHS itself producing one,
    # such as division-by-near-zero in a Michaelis-Menten term). Same
    # dummy-value treatment, so a NaN never reaches the loss.
    return torch.nan_to_num(ys, nan=_DIVERGED_VALUE, posinf=_DIVERGED_VALUE, neginf=-_DIVERGED_VALUE)


def _residuals(
    ys: torch.Tensor,  # (B, K, max_len, n_vars)
    initials: torch.Tensor,  # (B, K, n_vars)
    grid: _SubintervalGrid,
    data: torch.Tensor,  # (n_vars, T)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds the batched loss ingredients, analogous to `residuals` and
    `smoothness_penalty` in the scipy version:

    traj_res[b] : MEAN squared error (per time point, per variable)
                  between candidate b's stitched trajectory (segments
                  concatenated, dropping each segment's duplicated first
                  point, exactly like the scipy code) and the observed
                  `data`.

    cont_res[b] : MEAN of the multiple-shooting MATCHING CONDITION for
                  candidate b -- how far the ODE-propagated end of
                  segment i is from the free initial-condition parameter
                  that seeds segment i+1, averaged over segment gaps and
                  variables.

    cont_per_segment[b] : the same matching condition, per segment gap,
                  NOT averaged over gaps -- shape (B, K-1). This is what a
                  future adaptive-interval scheme would look at to decide
                  *which* segments need to be split further, instead of
                  only seeing the aggregate `cont_res`.

    Both `traj_res` and `cont_res` are means, not sums, so `A`/`B` in the
    "weighted_sum" method (and `gtol` for "constraints") mean roughly the
    same thing regardless of how many data points or segments you use.

    Returns (traj_res, cont_res, cont_per_segment).
    """
    B, K, _, n_vars = ys.shape
    T = data.shape[1]
    device, dtype = ys.device, ys.dtype

    stitched = torch.zeros(B, T, n_vars, device=device, dtype=dtype)
    for i in range(K):
        L = grid.lengths[i]
        start = int(grid.sub_indices[i])
        seg = ys[:, i, :L, :]  # (B, L, n_vars)
        if i == 0:
            stitched[:, start : start + L, :] = seg
        else:
            # skip the segment's first point: it duplicates the previous
            # segment's last point in the shared t_eval grid
            stitched[:, start + 1 : start + L, :] = seg[:, 1:, :]

    diff = stitched - data.T.unsqueeze(0)  # (1,T,n_vars) broadcast over B
    traj_res = (diff**2).mean(dim=(1, 2))  # (B,)

    if K > 1:
        cont_terms = []
        for i in range(K - 1):
            L = grid.lengths[i]
            pred_end = ys[:, i, L - 1, :]  # (B, n_vars): end of segment i, propagated
            seed_next = initials[:, i + 1, :]  # (B, n_vars): free var seeding segment i+1
            cont_terms.append(((pred_end - seed_next) ** 2).mean(dim=-1))  # (B,), mean over n_vars
        cont_per_segment = torch.stack(cont_terms, dim=1)  # (B, K-1)
        cont_res = cont_per_segment.mean(dim=1)  # (B,), mean over gaps too
    else:
        cont_per_segment = torch.zeros(B, 0, device=device, dtype=dtype)
        cont_res = torch.zeros(B, device=device, dtype=dtype)

    return traj_res, cont_res, cont_per_segment


def _stitch_trajectory(ys: torch.Tensor, grid: _SubintervalGrid) -> torch.Tensor:
    """Stitch segment solutions into a single trajectory of shape (B, T, n_vars)."""
    B, K, _, n_vars = ys.shape
    T = int(grid.sub_indices[-1]) + 1
    device, dtype = ys.device, ys.dtype

    stitched = torch.zeros(B, T, n_vars, device=device, dtype=dtype)
    for i in range(K):
        L = grid.lengths[i]
        start = int(grid.sub_indices[i])
        seg = ys[:, i, :L, :]
        if i == 0:
            stitched[:, start : start + L, :] = seg
        else:
            stitched[:, start + 1 : start + L, :] = seg[:, 1:, :]
    return stitched


def _sample_initial_params(
    model: InducedModel,
    t_eval: np.ndarray,
    n_subintervals: int,
    n_candidates: int,
    device,
    dtype,
    jitter: float = 0.1,
    seed: Optional[int] = None,
    sub_indices: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Naive multistart sampler -- a reasonable default, NOT the "avoid
    resampling near bad basins" scheme discussed earlier. Swap this out
    for e.g. a TikTak-style seeding once you have that in place; nothing
    downstream cares how `init_params` was produced. See also
    `pybm.estimate.gradient_matching.estimate_gradient_matching`, which
    replaces the naive uniform-random constants below with a fitted guess.

    Layout matches the scipy version: for each candidate, a flat vector
    [c_1..c_n, v_1(t_0)..v_m(t_0), ..., v_1(t_K)..v_m(t_K)], where v_1..v_m are the model's
    differential ("state") variables only - algebraic/frozen variables aren't part of `y` at all
    (see `_split_endo_vars`), so there is nothing to seed for them.

    - Constants: candidate 0 uses `const.initial_value` (or the midpoint
      of `const.range`); the rest are drawn uniformly from `const.range`
      (or a generic window around the base value if no range is given).
    - Segment seed states: drawn from the observed data at each segment's
      start time plus multiplicative-ish gaussian jitter (candidate 0 is
      kept exactly at the data-based guess, matching the scipy version's
      un-jittered initial guess).
    """
    rng = np.random.default_rng(seed)
    vars_, _, _ = _split_endo_vars(model)
    consts = list(model.consts.values())
    n_consts = len(consts)
    n_vars = len(vars_)

    const_ctx = torch.zeros(n_candidates, n_consts, dtype=dtype, device=device)
    for c in consts:
        assert c.index_in_ctx is not None
        if c.range is not None:
            lo, hi = c.range
        else:
            base = c.initial_value if c.initial_value is not None else 0.0
            width = abs(base) + 1.0
            lo, hi = base - width, base + width
        draws = rng.uniform(lo, hi, size=n_candidates)
        draws[0] = c.initial_value if c.initial_value is not None else 0.5 * (lo + hi)
        const_ctx[:, c.index_in_ctx] = torch.as_tensor(draws, dtype=dtype)

    if sub_indices is None:
        sub_indices = uniform_sub_indices(t_eval, n_subintervals)
    else:
        sub_indices = np.asarray(sub_indices, dtype=int)
    n_subintervals = len(sub_indices) - 1

    initials = torch.zeros(n_candidates, n_subintervals, n_vars, dtype=dtype, device=device)
    for i in range(n_subintervals):
        t0 = t_eval[sub_indices[i]]
        base_vals = np.array(
            [var.data(t0) if var.data is not None else (var.initial or 0.0) for var in vars_],
            dtype=float,
        )
        noise = rng.normal(scale=jitter * (np.abs(base_vals) + 1e-6), size=(n_candidates, n_vars))
        noise[0, :] = 0.0
        initials[:, i, :] = torch.as_tensor(base_vals[None, :] + noise, dtype=dtype)

    return torch.cat([const_ctx, initials.reshape(n_candidates, -1)], dim=1)


# ---------------------------------------------------------------------------
# "constraints" method: torchode for eval+grad, scipy trust-constr for the
# actual constrained optimization (one independent run per candidate).
# ---------------------------------------------------------------------------


@dataclass
class ConstraintsResult:
    consts : np.ndarray  # best constants
    x: np.ndarray  # best flat params
    cost: float  # trajectory residual at x
    continuity_violation: float
    cont_res_per_segment: np.ndarray  # (K-1,), per-segment breakdown of continuity_violation
    scipy_result: Any
    all_results: list[Any] = field(default_factory=list)


@dataclass
class _ForwardCache:
    """
    Single-slot cache keyed on the bytes of the trial point `x`. scipy's
    trust-constr evaluates `objective`, `constraint_fun` and
    `constraint_jac` as separate callbacks, but frequently at the same
    `x` within one iteration -- this avoids redoing the (adjoint-capable)
    forward solve up to 3x per iterate. A grad-enabled entry can serve a
    no-grad request, but not vice versa, so a no-grad hit never poisons a
    later request that actually needs `.grad`.
    """

    key: Optional[bytes] = None
    entry: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    requires_grad: bool = False

    def get(self, params_np: np.ndarray, need_grad: bool) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        key = params_np.tobytes()
        if self.key == key and self.entry is not None and (not need_grad or self.requires_grad):
            return self.entry
        return None

    def set(self, params_np: np.ndarray, params_t: torch.Tensor, traj_res: torch.Tensor, cont_res: torch.Tensor, requires_grad: bool) -> None:
        self.key = params_np.tobytes()
        self.entry = (params_t, traj_res, cont_res)
        self.requires_grad = requires_grad


def _make_plateau_callback(patience: int, tol: float) -> Callable[[np.ndarray, Any], bool]:
    """
    trust-constr callback that stops the run once `state.fun` (the
    objective, i.e. `traj_res`) hasn't improved by more than a `tol`
    relative amount over the last `patience` iterations.

    trust-constr's own `gtol`/`xtol` stop on local optimality / step size,
    not directly on "is the loss still going down" -- a run can plateau
    (e.g. fighting the constraint, or just slow numerically) without ever
    satisfying those, and would otherwise keep churning to `maxiter`.
    Returning `True` from a trust-constr callback terminates the run.
    """
    history: list[float] = []

    def callback(xk: np.ndarray, state: Any) -> bool:
        history.append(float(state.fun))
        if len(history) < patience + 1:
            return False
        window = history[-(patience + 1):]
        rel_change = abs(window[0] - window[-1]) / max(abs(window[0]), 1e-12)
        return rel_change < tol

    return callback


def _estimate_constraints_single(
    vars_: list[Var],
    grid: _SubintervalGrid,
    data: torch.Tensor,
    solver: to.AutoDiffAdjoint,
    init_params: np.ndarray,
    n_consts: int,
    n_subintervals: int,
    n_vars: int,
    dtype,
    maxiter: int,
    gtol: float,
    verbose: int,
    patience: int = 10,
):
    """
    One trust-constr run for a single candidate. Objective, constraint,
    and their exact Jacobians all go through ONE torchode call each
    (batched over this candidate's n_subintervals segments -- batch size
    here is just K, not K*n_candidates, since scipy drives one candidate
    at a time), reused via `_ForwardCache` when trust-constr asks for
    several of {value, jac} at the same x.
    """
    cache = _ForwardCache()

    def unpack(params_t: torch.Tensor):
        const_ctx = params_t[:n_consts].unsqueeze(0)  # (1, n_consts)
        initials = params_t[n_consts:].reshape(1, n_subintervals, n_vars)  # (1, K, n_vars)
        return const_ctx, initials

    def forward(params_np: np.ndarray, need_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = cache.get(params_np, need_grad)
        if cached is not None:
            return cached

        params_t = torch.as_tensor(params_np, dtype=dtype)
        if need_grad:
            params_t.requires_grad_(True)
            const_ctx, initials = unpack(params_t)
            ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
            traj_res, cont_res, _ = _residuals(ys, initials, grid, data)
        else:
            with torch.no_grad():
                const_ctx, initials = unpack(params_t)
                ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
                traj_res, cont_res, _ = _residuals(ys, initials, grid, data)

        cache.set(params_np, params_t, traj_res[0], cont_res[0], need_grad)
        return params_t, traj_res[0], cont_res[0]

    def objective(params_np: np.ndarray):
        params_t, traj_res, _ = forward(params_np, need_grad=True)
        (grad,) = torch.autograd.grad(traj_res, params_t, retain_graph=True)
        return traj_res.item(), grad.detach().cpu().numpy()

    def constraint_fun(params_np: np.ndarray):
        _, _, cont_res = forward(params_np, need_grad=False)
        return np.array([cont_res.item()])

    def constraint_jac(params_np: np.ndarray):
        params_t, _, cont_res = forward(params_np, need_grad=True)
        (grad,) = torch.autograd.grad(cont_res, params_t, retain_graph=True)
        return grad.detach().cpu().numpy().reshape(1, -1)

    # With a single segment (n_subintervals == 1, plain single-shooting) there is no continuity
    # gap to enforce at all - `_residuals` returns `cont_res = torch.zeros(...)` as a literal
    # constant in that case (see its own K==1 branch), disconnected from `params_t`'s graph, so
    # differentiating it below would raise ("does not have a grad_fn"). Skip the constraint
    # entirely rather than building one around a quantity that isn't a function of anything.
    constraints = (
        [NonlinearConstraint(constraint_fun, 0, 0, jac=constraint_jac)]  # type: ignore[arg-type]
        if n_subintervals > 1
        else []
    )
    result = minimize(
        fun=objective,
        x0=np.asarray(init_params, dtype=float),
        jac=True,
        constraints=constraints,
        method="trust-constr",
        options={"maxiter": maxiter, "verbose": verbose, "gtol": gtol},
        callback=_make_plateau_callback(patience, gtol),
    )
    return result


def _estimate_constraints(
    model: InducedModel,
    t_eval: np.ndarray,
    n_subintervals: int,
    init_params_batch: torch.Tensor,  # (B, n_consts + K*n_vars)
    device,
    dtype,
    solver_atol: float,
    solver_rtol: float,
    maxiter: int,
    gtol: float,
    verbose: int,
    executor=None,  # e.g. a concurrent.futures.Executor, to run candidates concurrently
    sub_indices: Optional[np.ndarray] = None,
    solver_max_steps: Optional[int] = 2000,
    solver_dt_min: Optional[float] = None,
    patience: int = 10,
) -> ConstraintsResult:
    vars_, n_vars, n_consts, grid, data, solver = _prepare_problem(
        model, t_eval, n_subintervals, device, dtype, solver_atol, solver_rtol,
        sub_indices=sub_indices, solver_max_steps=solver_max_steps, solver_dt_min=solver_dt_min,
    )
    n_subintervals = len(grid.sub_indices) - 1
    B = init_params_batch.shape[0]

    init_np = init_params_batch.detach().cpu().numpy()

    def run_one(b: int):
        return _estimate_constraints_single(
            vars_, grid, data, solver, init_np[b], n_consts, n_subintervals, n_vars,
            dtype, maxiter, gtol, verbose, patience=patience,
        )

    if executor is None:
        # Sequential multistart: B independent local searches, one per candidate.
        results = [run_one(b) for b in range(B)]
    else:
        # Caller-provided executor for concurrent candidates. Note: the torchode
        # `solver` / model closures capture torch tensors, so make sure your
        # executor (e.g. a thread pool) is compatible with how you've set up
        # torch's threading; a process pool would need everything to be
        # picklable, which the closures above are not, as written.
        results = list(executor.map(run_one, range(B)))

    best_idx = int(np.argmin([r.fun for r in results]))
    best = results[best_idx]

    # continuity violation at the winning x, for reporting
    params_t = torch.as_tensor(best.x, dtype=dtype)
    const_ctx = params_t[:n_consts].unsqueeze(0)
    initials = params_t[n_consts:].reshape(1, n_subintervals, n_vars)
    with torch.no_grad():
        ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
        _, cont_res_t, cont_per_segment_t = _residuals(ys, initials, grid, data)

    return ConstraintsResult(
        consts=best.x[:n_consts],
        x=best.x,
        cost=float(best.fun),
        continuity_violation=float(cont_res_t.item()),
        cont_res_per_segment=cont_per_segment_t[0].detach().cpu().numpy(),
        scipy_result=best,
        all_results=results,
    )


# ---------------------------------------------------------------------------
# "weighted_sum" method: pure torch/torchode, homotopy-annealed A/B, jointly
# batched over all n_candidates.
# ---------------------------------------------------------------------------


@dataclass
class WeightedSumResult:
    consts : torch.Tensor | Any # (B, n_consts), final constants for ALL candidates
    x : torch.Tensor | Any # (B, n_consts + K*n_vars), final flat params for ALL candidates
    params: torch.Tensor  # (B, n_consts + K*n_vars), final params for ALL candidates
    traj_res: torch.Tensor  # (B,) final trajectory residual per candidate
    cont_res: torch.Tensor  # (B,) final continuity residual per candidate
    cont_res_per_segment: torch.Tensor  # (B, K-1) per-segment breakdown of cont_res
    best_index: int
    n_iter_run: int  # actual iterations run (== max_iter unless gtol early-stopped it)
    history: list[dict]  # per-logged-iteration diagnostics


def _homotopy_weight(start: float, end: float, frac: float, mode: Literal["linear", "exp"]) -> float:
    """frac in [0,1]. 'exp' interpolates geometrically -- useful when end >> start."""
    if mode == "linear":
        return start + (end - start) * frac
    elif mode == "exp":
        s = max(start, 1e-8)
        e = max(end, 1e-8)
        return s * (e / s) ** frac
    else:
        raise ValueError(f"Unknown homotopy_schedule {mode!r}")


def _estimate_weighted_sum(
    model: InducedModel,
    t_eval: np.ndarray,
    n_subintervals: int,
    init_params_batch: torch.Tensor,  # (B, n_consts + K*n_vars), requires_grad set here
    device,
    dtype,
    solver_atol: float,
    solver_rtol: float,
    max_iter: int,
    optimizer_name: Literal["adam", "lbfgs"],
    lr: float,
    A_start: float,
    A_end: float,
    B_start: float,
    B_end: float,
    homotopy_schedule: Literal["linear", "exp"],
    gtol: float,
    log_every: int,
    verbose: int,
    sub_indices: Optional[np.ndarray] = None,
    solver_max_steps: Optional[int] = 2000,
    solver_dt_min: Optional[float] = None,
) -> WeightedSumResult:
    vars_, n_vars, n_consts, grid, data, solver = _prepare_problem(
        model, t_eval, n_subintervals, device, dtype, solver_atol, solver_rtol,
        sub_indices=sub_indices, solver_max_steps=solver_max_steps, solver_dt_min=solver_dt_min,
    )
    n_subintervals = len(grid.sub_indices) - 1
    B = init_params_batch.shape[0]

    # single leaf tensor holding ALL candidates' parameters -- gradients for
    # different candidates never mix (see _residuals: everything is batched
    # along dim 0 without cross terms), so one joint optimizer step is
    # exactly equivalent to B independent per-candidate steps.
    params = init_params_batch.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

    if optimizer_name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam([params], lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    else:
        raise ValueError(f"Unknown optimizer_name {optimizer_name!r}")

    history: list[dict] = []

    def compute_loss(A: float, Bw: float):
        const_ctx = params[:, :n_consts]
        initials = params[:, n_consts:].reshape(B, n_subintervals, n_vars)
        ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
        traj_res, cont_res, cont_per_segment = _residuals(ys, initials, grid, data)
        per_candidate = A * traj_res + Bw * cont_res
        return per_candidate.sum(), traj_res, cont_res, cont_per_segment

    prev_loss: Optional[float] = None
    n_iter_run = max_iter
    for it in range(max_iter):
        frac = it / max(max_iter - 1, 1)
        A = _homotopy_weight(A_start, A_end, frac, homotopy_schedule)
        Bw = _homotopy_weight(B_start, B_end, frac, homotopy_schedule)

        if optimizer_name == "adam":
            optimizer.zero_grad()
            loss, traj_res, cont_res, cont_per_segment = compute_loss(A, Bw)
            loss.backward()
            optimizer.step()  # type: ignore[call-arg]
        else:  # lbfgs

            def closure():
                optimizer.zero_grad()
                loss, _, _, _ = compute_loss(A, Bw)
                loss.backward()
                return loss

            optimizer.step(closure)
            # Don't rely on `optimizer.step(closure)`'s return value for
            # bookkeeping (LBFGS returns the *first* closure evaluation,
            # typed `Optional[float]` regardless) -- recompute at the
            # final iterate instead, same as the adam branch does anyway.
            with torch.no_grad():
                loss, traj_res, cont_res, cont_per_segment = compute_loss(A, Bw)

        if log_every and (it % log_every == 0 or it == max_iter - 1):
            entry = {
                "iter": it,
                "A": A,
                "B": Bw,
                "traj_res": traj_res.detach().cpu().numpy().copy(),
                "cont_res": cont_res.detach().cpu().numpy().copy(),
                "cont_res_per_segment": cont_per_segment.detach().cpu().numpy().copy(),
            }
            history.append(entry)
            if verbose:
                print(
                    f"[weighted_sum {it:5d}] A={A:.4g} B={Bw:.4g} "
                    f"traj_res(min/mean)={entry['traj_res'].min():.4g}/{entry['traj_res'].mean():.4g} "
                    f"cont_res(min/mean)={entry['cont_res'].min():.4g}/{entry['cont_res'].mean():.4g}"
                )

        # Early stopping, shared with "constraints"' `gtol`: only once the
        # homotopy schedule has fully annealed (frac >= 1) does a plateau
        # in the loss mean the fit itself has converged, rather than just
        # A/B still changing underneath it.
        loss_value = float(loss.detach())
        if frac >= 1.0 and prev_loss is not None:
            rel_change = abs(prev_loss - loss_value) / max(abs(prev_loss), 1e-12)
            if rel_change < gtol:
                n_iter_run = it + 1
                break
        prev_loss = loss_value

    with torch.no_grad():
        _, final_traj_res, final_cont_res, final_cont_per_segment = compute_loss(A_end, B_end)

    # pick the best candidate among those that are (approximately) actually
    # continuous; among the rest, rank by trajectory error. Threshold is the
    # batch MEDIAN (not mean) so a couple of badly diverged candidates don't
    # drag the "good enough" cutoff along with them.
    cont_ok = final_cont_res < (final_cont_res.median() + 1e-6)
    ranking_key = final_traj_res + torch.where(cont_ok, torch.zeros_like(final_cont_res), final_cont_res)
    best_index = int(torch.argmin(ranking_key).item())

    return WeightedSumResult(
        consts=params[:, :len(model.consts)].flatten().detach(),
        x=params.detach().flatten(),
        params=params.detach(),
        traj_res=final_traj_res.detach(),
        cont_res=final_cont_res.detach(),
        cont_res_per_segment=final_cont_per_segment.detach(),
        best_index=best_index,
        n_iter_run=n_iter_run,
        history=history,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def estimate_torch(
    model: InducedModel,
    t_eval,
    method: Literal["constraints", "weighted_sum"] = "weighted_sum",
    n_subintervals: int = 1,
    sub_indices: Optional[np.ndarray] = None,
    n_candidates: int = 1,
    init_params: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    solver_atol: float = 1e-8,
    solver_rtol: float = 1e-6,
    solver_max_steps: Optional[int] = 2000,
    solver_dt_min: Optional[float] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    # -- shared across both methods --
    max_iter: int = 200,
    gtol: float = 1e-6,
    # -- "constraints"-only knobs --
    executor=None,
    patience: int = 10,
    # -- "weighted_sum"-only knobs --
    optimizer_name: Literal["adam", "lbfgs"] = "adam",
    lr: float = 1e-2,
    A_start: float = 1.0,
    A_end: float = 1.0,
    B_start: float = 0.0,
    B_end: float = 1e3,
    homotopy_schedule: Literal["linear", "exp"] = "exp",
    log_every: int = 10,
):
    """
    Torch/torchode multishooting estimator, batched multistart over
    `n_candidates`.

    Parameters
    ----------
    model : InducedModel
        Must have been built with `engine="torch"`, with `Choose` objects
        already resolved (via `model.induce()`), and with every `Var.ode`
        written using differentiable torch operations.
    t_eval : array-like
        Time points to fit against (same role as in the scipy version).
    method : "constraints" | "weighted_sum"
        See the module docstring for the trade-offs.
    n_subintervals : int
        Number of shooting segments (K). K=1 recovers plain single-shooting.
        Ignored if `sub_indices` is given.
    sub_indices : array-like, optional
        Explicit (K+1,) segment-boundary indices into `t_eval`, overriding
        the uniform `n_subintervals` split (see `uniform_sub_indices`).
        The hook for a future adaptive-interval scheme -- everything else
        in this module treats the grid as opaque either way.
    n_candidates : int
        Number of multistart candidates (B) solved in one batched pass
        ("weighted_sum") or as B independent local searches
        ("constraints").
    init_params : (B, n_consts + K*n_vars) tensor, optional
        Initial guesses, same flat layout as the scipy version:
        [c_1..c_n, v_1(t_0)..v_m(t_0), ..., v_1(t_K)..v_m(t_K)] per
        candidate, where v_1..v_m are the model's differential ("state")
        variables only (see `_split_endo_vars`) - algebraic/frozen
        variables have nothing to seed. If omitted, a naive multistart
        sampler is used (see
        `_sample_initial_params`) -- for something better than uniform
        random constants, see `pybm.estimate.gradient_matching`, whose
        result's `.init_params(...)` builds exactly this layout.
    max_iter : int
        Shared "how long to optimize" budget: `trust-constr`'s `maxiter`
        for "constraints", or the gradient-descent loop count for
        "weighted_sum" (which can still stop earlier -- see `gtol`).
    gtol : float
        Shared convergence tolerance: `trust-constr`'s `gtol` for
        "constraints"; for "weighted_sum", once the A/B homotopy schedule
        has fully annealed, the loop stops early if the relative loss
        change drops below `gtol`. For "constraints", also reused as the
        plateau threshold for `patience` below.
    patience : int
        "constraints"-only. trust-constr's own `gtol`/`xtol` stop on local
        optimality / step size, not directly on "is the loss still going
        down" -- a run can plateau without satisfying those and otherwise
        keep churning to `max_iter`. This stops it once the objective
        hasn't improved by more than `gtol` (relative) over the last
        `patience` iterations (see `_make_plateau_callback`).
    solver_max_steps : int, optional
        Caps a single forward/adjoint-backward integration. An unstable
        trial point during optimizer search (e.g. trust-constr probing
        around a good guess) can otherwise make the adaptive step
        controller take an enormous number of steps chasing a blow-up
        instead of raising -- which looks like the optimizer being
        "stuck" (the outer iteration count doesn't move because it's
        still waiting on a single objective/constraint call). Set to
        `None` for torchode's own (unbounded) default.
    solver_dt_min : float, optional
        Complements `solver_max_steps`: fails a trajectory the moment the
        adaptive step controller would need a smaller step than this,
        usually the earliest sign of the same blow-up. No sane universal
        default -- it's in the same time units as your `t_eval`, so pass
        one that fits your model's timescale if you want this extra guard
        (see `_make_solver`).

    Returns
    -------
    ConstraintsResult or WeightedSumResult (see their docstrings/fields).

    Example: Adam warm start, then LBFGS polish
    --------------------------------------------
    >>> warm = estimate_torch(model, t_eval, method="weighted_sum",
    ...                        n_subintervals=5, n_candidates=20,
    ...                        optimizer_name="adam", max_iter=2000)
    >>> polished = estimate_torch(model, t_eval, method="weighted_sum",
    ...                            n_subintervals=5, n_candidates=20,
    ...                            init_params=warm.params,
    ...                            optimizer_name="lbfgs", max_iter=100,
    ...                            B_start=warm.history[-1]["B"], B_end=warm.history[-1]["B"])
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_torch requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )

    t_eval = np.asarray(t_eval, dtype=float)
    device = device or torch.device("cpu")

    if sub_indices is not None:
        sub_indices = np.asarray(sub_indices, dtype=int)
        n_subintervals = len(sub_indices) - 1

    if init_params is None:
        init_params = _sample_initial_params(
            model, t_eval, n_subintervals, n_candidates, device, dtype, seed=seed, sub_indices=sub_indices
        )
    else:
        init_params = init_params.to(device=device, dtype=dtype)
        if init_params.shape[0] != n_candidates:
            raise ValueError(
                f"init_params has batch size {init_params.shape[0]}, but n_candidates={n_candidates}."
            )

    if method == "constraints":
        return _estimate_constraints(
            model, t_eval, n_subintervals, init_params, device, dtype,
            solver_atol, solver_rtol, max_iter, gtol, verbose, executor=executor,
            sub_indices=sub_indices, solver_max_steps=solver_max_steps, solver_dt_min=solver_dt_min,
            patience=patience,
        )
    elif method == "weighted_sum":
        return _estimate_weighted_sum(
            model, t_eval, n_subintervals, init_params, device, dtype,
            solver_atol, solver_rtol, max_iter, optimizer_name, lr,
            A_start, A_end, B_start, B_end, homotopy_schedule, gtol, log_every, verbose,
            sub_indices=sub_indices, solver_max_steps=solver_max_steps, solver_dt_min=solver_dt_min,
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'constraints' or 'weighted_sum'.")
