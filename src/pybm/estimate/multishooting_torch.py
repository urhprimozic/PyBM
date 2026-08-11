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
    optimizations -- just done as one vectorized computation.

    Optimizer is selectable: "adam" (robust, good default, especially
    while B is still small / the landscape is easy) or "lbfgs" (fast local
    convergence once segments are nearly stitched). You can also run
    "adam" first and feed its result back in as `init_params` for a
    final "lbfgs" polish -- see `estimate_torch`'s docstring for an
    example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional
from types import SimpleNamespace

import numpy as np
import torch
import torchode as to
from scipy.optimize import NonlinearConstraint, minimize

from pybm.model import Choose, Const, Context, Model, Var


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _make_rhs(vars_: list[Var]) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Builds the right-hand side f(t, y, const_ctx) that torchode integrates.

    Shapes (N = mega-batch size = n_candidates * n_subintervals, flattened):
        t         : (N,)
        y         : (N, n_vars)
        const_ctx : (N, n_consts)
    Returns:
        dy/dt     : (N, n_vars)

    We transpose y and const_ctx to (n_vars, N) / (n_consts, N) before
    building the `Context` dict, because `Var.__call__` / `Const.__call__`
    index into `ctx["vars"][index]` / `ctx["consts"][index]` with a plain
    python int and expect back "the value of that single variable/constant
    across the batch" -- i.e. shape (N,). That only works if the *feature*
    axis is first, hence the transpose here (torchode itself always wants
    the batch axis first, so we transpose right back on the way out via
    `torch.stack(..., dim=-1)`).
    """

    def rhs(t: torch.Tensor, y: torch.Tensor, const_ctx_args: torch.Tensor) -> torch.Tensor:
        var_ctx = y.T  # (n_vars, N) -> var_ctx[idx] has shape (N,)
        const_ctx = const_ctx_args.T  # (n_consts, N) -> const_ctx[idx] has shape (N,)
        ctx: Context = {"vars": var_ctx, "consts": const_ctx}

        # Build derivatives out-of-place (via stack, not in-place index_put)
        # so we stay friendly to autograd / torch.compile.
        batch_size = y.shape[0]
        derivatives: list[torch.Tensor] = [torch.zeros(batch_size, dtype=y.dtype, device=y.device) for _ in vars_]
        for var in vars_:
            if var.ode is None:
                raise ValueError(f"Variable {var.name} has no ODE defined.")
            if isinstance(var.ode, Choose):
                raise ValueError(
                    f"Variable {var.name} still has an unresolved Choose() ODE. "
                    "Call model.induce() to pick a concrete model before estimating."
                )
            assert var.index_in_ctx is not None
            derivatives[var.index_in_ctx] = var.ode(t, ctx)
        return torch.stack(derivatives, dim=-1)  # (N, n_vars)

    return rhs


def _make_solver(vars_: list[Var], atol: float, rtol: float) -> to.AutoDiffAdjoint:
    term = to.ODETerm(_make_rhs(vars_), with_args=True)  # type: ignore[arg-type]
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    return to.AutoDiffAdjoint(step_method, step_size_controller)  # type: ignore[arg-type]


@dataclass
class _SubintervalGrid:
    """Precomputed, shared-across-candidates time layout for the shooting segments."""

    sub_indices: np.ndarray  # (K+1,) boundary indices into t_eval
    lengths: list[int]  # (K,) true (unpadded) number of points per segment
    max_len: int  # padded length used for the torchode batch call
    t_grid: torch.Tensor  # (K, max_len) padded per-segment time points


def _build_subinterval_grid(t_eval: np.ndarray, n_subintervals: int, device, dtype) -> _SubintervalGrid:
    """
    Splits t_eval into n_subintervals segments (same convention as the
    scipy version's `sub_indices = np.linspace(0, len(t_eval)-1, ...)`),
    and stacks their time points into a single padded (K, max_len) tensor
    so all K segments can be solved in one torchode batch call.

    Segments can have slightly different lengths (integer division
    remainder). Shorter segments are padded by repeating their own last
    time point -- torchode just takes zero-duration steps there, and the
    resulting (repeated) y-values are simply never read back (see
    `lengths` / `_stitch_trajectory`).
    """
    sub_indices = np.linspace(0, len(t_eval) - 1, n_subintervals + 1, dtype=int)
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
    return sol.ys.reshape(B, K, max_len, n_vars)


def _residuals(
    ys: torch.Tensor,  # (B, K, max_len, n_vars)
    initials: torch.Tensor,  # (B, K, n_vars)
    grid: _SubintervalGrid,
    data: torch.Tensor,  # (n_vars, T)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the two batched loss ingredients, analogous to `residuals` and
    `smoothness_penalty` in the scipy version:

    traj_res[b] : squared error between candidate b's stitched trajectory
                  (segments concatenated, dropping each segment's
                  duplicated first point, exactly like the scipy code)
                  and the observed `data`.

    cont_res[b] : the multiple-shooting MATCHING CONDITION for candidate
                  b -- how far the ODE-propagated end of segment i is from
                  the free initial-condition parameter that seeds segment
                  i+1. This is computed directly from the per-segment
                  solves we already have (rather than re-deriving it from
                  indices into a concatenated array), which is equivalent
                  in intent to the scipy version's `smoothness_penalty`
                  but more direct/robust.

    Returns (traj_res, cont_res), each of shape (B,).
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
    traj_res = (diff**2).sum(dim=(1, 2))  # (B,)

    if K > 1:
        cont_terms = []
        for i in range(K - 1):
            L = grid.lengths[i]
            pred_end = ys[:, i, L - 1, :]  # (B, n_vars): end of segment i, propagated
            seed_next = initials[:, i + 1, :]  # (B, n_vars): free var seeding segment i+1
            cont_terms.append(((pred_end - seed_next) ** 2).sum(dim=-1))
        cont_res = torch.stack(cont_terms, dim=0).sum(dim=0)  # (B,)
    else:
        cont_res = torch.zeros(B, device=device, dtype=dtype)

    return traj_res, cont_res


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


def simulate_multishooting(
    model: Model,
    t_eval,
    params,
    n_subintervals: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    solver_atol: float = 1e-8,
    solver_rtol: float = 1e-6,
):
    """
    Reconstruct a stitched multiple-shooting trajectory from a flat parameter vector.

    The parameter layout matches `estimate_torch`:
        [consts..., s_0, s_1, ..., s_{K-1}]
    where each shooting state s_i seeds segment i.
    """
    if model.engine != "torch":
        raise ValueError(
            f"simulate_multishooting requires a Model built with engine='torch', got engine={model.engine!r}."
        )

    vars_ = model.get_endo_variables()
    n_consts = len(model.consts)
    n_vars = len(vars_)
    device = device or torch.device("cpu")

    params_t = torch.as_tensor(params, dtype=dtype, device=device)
    if params_t.ndim == 1:
        params_t = params_t.unsqueeze(0)
    if params_t.ndim != 2:
        raise ValueError("params must be a 1D or 2D tensor/array.")

    if params_t.shape[1] < n_consts + n_subintervals * n_vars:
        raise ValueError("params has fewer entries than expected for the model and n_subintervals.")

    grid = _build_subinterval_grid(np.asarray(t_eval, dtype=float), n_subintervals, device, dtype)
    solver = _make_solver(vars_, solver_atol, solver_rtol)

    const_ctx = params_t[:, :n_consts]
    initials = params_t[:, n_consts : n_consts + n_subintervals * n_vars].reshape(
        params_t.shape[0], n_subintervals, n_vars
    )
    ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
    stitched = _stitch_trajectory(ys, grid)

    return SimpleNamespace(
        t=np.asarray(t_eval, dtype=float),
        y=stitched[0].detach().cpu().numpy().T,
        success=True,
        message="multiple-shooting reconstruction",
    )


def _sample_initial_params(
    model: Model,
    t_eval: np.ndarray,
    n_subintervals: int,
    n_candidates: int,
    device,
    dtype,
    jitter: float = 0.1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Naive multistart sampler -- a reasonable default, NOT the "avoid
    resampling near bad basins" scheme discussed earlier. Swap this out
    for e.g. a TikTak-style seeding once you have that in place; nothing
    downstream cares how `init_params` was produced.

    Layout matches the scipy version: for each candidate, a flat vector
    [c_1..c_n, v_1(t_0)..v_m(t_0), ..., v_1(t_K)..v_m(t_K)].

    - Constants: candidate 0 uses `const.initial_value` (or the midpoint
      of `const.range`); the rest are drawn uniformly from `const.range`
      (or a generic window around the base value if no range is given).
    - Segment seed states: drawn from the observed data at each segment's
      start time plus multiplicative-ish gaussian jitter (candidate 0 is
      kept exactly at the data-based guess, matching the scipy version's
      un-jittered initial guess).
    """
    rng = np.random.default_rng(seed)
    vars_ = model.get_endo_variables()
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

    sub_indices = np.linspace(0, len(t_eval) - 1, n_subintervals + 1, dtype=int)
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
    scipy_result: Any
    all_results: list[Any] = field(default_factory=list)


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
):
    """
    One trust-constr run for a single candidate. Objective, constraint,
    and their exact Jacobians all go through ONE torchode call each
    (batched over this candidate's n_subintervals segments -- batch size
    here is just K, not K*n_candidates, since scipy drives one candidate
    at a time).

    NOTE: `objective` and `constraint_fun`/`constraint_jac` each redo the
    forward solve at the given x (trust-constr calls them at possibly
    different x's, and doesn't expose a combined fun+constraint+jac
    callback). If profiling shows this dominates, add an x-keyed cache
    here (hash the bytes of x) to reuse the last forward pass.
    """

    def unpack(params_t: torch.Tensor):
        const_ctx = params_t[:n_consts].unsqueeze(0)  # (1, n_consts)
        initials = params_t[n_consts:].reshape(1, n_subintervals, n_vars)  # (1, K, n_vars)
        return const_ctx, initials

    def forward(params_np: np.ndarray):
        params_t = torch.as_tensor(params_np, dtype=dtype)
        params_t.requires_grad_(True)
        const_ctx, initials = unpack(params_t)
        ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
        traj_res, cont_res = _residuals(ys, initials, grid, data)
        return params_t, traj_res[0], cont_res[0]

    def objective(params_np: np.ndarray):
        params_t, traj_res, _ = forward(params_np)
        (grad,) = torch.autograd.grad(traj_res, params_t)
        return traj_res.item(), grad.detach().cpu().numpy()

    def constraint_fun(params_np: np.ndarray):
        _, _, cont_res = forward(params_np)
        return np.array([cont_res.item()])

    def constraint_jac(params_np: np.ndarray):
        params_t, _, cont_res = forward(params_np)
        (grad,) = torch.autograd.grad(cont_res, params_t)
        return grad.detach().cpu().numpy().reshape(1, -1)

    nlc = NonlinearConstraint(constraint_fun, 0, 0, jac=constraint_jac)  # type: ignore[arg-type]
    result = minimize(
        fun=objective,
        x0=np.asarray(init_params, dtype=float),
        jac=True,
        constraints=[nlc],
        method="trust-constr",
        options={"maxiter": maxiter, "verbose": verbose, "gtol": gtol},
    )
    return result


def _estimate_constraints(
    model: Model,
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
) -> ConstraintsResult:
    vars_ = model.get_endo_variables()
    n_vars = len(vars_)
    n_consts = len(model.consts)
    B = init_params_batch.shape[0]

    grid = _build_subinterval_grid(t_eval, n_subintervals, device, dtype)
    data = _get_data_tensor(vars_, t_eval, device, dtype)
    solver = _make_solver(vars_, solver_atol, solver_rtol)

    init_np = init_params_batch.detach().cpu().numpy()

    def run_one(b: int):
        return _estimate_constraints_single(
            vars_, grid, data, solver, init_np[b], n_consts, n_subintervals, n_vars,
            dtype, maxiter, gtol, verbose,
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
        _, cont_res_t = _residuals(ys, initials, grid, data)

    return ConstraintsResult(
        consts=best.x[:n_consts],
        x=best.x,
        cost=float(best.fun),
        continuity_violation=float(cont_res_t.item()),
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
    best_index: int
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
    model: Model,
    t_eval: np.ndarray,
    n_subintervals: int,
    init_params_batch: torch.Tensor,  # (B, n_consts + K*n_vars), requires_grad set here
    device,
    dtype,
    solver_atol: float,
    solver_rtol: float,
    n_iter: int,
    optimizer_name: Literal["adam", "lbfgs"],
    lr: float,
    A_start: float,
    A_end: float,
    B_start: float,
    B_end: float,
    homotopy_schedule: Literal["linear", "exp"],
    log_every: int,
    verbose: int,
) -> WeightedSumResult:
    vars_ = model.get_endo_variables()
    n_vars = len(vars_)
    n_consts = len(model.consts)
    B = init_params_batch.shape[0]

    grid = _build_subinterval_grid(t_eval, n_subintervals, device, dtype)
    data = _get_data_tensor(vars_, t_eval, device, dtype)
    solver = _make_solver(vars_, solver_atol, solver_rtol)

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
        traj_res, cont_res = _residuals(ys, initials, grid, data)
        per_candidate = A * traj_res + Bw * cont_res
        return per_candidate.sum(), traj_res, cont_res

    for it in range(n_iter):
        frac = it / max(n_iter - 1, 1)
        A = _homotopy_weight(A_start, A_end, frac, homotopy_schedule)
        Bw = _homotopy_weight(B_start, B_end, frac, homotopy_schedule)

        if optimizer_name == "adam":
            optimizer.zero_grad()
            loss, traj_res, cont_res = compute_loss(A, Bw)
            loss.backward()
            optimizer.step()  # type: ignore[call-arg]
        else:  # lbfgs

            def closure():
                optimizer.zero_grad()
                loss, _, _ = compute_loss(A, Bw)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            with torch.no_grad():
                _, traj_res, cont_res = compute_loss(A, Bw)

        if log_every and (it % log_every == 0 or it == n_iter - 1):
            entry = {
                "iter": it,
                "A": A,
                "B": Bw,
                "traj_res": traj_res.detach().cpu().numpy().copy(),
                "cont_res": cont_res.detach().cpu().numpy().copy(),
            }
            history.append(entry)
            if verbose:
                print(
                    f"[weighted_sum {it:5d}] A={A:.4g} B={Bw:.4g} "
                    f"traj_res(min/mean)={entry['traj_res'].min():.4g}/{entry['traj_res'].mean():.4g} "
                    f"cont_res(min/mean)={entry['cont_res'].min():.4g}/{entry['cont_res'].mean():.4g}"
                )

    with torch.no_grad():
        _, final_traj_res, final_cont_res = compute_loss(A_end, B_end)

    # pick the best candidate among those that are (approximately) actually
    # continuous; among the rest, rank by trajectory error.
    cont_ok = final_cont_res < (final_cont_res.mean() + 1e-6)
    ranking_key = final_traj_res + torch.where(cont_ok, torch.zeros_like(final_cont_res), final_cont_res)
    best_index = int(torch.argmin(ranking_key).item())

    return WeightedSumResult(
        consts=params[:, :len(model.consts)].flatten().detach(),
        x=params.detach().flatten(),
        params=params.detach(),
        traj_res=final_traj_res.detach(),
        cont_res=final_cont_res.detach(),
        best_index=best_index,
        history=history,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def estimate_torch(
    model: Model,
    t_eval,
    method: Literal["constraints", "weighted_sum"] = "weighted_sum",
    n_subintervals: int = 1,
    n_candidates: int = 1,
    init_params: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    solver_atol: float = 1e-8,
    solver_rtol: float = 1e-6,
    verbose: int = 0,
    seed: Optional[int] = None,
    # -- "constraints"-only knobs --
    maxiter: int = 100,
    gtol: float = 1e-6,
    executor=None,
    # -- "weighted_sum"-only knobs --
    n_iter: int = 500,
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
    model : Model
        Must have been built with `engine="torch"`, with `Choose` objects
        already resolved (via `model.induce()`), and with every `Var.ode`
        written using differentiable torch operations.
    t_eval : array-like
        Time points to fit against (same role as in the scipy version).
    method : "constraints" | "weighted_sum"
        See the module docstring for the trade-offs.
    n_subintervals : int
        Number of shooting segments (K). K=1 recovers plain single-shooting.
    n_candidates : int
        Number of multistart candidates (B) solved in one batched pass
        ("weighted_sum") or as B independent local searches
        ("constraints").
    init_params : (B, n_consts + K*n_vars) tensor, optional
        Initial guesses, same flat layout as the scipy version:
        [c_1..c_n, v_1(t_0)..v_m(t_0), ..., v_1(t_K)..v_m(t_K)] per
        candidate. If omitted, a naive multistart sampler is used (see
        `_sample_initial_params`) -- swap this for your own seeding
        strategy (e.g. TikTak-style) by just passing `init_params`.

    Returns
    -------
    ConstraintsResult or WeightedSumResult (see their docstrings/fields).

    Example: Adam warm start, then LBFGS polish
    --------------------------------------------
    >>> warm = estimate_torch(model, t_eval, method="weighted_sum",
    ...                        n_subintervals=5, n_candidates=20,
    ...                        optimizer_name="adam", n_iter=2000)
    >>> polished = estimate_torch(model, t_eval, method="weighted_sum",
    ...                            n_subintervals=5, n_candidates=20,
    ...                            init_params=warm.params,
    ...                            optimizer_name="lbfgs", n_iter=100,
    ...                            B_start=warm.history[-1]["B"], B_end=warm.history[-1]["B"])
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_torch requires a Model built with engine='torch', got engine={model.engine!r}."
        )

    t_eval = np.asarray(t_eval, dtype=float)
    device = device or torch.device("cpu")

    if init_params is None:
        init_params = _sample_initial_params(
            model, t_eval, n_subintervals, n_candidates, device, dtype, seed=seed
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
            solver_atol, solver_rtol, maxiter, gtol, verbose, executor=executor,
        )
    elif method == "weighted_sum":
        return _estimate_weighted_sum(
            model, t_eval, n_subintervals, init_params, device, dtype,
            solver_atol, solver_rtol, n_iter, optimizer_name, lr,
            A_start, A_end, B_start, B_end, homotopy_schedule, log_every, verbose,
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'constraints' or 'weighted_sum'.")