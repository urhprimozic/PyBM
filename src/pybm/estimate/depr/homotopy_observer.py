"""
Single-shooting homotopy/observer method - Vyasarayani, Uchida & McPhee, "Single-shooting homotopy
method for parameter identification in dynamical systems", Phys. Rev. E 85, 036201 (2012).

Method
------
The ODE is augmented with an observer-like feedback term pulling the simulated trajectory toward a
smooth interpolant of the data:

    dq/dt = f(q,theta,t) + lambda * Gamma * (y_hat(t) - q(t))

At `lambda=1` the feedback term dominates - the paper shows (their Eqs. 8-13, a singular-
perturbation/boundary-layer argument) that the resulting objective is then approximately QUADRATIC
in the parameter error, so it has one basin, no local minima to get stuck in. `lambda` is then
annealed geometrically down to a small (deliberately not exactly zero - see the paper's own
discussion after Eq. 14) floor, each step warm-started from the previous one's optimum - by the
time `lambda` is small, the optimizer is already in the right basin of the ORIGINAL (feedback-free)
single-shooting objective. Unlike `multishooting_adaptive.py`'s own continuation (which anneals the
WINDOW LENGTH), this anneals the STRENGTH OF A TERM ADDED TO THE DYNAMICS ITSELF - orthogonal ideas,
per the chat this followed.

Gamma (gain matrix)
--------------------
Diagonal, one gain per state: `Gamma = diag(gamma_1,...,gamma_n)`, feeding `e_i = y_hat_i(t)-q_i(t)`
into state i's OWN equation only. This matches the paper's Rossler example (every state measured,
diagonal gains) rather than its Lorenz example (only one state measured, so ALL rows of Gamma had
to borrow that one state's error signal) - every state in this codebase's benchmark systems (LV, PT,
single-var Bled) is fully observed, so there is no unmeasured state that would need that broadcast.

The paper hand-picks a fixed gamma per example (20 for Lorenz, 10 for Rossler) and only says gamma
should be "large enough that the error oscillates close to equilibrium" (increase the stiffness of
the error dynamics, Eq. 8-9) relative to the system's own eigenvalues. Here `gamma_i = gamma_scale /
lengthscale_i`, where `lengthscale_i` is state i's own GP-fitted RBF lengthscale (`fit_gps`, already
used throughout this package) - a natural, DATA-DRIVEN timescale, so `gamma_i` has units of a rate
(1/time) and `gamma_scale` (default 10, matching the paper's own Rossler value) controls how much
faster the error dynamics are forced to be than the state's own natural timescale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torchode as to

from pybm.estimate.depr.gradient_matching import (
    _qualified_var_names,
    estimate_gradient_matching,
    fit_gps,
)
from pybm.estimate.depr.multishooting_adaptive import _seed_at, _state_weights
from pybm.estimate.depr.multishooting_torch import (
    _build_subinterval_grid,
    _homotopy_weight,
    _solve_segments,
)
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _get_data_tensor, _make_rhs


def _interp1d(t_query: torch.Tensor, t_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of `y_grid` (n_grid, n_vars) at `t_query` (N,) - `t_grid` ascending,
    covering the full integration span (torchode's adaptive steps can query any `t` in that span,
    not just the raw data times). Returns (N, n_vars)."""
    idx = torch.searchsorted(t_grid, t_query.clamp(t_grid[0], t_grid[-1]))
    idx = idx.clamp(1, len(t_grid) - 1)
    t0, t1 = t_grid[idx - 1], t_grid[idx]
    y0, y1 = y_grid[idx - 1], y_grid[idx]
    frac = ((t_query - t0) / (t1 - t0).clamp_min(1e-12)).unsqueeze(-1)
    return y0 + frac * (y1 - y0)


def _make_observer_solver(
    state_vars: "list[Var]", algebraic_vars: "list[Var]", frozen_values: dict,
    gammas: torch.Tensor, t_grid: torch.Tensor, y_grid: torch.Tensor, lam: float,
    atol: float, rtol: float, max_steps: "int | None", dt_min: "float | None",
) -> to.AutoDiffAdjoint:
    """Same torchode solver machinery as `multishooting_torch._make_solver`, wrapping the model's
    own RHS with the observer feedback term (see module docstring). Rebuilt once per `lambda` value
    (cheap - it's just closing over a new Python float)."""
    base_rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    def rhs(t: torch.Tensor, y: torch.Tensor, const_ctx: torch.Tensor) -> torch.Tensor:
        y_hat = _interp1d(t, t_grid, y_grid)
        return base_rhs(t, y, const_ctx) + lam * gammas.unsqueeze(0) * (y_hat - y)

    term = to.ODETerm(rhs, with_args=True)  # type: ignore[arg-type]
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term, dt_min=dt_min)
    return to.AutoDiffAdjoint(step_method, step_size_controller, max_steps=max_steps)  # type: ignore[arg-type]


@dataclass
class HomotopyObserverResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray
    const_by_name: "dict[str, float]"
    x0: np.ndarray
    gammas: "dict[str, float]"
    loss_history: "list[dict]"


def estimate_homotopy_observer(
    model: InducedModel,
    t_eval,
    gamma_scale: float = 10.0,
    lambda_start: float = 1.0,
    lambda_end: float = 1e-3,
    n_lambda_steps: int = 12,
    iters_per_lambda: int = 40,
    lr: float = 1e-2,
    weight_by_uncertainty: bool = True,
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    n_grid: int = 500,
    solver_atol: float = 1e-6,
    solver_rtol: float = 1e-4,
    solver_max_steps: "int | None" = 1000,
    solver_dt_min: "float | None" = None,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    torch_threads: "int | None" = None,
    homotopy_schedule: Literal["linear", "exp"] = "exp",
    norm: Literal["l2", "l1"] = "l2",
    verbose: int = 0,
) -> HomotopyObserverResult:
    """
    Single-shooting homotopy/observer method - see module docstring for the method and how `Gamma`
    is chosen. Constants and the initial state `x0` are fitted jointly at every `lambda`, warm-
    started from `estimate_gradient_matching` / the GP mean at `t_eval[0]` (same convention as
    `multishooting_adaptive.py`) at `lambda=lambda_start`, then carried over between steps.

    Parameters
    ----------
    gamma_scale : float, optional
        `gamma_i = gamma_scale / lengthscale_i` - see module docstring.
    lambda_start, lambda_end, n_lambda_steps : optional
        Homotopy schedule. `lambda_end` is deliberately not exactly 0 (the paper's own finding: a
        vanishing feedback on long-duration data lets the model drift away again before the fit is
        read off - see the discussion after their Eq. 14).
    iters_per_lambda, lr : optional
        Adam steps and learning rate used at EACH `lambda` value.
    weight_by_uncertainty : bool, optional
        Same convention as `multishooting_adaptive.py`: divide each residual by the GP's own
        posterior state standard deviation (`_FittedGP.var`) - default `True`.
    norm : "l2" | "l1", optional
        Per-element penalty in `compute_loss`. "l1" uses RAW `abs()`, not smoothed - safe here
        since this method only ever takes a first-order gradient through the loss (Adam), never a
        Hessian (contrast `multishooting_adaptive._penalty`, used by the Newton-based methods).

    Returns
    -------
    HomotopyObserverResult
        `loss_history`: one dict per `lambda` step (`{"lambda", "loss"}`), for inspecting how the
        fit evolves as the feedback anneals away.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_homotopy_observer requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    n_vars = len(state_vars)
    n_consts = len(model.consts)
    qualified = _qualified_var_names(model)

    state_var_names = {qualified[id(v)]: v for v in state_vars}
    gps = fit_gps(state_var_names, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)  # (n_vars, T)
    weight = _state_weights(gps, qualified, state_vars, t_eval, device, dtype) if weight_by_uncertainty else None

    t_fine = np.linspace(t_eval[0], t_eval[-1], n_grid)
    y_grid = torch.stack(
        [torch.as_tensor(gps[qualified[id(v)]].mean(t_fine), dtype=dtype, device=device) for v in state_vars],
        dim=1,
    )  # (n_grid, n_vars)
    t_grid = torch.as_tensor(t_fine, dtype=dtype, device=device)

    gammas = torch.as_tensor(
        [gamma_scale / gps[qualified[id(v)]].lengthscale for v in state_vars], dtype=dtype, device=device
    )

    gm0 = estimate_gradient_matching(
        model, t_eval, gps=gps, weight_by_uncertainty=weight_by_uncertainty, device=device, dtype=dtype, verbose=verbose,
    )
    theta = torch.as_tensor(gm0.consts, dtype=dtype, device=device)
    x0 = _seed_at(gps, qualified, state_vars, t_eval[0], device, dtype)
    params = torch.cat([theta, x0])  # (n_consts + n_vars,)

    grid = _build_subinterval_grid(t_eval, 1, device, dtype)
    data_t = data.T  # (T, n_vars), matching _solve_segments' own (B,K,max_len,n_vars) trajectory layout

    loss_history: "list[dict]" = []
    for step in range(n_lambda_steps):
        frac = step / max(n_lambda_steps - 1, 1)
        lam = _homotopy_weight(lambda_start, lambda_end, frac, homotopy_schedule)
        solver = _make_observer_solver(
            state_vars, algebraic_vars, frozen_values, gammas, t_grid, y_grid, lam,
            solver_atol, solver_rtol, solver_max_steps, solver_dt_min,
        )
        params = params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)

        def compute_loss():
            const_ctx = params[:n_consts].unsqueeze(0)
            initials = params[n_consts:].reshape(1, 1, n_vars)
            ys = _solve_segments(state_vars, const_ctx, initials, grid, solver)
            diff = ys[0, 0, : len(t_eval), :] - data_t
            if weight is not None:
                diff = diff * weight
            return diff.abs().mean() if norm == "l1" else (diff**2).mean()

        for _ in range(iters_per_lambda):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_loss = float(compute_loss())
        params = params.detach()
        loss_history.append({"lambda": lam, "loss": final_loss})
        if verbose:
            print(f"[homotopy_observer step {step:3d}] lambda={lam:.4g} loss={final_loss:.4g}")

    const_by_name = {name: float(params[c.index_in_ctx]) for name, c in model.consts.items()}
    gamma_by_name = {qualified[id(v)]: float(gammas[i]) for i, v in enumerate(state_vars)}

    return HomotopyObserverResult(
        model=model,
        t_eval=t_eval,
        consts=params[:n_consts].detach().cpu().numpy(),
        const_by_name=const_by_name,
        x0=params[n_consts:].detach().cpu().numpy(),
        gammas=gamma_by_name,
        loss_history=loss_history,
    )
