from dataclasses import dataclass

import torch
from tqdm import tqdm
from typing import Any, Callable, Literal
import numpy as np
from scipy.optimize import least_squares

from pybm.estimate.gradient_matching import estimate_gradient_matching
from pybm.estimate.multishooting_torch import (
    _build_subinterval_grid,
    _make_solver,
    _residuals,
    _solve_segments,
    _split_endo_vars,
)
from pybm.estimate.results import ParamEstimationResults
from pybm.model import Context, InducedModel, Model, Var
from pybm.simulate.initials import get_ctx
from pybm.simulate.trajectory import simulate as single_shooting

recepies = Literal["gp", "gp+ms", "ms"]


def _gp(model, t_eval, *args, **kwargs) -> ParamEstimationResults:
    result = estimate_gradient_matching(model, t_eval, *args, **kwargs)
    return ParamEstimationResults(
        model=result.model, consts=result.consts, const_by_name=result.const_by_name
    )


def _gp_plus_ms(model, t_eval, *args, **kwargs):
    raise NotImplementedError(
        "Gaussian Process + Multishooting is not implemented yet."
    )


def _ms(model, t_eval, *args, **kwargs):
    raise NotImplementedError("Multishooting is not implemented yet.")


def _singleshooting_loss(
    model: InducedModel,
    t_eval,
    const_ctx,
    method: Literal["scipy", "torch"] = "scipy",
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    """
    Returns the minimal loss (MSE) of a single, whole-horizon trajectory simulated with the given
    `const_ctx`. We don't know a good initial state upfront, so this treats it as a nuisance
    parameter and optimizes over it too - the returned loss is `min_{x0} MSE(trajectory(const_ctx,
    x0))`, i.e. the best this `const_ctx` could possibly do, not just whatever one guess gives.

    Only the model's differential ("state", `.ode is not None`) variables need an initial value at
    all (algebraic/frozen variables are re-derived/held fixed regardless - see
    `pybm.simulate.trajectory.simulate`) - the optimizer's own starting guess for each is its OWN
    observed data at `t_eval[0]` (interpolated - see `pybm.simulate.initials.get_ctx`), a
    reasonable prior since the state IS what's being observed, just possibly off due to noise.

    `method` picks the optimization backend:
    - "scipy": `scipy.optimize.least_squares` around `pybm.simulate.trajectory.simulate`
      (numeric Jacobian, works with any model - no differentiability requirement).
    - "torch": gradient descent (Adam) with exact gradients through a torchode integration -
      requires `model.engine == "torch"` and every equation written in differentiable torch ops
      (same precondition as `pybm.estimate.multishooting_torch.estimate_torch`), but is typically
      much faster since it doesn't need to re-simulate the whole trajectory per finite-difference
      probe.

    Returns a plain `float` either way.
    """
    state_vars = [var for var in model.get_endo_variables() if var.ode is not None]
    if not state_vars:
        raise ValueError(
            f"Model {model} has no differential (state) variables to shoot from."
        )
    for var in state_vars:
        if var.data is None:
            raise ValueError(
                f"Variable {var.name} has no data - can't score a trajectory against it."
            )

    t_eval = np.asarray(t_eval, dtype=float)
    data = np.stack(
        [[float(var.data(t)) for t in t_eval] for var in state_vars]
    )  # (n_state, T)

    # a full, correctly-shaped starting var context (frozen vars at their real initial, algebraic
    # slots at a harmless placeholder - see get_ctx) - only the state-var slots are actually
    # optimized below; the rest is reused as-is.
    base_var_ctx = get_ctx(model, t=float(t_eval[0]))["vars"]
    x0 = np.array([base_var_ctx[var.index_in_ctx] for var in state_vars], dtype=float)

    if method == "scipy":
        return _singleshooting_loss_scipy(
            model, t_eval, const_ctx, state_vars, data, base_var_ctx, x0
        )
    elif method == "torch":
        return _singleshooting_loss_torch(
            model, t_eval, const_ctx, state_vars, data, x0, max_iter=max_iter, lr=lr
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'scipy' or 'torch'.")


def _singleshooting_loss_scipy(
    model: InducedModel,
    t_eval: np.ndarray,
    const_ctx,
    state_vars: "list[Var]",
    data: np.ndarray,
    base_var_ctx,
    x0: np.ndarray,
) -> float:
    const_ctx = np.asarray(
        const_ctx.detach().cpu().numpy() if torch.is_tensor(const_ctx) else const_ctx,
        dtype=float,
    )
    n_points = data.size

    def residuals(x: np.ndarray) -> np.ndarray:
        var_ctx = np.array(base_var_ctx, dtype=float, copy=True)
        for i, var in enumerate(state_vars):
            var_ctx[var.index_in_ctx] = x[i]
        sol = single_shooting(model, t_eval, Context(vars=var_ctx, consts=const_ctx))
        if not sol.success:
            return np.full(n_points, 1e6)
        pred = np.stack([sol.y[var.index_in_ctx] for var in state_vars])
        return (pred - data).ravel()

    result = least_squares(residuals, x0=x0, method="trf")
    return float(np.mean(result.fun**2))


def _singleshooting_loss_torch(
    model: InducedModel,
    t_eval: np.ndarray,
    const_ctx,
    state_vars: "list[Var]",
    data: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    lr: float,
) -> float:
    if model.engine != "torch":
        raise ValueError(
            f"_singleshooting_loss(method='torch') requires an InducedModel built with "
            f"engine='torch', got engine={model.engine!r}."
        )
    device = torch.device("cpu")
    dtype = torch.float64

    _, algebraic_vars, frozen_values = _split_endo_vars(model)
    solver = _make_solver(
        state_vars, algebraic_vars, frozen_values, atol=1e-8, rtol=1e-6
    )
    grid = _build_subinterval_grid(t_eval, n_subintervals=1, device=device, dtype=dtype)
    data_t = torch.as_tensor(data, dtype=dtype, device=device)  # (n_state, T)
    const_ctx_t = torch.as_tensor(
        const_ctx.detach().cpu().numpy() if torch.is_tensor(const_ctx) else const_ctx,
        dtype=dtype,
        device=device,
    ).unsqueeze(
        0
    )  # (1, n_consts)

    x = (
        torch.as_tensor(x0, dtype=dtype, device=device)
        .unsqueeze(0)
        .clone()
        .requires_grad_(True)
    )  # (1, n_state)
    optimizer = torch.optim.Adam([x], lr=lr)

    def loss_fn() -> torch.Tensor:
        initials = x.reshape(1, 1, len(state_vars))  # (B=1, K=1, n_state)
        ys = _solve_segments(state_vars, const_ctx_t, initials, grid, solver)
        traj_res, _, _ = _residuals(ys, initials, grid, data_t)
        return traj_res.sum()  # single candidate - sum == that candidate's own loss

    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return float(loss_fn().item())

@dataclass
class FullEstimationResults:
    best_consts : np.ndarray | torch.Tensor | Any
    best_const_by_name : dict[str, float | Any]
    best_loss : float | np.ndarray | torch.Tensor | Any
    all_results : list[ParamEstimationResults]

def estimate_model(
    incomplete_model,
    t_eval: np.ndarray | torch.Tensor | Any,
    recepie: recepies = "gp",
    parallel=False,
    n_intervals_ms=50,
    engine: Literal["torch", "scipy"] = "torch",
    *args,
    **kwargs,
):
    """
    Induces all possible models, estimates the parameters and returns the best model.

    Parameters
    ----------
    model : Model
        The model to be estimated.
    recepie : str, optional
        The estimation method to be used. Can be
            - "gp" for Gaussian Process
            - "gp+ms" for Gaussian Process + Multishooting
            - "ms" for Multishooting.

        Default is "gp".
    parallel : bool, optional
        If true, different models will be estimated in parallel. Default is false
    n_intervals_ms : int, optional
        The number of intervals for multishooting. Default is 50.
    *args :
        Additional positional arguments to be passed to the estimator function.
    **kwargs :
        Additional keyword arguments to be passed to the estimator function.
    """
    # get estimator function based on recepie
    estimator: Callable[[Model, Any], ParamEstimationResults] | Any
    if recepie == "gp":
        estimator = _gp
    elif recepie == "gp+ms":
        estimator = _gp_plus_ms
    elif recepie == "ms":
        estimator = _ms
    else:
        raise ValueError(f"Unknown recepie: {recepie}")

    # induce models
    models = incomplete_model.induce()

    if parallel:
        raise NotImplementedError("Not yet implemented")

    # estimate parameters for each model
    all_results = []
    for model in tqdm(models, total=len(models), desc="Structure estimation"):
        try:
            model.switch_engine(engine)
            results: ParamEstimationResults = estimator(model, t_eval, *args, **kwargs)
            results.loss = _singleshooting_loss(model, t_eval, results.consts, method=engine)
        except Exception as e:
            print(f"Error occurred while estimating model: {e}.")
            # store a result with infinite loss to indicate failure, instead of dropping it -
            # keeps all_results aligned 1:1 with `models` and never empty just because every
            # candidate happened to fail.
            results = ParamEstimationResults(
                model=model, consts=np.nan, loss=np.inf, const_by_name={}
            )
        all_results.append(results)

    # find the best model
    best_result = min(all_results, key=lambda r: r.loss)

    # return the best model and all results
    return FullEstimationResults(
        best_consts=best_result.consts,
        best_loss=best_result.loss,
        best_const_by_name=best_result.const_by_name,
        all_results=all_results
    )
