from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import torch
from tqdm import tqdm
from typing import Any, Callable, Literal
import numpy as np
from scipy.optimize import least_squares

from pybm.estimate.gradient_matching import estimate_gradient_matching, fit_gps
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
    device: "torch.device | str | None" = None,
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
      (numeric Jacobian, works with any model - no differentiability requirement). `device` is
      ignored - this path is plain numpy, always CPU.
    - "torch": gradient descent (Adam) with exact gradients through a torchode integration -
      requires `model.engine == "torch"` and every equation written in differentiable torch ops
      (same precondition as `pybm.estimate.multishooting_torch.estimate_torch`), but is typically
      much faster since it doesn't need to re-simulate the whole trajectory per finite-difference
      probe. `device` (default CPU) is where the solve runs - `model` itself should already be on
      the same device (see `InducedModel.switch_engine`'s `device` argument), or an exogenous
      variable's data and this call's own tensors will collide the moment an equation reads both.

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
            model, t_eval, const_ctx, state_vars, data, x0, max_iter=max_iter, lr=lr, device=device
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
    device: "torch.device | str | None" = None,
) -> float:
    if model.engine != "torch":
        raise ValueError(
            f"_singleshooting_loss(method='torch') requires an InducedModel built with "
            f"engine='torch', got engine={model.engine!r}."
        )
    device = torch.device(device) if device is not None else torch.device("cpu")
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


def _worker_init():
    """
    Runs once per worker process, before it picks up any job. Without this, each of the N worker
    PROCESSES would also let torch spin up its own multi-threaded BLAS pool - N processes x M
    threads each easily oversubscribes the machine's actual core count, which shows up as
    everything getting slower once you add workers, not faster. Since the parallelism here is
    already across processes (one model per worker), each worker only needs 1 torch thread.
    """
    torch.set_num_threads(1)


def _estimate_one(
    index: int,
    model: InducedModel,
    t_eval,
    estimator: Callable,
    engine: Literal["torch", "scipy"],
    max_iter_loss: int,
    verbose: int,
    args: tuple,
    kwargs: dict,
    device: "torch.device | str | None" = None,
    gps: "dict | None" = None,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
) -> ParamEstimationResults:
    """
    Estimates a single (already induced) model's constants and scores it - the per-model unit of
    work `estimate_model` maps over `models`, sequentially or (via `parallel=True`) across
    processes. Module-level (not a closure) specifically so it - and the `InducedModel` it's
    called with - can be pickled and sent to a worker process; see `pybm.func`'s
    `_apply_dispatched_math`/`_apply_dispatched_binary` for the matching fix on the model side
    (an `InducedModel` built from a converted ProBMoT library embeds `exp`/`pow`/... callables in
    its equations, which also need to survive that same pickling).

    `device` (torch-only, ignored for `engine="scipy"`) is applied to `model` itself first (see
    `InducedModel.switch_engine`), then threaded into both the estimator call and the loss
    computation, so every tensor this model ever touches lives on the same device throughout -
    otherwise an exogenous variable's data left on one device and a state/const tensor on another
    collide the moment an equation reads both.

    `gps` (recipe "gp" only) is forwarded straight to `estimate_gradient_matching` - see
    `estimate_model`'s docstring for why this is fit once, up front, shared across every model
    instead of being redone here per structural candidate.

    `ftol`/`xtol`/`gtol`/`max_nfev` (recipe "gp" only) are forwarded to `estimate_gradient_matching`
    -> `scipy.optimize.least_squares`, the constant fit's own convergence knobs.
    """
    try:
        if verbose >= 2:
            print(f"Estimating model #{index}")
        if engine == "torch":
            model.switch_engine(engine, device=device)
        else:
            model.switch_engine(engine)
        results: ParamEstimationResults = estimator(
            model, t_eval, *args, device=device, gps=gps,
            ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev, **kwargs,
        )
        if verbose >= 2:
            print("Calculating model loss..")
        results.loss = _singleshooting_loss(
            model, t_eval, results.consts, method=engine, max_iter=max_iter_loss, device=device
        )
    except Exception as e:
        print(f"Error occurred while estimating model #{index}: {e}.")
        # store a result with infinite loss to indicate failure, instead of dropping it - keeps
        # all_results aligned 1:1 with `models` and never empty just because every candidate
        # happened to fail.
        results = ParamEstimationResults(model=model, consts=np.nan, loss=np.inf, const_by_name={})
    return results


def estimate_model(
    incomplete_model,
    t_eval: np.ndarray | torch.Tensor | Any,
    recepie: recepies = "gp",
    parallel=False,
    max_workers: "int | None" = None,
    n_intervals_ms=50,
    engine: Literal["torch", "scipy"] = "torch",
    device: "torch.device | str | None" = None,
    max_iter_loss : int = 100,
    verbose : int = 0,
    max_gp_iter: int = 100,
    max_gp_points: int = 300,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
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
        If true, different structural candidates are estimated in separate worker PROCESSES
        (`concurrent.futures.ProcessPoolExecutor`), not threads - this is CPU-bound work (GP
        fitting, least_squares, torch forward/backward passes), and threads would still fight over
        the GIL for the pure-Python parts. Requires everything reachable from an `InducedModel`
        (its `Var`/`Const`/`Expr` tree, any exogenous `TimeSeries` data) to be picklable to cross
        the process boundary - true for equations built via `pybm.func`'s `exp`/`sin`/.../`If`, or
        converted from a ProBMoT library (see `_estimate_one`'s docstring) - but a hand-written
        equation using a local lambda/closure as its callable would break this.
        Default is false.
    max_workers : int, optional
        Worker process count for `parallel=True` (default: `os.cpu_count()`, via
        `ProcessPoolExecutor`'s own default). Ignored if `parallel` is false.
    n_intervals_ms : int, optional
        The number of intervals for multishooting. Default is 50.
    device : torch.device | str, optional
        Where to run everything torch-based (`engine="torch"`) - pass e.g. `"cuda"` to use a GPU,
        or explicitly pass `"cpu"` to force CPU even if a GPU is available (default is already CPU
        when omitted - `None` never means "auto-pick a GPU"). Applied consistently to every
        induced model (`InducedModel.switch_engine`), the estimator, and the loss computation, so
        an exogenous variable's data and the state/const tensors it's read alongside never end up
        on different devices. Ignored when `engine="scipy"` (that path is plain numpy, always CPU)
        - with `parallel=True` and multiple GPU workers you don't want oversubscribing a single
        device, pick one device per run (or drive that split yourself, outside this function).
    max_gp_iter, max_gp_points : int, optional
        Passed to `fit_gps` (recipe "gp" only) - see `pybm.estimate.gradient_matching.fit_gps`.
    ftol, xtol, gtol, max_nfev : optional
        Recipe "gp" only - forwarded to `scipy.optimize.least_squares` for the constant fit (see
        `pybm.estimate.gradient_matching.estimate_gradient_matching`). Loosen these to trade fit
        accuracy for speed - this recipe is only a fast initializer to begin with (see that
        module's docstring), so an approximate answer is often good enough.
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

    # Step 1 of gradient matching (the GP fit) depends only on a variable's own observed data,
    # never on which structural candidate is chosen elsewhere in the model - fit it ONCE here,
    # against the ORIGINAL incomplete_model's vars (data is attached before induce() and
    # deep-copied identically into every branch, so this is exactly the same fit every branch
    # would otherwise redo from scratch). Matched by variable name in estimate_gradient_matching,
    # so it doesn't matter whether a given variable ends up state/algebraic/frozen in any specific
    # branch - an unused entry here is simply never looked up.
    gps = None
    if recepie == "gp":
        vars_with_data = [var for var in incomplete_model.vars.values() if var.data is not None]
        if vars_with_data:
            gps = fit_gps(vars_with_data, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
            future_to_index = {
                executor.submit(
                    _estimate_one, index, model, t_eval, estimator, engine, max_iter_loss, verbose, args, kwargs,
                    device, gps, ftol, xtol, gtol, max_nfev,
                ): index
                for index, model in enumerate(models)
            }
            indexed_results: "list[tuple[int, ParamEstimationResults]]" = []
            for future in tqdm(
                as_completed(future_to_index), total=len(future_to_index),
                desc="Structure estimation (parallel)", disable=verbose == 0,
            ):
                indexed_results.append((future_to_index[future], future.result()))
        # as_completed() finishes in COMPLETION order, not submission order - restore the
        # original models[] order so all_results stays aligned with it either way.
        all_results = [result for _, result in sorted(indexed_results, key=lambda pair: pair[0])]
    else:
        all_results = [
            _estimate_one(
                index, model, t_eval, estimator, engine, max_iter_loss, verbose, args, kwargs,
                device, gps, ftol, xtol, gtol, max_nfev,
            )
            for index, model in tqdm(
                enumerate(models), total=len(models), desc="Structure estimation", disable=verbose == 0
            )
        ]

    # find the best model
    best_result = min(all_results, key=lambda r: r.loss)

    # return the best model and all results
    return FullEstimationResults(
        best_consts=best_result.consts,
        best_loss=best_result.loss,
        best_const_by_name=best_result.const_by_name,
        all_results=all_results
    )
