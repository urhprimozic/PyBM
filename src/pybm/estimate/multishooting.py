from typing import Literal

from odestimate.latent.multishooting import ms
import numpy as np
import torch
from scipy.optimize import Bounds

from pybm.estimate.gradient_matching import build_observation, build_problem
from pybm.estimate.preprocess import _const_bounds, _initial_const_guess
from pybm.model import InducedModel, _get_data_tensor


def multishooting(model: InducedModel, t_eval, n_subintervals: int, engine: Literal["torch", "scipy"] = "torch",
                  unobserved_inits=None, device: "torch.device | None" = None, **kwargs):
    """
    Estimate a candidate's constants by multiple shooting - a thin PyBM-side adapter around
    `odestimate.latent.multishooting.ms`, which does the actual work. Turns the `InducedModel`
    into what `ms` takes:

    - the loss compares every endogenous variable that has data - state or algebraic - with its
      data read at `t_eval`: state variables from the integrated trajectory, algebraic ones from
      their equations at the integrated states (`build_observation`). E.g. an observed flow that
      is computed from the water stored in the states.
    - `f(x, t, θ)` over the constants that appear in this candidate's equations (`build_problem`).
    - the data of the state variables seeds the node states; state variables without data start
      from `unobserved_inits` and are carried forward (see `ms`).
    - the constants' `range`s as bounds, so a fit can't leave a constant's declared range (same
      reason as in `pybm.estimate.gradient_matching.gradient_matching`).

    Parameters
    ----------
    model : InducedModel
        Its data is switched to `engine` (`switch_engine`), for every candidate sharing it.
    t_eval : array-like
        Time points: where the data is read and the model is compared with it.
    n_subintervals : int
        Number of shooting subintervals, see `ms`.
    engine : "torch" or "scipy"
        See `ms`. "torch" integrates with fixed-step Euler (`step`, default 1), "scipy" with
        anything `ms` accepts in **kwargs.
    unobserved_inits : array-like, optional
        Initial guess at t_eval[0] for the state variables without data, in state-variable order
        (see `ms`). Default: their declared `initial` values, where set - meant for the start of
        the model; if t_eval starts later, pass a guess for that time.
    device : torch.device, optional
        engine="torch": where the model's data lives. Defaults to CPU.
    **kwargs :
        Forwarded to `ms`: integration settings (`method`, `step`, `rtol`, ...), `minimizer_kwargs`,
        `weights`, `huber_delta`, `int_threshold`. Bounds from the constants' ranges are added to
        `minimizer_kwargs` unless it already has `bounds`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        `ms`'s own result (`.theta` the fitted free constants, `.inits` the node states, columns in
        `.state_vars` order), with `.model`, `.consts` (every constant, in `index_in_ctx` order),
        `.const_by_name`, `.observed_vars` (the compared variables, in `.weights` order) and
        `.cost` (the minimized loss, `.fun`) attached - the same convention as
        `pybm.estimate.gradient_matching.gradient_matching`.
    """
    if engine == "torch":
        model.switch_engine("torch", device=device or torch.device("cpu"))
    else:
        model.switch_engine("scipy")
    t_eval = np.asarray(t_eval, dtype=float)
    names = {id(v): name for name, v in model.vars.items()}

    read = lambda vars_: _get_data_tensor(vars_, t_eval, torch.device("cpu"), torch.float64).numpy().T
    observed_vars = [v for v in model.get_endo_variables() if v.data is not None]
    if not observed_vars:
        raise ValueError("multishooting compares the model with data, and no endogenous variable has any.")
    observed_names = [names[id(v)] for v in observed_vars]
    ys = read(observed_vars)

    state_vars, _, _ = model.split_endo_vars()
    xs = np.full((len(t_eval), len(state_vars)), np.nan)
    with_data = [i for i, v in enumerate(state_vars) if v.data is not None]
    if with_data:
        xs[:, with_data] = read([state_vars[i] for i in with_data])
    hidden = [i for i, v in enumerate(state_vars) if v.data is None]
    if unobserved_inits is None and hidden and all(state_vars[i].initial is not None for i in hidden):
        unobserved_inits = [state_vars[i].initial for i in hidden]

    f, theta_0, (lo, hi), free_idx = build_problem(model, engine)
    observe = build_observation(model, observed_names, engine)
    n_nodes = n_subintervals * len(state_vars)
    minimizer_kwargs = dict(kwargs.pop("minimizer_kwargs", None) or {})
    minimizer_kwargs.setdefault("bounds", Bounds(np.r_[np.full(n_nodes, -np.inf), lo], np.r_[np.full(n_nodes, np.inf), hi]))
    if engine == "scipy":
        kwargs.setdefault("batched_theta", True)  # build_problem's numpy f takes one θ per row

    result = ms(f, t_eval, xs, n_subintervals, theta_0, unobserved_inits=unobserved_inits, engine=engine,
                minimizer_kwargs=minimizer_kwargs, observe=observe, ys=ys, **kwargs)

    all_lo, all_hi = _const_bounds(model)
    consts = np.clip(_initial_const_guess(model), all_lo, all_hi)
    consts[free_idx] = result.theta
    result.model = model
    result.consts = consts
    result.const_by_name = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    result.state_vars = [names[id(v)] for v in state_vars]
    result.observed_vars = observed_names
    result.cost = result.fun
    return result
