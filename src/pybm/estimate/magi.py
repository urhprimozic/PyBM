from typing import Literal

import numpy as np
import torch
from odestimate.latent.magi import magi as _magi
from odestimate.latent.multishooting import _scipy_trajectory

from pybm.estimate.gradient_matching import build_observation, build_problem
from pybm.estimate.preprocess import _const_bounds, _initial_const_guess
from pybm.model import InducedModel, _get_data_tensor


def magi(model: InducedModel, t_eval, method: Literal["max_likelihood", "bayes"] = "max_likelihood",
         unobserved_inits=None, init_integration: "dict | None" = None, **kwargs):
    """
    Estimate a candidate's constants with MAGI - a thin PyBM-side adapter around
    `odestimate.latent.magi.magi`, which does the actual work (see PyBM notes/magi.md). Turns the
    `InducedModel` into what `magi` takes:

    - the data of every endogenous variable that has data, read at `t_eval`. If all of them are
      state variables, they are compared with the states directly; if any is algebraic (e.g. an
      observed flow computed from the water stored in the states), every one is compared through
      the model's equations (`build_observation`).
    - the data of the state variables sets their GP prior and starting trajectory; state variables
      without data are unobserved.
    - `f(x, t, θ)` over the constants that appear in this candidate's equations (`build_problem`).
    - the constants' `range`s as bounds.

    Parameters
    ----------
    model : InducedModel
        Its data is switched to numpy (`switch_engine("scipy")`), for every candidate sharing it.
    t_eval : array-like
        Data times: where the data is read. Also the seeds, unless `n_seeds` or `grid` is passed in
        **kwargs.
    method : "max_likelihood" or "bayes"
        See `odestimate.latent.magi.magi`.
    unobserved_inits : array-like, optional
        Starting trajectory (or constant) of the state variables without data, see `magi`. Default:
        one simulation with the initial constants from the variables' values at t_eval[0] - data
        for the observed ones, the declared `initial` for the others (meant for the start of the
        model; if t_eval starts later, pass a guess). The GP prior of those variables is then
        fitted to this trajectory.
    init_integration : dict, optional
        Settings of that simulation, see `odestimate.latent.multishooting._scipy_trajectory`
        (e.g. {"method": "euler", "step": 1.0}). Default: solve_ivp's RK45.
    **kwargs :
        Forwarded to `magi`: n_seeds, grid, ode_points, midpoints, kernel, phi, sigma, mu, beta,
        minimizer_kwargs, hmc_kwargs, seed, gp_restarts.

    Returns
    -------
    scipy.optimize.OptimizeResult
        `magi`'s own result (`.theta` the fitted free constants, `.x` the states on `.grid`,
        columns in `.state_vars` order), with `.model`, `.consts` (every constant, in `index_in_ctx`
        order), `.const_by_name`, `.observed_vars` and `.cost` (the minimized loss of the MAP)
        attached - the same convention as `pybm.estimate.multishooting.multishooting`.
    """
    model.switch_engine("scipy")
    t_eval = np.asarray(t_eval, dtype=float)
    names = {id(v): name for name, v in model.vars.items()}
    read = lambda vars_: _get_data_tensor(vars_, t_eval, torch.device("cpu"), torch.float64).numpy().T

    observed_vars = [v for v in model.get_endo_variables() if v.data is not None]
    if not observed_vars:
        raise ValueError("magi compares the model with data, and no endogenous variable has any.")
    observed_names = [names[id(v)] for v in observed_vars]

    state_vars, _, _ = model.split_endo_vars()
    xs = np.full((len(t_eval), len(state_vars)), np.nan)
    with_data = [i for i, v in enumerate(state_vars) if v.data is not None]
    if with_data:
        xs[:, with_data] = read([state_vars[i] for i in with_data])
    hidden = [i for i, v in enumerate(state_vars) if v.data is None]

    f, theta_0, (lo, hi), free_idx = build_problem(model, "scipy")
    observe = ys = None
    if any(v not in state_vars for v in observed_vars):
        observe = build_observation(model, observed_names, "scipy")
        ys = read(observed_vars)

    if unobserved_inits is None and hidden and all(state_vars[i].initial is not None for i in hidden):
        x0 = np.array([xs[0, i] if i in with_data else state_vars[i].initial for i, v in enumerate(state_vars)], dtype=float)
        trajectory, _ = _scipy_trajectory(f, t_eval[0], t_eval[-1], t_eval, x0, theta_0, 1e6, **(init_integration or {}))
        unobserved_inits = trajectory[:, hidden]  # on the data times; magi interpolates it to its seeds

    result = _magi(f, t_eval, xs, theta_0, unobserved_inits=unobserved_inits, method=method, bounds=(lo, hi),
                   observe=observe, ys=ys, **kwargs)

    all_lo, all_hi = _const_bounds(model)
    consts = np.clip(_initial_const_guess(model), all_lo, all_hi)
    consts[free_idx] = result.theta
    result.model = model
    result.consts = consts
    result.const_by_name = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    result.state_vars = [names[id(v)] for v in state_vars]
    result.observed_vars = observed_names
    result.cost = result.cost if method == "max_likelihood" else result.map.cost
    return result
