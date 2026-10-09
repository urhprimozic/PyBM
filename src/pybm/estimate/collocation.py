import numpy as np
import torch
from odestimate.latent.collocation import collocation as _collocation

from pybm.estimate.gradient_matching import build_observation, build_problem
from pybm.estimate.preprocess import _const_bounds, _initial_const_guess
from pybm.model import InducedModel, _get_data_tensor


def collocation(model: InducedModel, t_eval, theta_0=None, unobserved_inits=None, **kwargs):
    """
    Estimate a candidate's constants by collocation - a thin PyBM-side adapter around
    `odestimate.latent.collocation.collocation`, which does the actual work. Turns the
    `InducedModel` into what `collocation` takes:

    - the data of every endogenous variable that has data, read at `t_eval`. If all of them are
      state variables, they are compared with the states directly; if any is algebraic (e.g. an
      observed flow computed from the water stored in the states), every one is compared through
      the model's equations (`build_observation`).
    - the data of the state variables gives their starting values; state variables without data
      are unobserved.
    - `f(x, t, θ)` over the constants that appear in this candidate's equations (`build_problem`).
    - the constants' `range`s as bounds.

    Parameters
    ----------
    model : InducedModel
        Its data is switched to numpy (`switch_engine("scipy")`), for every candidate sharing it.
    t_eval : array-like
        Data times: where the data is read and the loss is evaluated. Also the seeds, unless
        `n_seeds` or `seeds` is passed in **kwargs.
    theta_0 : array-like, optional
        Start for the free constants, in `build_problem` order (e.g. another method's `.theta`).
        Default: their initial values.
    unobserved_inits : array-like, optional
        Start for the state variables without data, see `collocation`. Default: their declared
        `initial` values at t_eval[0], carried forward with the initial constants - meant for the
        start of the model; if t_eval starts later, pass a guess for that time.
    **kwargs :
        Forwarded to `collocation`: n_seeds, seeds, n_ode_points, ode_points, sigma, and anything
        for `scipy.optimize.minimize` (e.g. options={"maxiter": 500}).

    Returns
    -------
    scipy.optimize.OptimizeResult
        `collocation`'s own result (`.theta` the fitted free constants, `.x` the states at
        `.seeds`, `.x_ts` at t_eval, columns in `.state_vars` order), with `.model`, `.consts`
        (every constant, in `index_in_ctx` order), `.const_by_name` and `.observed_vars` attached -
        the same convention as `pybm.estimate.multishooting.multishooting`.
    """
    model.switch_engine("scipy")
    t_eval = np.asarray(t_eval, dtype=float)
    names = {id(v): name for name, v in model.vars.items()}
    read = lambda vars_: _get_data_tensor(vars_, t_eval, torch.device("cpu"), torch.float64).numpy().T

    observed_vars = [v for v in model.get_endo_variables() if v.data is not None]
    if not observed_vars:
        raise ValueError("collocation compares the model with data, and no endogenous variable has any.")
    observed_names = [names[id(v)] for v in observed_vars]

    state_vars, _, _ = model.split_endo_vars()
    xs = np.full((len(t_eval), len(state_vars)), np.nan)
    with_data = [i for i, v in enumerate(state_vars) if v.data is not None]
    if with_data:
        xs[:, with_data] = read([state_vars[i] for i in with_data])
    hidden = [i for i, v in enumerate(state_vars) if v.data is None]
    if unobserved_inits is None and hidden and all(state_vars[i].initial is not None for i in hidden):
        unobserved_inits = [state_vars[i].initial for i in hidden]

    f, initial_consts, (lo, hi), free_idx = build_problem(model, "scipy")
    theta_0 = initial_consts if theta_0 is None else np.asarray(theta_0, float)
    observe = ys = None
    if any(v not in state_vars for v in observed_vars):
        observe = build_observation(model, observed_names, "scipy")
        ys = read(observed_vars)

    result = _collocation(f, t_eval, xs, theta_0, unobserved_inits=unobserved_inits, observe=observe, ys=ys,
                          bounds=(lo, hi), **kwargs)

    all_lo, all_hi = _const_bounds(model)
    consts = np.clip(_initial_const_guess(model), all_lo, all_hi)
    consts[free_idx] = result.theta
    result.model = model
    result.consts = consts
    result.const_by_name = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    result.state_vars = [names[id(v)] for v in state_vars]
    result.observed_vars = observed_names
    return result
