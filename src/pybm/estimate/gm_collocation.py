import numpy as np
import torch
from odestimate.matching import matching

from pybm.estimate.gradient_matching import build_observation, build_problem
from pybm.estimate.preprocess import _const_bounds, _initial_const_guess
from pybm.estimate.restarts import best_of
from pybm.model import InducedModel, _get_data_tensor


def gm_collocation(model: InducedModel, t_eval, gp=None, theta_0=None, unobserved_inits=None, n_restarts: int = 1,
                   seed=None, **kwargs):
    """
    Estimate a candidate's constants by gradient matching on its observed state variables and
    collocation for the unobserved ones - a thin PyBM-side adapter around
    `odestimate.matching.gm_collocation`, which does the actual work. Turns the `InducedModel` into
    what it takes:

    - the data of the state variables, read at `t_eval` (NaN where missing): state variables with
      data are matched through an interpolant of it (a GP), the others are free.
    - observed algebraic variables (e.g. a flow computed from the water stored in the states): a
      data term through the model's equations (`build_observation`), their data read at `t_eval`.
    - `f(x, t, θ)` over the constants that appear in this candidate's equations (`build_problem`).
    - the constants' `range`s as bounds.

    Parameters
    ----------
    model : InducedModel
        Its data is switched to numpy (`switch_engine("scipy")`), for every candidate sharing it.
    t_eval : array-like
        Data times.
    gp : optional
        Interpolant of the observed state variables (in state-variable order) fitted elsewhere, e.g.
        a GP once for every candidate. Default: a GP fitted here, see `odestimate.matching.core.matching`.
    theta_0 : array-like, optional
        Start for the free constants, in `build_problem` order. Default: their initial values.
    unobserved_inits : array-like, optional
        Start for the state variables without data. Default: their declared `initial` values at
        t_eval[0].
    n_restarts : int
        Number of starts: 1 is theta_0 only; otherwise also n_restarts - 1 starts drawn uniformly
        within the constants' ranges (`pybm.estimate.restarts`), and the fit with the lowest cost
        is returned, with .restart_costs.
    seed : int, optional
        Seed of those draws.
    **kwargs :
        Forwarded to `odestimate.matching.core.matching`: points, n_points, n_seeds, seeds,
        ode_points, sigma, s, power, gp_kwargs, and the optimizer's arguments.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The estimator's result, with `.model`, `.consts` (every constant, in `index_in_ctx` order),
        `.const_by_name`, `.state_vars` and `.observed_vars` attached - the same convention as
        `pybm.estimate.collocation.collocation`.
    """
    return _fit(model, t_eval, "derivative", gp, theta_0, unobserved_inits, n_restarts, seed, kwargs)


def im_collocation(model: InducedModel, t_eval, gp=None, theta_0=None, unobserved_inits=None, n_restarts: int = 1,
                   seed=None, **kwargs):
    """`gm_collocation` with integral matching on the observed state variables (window, n_quad in **kwargs)."""
    return _fit(model, t_eval, "integral", gp, theta_0, unobserved_inits, n_restarts, seed, kwargs)


def _fit(model, t_eval, form, gp, theta_0, unobserved_inits, n_restarts, seed, kwargs):
    model.switch_engine("scipy")
    t_eval = np.asarray(t_eval, dtype=float)
    names = {id(v): name for name, v in model.vars.items()}
    read = lambda vars_: _get_data_tensor(vars_, t_eval, torch.device("cpu"), torch.float64).numpy().T

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
    outputs = [v for v in model.get_endo_variables() if v.data is not None and v not in state_vars]
    observe = ys = None
    if outputs:
        observe = build_observation(model, [names[id(v)] for v in outputs], "scipy")
        ys = read(outputs)

    fit = lambda start: matching(f, t_eval, xs, start, matched="observed", form=form, gp=gp,
                                 unobserved_inits=unobserved_inits, observe=observe, ys=ys, bounds=(lo, hi), **kwargs)
    result = fit(theta_0) if n_restarts == 1 else best_of(fit, model, n_restarts, seed, first=theta_0)

    all_lo, all_hi = _const_bounds(model)
    consts = np.clip(_initial_const_guess(model), all_lo, all_hi)
    consts[free_idx] = result.theta
    result.model = model
    result.consts = consts
    result.const_by_name = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    result.state_vars = [names[id(v)] for v in state_vars]
    result.observed_vars = [names[id(state_vars[i])] for i in with_data] + [names[id(v)] for v in outputs]
    return result
