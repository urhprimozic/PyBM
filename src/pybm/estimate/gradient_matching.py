import numpy as np
import torch
from odestimate.gp.regressor import GP
from odestimate.gradient_matching import gradient_matching as _gradient_matching
from odestimate.gradient_matching import uniform_trust_region
from pybm.model import InducedModel, _get_data_tensor, _make_observation, _make_rhs, _make_rhs_numpy
from pybm.model_graph import collect_leaves
from pybm.estimate.preprocess import _const_bounds, _initial_const_guess, _used_const_indices



def build_problem(model: InducedModel, engine: str = "torch"):
    """
    Turns a candidate into the `(f, theta0, bounds)` that `odestimate.gradient_matching`'s
    `gradient_matching`/`confidence_interval` take - over only the constants that actually appear
    in this candidate's equations. The model declares every constant of every library process (on
    Bled: 16, of which a candidate uses 4-7); the rest have an identically zero Jacobian column, so
    fitting them only costs Jacobian time and conditioning.

    Parameters
    ----------
    model : InducedModel
        Its data must already be in `engine`'s format (`model.switch_engine`).
    engine : "torch" or "scipy"
        Build `f` from torch (`_make_rhs`) or numpy (`_make_rhs_numpy`) operations.

    Returns
    -------
    f : callable
        `f(x, t, theta) -> dx/dt`, `x` shape `(n, n_state_vars)`, `theta` the free constants only.
        For engine="scipy", `theta` may also be `(n, n_free)`: one parameter vector per row.
    theta0 : np.ndarray
        Initial guess for the free constants (clipped into their bounds).
    bounds : tuple[np.ndarray, np.ndarray]
        `(lo, hi)` for the free constants - pass to `least_squares` as `bounds=`.
    free_idx : np.ndarray
        `theta[k]` is the constant with `index_in_ctx == free_idx[k]`. Every other constant is
        held at its initial guess (it doesn't enter any equation, so its value is irrelevant).
    """
    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    free_idx, full0, lo, hi = _free_consts(model)
    if engine == "scipy":
        f = _with_free_consts(_make_rhs_numpy(state_vars, algebraic_vars, frozen_values), engine, free_idx, full0)
    else:
        f = _with_free_consts(_make_rhs(state_vars, algebraic_vars, frozen_values), engine, free_idx, full0)
    return f, full0[free_idx], (lo[free_idx], hi[free_idx]), free_idx


def build_observation(model: InducedModel, var_names: "list[str]", engine: str = "torch"):
    """
    `observe(x, t, theta) -> (n, len(var_names))`: the values of the endogenous variables
    `var_names` (state or algebraic) at states x, over the same free constants as `build_problem`'s
    `f` (same `theta` order, same shapes). For estimators that compare observed variables which
    aren't states, e.g. a flow computed from the water stored in the states.
    """
    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    free_idx, full0, _, _ = _free_consts(model)
    observe = _make_observation(state_vars, algebraic_vars, frozen_values, [model.vars[n] for n in var_names])
    return _with_free_consts(observe, engine, free_idx, full0)


def _free_consts(model: InducedModel):
    """(free_idx, full0, lo, hi): the constants the equations use, and every constant's initial
    guess (clipped into its bounds) and bounds."""
    state_vars, algebraic_vars, _ = model.split_endo_vars()
    lo, hi = _const_bounds(model)
    return _used_const_indices(state_vars + algebraic_vars), np.clip(_initial_const_guess(model), lo, hi), lo, hi


def _with_free_consts(g, engine: str, free_idx: np.ndarray, full0: np.ndarray):
    """Turns g(t, y, consts) over every constant into fun(x, t, theta) over the free ones only;
    the others stay at full0."""
    if engine == "scipy":
        def fun(x, t, theta):
            theta = np.asarray(theta, dtype=float)
            if theta.ndim == 2:  # one parameter vector per row: each const reads as an (m,) array
                consts = np.repeat(full0[:, None], len(theta), axis=1)
                consts[free_idx] = theta.T
            else:
                consts = full0.copy()
                consts[free_idx] = theta
            return g(t, x, consts)
        return fun

    base = torch.as_tensor(full0, dtype=torch.float64)
    index = torch.as_tensor(free_idx)

    def fun(x, t, theta):
        consts = base.index_put((index,), theta)  # out-of-place, autodiff-friendly
        return g(t, x, consts.unsqueeze(0).expand(x.shape[0], -1))
    return fun


def gradient_matching(model: InducedModel, t_eval, gp: "GP | None" = None, device: "torch.device | None" = None,
                      n_samples: "int | None" = None, n_restarts: int = 1, seed=None, power: float = 2.0, **kwargs):
    """
    Estimate a candidate's constants by gradient matching against a GP of its state variables - a
    thin PyBM-side adapter around `odestimate.gradient_matching`. Two modes:

    - default (`n_samples` None): least squares between the GP's derivative and f at the GP's mean,
      at the points `t_eval`, over the constants that appear in this candidate's equations
      (`build_problem`, numpy), within their ranges, from `n_restarts` starts.
    - `n_samples` given: the screening pass of `pybm.estimate.estimate.estimate_models_gradient_matching`
      (`_screening`): `odestimate.gradient_matching.uniform_trust_region` on `n_samples` random points
      of [t_eval[0], t_eval[-1]], torch, every constant, plus a confidence interval (`.interval`).

    Parameters
    ----------
    model : InducedModel
    t_eval : array-like
        Default mode: where the derivatives are matched, and where the state variables' data is read
        to fit a GP when `gp` is None. Screening mode: see `_screening`.
    gp : odestimate.gp.regressor.GP, optional
        A GP of the state variables (state-variable order) fitted elsewhere, e.g. once for every
        candidate. Default: fitted on the state variables' data at t_eval.
    device : torch.device, optional
        Screening mode only.
    n_samples : int, optional
        Switches to the screening mode.
    n_restarts : int
        Default mode: number of starts - 1 is the usual start only (initial values, else the middle
        of the ranges); otherwise also n_restarts - 1 starts drawn uniformly within the ranges, and
        the fit with the lowest cost is returned, with .restart_costs (`pybm.estimate.restarts`).
    seed : int, optional
        Seed of those draws.
    power : float
        Default mode: the loss ½ Σ |r|^power (`odestimate.gradient_matching.powered`).
    **kwargs :
        Default mode: forwarded to `scipy.optimize.least_squares` (ftol, xtol, gtol, max_nfev, ...).
        Screening mode: to `uniform_trust_region`.

    Returns
    -------
    scipy.optimize.OptimizeResult
        least_squares' result; .x (and .theta) the fitted constants - in the default mode only the
        ones the equations use, in `build_problem` order - with .model, .consts (every constant, in
        `index_in_ctx` order) and .const_by_name attached.
    """
    if n_samples is not None:
        return _screening(model, t_eval, gp, device, n_samples=n_samples, **kwargs)
    from pybm.estimate.restarts import best_of  # imports build_problem from here

    model.switch_engine("scipy")
    t_eval = np.asarray(t_eval, dtype=float)
    state_vars, _, _ = model.split_endo_vars()
    if gp is None:
        gp = GP(t_eval, _get_data_tensor(state_vars, t_eval, torch.device("cpu"), torch.float64).numpy())
    if len(gp) != len(state_vars):
        raise ValueError(f"gp has {len(gp)} component(s), the model {len(state_vars)} state variable(s).")

    f, theta_0, (lo, hi), free_idx = build_problem(model, "scipy")
    xs, dxs = gp.mean(t_eval), gp.derivative(t_eval)
    fit = lambda start: _gradient_matching(t_eval, xs, dxs, f, start, engine="scipy", power=power, bounds=(lo, hi), **kwargs)
    result = fit(theta_0) if n_restarts == 1 else best_of(fit, model, n_restarts, seed)

    all_lo, all_hi = _const_bounds(model)
    consts = np.clip(_initial_const_guess(model), all_lo, all_hi)
    consts[free_idx] = result.x
    result.theta = result.x
    result.model = model
    result.consts = consts
    result.const_by_name = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    return result


def _screening(model: InducedModel, t_eval, gp: "GP | None" = None, device: "torch.device | None" = None, **kwargs):
    """
    The screening mode of `gradient_matching` (with `n_samples`): estimate parameters of a model
    using gradient matching - a thin PyBM-side adapter around
    `odestimate.gradient_matching.uniform_trust_region`, which does the actual work (GP fit +
    least-squares against the GP-implied derivative, plus a confidence interval - see
    `notes/gradient-matching-pruning.md`). All this function does is turn a PyBM `InducedModel`
    into the plain `(t_obs, x_obs, f, theta_0)` shape that call needs:

    - `x_obs`: every STATE (differential) variable's own observed data, stacked `(n_vars, n)`
      (`_get_data_tensor` - same convention `odestimate.gp.regressor.GP` expects).
    - `f(x, t, theta)`: wraps PyBM's own `_make_rhs` (the same right-hand side `multishooting_torch`
      integrates), broadcasting the single `theta` vector `gradient_matching` passes into the
      `(N, n_consts)` "same constants at every point" context `_make_rhs` expects - the ONLY real
      adaptation needed, since `_make_rhs`'s own `(t, y, const_ctx)` shapes already match
      `odestimate`'s `(x, t, theta)` batched convention.

    Parameters
    ----------
    model : InducedModel
        Must have `engine == "torch"`, every `Var.ode` in differentiable torch ops, and every
        STATE variable's own `.data` set (gradient matching needs an observed trajectory for each
        one - same requirement `_get_data_tensor` already enforces).
    t_eval : array-like
        Time points - where `x_obs` is read from (to fit a fresh GP, only if `gp` is not given)
        and the range `uniform_trust_region`'s own test points are sampled from.
    gp : odestimate.gp.regressor.GP, optional
        A GP already fit elsewhere - reuse it instead of fitting a fresh one on `model`'s own data.
        Lets a caller (e.g. `pybm.estimate.estimate.estimate_models`) fit ONE GP ONCE and share it
        across many candidate `model`s that all have the SAME state variables (gradient matching's
        interpolant never depends on which candidate RHS is chosen, only on each state variable's
        own observed data - see the chat this followed). When given, `x_obs` is never built at all
        (not just unused) - skips `_get_data_tensor` entirely, since it would only have been used
        to fit the GP this call is now skipping.
    device : torch.device, optional
        Where `model`'s own data lives (`model.switch_engine`) - defaults to CPU. Kept separate
        from `odestimate.gradient_matching`'s own internal device/dtype choice (`DEVICE`/`DTYPE`,
        `torch.float64` and CUDA-if-available) - the two never need to match: `f` only ever
        operates on whatever tensors it's actually given, never allocates on a hardcoded device.
    **kwargs :
        Forwarded to `odestimate.gradient_matching.uniform_trust_region` (e.g. `n_samples`,
        `p_value` - see its own docstring).

    Returns
    -------
    scipy.optimize.OptimizeResult
        `uniform_trust_region`'s own result, with `.model` (this same `model`), `.consts` (an
        alias for `.x`) and `.const_by_name` (`{name: value}`) attached on top - PyBM's usual
        `ParamEstimationResults`-like convention. `.interval` (from `uniform_trust_region`) is
        also attached, unchanged.
    """
    if model.engine != "torch":
        raise ValueError(f"gradient_matching requires an InducedModel built with engine='torch', got engine={model.engine!r}.")
    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)

    def f(x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        const_ctx = theta.unsqueeze(0).expand(x.shape[0], -1)  # same theta at every point
        return rhs(t, x, const_ctx)

    if gp is not None and len(gp) != len(state_vars):
        raise ValueError(
            f"gp was fit on {len(gp)} state variable(s) but {model!r} has {len(state_vars)} - a "
            f"shared gp is only valid across candidates with the SAME state variables, in the "
            f"same order (see pybm.estimate.estimate._check_shared_state_vars, which should have "
            f"caught this before calling here)."
        )
    # `x_obs` is only ever needed to FIT a fresh GP - skip building it entirely when reusing one.
    x_obs = None if gp is not None else _get_data_tensor(state_vars, t_eval, device, torch.float64).cpu().numpy()

    # collect initial guesses and bounds for constants
    theta0 = _initial_const_guess(model)
    lo, hi = _const_bounds(model)
    theta0 = np.clip(theta0, lo, hi)

    # `bounds=` MUST reach `scipy.optimize.least_squares` itself, not just clip theta0 - checked
    # directly: without this, an unconstrained fit can push a constant PAST its declared range
    # (e.g. a rate meant to be >=0 converging to a negative value), silently making two
    # structurally DIFFERENT candidates (e.g. a Choose between +k*x and -k*x) converge to the SAME
    # fit - which defeats the whole point of comparing/pruning between them.
    result, interval = uniform_trust_region(t_eval, x_obs, f, theta0, gp=gp, bounds=(lo, hi), **kwargs)
    result.model = model
    result.consts = result.x
    result.const_by_name = {name: float(result.x[c.index_in_ctx]) for name, c in model.consts.items()}
    result.interval = interval
    return result

