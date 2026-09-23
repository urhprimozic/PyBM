import numpy as np
import torch
from odestimate.gp.regressor import GP
from odestimate.gradient_matching import uniform_trust_region

from pybm.model import InducedModel, _get_data_tensor, _make_rhs


def _const_bounds(model: InducedModel) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(model.consts), -np.inf)
    hi = np.full(len(model.consts), np.inf)
    for const in model.consts.values():
        if const.range is not None:
            lo[const.index_in_ctx] = const.range[0]
            hi[const.index_in_ctx] = const.range[1]
    return lo, hi
def _initial_const_guess(model: InducedModel) -> np.ndarray:
    """
    Starting guess for the constant fit: `initial_value` where known, otherwise the midpoint of
    `range` (falling back to 0.0 only when neither is available). A blanket 0.0 default (as
    `int_scipy.get_initial_const_ctx` uses) can land a constant that appears as a divisor in some
    equation (e.g. a `refTemp`) at exactly zero, which makes the very first residual evaluation
    NaN/inf - `least_squares` refuses to even start from a point like that.
    """
    c0 = np.zeros(len(model.consts), dtype=float)
    for const in model.consts.values():
        if const.initial_value is not None:
            c0[const.index_in_ctx] = const.initial_value
        elif const.range is not None and np.isfinite(const.range[0]) and np.isfinite(const.range[1]):
            c0[const.index_in_ctx] = 0.5 * (const.range[0] + const.range[1])
        else:
            c0[const.index_in_ctx] = 0.0
    return c0

def gradient_matching(model: InducedModel, t_eval, gp: "GP | None" = None, device: "torch.device | None" = None, **kwargs):
    """
    Estimate parameters of a model using gradient matching - a thin PyBM-side adapter around
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

