import numpy as np
import torch
from pybm.model import InducedModel, _get_data_tensor, _make_rhs, _make_rhs_numpy
from pybm.model_graph import collect_leaves

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


def _used_const_indices(equation_vars: "list") -> np.ndarray:
    """Sorted `index_in_ctx` of every Const that appears in at least one of these vars' equations."""
    consts = {}
    for var in equation_vars:
        _, found = collect_leaves(var.ode if var.ode is not None else var.algebraic)
        consts.update(found)
    return np.array(sorted(c.index_in_ctx for c in consts.values()), dtype=int)
