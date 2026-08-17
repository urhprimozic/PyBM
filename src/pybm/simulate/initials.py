"""
Builds a ready-to-simulate `Context` for an induced model whose variables already carry observed
data (e.g. a model converted from ProBMoT via `set_data()`), by picking reasonable initial values
for everything `simulate()` needs.
"""

from __future__ import annotations

import numpy as np

from pybm.model import Context, InducedModel


def _sample_value(rng: np.random.Generator, range_, fallback_range: "tuple[float, float]") -> float:
    """
    Draws a value uniformly from `range_` (a `(lo, hi)` tuple). `fallback_range` substitutes for a
    missing bound - `range_` itself being `None`, or either bound being non-finite (there's no
    meaningful "uniform over all reals" to draw from). If the result would still be an empty/
    inverted interval (e.g. a genuine lower bound already above the whole fallback window),
    `fallback_range`'s width is used above that lower bound instead.
    """
    lo, hi = range_ if range_ is not None else fallback_range
    if lo is None or not np.isfinite(lo):
        lo = fallback_range[0]
    if hi is None or not np.isfinite(hi):
        hi = fallback_range[1]
    if hi <= lo:
        hi = lo + (fallback_range[1] - fallback_range[0])
    return float(rng.uniform(lo, hi))


def get_ctx(
    model: InducedModel,
    t: "float | None" = None,
    seed: "int | None" = None,
    fallback_range: "tuple[float, float]" = (0.0, 10.0),
) -> Context:
    """
    Builds an initial `Context` for `model` (the output of `Model.induce()`).

    - Endogenous variables: value read from `var.data` at time `t` (linearly interpolated - see
      `TimeSeries.__call__`). If `t` is None, each variable uses the earliest time point in its
      OWN data (variables don't need to share one time grid). A variable with no data at all
      falls back to its declared `.initial`; if that's also unset, a differential (`.ode`)
      variable raises a ValueError - there is nothing sensible to guess for an integrated state.
      An algebraic-only variable (no `.ode`) instead gets 0.0 - `simulate()` re-derives it from
      scratch at every step (see `_settle_algebraic` in predict.py) rather than integrating it, so
      this slot is never actually read.
    - Constants: value read from `.initial_value` if set, otherwise drawn uniformly from `.range`;
      if `.range` is also unset (or has an infinite bound), `fallback_range` is used instead (see
      `_sample_value`).

    `seed`, if given, makes constants drawn for missing values reproducible.
    """
    rng = np.random.default_rng(seed)

    endo_vars = model.get_endo_variables()
    var_ctx = np.zeros(len(endo_vars), dtype=float)
    for var in endo_vars:
        if var.data is not None:
            var_t = t if t is not None else var.data.t[0]
            var_ctx[var.index_in_ctx] = var.data(var_t)
        elif var.initial is not None:
            var_ctx[var.index_in_ctx] = var.initial
        elif var.ode is None:
            # algebraic-only variable (no .ode) - simulate() re-derives it from scratch at every
            # step instead of integrating it (see _settle_algebraic in predict.py, which always
            # starts that pass from zero), so nothing ever actually reads this slot's "initial"
            # value - 0.0 is just a harmless placeholder.
            var_ctx[var.index_in_ctx] = 0.0
        else:
            raise ValueError(
                f"Variable {var.name!r} has neither data nor an initial value - nothing to seed "
                "its initial (integrated) state with."
            )

    const_ctx = np.zeros(len(model.consts), dtype=float)
    for const in model.consts.values():
        if const.initial_value is not None:
            const_ctx[const.index_in_ctx] = const.initial_value
        else:
            const_ctx[const.index_in_ctx] = _sample_value(rng, const.range, fallback_range)

    return Context(vars=var_ctx, consts=const_ctx)
