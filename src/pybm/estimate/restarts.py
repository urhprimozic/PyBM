"""
Several starts for an estimator: the usual start, then starts drawn within the constants' ranges.
For fits that end in different local minima depending on where they start.
"""

import numpy as np

from pybm.estimate.gradient_matching import build_problem
from pybm.model import InducedModel


def starts(model: InducedModel, n_restarts: int, seed=None, first=None) -> np.ndarray:
    """
    (n_restarts, n_free) starting values of the constants the equations use (`build_problem`
    order). The first is `first`, by default the usual start (initial values, else the middle of
    the ranges); the other n_restarts - 1 are drawn uniformly within the ranges. A constant with an
    infinite range keeps the first start's value.
    """
    _, theta_0, (lo, hi), _ = build_problem(model, "scipy")
    theta_0 = theta_0 if first is None else np.asarray(first, float)
    finite = np.isfinite(lo) & np.isfinite(hi)
    draws = np.tile(theta_0, (n_restarts, 1))
    draws[1:, finite] = np.random.default_rng(seed).uniform(lo[finite], hi[finite], size=(n_restarts - 1, finite.sum()))
    return draws


def best_of(estimate, model: InducedModel, n_restarts: int = 1, seed=None, first=None):
    """
    Runs estimate(theta_0) -> OptimizeResult (with .cost) from every start of `starts`; returns the
    result with the lowest cost, with .restart_costs (every start's cost, inf where it failed).
    """
    best, costs = None, []
    for theta_0 in starts(model, n_restarts, seed, first):
        try:
            result = estimate(theta_0)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):  # a start where the model can't be evaluated
            costs.append(np.inf)
            continue
        cost = result.cost if np.isfinite(result.cost) else np.inf
        costs.append(cost)
        if best is None or cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError(f"the estimation failed from all {n_restarts} starts")
    best.restart_costs = costs
    return best
