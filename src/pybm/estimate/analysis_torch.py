"""
Post-fit analysis helpers for the torch multishooting pipeline:
trajectory reconstruction (single- and multiple-shooting) and a data-fit
MSE metric. Kept separate from `multishooting_torch.py` since none of
this is part of *fitting* -- it's what you run after `estimate_torch` to
look at / sanity-check the result.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from pybm.estimate.multishooting_torch import (
    _build_subinterval_grid,
    _make_solver,
    _solve_segments,
    _stitch_trajectory,
    uniform_sub_indices,
)
from pybm.model import Model


def simulate_multishooting(
    model: Model,
    t_eval,
    params,
    n_subintervals: int,
    sub_indices: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    solver_atol: float = 1e-8,
    solver_rtol: float = 1e-6,
    solver_max_steps: Optional[int] = 2000,
    solver_dt_min: Optional[float] = None,
):
    """
    Reconstruct a stitched multiple-shooting trajectory from a flat parameter vector.

    The parameter layout matches `estimate_torch`:
        [consts..., s_0, s_1, ..., s_{K-1}]
    where each shooting state s_i seeds segment i.

    Pass the same `sub_indices` used to fit `params` if it was fit with an
    explicit (non-uniform) segment layout -- otherwise the reconstruction
    would use different segment boundaries than the fit did.
    """
    if model.engine != "torch":
        raise ValueError(
            f"simulate_multishooting requires a Model built with engine='torch', got engine={model.engine!r}."
        )

    vars_ = model.get_endo_variables()
    n_consts = len(model.consts)
    n_vars = len(vars_)
    device = device or torch.device("cpu")

    if sub_indices is not None:
        n_subintervals = len(np.asarray(sub_indices)) - 1

    params_t = torch.as_tensor(params, dtype=dtype, device=device)
    if params_t.ndim == 1:
        params_t = params_t.unsqueeze(0)
    if params_t.ndim != 2:
        raise ValueError("params must be a 1D or 2D tensor/array.")

    if params_t.shape[1] < n_consts + n_subintervals * n_vars:
        raise ValueError("params has fewer entries than expected for the model and n_subintervals.")

    grid = _build_subinterval_grid(
        np.asarray(t_eval, dtype=float), n_subintervals, device, dtype, sub_indices=sub_indices
    )
    solver = _make_solver(vars_, solver_atol, solver_rtol, max_steps=solver_max_steps, dt_min=solver_dt_min)

    const_ctx = params_t[:, :n_consts]
    initials = params_t[:, n_consts : n_consts + n_subintervals * n_vars].reshape(
        params_t.shape[0], n_subintervals, n_vars
    )
    ys = _solve_segments(vars_, const_ctx, initials, grid, solver)
    stitched = _stitch_trajectory(ys, grid)

    return SimpleNamespace(
        t=np.asarray(t_eval, dtype=float),
        y=stitched[0].detach().cpu().numpy().T,
        success=True,
        message="multiple-shooting reconstruction",
    )


def simulate(model: Model, t_eval, consts, initial=None, **kwargs):
    """
    Single-shooting simulation: one global initial condition, forward-
    simulated across the WHOLE horizon with `consts` -- just
    `simulate_multishooting` with `n_subintervals=1` (see its docstring
    for what's returned and for `**kwargs`), so it's exactly as sensitive
    to a bad `consts` guess as any single-shooting integration is.

    `initial`, if omitted, is read off each endogenous variable's observed
    data at `t_eval[0]` (falling back to its declared `.initial` if it has
    no data) -- the same convention `_sample_initial_params` uses.
    """
    vars_ = model.get_endo_variables()
    t_eval = np.asarray(t_eval, dtype=float)

    if initial is None:
        t0 = t_eval[0]
        initial = [
            float(var.data(t0)) if var.data is not None else (var.initial or 0.0)
            for var in vars_
        ]

    params = np.concatenate([np.asarray(consts, dtype=float), np.asarray(initial, dtype=float)])
    return simulate_multishooting(model, t_eval, params, n_subintervals=1, **kwargs)


def MSE(model: Model, t_eval, n_subintervals: int, consts, sub_indices: Optional[np.ndarray] = None, **kwargs) -> float:
    """
    Mean squared error between the observed data and an
    `n_subintervals`-segment reconstruction using `consts`, where every
    segment is reset to the OBSERVED data at its start -- not a fitted
    free shooting state, since only `consts` is given here.

    This is an n-step-ahead prediction error: how well do `consts` explain
    the data over segments of this length, decoupled from how far a fit
    would additionally have to stretch to stitch far-apart segments
    together (that stitching error is what `estimate_torch`'s `cont_res`
    tracks instead). `n_subintervals=1` recovers the MSE of a single,
    full-horizon forward simulation (see `simulate`) -- the harshest,
    most single-shooting-sensitive setting; larger `n_subintervals` gives
    a more local, more forgiving picture of the same `consts`.
    """
    vars_ = model.get_endo_variables()
    t_eval = np.asarray(t_eval, dtype=float)

    if sub_indices is None:
        sub_indices = uniform_sub_indices(t_eval, n_subintervals)
    else:
        sub_indices = np.asarray(sub_indices, dtype=int)
        n_subintervals = len(sub_indices) - 1

    seeds = np.array(
        [[float(var.data(t_eval[sub_indices[i]])) for var in vars_] for i in range(n_subintervals)],
        dtype=float,
    )
    params = np.concatenate([np.asarray(consts, dtype=float), seeds.reshape(-1)])

    sim = simulate_multishooting(
        model, t_eval, params, n_subintervals=n_subintervals, sub_indices=sub_indices, **kwargs
    )
    observed = np.stack([[float(var.data(t)) for t in t_eval] for var in vars_])
    return float(np.mean((sim.y - observed) ** 2))
