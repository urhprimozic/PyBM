"""
Growing-horizon single-shooting via predictor-corrector continuation (Davidenko's equation) - the
user's own design (see the chat this followed), replacing `single_shooting_continuation.py`'s
Adam-per-round approach.

Idea: define `o(T) = argmin_theta L_T(theta)`, the optimal fit on the window `[t0, T]`. Since
`d/dtheta L_T(o(T)) = 0` for every `T`, differentiating through `T` gives (Davidenko's equation):

    H(T) . do/dT + d/dT[grad_theta L_T](o(T)) = 0     =>     do/dT = -H(T)^-1 . d/dT[grad_theta L_T]

where `H(T) = d^2/dtheta^2 L_T(o(T))` is the Hessian. Instead of re-running a whole optimizer at
each new `T` (as `single_shooting_continuation.py` did), we:

1. PREDICT: take one step of this ODE to guess `o(T+dT)` from `o(T)`.
2. CORRECT: a few Newton steps (`theta <- theta - H^-1 grad`) at the new `T` to snap back onto the
   true `o(T+dT)` (the predictor alone drifts - standard practice in ALL real continuation software,
   see the chat's own reference list: Allgower & Georg; Diehl, Bock & Schloder's real-time iteration).
3. `lambda_min(H) <= 0` (a bifurcation - `o(T)` cannot be tracked reliably past this point) is the
   natural stopping signal - no separate bisection/patience heuristic needed.

Needs a SMOOTH loss in `T` (`d/dT[grad_theta L_T]` must exist) - reuses the sigmoid window from
`single_shooting_continuation.py`: `w(t;T,eps) = sigmoid((T-t)/eps)`. `H(T)` is computed exactly via
`multishooting_adaptive`'s forward/double-backprop machinery (already validated correct for a single
segment - see the chat this followed).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import torch

from pybm.estimate.depr.gradient_matching import (
    _const_bounds,
    _qualified_var_names,
    estimate_gradient_matching,
    fit_gps,
)
from pybm.estimate.multishooting_adaptive import (
    _L1_SMOOTH_EPS,
    _block_loss,
    _seed_at,
    _state_weights,
)
from pybm.estimate.multishooting_torch import _homotopy_weight, _make_solver
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Var, _get_data_tensor


def _window_weight(t: torch.Tensor, T: float, eps: float) -> torch.Tensor:
    """`sigmoid((T-t)/eps)` - see module docstring."""
    return torch.sigmoid((T - t) / eps)


def _loss_and_grad(
    z: np.ndarray, n_consts: int, n_vars: int, state_vars, solver, t_window, data_window, weight, device, dtype,
    norm: Literal["l2", "l1"] = "l2", l1_eps: float = _L1_SMOOTH_EPS, s_t: "torch.Tensor | None" = None,
) -> "tuple[float, np.ndarray]":
    """One first-order backward through `_block_loss` (K=1: `has_prev=False`, no `prev_seed`). `z`
    is the corrector's own free variable - the actual `[theta,s_new]` fed to `_block_loss` is `z*s_t`
    elementwise (`s_t=None`, the default, behaves as an all-ones scale - see `precondition`);
    differentiating w.r.t. `z` (not the scaled quantity) folds that chain-rule factor into the
    gradient/Hessian automatically via autograd."""
    z_t = torch.as_tensor(z, dtype=dtype, device=device).requires_grad_(True)
    z_full = z_t if s_t is None else s_t * z_t
    loss = _block_loss(
        z_full, n_consts, n_vars, False, None, state_vars, solver,
        t_window, np.array([0, len(t_window) - 1]), data_window, weight, device, dtype, norm, l1_eps,
    )
    (grad,) = torch.autograd.grad(loss, z_t)
    return float(loss.item()), grad.detach().cpu().numpy()


def _loss_only(
    z: np.ndarray, n_consts: int, n_vars: int, state_vars, solver, t_window, data_window, weight, device, dtype,
    norm: Literal["l2", "l1"] = "l2", l1_eps: float = _L1_SMOOTH_EPS, s_t: "torch.Tensor | None" = None,
) -> float:
    """Forward-only evaluation of `_block_loss` (`torch.no_grad`, no autograd graph built at all) -
    for backtracking trial points, which only ever need the loss VALUE, not its gradient. Calling
    `_loss_and_grad` there (as an earlier version of this module did) wastes a full backward pass on
    every one of up to 10 backtracking tries per Newton step - a real, measured cost, not a
    hypothetical one (see the chat this followed)."""
    z_t = torch.as_tensor(z, dtype=dtype, device=device)
    z_full = z_t if s_t is None else s_t * z_t
    with torch.no_grad():
        loss = _block_loss(
            z_full, n_consts, n_vars, False, None, state_vars, solver,
            t_window, np.array([0, len(t_window) - 1]), data_window, weight, device, dtype, norm, l1_eps,
        )
    return float(loss.item())


def _hessian(
    z: np.ndarray, n_consts: int, n_vars: int, state_vars, solver, t_window, data_window, weight, device, dtype,
    norm: Literal["l2", "l1"] = "l2", l1_eps: float = _L1_SMOOTH_EPS, s_t: "torch.Tensor | None" = None,
) -> np.ndarray:
    """Exact Hessian via double-backprop w.r.t. `z` (the corrector's own, possibly-preconditioned
    free variable - see `_loss_and_grad`). Reimplements `_exact_block_hessian`'s own double-backprop
    directly around `_block_loss` (rather than calling it) so the `s_t` chain-rule factor can be
    folded in the SAME way as `_loss_and_grad` - mathematically identical to
    `_exact_block_hessian(z,...)` when `s_t=None` (verified by regression test, see the chat this
    followed)."""
    z_t = torch.as_tensor(z, dtype=dtype, device=device).requires_grad_(True)
    z_full = z_t if s_t is None else s_t * z_t
    loss = _block_loss(
        z_full, n_consts, n_vars, False, None, state_vars, solver,
        t_window, np.array([0, len(t_window) - 1]), data_window, weight, device, dtype, norm, l1_eps,
    )
    (grad,) = torch.autograd.grad(loss, z_t, create_graph=True)
    n = len(z)
    H = torch.zeros(n, n, dtype=dtype, device=device)
    for i in range(n):
        (row,) = torch.autograd.grad(grad[i], z_t, retain_graph=True)
        H[i] = row
    H = 0.5 * (H + H.T)
    return H.detach().cpu().numpy()


@dataclass
class GrowingHorizonResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray
    const_by_name: "dict[str, float]"
    x0: np.ndarray
    status: str  # "completed" | "stalled"
    history: "list[dict]"


def estimate_growing_horizon(
    model: InducedModel,
    t_eval,
    init_points: int = 6,
    growth_factor: float = 1.6,
    eps_frac_start: float = 0.5,
    eps_frac_end: float = 0.02,
    k_eps: float = 4.0,
    corrector: "Literal['newton', 'adam', 'sgd']" = "newton",
    corrector_steps: int = 10,
    corrector_lr: float = 1e-2,
    pd_margin_rel: float = 1e-5,
    pd_check: bool = True,
    fd_frac: float = 0.01,
    max_rounds: int = 30,
    outer_solver: "Literal['euler_newton', 'ode', 'ode_corrected']" = "euler_newton",
    corrector_every: int = 1,
    norm: "Literal['l2', 'l1']" = "l2",
    l1_eps: "float | None" = None,
    precondition: bool = False,
    precondition_floor: float = 1e-3,
    weight_by_uncertainty: bool = True,
    max_gp_points: int = 300,
    max_gp_iter: int = 200,
    solver_atol: float = 1e-6,
    solver_rtol: float = 1e-4,
    solver_max_steps: "int | None" = 1000,
    solver_dt_min: "float | None" = None,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    torch_threads: "int | None" = None,
    verbose: int = 0,
) -> GrowingHorizonResult:
    """
    See module docstring. Warm start (theta, x0) exactly as `single_shooting_continuation.py` did:
    `estimate_gradient_matching` on the short first window, `x0` from the GP mean at `t_eval[0]`.

    Parameters
    ----------
    corrector : "newton" | "adam" | "sgd", optional
        How the corrector snaps the predictor's guess back onto `o(T)`. "newton" (default): damped
        Newton with backtracking (see `correct`'s own docstring) - needs very few steps, since a
        good predictor already lands close. "adam"/"sgd": plain first-order `torch.optim` steps on
        the same windowed loss instead - no Hessian needed for the STEP itself (still computed
        afterward, for the stall check), but typically needs many more `corrector_steps` than Newton
        to reach a comparable optimum - see `corrector_lr`.
    corrector_steps : int, optional
        Fixed number of corrector steps (Newton, or Adam/SGD) run at each new `T` - no convergence
        check, kept fixed for simplicity. Newton needs very few (a good predictor lands close);
        Adam/SGD need far more (pass a much larger value when using them).
    corrector_lr : float, optional
        Learning rate for `corrector="adam"`/`"sgd"` - ignored for `"newton"`.
    pd_margin_rel : float, optional
        Growth stops (`status="stalled"`) the moment `lambda_min(H) < pd_margin_rel * lambda_max(H)`
        - the bifurcation signal from the module docstring. Ignored entirely when `pd_check=False`.
    pd_check : bool, optional
        Default `True`. When `False`, SKIPS the extra Hessian evaluation `correct` otherwise does
        purely to report `lambda_min`/`lambda_max` (the corrector's OWN Hessian, needed for the
        Newton step itself, is still computed regardless - this only removes the one-off,
        diagnostic-only evaluation after the corrector has already converged, and disables the
        `"ode"`/`"ode_corrected"` outer solvers' `events=` bifurcation detection in `solve_ivp`,
        which would otherwise call this same Hessian purely to check the same threshold).
        `lambda_min`/`lambda_max` become `nan` and every stall check based on them becomes a no-op
        (growth never stops early) - growth then always runs to the full horizon. Measured directly
        (see the chat this followed): this Hessian requires an ACTUAL ODE integration + double
        backprop, ~2s/call on Protein Transduction - a real saving when the diagnostic won't be
        used, unlike the analogous option on `im_growing_horizon.py` where the same Hessian is
        cheap (no ODE solve) and skipping it saves almost nothing. Turn this off when structural
        non-identifiability is expected/accepted (silently landing on an arbitrary point of a
        degenerate parameter combination, same as `gradient_matching`/Adam-based methods already
        do - see the chat).
    fd_frac : float, optional
        `d/dT[grad_theta L_T]` is estimated by a central difference in `T` with step
        `fd_frac * dT` (`dT` = the current growth step) - avoids a second nested double-backprop.
    outer_solver : "euler_newton" | "ode" | "ode_corrected", optional
        "euler_newton" (default): the round-based loop below - one fixed (geometrically growing)
        Euler step of the Davidenko equation per round, then `corrector` snaps it back onto `o(T)`.
        "ode": experiment (1) - NO corrector at all, ever. Integrates the Davidenko equation
        `do/dT = -H(T)^-1 d/dT[grad_theta L_T]` directly as a real ODE via `scipy.integrate.
        solve_ivp` (adaptive step, not a single fixed Euler step) from `T_cur` all the way to
        `t_eval[-1]`, with an `events=` callback that stops the integration the EXACT instant
        `lambda_min(H)` crosses the same PD margin used elsewhere.
        "ode_corrected": experiment (2) - a middle ground. Same adaptive `solve_ivp` integration,
        but chunked into the SAME geometrically-growing windows `euler_newton` uses; `corrector` is
        called only after every `corrector_every`-th chunk (see `corrector_every`), not after every
        adaptive step (that would be needless - Newton has nothing to fix if the chunk's own
        adaptive integration already tracked `o(T)` accurately) and not never (drift still
        eventually needs snapping back).

        Both ODE-based modes track an EXTRA scalar `ell(T) := L_T(o(T))` alongside `o(T)` in the
        same augmented state (a nearly-free addition - the envelope theorem, see the chat this
        followed for a reference: `d(ell)/dT = d(L_T)/dT` evaluated at FIXED theta, no `do/dT` cross
        term needed - and that quantity is already computed, then discarded, as `loss_plus`/
        `loss_minus` while estimating `d/dT[grad_theta L_T]` by finite differences).
    corrector_every : int, optional
        "ode_corrected"-only: run `corrector` after every this-many chunks (default 1 - after every
        chunk; use a larger value to correct less often).
    norm : "l2" | "l1", optional
        Per-element penalty in the windowed loss (see `multishooting_adaptive._penalty`). "l1" is a
        SMOOTHED absolute value (pseudo-Huber), not raw `abs()` - the Newton corrector needs a
        genuine, non-degenerate Hessian, which raw L1 does not have (its second derivative is a
        Dirac delta at zero residual).
    l1_eps : float, optional
        The pseudo-Huber smoothing width for `norm="l1"` - ignored for `norm="l2"`. If `None`
        (default), calibrated automatically from the data's own noise scale: the median absolute
        deviation between each state var's raw data and its GP mean (already fit below), pooled
        across vars. This MATTERS - a fixed `eps` that's far smaller than the data's actual residual
        scale at convergence makes pseudo-Huber's second derivative collapse to ~0 almost everywhere
        (it's `eps**2/(diff**2+eps**2)**1.5`, i.e. ~`eps**2/|diff|**3` once `|diff| >> eps`), which
        starves the Newton corrector's Hessian of curvature and can trigger the PD-margin stall
        check as a false positive (verified directly: on Lotka-Volterra, noise_std=0.1 vs a fixed
        `eps=1e-4` gave a 1000x-too-small smoothing width and a badly premature stop; on Protein
        Transduction, noise_std=0.001 happened to be close to that same fixed value, so it looked
        fine there - the point is this must track the noise scale, not be a global constant).
    precondition : bool, optional
        Diagonal (Jacobi-style) preconditioning of the CONSTANT part of `z=[theta,x0]` only (the
        seed `x0` keeps scale 1 - it already comes from the GP mean, on the data's own scale): the
        corrector's free variable becomes `phi=[theta/s, x0]` (`s_i = max(|theta0_i|,
        precondition_floor)`, estimated once from the initial GM warm start `theta0`). See
        `pybm.estimate.im_growing_horizon.estimate_im_growing_horizon`'s own `precondition`
        docstring for the full motivation (a direct comparison across this codebase's methods on
        Protein Transduction) and the caveat that full Newton (exact Hessian) is itself invariant to
        this reparameterization - what changes is the isotropic damping `reg*I` and the
        `pd_margin_rel` ratio check, both of which operate in `phi`-units instead of raw `theta`-
        units. Requires `corrector="newton"` (raises `ValueError` otherwise - "adam"/"sgd" don't
        route through the `s`-aware wrapper). Default `False` reproduces this module's original
        behavior exactly.
    precondition_floor : float, optional
        Floor for `s_i` above. Default `1e-3`.
    """
    if model.engine != "torch":
        raise ValueError(
            f"estimate_growing_horizon requires an InducedModel built with engine='torch', got engine={model.engine!r}."
        )
    if precondition and corrector != "newton":
        raise ValueError(f"precondition=True requires corrector='newton', got corrector={corrector!r}.")
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    T_full = len(t_eval)
    if init_points < 2:
        raise ValueError(f"init_points must be >= 2, got {init_points}.")

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    n_vars = len(state_vars)
    n_consts = len(model.consts)
    qualified = _qualified_var_names(model)

    state_var_names = {qualified[id(v)]: v for v in state_vars}
    gps = fit_gps(state_var_names, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)
    unc_weight = _state_weights(gps, qualified, state_vars, t_eval, device, dtype) if weight_by_uncertainty else None

    if norm == "l1" and l1_eps is None:
        # Calibrate the pseudo-Huber smoothing width to the data's own noise scale (median absolute
        # deviation of raw data from the GP mean, pooled across state vars) instead of using some
        # fixed constant - see the docstring above for why a mismatched fixed `eps` silently breaks
        # the Newton corrector's Hessian.
        gp_mean = torch.stack(
            [torch.as_tensor(gps[qualified[id(v)]].mean(t_eval), dtype=dtype, device=device) for v in state_vars],
            dim=0,
        )  # (n_vars, T), matches `data`'s own layout
        l1_eps = max(float((data - gp_mean).abs().median()), 1e-8)
        if verbose:
            print(f"[growing_horizon] norm='l1': auto-calibrated l1_eps={l1_eps:.4g} from data noise scale")
    elif l1_eps is None:
        l1_eps = _L1_SMOOTH_EPS

    solver = _make_solver(
        state_vars, algebraic_vars, frozen_values, solver_atol, solver_rtol,
        max_steps=solver_max_steps, dt_min=solver_dt_min,
    )

    idx0 = min(init_points - 1, T_full - 1)
    gm0 = estimate_gradient_matching(
        model, t_eval, collocation_times=t_eval[: idx0 + 1], gps=gps,
        weight_by_uncertainty=weight_by_uncertainty, device=device, dtype=dtype, verbose=verbose,
    )
    z_raw = np.concatenate([gm0.consts, _seed_at(gps, qualified, state_vars, t_eval[0], device, dtype).cpu().numpy()])

    # Diagonal preconditioner `s_full` - see `precondition`'s own docstring. Only the CONSTANT part
    # is scaled (`s_full[:n_consts]`, from `gm0.consts`); the seed part keeps scale 1 (it already
    # lives on the data's own scale, from the GP mean). `z` from here on is `phi = z_raw/s_full`,
    # the corrector's own free variable - converted back to real units only once, at the very end.
    if precondition:
        s_full = np.concatenate([np.maximum(np.abs(gm0.consts), precondition_floor), np.ones(n_vars)])
        if verbose:
            print(f"[growing_horizon] precondition=True: s={s_full[:n_consts]}")
    else:
        s_full = np.ones(n_consts + n_vars)
    s_full_t = torch.as_tensor(s_full, dtype=dtype, device=device)
    z = z_raw / s_full

    # User-supplied constant bounds (`Const.range`, same ones `gradient_matching`/
    # `integral_matching` already use via `_const_bounds`) - unbounded (+-inf) where not given, and
    # ALWAYS unbounded for the seed part of `z` (no analogous range exists for a state value here).
    # Converted once to `phi`-space (`s_full` is always positive, so division preserves ordering)
    # and enforced by clipping every trial point in `correct_newton` - see its own docstring for why
    # this matters once `pd_check=False` removes the PD-margin safety net: without a bound, a
    # genuinely unidentified direction (see the chat's own c1*c2 toy example) lets the corrector run
    # off to an arbitrarily extreme value along it instead of landing somewhere reasonable.
    lo_c, hi_c = _const_bounds(model)
    lo_z = np.concatenate([lo_c, np.full(n_vars, -np.inf)])
    hi_z = np.concatenate([hi_c, np.full(n_vars, np.inf)])
    lo_phi = lo_z / s_full
    hi_phi = hi_z / s_full

    # Rebind these three names LOCALLY with `norm`/`l1_eps`/`s_t` baked in via `functools.partial` -
    # every call site below (already written, unchanged) then automatically uses the requested
    # norm/eps/scale without having to thread them through each one individually. `globals()[...]`
    # (not the bare name) on the right-hand side is required here: assigning to `_loss_and_grad`
    # anywhere in this function makes Python treat it as local for the WHOLE function body, so the
    # bare name on the right would raise UnboundLocalError instead of resolving to the module-level
    # function.
    _loss_and_grad = partial(globals()["_loss_and_grad"], norm=norm, l1_eps=l1_eps, s_t=s_full_t)
    _loss_only = partial(globals()["_loss_only"], norm=norm, l1_eps=l1_eps, s_t=s_full_t)
    _hessian = partial(globals()["_hessian"], norm=norm, l1_eps=l1_eps, s_t=s_full_t)

    span = t_eval[-1] - t_eval[0]
    T_cur = t_eval[idx0]
    dT = max(T_cur - t_eval[0], 1e-9)

    def window(T: float, eps: float):
        """(t_window, data_window, weight) truncated to [0, T + k_eps*eps]."""
        T_integrate = min(T + k_eps * eps, t_eval[-1])
        idx_end = min(int(np.searchsorted(t_eval, T_integrate)) + 1, T_full - 1)
        t_window = t_eval[: idx_end + 1]
        data_window = data[:, : idx_end + 1]
        t_window_t = torch.as_tensor(t_window, dtype=dtype, device=device)
        w = _window_weight(t_window_t, T, eps).unsqueeze(-1)
        if unc_weight is not None:
            w = w * unc_weight[: idx_end + 1, :]
        return t_window, data_window, w

    def eps_at(T: float) -> float:
        frac = min((T - t_eval[0]) / span, 1.0)
        eps_frac = _homotopy_weight(eps_frac_start, eps_frac_end, frac, "exp")
        return max(eps_frac * (T - t_eval[0]), 1e-9)

    def correct_newton(z0: np.ndarray, t_window, data_window, weight) -> "tuple[np.ndarray, float, float]":
        """Up to `corrector_steps` DAMPED Newton steps (`z <- z - step*(H+reg*I)^-1 grad`), stopping
        EARLY the moment `||grad||` is already tiny (a well warm-started Newton step converges in a
        handful of iterations - burning through the full `corrector_steps` budget regardless wastes
        a full exact Hessian (an `n`-fold double-backprop) per leftover iteration for no benefit;
        checked directly, this and the `_loss_only` backtracking below cut wall-clock time roughly in
        half on LV without changing a single result). `reg` is a tiny fixed relative damping so a
        momentarily near-singular `H` can't blow up the step direction; `step` is found by simple
        backtracking (halve until the loss actually decreases, checked via `_loss_only` - a
        backtracking TRY never needs the gradient, only the loss value) - plain (undamped, step=1)
        Newton is only LOCALLY convergent and, tried directly, overshoots badly this far from the
        optimum (checked directly: it drove `theta1` from a reasonable warm start to -169).

        Returns `(z, grad_norm, step_norm)` - the LAST `||grad||` (how close to a stationary point)
        and `||delta||` (the un-taken-or-just-taken Newton step size) seen. Both are diagnostics of
        the corrector's OWN remaining numerical error, distinct from the noise-driven statistical
        uncertainty `sigma_worst` computed elsewhere: Newton converges quadratically, so
        `||z_final - o(T)|| = O(step_norm^2) << step_norm` - `step_norm` is therefore a cheap,
        already-computed (no extra ODE solves), conservative estimate of how far off `z` still is.

        Every point this loop ever evaluates is clipped to `[lo_phi,hi_phi]` (the CONSTANT part
        only - the seed part is unbounded - see the bounds' own definition above for why this
        matters, especially with `pd_check=False`)."""
        z_c = np.clip(z0, lo_phi, hi_phi)
        grad_norm = step_norm = float("nan")
        for _ in range(corrector_steps):
            H = _hessian(z_c, n_consts, n_vars, state_vars, solver, t_window, data_window, weight, device, dtype)
            reg = 1e-8 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            loss0, g = _loss_and_grad(z_c, n_consts, n_vars, state_vars, solver, t_window, data_window, weight, device, dtype)
            grad_norm = float(np.linalg.norm(g))
            delta = np.linalg.solve(H + reg * np.eye(len(H)), g)
            step_norm = float(np.linalg.norm(delta))
            if grad_norm < 1e-8 * max(np.linalg.norm(z_c), 1.0):
                break
            step = 1.0
            z_try = np.clip(z_c - step * delta, lo_phi, hi_phi)
            for _ in range(10):
                loss_try = _loss_only(z_try, n_consts, n_vars, state_vars, solver, t_window, data_window, weight, device, dtype)
                if loss_try < loss0:
                    break
                step *= 0.5
                z_try = np.clip(z_c - step * delta, lo_phi, hi_phi)
            z_c = z_try
        return z_c, grad_norm, step_norm

    def correct_first_order(z0: np.ndarray, t_window, data_window, weight) -> np.ndarray:
        """`corrector_steps` plain `torch.optim` ("adam" or "sgd") steps on the windowed loss - no
        Hessian used for the step itself (only afterward, for the stall check - see `correct`).
        Returns `(z, grad_norm)` - no analogue of Newton's `step_norm` exists here (there is no
        single well-defined "next step" the way `H^-1 grad` is for Newton)."""
        z_t = torch.as_tensor(z0, dtype=dtype, device=device).requires_grad_(True)
        opt_cls = torch.optim.Adam if corrector == "adam" else torch.optim.SGD
        optimizer = opt_cls([z_t], lr=corrector_lr)
        local_sub_indices = np.array([0, len(t_window) - 1])
        grad_norm = float("nan")
        for _ in range(corrector_steps):
            optimizer.zero_grad()
            loss = _block_loss(
                z_t, n_consts, n_vars, False, None, state_vars, solver,
                t_window, local_sub_indices, data_window, weight, device, dtype, norm, l1_eps,
            )
            loss.backward()
            grad_norm = float(z_t.grad.norm().item())
            optimizer.step()
        return z_t.detach().cpu().numpy(), grad_norm

    def correct(z0: np.ndarray, T: float, eps: float) -> "tuple[np.ndarray, float, float, float, float]":
        """Snaps `z0` (the predictor's guess, or the raw warm start before growth begins) back onto
        `o(T)` via `corrector` ("newton", "adam" or "sgd" - see `correct_newton`/`correct_first_order`
        above). Returns `(z, lambda_min, lambda_max, grad_norm, step_norm)` - the Hessian's extreme
        eigenvalues at the FINAL point (used both for this snapping and to first establish a genuine
        `o(T_cur)` before growth starts at all - the warm start from `estimate_gradient_matching`
        optimizes a DIFFERENT loss, so isn't yet a real `o(T_cur)`), plus the corrector's own final
        `grad_norm`/`step_norm` diagnostics (`step_norm` is `nan` for "adam"/"sgd" - see
        `correct_first_order`)."""
        t_window, data_window, weight = window(T, eps)
        if corrector == "newton":
            z_c, grad_norm, step_norm = correct_newton(z0, t_window, data_window, weight)
        else:
            z_c, grad_norm = correct_first_order(z0, t_window, data_window, weight)
            step_norm = float("nan")
        if not pd_check:
            # Skip the Hessian entirely - here it's ONLY used to report lambda_min/lambda_max for
            # the stall checks below, all of which compare `lambda_min < threshold` and so become
            # no-ops (nan is never < anything) - see `pd_check`'s own docstring. Unlike `im_gh`,
            # this Hessian requires a real ODE integration + double backprop (measured ~2s/call on
            # Protein Transduction) - skipping it is a genuine saving here, not a micro-optimization.
            return z_c, float("nan"), float("nan"), grad_norm, step_norm
        H_final = _hessian(z_c, n_consts, n_vars, state_vars, solver, t_window, data_window, weight, device, dtype)
        eig = np.linalg.eigvalsh(H_final)
        return z_c, float(eig[0]), float(eig[-1]), grad_norm, step_norm

    status = "completed"
    history: "list[dict]" = []

    # Establish a genuine o(T_cur) on the first (short, near-convex) window before growing at all.
    z, lambda_min, lambda_max, grad_norm, step_norm = correct(z, T_cur, eps_at(T_cur))
    if lambda_min < pd_margin_rel * max(lambda_max, 1e-12):
        status = "stalled"
        if verbose:
            print(f"[growing_horizon] initial window already non-convex (lambda_min={lambda_min:.4g}, lambda_max={lambda_max:.4g}) - stalled immediately")

    def make_outer_ode(z_dim: int):
        """Builds `(rhs, event)` for the augmented state `y = [z (z_dim,), ell (1,)]` - `ell(T)`
        tracks `L_T(o(T))` alongside `o(T)` itself via the envelope theorem (see module/function
        docstrings): `d(ell)/dT = d(L_T)/dT` at FIXED `theta` - no `do/dT` cross term needed, and
        the two loss VALUES it needs (`loss_plus`/`loss_minus`) are already computed, then
        discarded, while estimating `d/dT[grad_theta L_T]` below - this is nearly free."""

        def rhs(T: float, y: np.ndarray) -> np.ndarray:
            z_y = y[:z_dim]
            eps = eps_at(T)
            H = _hessian(z_y, n_consts, n_vars, state_vars, solver, *window(T, eps), device, dtype)
            dT_fd = fd_frac * eps
            loss_plus, g_plus = _loss_and_grad(z_y, n_consts, n_vars, state_vars, solver, *window(T + dT_fd, eps), device, dtype)
            loss_minus, g_minus = _loss_and_grad(z_y, n_consts, n_vars, state_vars, solver, *window(T - dT_fd, eps), device, dtype)
            dgdT = (g_plus - g_minus) / (2 * dT_fd)
            reg = 1e-8 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            do_dT = -np.linalg.solve(H + reg * np.eye(len(H)), dgdT)
            dell_dT = (loss_plus - loss_minus) / (2 * dT_fd)
            return np.concatenate([do_dT, [dell_dT]])

        def event(T: float, y: np.ndarray) -> float:
            eps = eps_at(T)
            H = _hessian(y[:z_dim], n_consts, n_vars, state_vars, solver, *window(T, eps), device, dtype)
            eig = np.linalg.eigvalsh(H)
            return eig[0] - pd_margin_rel * max(eig[-1], 1e-12)

        event.terminal = True  # type: ignore[attr-defined]
        event.direction = -1  # type: ignore[attr-defined]
        return rhs, event

    if status != "stalled" and outer_solver == "ode":
        # --- experiment (1): pure ODE tracking, no corrector at all, ever ---
        from scipy.integrate import solve_ivp

        outer_rhs, bifurcation_event = make_outer_ode(len(z))
        loss0, _ = _loss_and_grad(z, n_consts, n_vars, state_vars, solver, *window(T_cur, eps_at(T_cur)), device, dtype)
        y0 = np.concatenate([z, [loss0]])

        sol = solve_ivp(outer_rhs, (T_cur, t_eval[-1]), y0, method="RK45", events=(bifurcation_event if pd_check else None))
        z = sol.y[:-1, -1]
        T_cur = float(sol.t[-1])
        status = "completed" if T_cur >= t_eval[-1] - 1e-9 else "stalled"
        history = [{"T": float(t), "loss": float(ell)} for t, ell in zip(sol.t, sol.y[-1])]
        if verbose:
            reason = "full span" if status == "completed" else "bifurcation event"
            print(f"[growing_horizon ode] reached T={T_cur:.4g} ({reason}), {len(sol.t)} accepted steps")

    elif status != "stalled" and outer_solver == "ode_corrected":
        # --- experiment (2): adaptive integration WITHIN each chunk, Newton correction only every
        # `corrector_every`-th chunk boundary - see the chat this followed ---
        from scipy.integrate import solve_ivp

        outer_rhs, bifurcation_event = make_outer_ode(len(z))
        chunk_idx = 0
        for round_idx in range(max_rounds):
            if status == "stalled" or T_cur >= t_eval[-1]:
                break
            T_new = min(T_cur + dT, t_eval[-1])

            loss0, _ = _loss_and_grad(z, n_consts, n_vars, state_vars, solver, *window(T_cur, eps_at(T_cur)), device, dtype)
            y0 = np.concatenate([z, [loss0]])
            sol = solve_ivp(outer_rhs, (T_cur, T_new), y0, method="RK45", events=(bifurcation_event if pd_check else None))
            z = sol.y[:-1, -1]
            T_cur = float(sol.t[-1])
            history.extend({"T": float(t), "loss": float(ell)} for t, ell in zip(sol.t, sol.y[-1]))

            if T_cur < T_new - 1e-9:
                status = "stalled"  # bifurcation event fired mid-chunk
                if verbose:
                    print(f"[growing_horizon ode_corrected round {round_idx:3d}] bifurcation mid-chunk at T={T_cur:.4g}")
                break

            chunk_idx += 1
            if chunk_idx % corrector_every == 0:
                z, lambda_min, lambda_max, grad_norm, step_norm = correct(z, T_cur, eps_at(T_cur))
                if history:
                    history[-1]["grad_norm"] = grad_norm
                    history[-1]["step_norm"] = step_norm
                if lambda_min < pd_margin_rel * max(lambda_max, 1e-12):
                    status = "stalled"
                    if verbose:
                        print(f"[growing_horizon ode_corrected round {round_idx:3d}] T={T_cur:.4g} lambda_min={lambda_min:.4g} - stalled after correction")
                    break
                if verbose:
                    print(f"[growing_horizon ode_corrected round {round_idx:3d}] T={T_cur:.4g} lambda_min={lambda_min:.4g} (corrected)")
            dT *= growth_factor

    elif status != "stalled":
        for round_idx in range(max_rounds):
            if status == "stalled" or T_cur >= t_eval[-1]:
                break

            eps = eps_at(T_cur)

            # --- predictor: d/dT[grad_theta L_T] via a central difference in T, then one Euler step ---
            # Damped (same convention as `correct_newton`'s own solve) - with `pd_check=False`,
            # growth no longer stops the moment `H` gets close to singular, so an UNDAMPED solve
            # here can hit an exactly-singular matrix and raise (confirmed directly on Protein
            # Transduction: `LinAlgError: Singular matrix`).
            H = _hessian(z, n_consts, n_vars, state_vars, solver, *window(T_cur, eps), device, dtype)
            dT_fd = fd_frac * dT
            _, g_plus = _loss_and_grad(z, n_consts, n_vars, state_vars, solver, *window(T_cur + dT_fd, eps), device, dtype)
            _, g_minus = _loss_and_grad(z, n_consts, n_vars, state_vars, solver, *window(T_cur - dT_fd, eps), device, dtype)
            dgdT = (g_plus - g_minus) / (2 * dT_fd)
            reg = 1e-8 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            do_dT = -np.linalg.solve(H + reg * np.eye(len(H)), dgdT)

            T_new = min(T_cur + dT, t_eval[-1])
            z_pred = z + do_dT * (T_new - T_cur)

            # --- corrector: a few Newton steps at T_new to snap back onto o(T_new) ---
            z, lambda_min, lambda_max, grad_norm, step_norm = correct(z_pred, T_new, eps_at(T_new))
            T_cur = T_new

            if lambda_min < pd_margin_rel * max(lambda_max, 1e-12):
                status = "stalled"
                if verbose:
                    print(f"[growing_horizon round {round_idx:3d}] T={T_cur:.4g} lambda_min={lambda_min:.4g} - stalled (bifurcation)")
                break

            loss_new, _ = _loss_and_grad(z, n_consts, n_vars, state_vars, solver, *window(T_cur, eps_at(T_cur)), device, dtype)
            history.append({
                "T": float(T_cur), "lambda_min": lambda_min, "loss": loss_new,
                "grad_norm": grad_norm, "step_norm": step_norm,
            })
            if verbose:
                print(
                    f"[growing_horizon round {round_idx:3d}] T={T_cur:.4g} lambda_min={lambda_min:.4g} "
                    f"loss={loss_new:.4g} step_norm={step_norm:.4g}"
                )

            dT *= growth_factor

    z = s_full * z  # phi -> real [theta,x0] units (identity when precondition=False, s_full all-ones)
    const_by_name = {name: float(z[c.index_in_ctx]) for name, c in model.consts.items()}
    return GrowingHorizonResult(
        model=model,
        t_eval=t_eval,
        consts=z[:n_consts],
        const_by_name=const_by_name,
        x0=z[n_consts:],
        status=status,
        history=history,
    )
