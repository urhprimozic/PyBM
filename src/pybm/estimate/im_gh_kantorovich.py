"""
Lockstep multi-origin integral matching with Newton-Kantorovich-gated correction
(`im_gh_kantorovich`) - the user's own design (see the chat this followed) for two problems left
open by `pybm.estimate.im_growing_horizon`:

1. How to pick WHERE to anchor integral-matching windows and how far a single window may safely
   grow, using the same Gronwall-type interpolation-error bound `im_growing_horizon.epsilon_tol`
   already logs (there: diagnostic only). Here it is used PROACTIVELY, before any optimization
   starts, to partition the whole horizon into origins `t_1 < t_2 < ... < t_n` such that panel `j`
   (`[t_j, t_{j+1}]`) never accumulates more than `epsilon_tol` of estimated interpolation error.
2. How often a predictor-corrector continuation needs to actually STOP and correct, instead of
   free-running the outer (Davidenko) ODE. Answered here via a SECOND, different Gronwall-type
   argument (not about interpolation error - about the continuation's OWN numerical drift) combined
   with the classical Newton-Kantorovich local convergence radius.

Loss - one shared `theta`, `n` origins, SAME width `T` grown in lockstep
--------------------------------------------------------------------------
    r_i(T;theta) = x_hat(t_i+T) - x_hat(t_i) - integral_{t_i}^{t_i+T} F(x_hat(tau),tau,theta) dtau
    L_T(theta)   = pooled penalty over {r_i(T;theta) : i=1..n}      (see `_penalty`, pooled mean)
    o(T)         = argmin_theta L_T(theta)

This differs from `im_growing_horizon`'s own `mode="multi_origin"` in one essential way: THERE,
each origin grows to its own natural stopping point INDEPENDENTLY, and origins are pooled only
ONCE, at the very end. HERE, every origin's window grows by the SAME width increment every round,
sharing ONE `theta` continuously throughout growth - a genuine joint continuation in the shared
parameter `T`, not "fit n times, then average".

Davidenko's equation, boundary term (exact, no finite differences)
--------------------------------------------------------------------
Exactly as derived for the single-origin case (see `im_growing_horizon`'s own module docstring),
because `T` only ever enters as the UPPER LIMIT of each origin's own integral:

    d/dT[grad_theta L_T](o(T)) = mean_i grad_theta l_{t_i+T}(o(T), t_i)

so the predictor step below never needs anything beyond an ordinary gradient evaluation on the
GROWN spans - no re-derivation needed for the multi-origin, lockstep case.

Origin selection (`_find_breakpoints`)
----------------------------------------
Computed ONCE, at a reference `theta` (a plain gradient-matching fit over the WHOLE horizon), using
exactly the per-node quantities `im_growing_horizon._error_diagnostics` already computes
(`||dF/dx||` via `vmap(jacrev(...))`, `sigma_GP` from the fitted GP's own posterior variance):
panel `j`'s own contribution `E_j = sum_{nodes in panel j} ||dF/dx|| * z * sigma_GP * quadrature
weight` is independent of where a window STARTS, so it is computed for every panel in the grid in
ONE batched pass, then breakpoints are found by a cheap cumulative-sum threshold walk (`E_j` summed
until `>= epsilon_tol`, next breakpoint recorded, repeat) - no repeated autodiff calls.

When to correct: a second, DIFFERENT Gronwall bound + Newton-Kantorovich
----------------------------------------------------------------------------
Let `theta(T)` be the PREDICTOR-only path (no correction) and `o(T)` the true continuation. Both
satisfy `theta' ~ g(T,theta) := -H(T,theta)^{-1} b(T,theta)` (`b` the boundary term above), `o(T)`
exactly, `theta(T)` up to the predictor's own per-step numerical error `eps_num`. Subtracting the
two ODEs and linearizing around `o(T)` gives, for the drift `Delta(T):=theta(T)-o(T)`:

    Delta'(T) ~ L_g . Delta(T) + eps_num(T),      L_g := sup||dg/dtheta||

Gronwall on this LINEAR ODE gives EXPONENTIAL (not linear) growth:

    ||Delta(T)|| <= (eps_num_bar/L_g) * (exp(L_g*(T-T0)) - 1)

A subsequent Newton correction is GUARANTEED to converge back to the true `o(T)` (not some other
stationary point) as long as `||Delta(T)|| < rho`, the classical Newton-Kantorovich radius for a
locally `m`-strongly-convex, `M`-smooth, `L`-Hessian-Lipschitz objective (Boyd & Vandenberghe,
*Convex Optimization*, SS9.5.3): `rho ~ m^2/(L*M)`. This module does not attempt to track
`Delta(T)` itself (no independent estimate of `eps_num`/`L_g` is cheaply available) - instead it
uses the SAME `H_old` already computed every round for the ordinary secant predictor step (see
`im_growing_horizon`'s own module docstring) as a free, per-round estimate of `(m,M)`
(`eigvalsh(H_old)[0]`/`[-1]`), and estimates `L` (the Hessian's own Lipschitz constant) from the
observed rate of change of that SAME Hessian across consecutive rounds:

    L_round = ||H_old(round_k) - H_old(round_k-1)||_F / ||theta(round_k) - theta(round_k-1)||
    L_running = max(L_running, L_round)          (running, never-decreasing estimate)

`drift_accum` (the sum of predictor-only step norms `||theta_pred - theta_prev||` since the last
correction) is then compared against `kantorovich_safety * rho` each round - a cheap, EMPIRICAL
proxy for `||Delta(T)||` (predictor steps ARE the only source of drift when no correction has run
in between). Once `drift_accum` would exceed this budget, correction is no longer optional -
`correct_spans` (damped Newton, reusing `im_growing_horizon`'s own machinery) is called, and the
accumulator resets to zero. `m<=0` (a locally non-convex/indefinite Hessian - Kantorovich's own
strong-convexity premise fails) is treated as `rho=0`, forcing correction on every such round -
the safe, conservative fallback exactly where the underlying theorem no longer applies.

Newton (not Adam) is used for the corrector BY DESIGN here - the whole point of this module is to
size the SAFE reuse interval for a Newton step via its own Kantorovich radius, which has no Adam
analogue. See the chat this followed for the (well-known) saddle-attraction concern with
UNDAMPED Newton, and why the damped, bounded, optionally-Gauss-Newton corrector already built for
`im_growing_horizon` (reused unmodified here) sidesteps it.

`outer_solver="euler"` vs `"dopri"` - two different predictors for the SAME Kantorovich-gated
continuation, plus one honest caveat
--------------------------------------------------------------------------------------------------
`"euler"` (default): the discrete secant/quasi-Newton predictor above, one step per chunk (`i0,i1`
GRID-INDEX spans, `_loss_spans` etc. below) - reuses `im_growing_horizon`'s own windowed-MEAN loss
machinery UNCHANGED (mean over every grid point between each origin and its current end, not just
the single endpoint `T`). `"dopri"`: replaces that single crude step with a genuinely CONTINUOUS,
adaptive integration of the SAME Davidenko ODE across each chunk, via `scipy.integrate.solve_ivp
(method="RK45")` (the Dormand-Prince 5(4) pair - scipy's `"RK45"` name for it) - see the chat this
followed. This requires `L_T` to actually be smooth in continuous `T` (a step function, constant
between grid crossings, has zero derivative almost everywhere - useless as an ODE right-hand
side), so `"dopri"` uses the module docstring's OWN literal formula instead - a single-point
residual per origin (`_residual_continuous`), evaluated at ANY real `T` via a fixed Gauss-Legendre
quadrature (`n_quad` nodes, rescaled to `[t_i, t_i+T]` - `numpy.polynomial.legendre.leggauss`) -
NOT the windowed-mean loss `"euler"` actually optimizes. **The two `outer_solver`s are therefore
not integrating the same loss surface** - a genuine, worth-reporting difference on top of the
integrator itself, not just an implementation detail. `"dopri"` still uses finite differences
(`fd_frac`, same convention as `growing_horizon.py`'s own `make_outer_ode`) for `d[grad_theta L_T]
/dT` rather than the exact closed form the module docstring derives - a deliberate simplification
for a first head-to-head comparison, not a claim that FD is necessary here (it manifestly is not,
per the boundary-term derivation above - revisit if `"dopri"` looks promising enough to invest in
the exact version). All `n` origins share ONE `T_max = min_i(t_eval[-1]-t_i)` under `"dopri"` (no
per-origin freezing/dropout mid-integration, unlike `"euler"` - a `solve_ivp` event to freeze
individual origins as they run out of room is future work, not implemented here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import norm as _normal_dist
from torch.func import jacfwd, jacrev, vmap

from pybm.estimate.depr.gradient_matching import (
    _FittedGP,
    _const_bounds,
    _qualified_var_names,
    estimate_gradient_matching,
    fit_gps,
)
from pybm.estimate.integral_matching import _build_windows, _guard_rhs, _simpson_nodes_weights
from pybm.estimate.multishooting_adaptive import _L1_SMOOTH_EPS, _penalty
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, _get_data_tensor, _make_rhs

Span = "tuple[int, int]"


@dataclass
class IMGHKantorovichResult(ParamEstimationResults):
    model: InducedModel
    t_eval: np.ndarray
    consts: np.ndarray
    const_by_name: "dict[str, float]"
    status: str  # "completed" | "stalled"
    history: "list[dict]"
    origins: "list[float]"  # t-values of the adaptively-chosen window origins
    final_spans: "list[Span]"
    n_newton_calls: int
    gps: "dict[str, _FittedGP]" = field(default_factory=dict)


def estimate_im_gh_kantorovich(
    model: InducedModel,
    t_eval,
    outer_solver: "str" = "euler",
    init_points: int = 6,
    growth_factor: float = 1.6,
    corrector_steps: int = 10,
    max_rounds: int = 300,
    max_rounds_between_correct: int = 20,
    kantorovich_safety: float = 0.5,
    norm: "str" = "l2",
    l1_eps: "float | None" = None,
    hessian_mode: "str" = "gauss_newton",
    precondition: bool = False,
    precondition_floor: float = 1e-3,
    epsilon_tol: float = 1e-3,
    gronwall_p: float = 0.05,
    breakpoint_min_points: int = 2,
    n_panels: int = 1,
    max_rhs: float = 1e4,
    weight_by_uncertainty: bool = True,
    max_gp_points: int = 500,
    max_gp_iter: int = 200,
    fd_frac: float = 1e-3,
    n_quad: int = 12,
    device: "torch.device | None" = None,
    dtype: torch.dtype = torch.float64,
    torch_threads: "int | None" = None,
    verbose: int = 0,
) -> IMGHKantorovichResult:
    """See module docstring for the method, and its own `outer_solver` section for `"euler"` vs
    `"dopri"` (`fd_frac`/`n_quad` only apply to `"dopri"`). `corrector_steps`, `max_rounds_between_correct`: hard
    safety caps (the second one on top of the Kantorovich-gated trigger, in case `rho` estimates
    stay large for a long stretch - never go longer than this many rounds without a fresh
    correction). `kantorovich_safety` in `(0,1]`: fraction of the theoretical radius `rho` allowed
    to accumulate before a correction is forced - `1.0` uses the full (asymptotic) Kantorovich
    bound, smaller values correct earlier/more conservatively. `breakpoint_min_points`: minimum
    grid points per adaptively-chosen origin segment (avoids a degenerate near-zero-width origin
    purely from `epsilon_tol` being very tight). Other parameters match
    `im_growing_horizon.estimate_im_growing_horizon`'s own (same names, same meaning)."""
    if model.engine != "torch":
        raise ValueError(f"estimate_im_gh_kantorovich requires engine='torch', got {model.engine!r}.")
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    n_grid = len(t_eval)
    if init_points < 2:
        raise ValueError(f"init_points must be >= 2, got {init_points}.")
    max_i1 = n_grid - 1

    state_vars, algebraic_vars, frozen_values = model.split_endo_vars()
    n_consts = len(model.consts)
    qualified = _qualified_var_names(model)
    var_names = [qualified[id(v)] for v in state_vars]

    state_var_names = {name: v for name, v in zip(var_names, state_vars)}
    gps = fit_gps(state_var_names, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)
    data = _get_data_tensor(state_vars, t_eval, device, dtype)

    if norm == "l1" and l1_eps is None:
        gp_mean = torch.stack(
            [torch.as_tensor(gps[name].mean(t_eval), dtype=dtype, device=device) for name in var_names], dim=0,
        )
        l1_eps = max(float((data - gp_mean).abs().median()), 1e-8)
        if verbose:
            print(f"[im_gh_kantorovich] norm='l1': auto-calibrated l1_eps={l1_eps:.4g}")
    elif l1_eps is None:
        l1_eps = _L1_SMOOTH_EPS

    windows = _build_windows(t_eval, stride=1)
    nodes, panel_weights = _simpson_nodes_weights(windows, n_panels=n_panels)
    n_panel_windows, n_nodes = nodes.shape
    nodes_flat = nodes.reshape(-1)
    x_nodes = torch.stack(
        [torch.as_tensor(gps[name].mean(nodes_flat), dtype=dtype, device=device) for name in var_names], dim=1,
    )
    t_nodes = torch.as_tensor(nodes_flat, dtype=dtype, device=device)
    panel_weights_t = torch.as_tensor(panel_weights, dtype=dtype, device=device)

    x_all = torch.stack(
        [torch.as_tensor(gps[name].mean(t_eval), dtype=dtype, device=device) for name in var_names], dim=1,
    )

    rhs = _make_rhs(state_vars, algebraic_vars, frozen_values)
    _gronwall_z = float(_normal_dist.ppf(1 - gronwall_p / 2))

    # ------------------------------------------------------------- adaptive origin selection ---

    def _panel_error_contrib(theta_t: torch.Tensor) -> "tuple[np.ndarray, float]":
        """`(panel_E, L_F_max)` - `panel_E[j]` = this panel's own contribution to the Gronwall
        interpolation-error integral (see module docstring), for EVERY panel `j=0..n_grid-2` at
        once (one batched Jacobian pass, reused by the cumulative-sum breakpoint walk below)."""
        const_ctx = theta_t.unsqueeze(0).expand(t_nodes.shape[0], -1)

        def f_single(x_i, t_i, c_i):
            return rhs(t_i.unsqueeze(0), x_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

        J = vmap(jacrev(f_single, argnums=0))(x_nodes, t_nodes, const_ctx)
        Fx_norm = torch.linalg.matrix_norm(J, ord=2)  # (n_panel_windows*n_nodes,)
        t_np = t_nodes.detach().cpu().numpy()
        sigma_np = np.zeros(len(t_np))
        for name in var_names:
            sigma_np = np.maximum(sigma_np, np.sqrt(np.maximum(gps[name].var(t_np), 0.0)))
        sigma_t = torch.as_tensor(sigma_np, dtype=dtype, device=device)
        contrib = (Fx_norm * _gronwall_z * sigma_t).reshape(n_panel_windows, n_nodes)
        panel_E = (contrib * panel_weights_t).sum(dim=1)
        return panel_E.detach().cpu().numpy(), float(Fx_norm.max())

    def _find_breakpoints(theta_ref: np.ndarray) -> "list[int]":
        theta_t = torch.as_tensor(theta_ref, dtype=dtype, device=device)
        panel_E, _ = _panel_error_contrib(theta_t)
        breakpoints = [0]
        i0 = 0
        while i0 < max_i1:
            target = min(i0 + breakpoint_min_points, max_i1)
            i1, acc = i0, 0.0
            while i1 < target:
                acc += panel_E[i1]
                i1 += 1
            while i1 < max_i1 and acc < epsilon_tol:
                acc += panel_E[i1]
                i1 += 1
            breakpoints.append(i1)
            i0 = i1
        return breakpoints

    # ------------------------------------------------------------------ core span machinery ---
    # (identical to `im_growing_horizon`'s own - see that module for the full reasoning)

    def _residuals_full(theta_t: torch.Tensor) -> torch.Tensor:
        const_ctx = theta_t.unsqueeze(0).expand(t_nodes.shape[0], -1)
        f_vals = _guard_rhs(rhs(t_nodes, x_nodes, const_ctx), max_rhs).reshape(n_panel_windows, n_nodes, len(state_vars))
        panel_I = (panel_weights_t.unsqueeze(-1) * f_vals).sum(dim=1)
        cum_I = torch.cumsum(panel_I, dim=0)
        zeros = torch.zeros(1, len(state_vars), dtype=dtype, device=device)
        return torch.cat([zeros, cum_I], dim=0)

    def _span_residual(cum_I_ext: torch.Tensor, i0: int, i1: int) -> torch.Tensor:
        return (x_all[i0 + 1 : i1 + 1] - x_all[i0]) - (cum_I_ext[i0 + 1 : i1 + 1] - cum_I_ext[i0])

    def _span_weight(i0: int, i1: int) -> "torch.Tensor | None":
        if not weight_by_uncertainty:
            return None
        wv = np.stack(
            [gps[name].window_var(np.full(i1 - i0, t_eval[i0]), t_eval[i0 + 1 : i1 + 1]) for name in var_names],
            axis=1,
        )
        return torch.as_tensor(1.0 / np.sqrt(np.maximum(wv, 1e-12)), dtype=dtype, device=device)

    def _residuals_flat_spans(theta_t: torch.Tensor, spans: "list[Span]", w: "list[torch.Tensor | None]") -> torch.Tensor:
        cum_I_ext = _residuals_full(theta_t)
        parts = []
        for (i0, i1), wi in zip(spans, w):
            r = _span_residual(cum_I_ext, i0, i1)
            if wi is not None:
                r = r * wi
            parts.append(r.reshape(-1))
        return torch.cat(parts)

    def _loss_spans(theta_t: torch.Tensor, spans: "list[Span]", w: "list[torch.Tensor | None]") -> torch.Tensor:
        return _penalty(_residuals_flat_spans(theta_t, spans, w), norm, l1_eps).mean()

    def _loss_and_grad_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[float, np.ndarray]":
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device).requires_grad_(True)
        loss = _loss_spans(s_t * phi_t, spans, w)
        (grad,) = torch.autograd.grad(loss, phi_t)
        return float(loss.item()), grad.detach().cpu().numpy()

    def _loss_only_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> float:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        with torch.no_grad():
            return float(_loss_spans(s_t * phi_t, spans, w).item())

    def _exact_hessian_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device).requires_grad_(True)
        loss = _loss_spans(s_t * phi_t, spans, w)
        (grad,) = torch.autograd.grad(loss, phi_t, create_graph=True)
        H = torch.zeros(n_consts, n_consts, dtype=dtype, device=device)
        for i in range(n_consts):
            (row,) = torch.autograd.grad(grad[i], phi_t, retain_graph=True)
            H[i] = row
        return (0.5 * (H + H.T)).detach().cpu().numpy()

    def _gauss_newton_hessian_spans(phi: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        J = jacfwd(lambda p: _residuals_flat_spans(s_t * p, spans, w))(phi_t)
        rho = _residuals_flat_spans(s_t * phi_t, spans, w)
        gpp = torch.full_like(rho, 2.0) if norm == "l2" else l1_eps**2 / (rho**2 + l1_eps**2) ** 1.5
        H = (J.T * gpp.unsqueeze(0)) @ J / rho.numel()
        return H.detach().cpu().numpy()

    def _hessian_spans(theta: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> np.ndarray:
        return (
            _exact_hessian_spans(theta, spans, w)
            if hessian_mode == "exact"
            else _gauss_newton_hessian_spans(theta, spans, w)
        )

    def correct_newton_spans(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, float]":
        theta_c = np.clip(theta0, lo_phi, hi_phi)
        grad_norm = step_norm = float("nan")
        for _ in range(corrector_steps):
            H = _hessian_spans(theta_c, spans, w)
            reg = 1e-4 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            loss0, g = _loss_and_grad_spans(theta_c, spans, w)
            grad_norm = float(np.linalg.norm(g))
            delta = np.linalg.solve(H + reg * np.eye(len(H)), g)
            step_norm = float(np.linalg.norm(delta))
            if grad_norm < 1e-8 * max(np.linalg.norm(theta_c), 1.0):
                break
            step = 1.0
            theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            for _ in range(10):
                loss_try = _loss_only_spans(theta_try, spans, w)
                if loss_try < loss0:
                    break
                step *= 0.5
                theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            theta_c = theta_try
        return theta_c, grad_norm, step_norm

    def correct_spans(theta0: np.ndarray, spans: "list[Span]", w: "list[torch.Tensor | None]") -> "tuple[np.ndarray, float, float, float, float]":
        theta_c, grad_norm, step_norm = correct_newton_spans(theta0, spans, w)
        H_final = _hessian_spans(theta_c, spans, w)
        eig = np.linalg.eigvalsh(H_final)
        return theta_c, float(eig[0]), float(eig[-1]), grad_norm, step_norm

    def _warm_start(i0: int, i1: int) -> np.ndarray:
        gm = estimate_gradient_matching(
            model, t_eval, collocation_times=t_eval[i0 : i1 + 1], gps=gps,
            weight_by_uncertainty=weight_by_uncertainty, device=device, dtype=dtype, verbose=verbose,
        )
        return np.asarray(gm.consts, dtype=float)

    # ------------------------------------------------------------------- continuous-T machinery ---
    # (`outer_solver="dopri"` only - see module docstring's own section on the two `outer_solver`s.
    # Genuinely continuous in `T` (fixed Gauss-Legendre quadrature per call, no grid dependency at
    # all) so `scipy.integrate.solve_ivp` can evaluate it at ANY intermediate `T` it likes.)

    _gl_x, _gl_w = np.polynomial.legendre.leggauss(n_quad)  # nodes/weights on [-1,1]

    def _quad_nodes_weights(t0: float, T: float) -> "tuple[np.ndarray, np.ndarray]":
        half = T / 2.0
        return t0 + half + half * _gl_x, half * _gl_w

    def _residual_continuous(theta_t: torch.Tensor, origin_t: float, T: float) -> torch.Tensor:
        """Single-point residual `r_i(T;theta)` for ONE origin (module docstring's own formula) -
        `x_hat`/the integral both evaluated at genuinely continuous `t`, never snapped to `t_eval`."""
        nodes, weights = _quad_nodes_weights(origin_t, T)
        x_q = torch.stack([torch.as_tensor(gps[name].mean(nodes), dtype=dtype, device=device) for name in var_names], dim=1)
        t_q = torch.as_tensor(nodes, dtype=dtype, device=device)
        const_ctx = theta_t.unsqueeze(0).expand(n_quad, -1)
        f_vals = _guard_rhs(rhs(t_q, x_q, const_ctx), max_rhs)
        w_t = torch.as_tensor(weights, dtype=dtype, device=device)
        integral = (w_t.unsqueeze(-1) * f_vals).sum(dim=0)
        endpoints = np.array([origin_t, origin_t + T])
        x_end = torch.stack([torch.as_tensor(gps[name].mean(endpoints), dtype=dtype, device=device) for name in var_names], dim=1)
        return (x_end[1] - x_end[0]) - integral

    def _weight_continuous(origin_t: float, T: float) -> "torch.Tensor | None":
        if not weight_by_uncertainty:
            return None
        wv = np.array([gps[name].window_var(np.array([origin_t]), np.array([origin_t + T]))[0] for name in var_names])
        return torch.as_tensor(1.0 / np.sqrt(np.maximum(wv, 1e-12)), dtype=dtype, device=device)

    def _residuals_flat_continuous(theta_t: torch.Tensor, origins_t: "list[float]", T: float) -> torch.Tensor:
        parts = []
        for o in origins_t:
            r = _residual_continuous(theta_t, o, T)
            wi = _weight_continuous(o, T)
            if wi is not None:
                r = r * wi
            parts.append(r)
        return torch.cat(parts)

    def _loss_continuous(theta_t: torch.Tensor, origins_t: "list[float]", T: float) -> torch.Tensor:
        return _penalty(_residuals_flat_continuous(theta_t, origins_t, T), norm, l1_eps).mean()

    def _loss_and_grad_continuous(phi: np.ndarray, origins_t: "list[float]", T: float) -> "tuple[float, np.ndarray]":
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device).requires_grad_(True)
        loss = _loss_continuous(s_t * phi_t, origins_t, T)
        (grad,) = torch.autograd.grad(loss, phi_t)
        return float(loss.item()), grad.detach().cpu().numpy()

    def _loss_only_continuous(phi: np.ndarray, origins_t: "list[float]", T: float) -> float:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        with torch.no_grad():
            return float(_loss_continuous(s_t * phi_t, origins_t, T).item())

    def _hessian_continuous(phi: np.ndarray, origins_t: "list[float]", T: float) -> np.ndarray:
        phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
        if hessian_mode == "exact":
            phi_g = phi_t.clone().requires_grad_(True)
            loss = _loss_continuous(s_t * phi_g, origins_t, T)
            (grad,) = torch.autograd.grad(loss, phi_g, create_graph=True)
            H = torch.zeros(n_consts, n_consts, dtype=dtype, device=device)
            for i in range(n_consts):
                (row,) = torch.autograd.grad(grad[i], phi_g, retain_graph=True)
                H[i] = row
            return (0.5 * (H + H.T)).detach().cpu().numpy()
        J = jacfwd(lambda p: _residuals_flat_continuous(s_t * p, origins_t, T))(phi_t)
        rho = _residuals_flat_continuous(s_t * phi_t, origins_t, T)
        gpp = torch.full_like(rho, 2.0) if norm == "l2" else l1_eps**2 / (rho**2 + l1_eps**2) ** 1.5
        H = (J.T * gpp.unsqueeze(0)) @ J / rho.numel()
        return H.detach().cpu().numpy()

    def correct_continuous(theta0: np.ndarray, origins_t: "list[float]", T: float) -> "tuple[np.ndarray, float, float, float, float]":
        theta_c = np.clip(theta0, lo_phi, hi_phi)
        grad_norm = step_norm = float("nan")
        for _ in range(corrector_steps):
            H = _hessian_continuous(theta_c, origins_t, T)
            reg = 1e-4 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            loss0, g = _loss_and_grad_continuous(theta_c, origins_t, T)
            grad_norm = float(np.linalg.norm(g))
            delta = np.linalg.solve(H + reg * np.eye(len(H)), g)
            step_norm = float(np.linalg.norm(delta))
            if grad_norm < 1e-8 * max(np.linalg.norm(theta_c), 1.0):
                break
            step = 1.0
            theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            for _ in range(10):
                if _loss_only_continuous(theta_try, origins_t, T) < loss0:
                    break
                step *= 0.5
                theta_try = np.clip(theta_c - step * delta, lo_phi, hi_phi)
            theta_c = theta_try
        H_final = _hessian_continuous(theta_c, origins_t, T)
        eig = np.linalg.eigvalsh(H_final)
        return theta_c, float(eig[0]), float(eig[-1]), grad_norm, step_norm

    def _make_outer_rhs(origins_t: "list[float]"):
        def rhs_ode(T: float, y: np.ndarray) -> np.ndarray:
            phi = y[:n_consts]
            H = _hessian_continuous(phi, origins_t, T)
            dT_fd = fd_frac * max(T, 1e-6)
            loss_plus, g_plus = _loss_and_grad_continuous(phi, origins_t, T + dT_fd)
            loss_minus, g_minus = _loss_and_grad_continuous(phi, origins_t, max(T - dT_fd, 1e-9))
            dgdT = (g_plus - g_minus) / (2 * dT_fd)
            reg = 1e-4 * max(np.linalg.eigvalsh(H)[-1], 1.0)
            do_dT = -np.linalg.solve(H + reg * np.eye(len(H)), dgdT)
            dell_dT = (loss_plus - loss_minus) / (2 * dT_fd)
            return np.concatenate([do_dT, [dell_dT]])
        return rhs_ode

    # ------------------------------------------------------------------------------- setup ---

    if precondition:
        theta0_ref = _warm_start(0, min(init_points - 1, max_i1))
        s = np.maximum(np.abs(theta0_ref), precondition_floor)
    else:
        s = np.ones(n_consts)
    s_t = torch.as_tensor(s, dtype=dtype, device=device)

    lo, hi = _const_bounds(model)
    lo_phi = lo / s
    hi_phi = hi / s

    # Reference theta for origin selection: a single GM fit over the WHOLE horizon (cheap relative
    # to everything downstream, and only needs to be roughly right - `panel_E` only needs the right
    # ORDER of magnitude of `||dF/dx||`, not a converged fit).
    theta_ref = _warm_start(0, max_i1)
    breakpoints = _find_breakpoints(theta_ref)
    origins = [b for b in breakpoints[:-1] if max_i1 - b >= 1]
    n_origins = len(origins)
    if verbose:
        print(f"[im_gh_kantorovich] {n_origins} adaptive origin(s) at t={[round(t_eval[o],3) for o in origins]}")

    if outer_solver == "euler":
        idx0 = init_points - 1
        i1_init = [min(o + idx0, max_i1) for o in origins]
        spans: "list[Span]" = [(origins[j], i1_init[j]) for j in range(n_origins)]
        w = [_span_weight(*sp) for sp in spans]

        thetas0 = [_warm_start(*sp) / s for sp in spans]
        theta0_pooled = np.mean(np.stack(thetas0), axis=0)
        theta, lmin0, lmax0, gn0, sn0 = correct_spans(theta0_pooled, spans, w)
        n_newton_calls = 1
        history: "list[dict]" = [{
            "round": 0, "corrected": True, "n_active": n_origins, "T": float(np.mean([
                t_eval[spans[j][1]] - t_eval[origins[j]] for j in range(n_origins)
            ])),
            "m": lmin0, "M": lmax0, "L_running": float("nan"), "rho": float("nan"), "drift_accum": 0.0,
            "loss": _loss_only_spans(theta, spans, w), "grad_norm": gn0, "step_norm": sn0,
        }]

        # -------------------------------------------------------------------- lockstep growth ---

        T_shared = max(t_eval[spans[j][1]] - t_eval[origins[j]] for j in range(n_origins))
        dT = max(T_shared, 1e-9)
        L_running = 0.0
        H_prev = None
        theta_prev = None
        drift_accum = 0.0
        rounds_since_correct = 0
        status = "completed"

        for round_idx in range(1, max_rounds + 1):
            active = [j for j in range(n_origins) if spans[j][1] < max_i1]
            if not active:
                break

            T_new = T_shared + dT
            spans_new: "list[Span]" = []
            for j in range(n_origins):
                if spans[j][1] >= max_i1:
                    spans_new.append(spans[j])  # this origin already frozen at the end of the horizon
                    continue
                i1_new = min(int(np.searchsorted(t_eval, t_eval[origins[j]] + T_new, side="left")), max_i1)
                if i1_new <= spans[j][1]:
                    i1_new = min(spans[j][1] + 1, max_i1)
                spans_new.append((origins[j], i1_new))
            w_new = [_span_weight(*sp) for sp in spans_new]

            H_old = _hessian_spans(theta, spans, w)
            eig_old = np.linalg.eigvalsh(H_old)
            m, M = float(eig_old[0]), float(eig_old[-1])
            reg = 1e-4 * max(M, 1.0)
            _, g_new = _loss_and_grad_spans(theta, spans_new, w_new)
            theta_pred = theta - np.linalg.solve(H_old + reg * np.eye(len(H_old)), g_new)

            if theta_prev is not None:
                dtheta = np.linalg.norm(theta - theta_prev)
                if dtheta > 1e-12:
                    L_running = max(L_running, float(np.linalg.norm(H_old - H_prev)) / dtheta)
            rho = (m**2) / (L_running * M) if (m > 0 and L_running > 0 and M > 0) else 0.0

            step_norm_pred = float(np.linalg.norm(theta_pred - theta))
            drift_accum += step_norm_pred
            rounds_since_correct += 1
            all_frozen_next = all(spans_new[j][1] >= max_i1 for j in range(n_origins))
            do_correct = (
                m <= 0
                or drift_accum >= kantorovich_safety * rho
                or rounds_since_correct >= max_rounds_between_correct
                or all_frozen_next
            )

            if do_correct:
                theta_new, lmin, lmax, gn, sn = correct_spans(theta_pred, spans_new, w_new)
                n_newton_calls += 1
                drift_accum = 0.0
                rounds_since_correct = 0
            else:
                theta_new, lmin, lmax, gn, sn = theta_pred, m, M, float("nan"), float("nan")

            H_prev, theta_prev = H_old, theta
            theta = theta_new
            spans, w = spans_new, w_new
            T_shared = T_new
            dT *= growth_factor

            loss_val = _loss_only_spans(theta, spans, w)
            history.append({
                "round": round_idx, "corrected": bool(do_correct), "n_active": len(active),
                "T": float(np.mean([t_eval[spans[j][1]] - t_eval[origins[j]] for j in range(n_origins)])),
                "m": m, "M": M, "L_running": L_running, "rho": rho, "drift_accum": drift_accum,
                "loss": loss_val, "grad_norm": gn, "step_norm": sn,
            })
            if verbose:
                print(
                    f"[im_gh_kantorovich] round {round_idx:3d} n_active={len(active)}/{n_origins} "
                    f"T~{history[-1]['T']:.4g} m={m:.4g} M={M:.4g} L~{L_running:.4g} rho={rho:.4g} "
                    f"drift={drift_accum:.4g} corrected={do_correct} loss={loss_val:.4g}"
                )

    elif outer_solver == "dopri":
        origins_t = [float(t_eval[o]) for o in origins]
        T_max = min(t_eval[-1] - o for o in origins_t)
        idx0 = init_points - 1
        T0 = min(max(float(np.mean(
            [t_eval[min(o_idx + idx0, max_i1)] - t_eval[o_idx] for o_idx in origins]
        )), 1e-6), T_max)

        thetas0 = [_warm_start(o_idx, min(o_idx + idx0, max_i1)) / s for o_idx in origins]
        theta0_pooled = np.mean(np.stack(thetas0), axis=0)
        theta, lmin0, lmax0, gn0, sn0 = correct_continuous(theta0_pooled, origins_t, T0)
        n_newton_calls = 1
        history: "list[dict]" = [{
            "chunk": 0, "corrected": True, "T": T0, "m": lmin0, "M": lmax0,
            "L_running": float("nan"), "rho": float("nan"), "drift_accum": 0.0,
            "loss": _loss_only_continuous(theta, origins_t, T0), "grad_norm": gn0, "step_norm": sn0,
        }]

        outer_rhs = _make_outer_rhs(origins_t)
        T_cur = T0
        dT_chunk = T0
        L_running = 0.0
        H_prev = None
        theta_prev = None
        drift_accum = 0.0
        rounds_since_correct = 0
        status = "completed"

        for chunk_idx in range(1, max_rounds + 1):
            if T_cur >= T_max - 1e-9:
                break
            T_new = min(T_cur + dT_chunk, T_max)
            ell0 = _loss_only_continuous(theta, origins_t, T_cur)
            y0 = np.concatenate([theta, [ell0]])
            sol = solve_ivp(outer_rhs, (T_cur, T_new), y0, method="RK45")
            theta_pred = sol.y[:-1, -1]

            H_old = _hessian_continuous(theta_pred, origins_t, T_new)
            eig_old = np.linalg.eigvalsh(H_old)
            m, M = float(eig_old[0]), float(eig_old[-1])
            if theta_prev is not None:
                dtheta = np.linalg.norm(theta_pred - theta_prev)
                if dtheta > 1e-12:
                    L_running = max(L_running, float(np.linalg.norm(H_old - H_prev)) / dtheta)
            rho = (m**2) / (L_running * M) if (m > 0 and L_running > 0 and M > 0) else 0.0

            step_norm_pred = float(np.linalg.norm(theta_pred - theta))
            drift_accum += step_norm_pred
            rounds_since_correct += 1
            do_correct = (
                m <= 0
                or drift_accum >= kantorovich_safety * rho
                or rounds_since_correct >= max_rounds_between_correct
                or T_new >= T_max - 1e-9
            )

            if do_correct:
                theta_new, lmin, lmax, gn, sn = correct_continuous(theta_pred, origins_t, T_new)
                n_newton_calls += 1
                drift_accum = 0.0
                rounds_since_correct = 0
            else:
                theta_new, lmin, lmax, gn, sn = theta_pred, m, M, float("nan"), float("nan")

            H_prev, theta_prev = H_old, theta
            theta = theta_new
            T_cur = T_new
            dT_chunk *= growth_factor

            loss_val = _loss_only_continuous(theta, origins_t, T_cur)
            history.append({
                "chunk": chunk_idx, "corrected": bool(do_correct), "T": T_cur, "n_solver_steps": len(sol.t),
                "m": m, "M": M, "L_running": L_running, "rho": rho, "drift_accum": drift_accum,
                "loss": loss_val, "grad_norm": gn, "step_norm": sn,
            })
            if verbose:
                print(
                    f"[im_gh_kantorovich dopri] chunk {chunk_idx:3d} T={T_cur:.4g}/{T_max:.4g} "
                    f"m={m:.4g} M={M:.4g} L~{L_running:.4g} rho={rho:.4g} drift={drift_accum:.4g} "
                    f"corrected={do_correct} loss={loss_val:.4g} (RK45 accepted {len(sol.t)} steps)"
                )

        spans = [(o, int(np.searchsorted(t_eval, t_eval[o] + T_cur))) for o in origins]
    else:
        raise ValueError(f"Unknown outer_solver {outer_solver!r}, use 'euler' or 'dopri'.")

    theta = s * theta
    const_by_name = {name: float(theta[c.index_in_ctx]) for name, c in model.consts.items()}
    return IMGHKantorovichResult(
        model=model,
        t_eval=t_eval,
        consts=theta,
        const_by_name=const_by_name,
        status=status,
        history=history,
        origins=[float(t_eval[o]) for o in origins],
        final_spans=spans,
        n_newton_calls=n_newton_calls,
        gps=gps,
    )
