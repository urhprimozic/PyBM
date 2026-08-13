import numpy as np
from pybm.estimate.int_scipy import minimal_var_ctx
from pybm.model import Choose, Context
from scipy.integrate import solve_ivp


def _settle_algebraic(algebraic_vars, t, var_ctx, const_ctx):
    """
    Evaluates all algebraic (non-differential) endogenous variables into `var_ctx` in place, so
    that differential equations (and other algebraic equations) reading them see up-to-date values.

    Algebraic variables can depend on each other (e.g. growthRate depends on tempGrowthLim, lightLim
    and nutrientLim, which are themselves algebraic) - `Var`/`Process` don't expose a dependency graph
    to sort by, so this settles via repeated fixed-point passes instead: a variable that reads another
    not-yet-updated algebraic variable simply sees last pass's value and converges to the correct one
    within (dependency-chain depth) passes. `len(algebraic_vars)` is a safe upper bound on that depth
    for any acyclic dependency graph over that many variables; one extra pass on top of that is needed
    to actually *observe* "nothing changed anymore" and stop, so the loop runs up to
    `len(algebraic_vars) + 1` times - guaranteed to converge (and be detected as converged) unless
    there's a cycle.
    """
    max_passes = len(algebraic_vars) + 1
    for _ in range(max_passes):
        changed = False
        new_ctx = Context(vars=var_ctx, consts=const_ctx)
        for var in algebraic_vars:
            value = var.algebraic(t, new_ctx)
            if var_ctx[var.index_in_ctx] != value:
                changed = True
            var_ctx[var.index_in_ctx] = value
        if not changed:
            return
    if algebraic_vars:
        raise ValueError(
            f"Algebraic variables did not converge within {max_passes} passes - check for a "
            "circular dependency between algebraic equations (e.g. two algebraic variables that "
            "each read the other)."
        )


def simulate(model, t_eval, context : Context, **kwargs):
    """
    Simulate the model using the given context and time evaluation points.

    Parameters:
    - model: The model to simulate.
    - t_eval: Array of time points at which to store the computed solution.
    - context: A Context object containing the values of constants and variables. If context.vars is None, the initial values of the variables will be set to their default initial values.
    - kwargs: Additional keyword arguments to pass to the ODE solver (solve_ivp).

    Returns:
    - sol: The solution object returned by the ODE solver. `sol.y` is indexed exactly like before -
      `sol.y[var.index_in_ctx]` gives the trajectory of any endogenous variable, state (differential)
      or algebraic - only the differential variables are actually integrated by the solver; algebraic
      variables are re-derived at each time point from the current state (see `_settle_algebraic`).
    """
    all_vars = model.get_endo_variables()
    state_vars = [var for var in all_vars if var.ode is not None]
    algebraic_vars = [var for var in all_vars if var.ode is None and var.algebraic is not None]
    missing = [var.name for var in all_vars if var.ode is None and var.algebraic is None]
    if missing:
        raise ValueError(f"Variable(s) {missing} do not have an ODE or an algebraic equation defined.")
    for var in all_vars:
        eq = var.ode if var.ode is not None else var.algebraic
        if isinstance(eq, Choose):
            raise ValueError(
                f"Variable {var.name} has a Choose object as its equation. First use model.induce() to get a list of models without Choose objects."
            )

    const_ctx = context["consts"]
    initial_var_ctx = context["vars"]
    full_size = len(all_vars)  # the shared index space spans every endogenous var, state + algebraic

    def f(t, y):
        var_ctx = np.zeros(full_size, dtype=float)
        for i, var in enumerate(state_vars):
            var_ctx[var.index_in_ctx] = y[i]
        _settle_algebraic(algebraic_vars, t, var_ctx, const_ctx)
        new_ctx = Context(vars=var_ctx, consts=const_ctx)
        derivatives = np.zeros(len(state_vars), dtype=float)
        for i, var in enumerate(state_vars):
            derivatives[i] = var.ode(t, new_ctx)
        return derivatives

    if initial_var_ctx is None:
        initial_values_full = minimal_var_ctx(*all_vars, set_initials=True, fixed_lenght=full_size)
    else:
        initial_values_full = initial_var_ctx
    y0 = np.array([initial_values_full[var.index_in_ctx] for var in state_vars], dtype=float)

    # solve GRAD(state_vars) = F(t, ctx)
    sol = solve_ivp(fun=f, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, **kwargs)

    # reconstruct the full (state + algebraic) trajectory at every output time point, in the
    # model's original index_in_ctx layout, so callers can keep indexing sol.y by any endo var
    # exactly as before this change.
    full_y = np.zeros((full_size, sol.y.shape[1]))
    for col in range(sol.y.shape[1]):
        var_ctx = np.zeros(full_size, dtype=float)
        for i, var in enumerate(state_vars):
            var_ctx[var.index_in_ctx] = sol.y[i, col]
        _settle_algebraic(algebraic_vars, sol.t[col], var_ctx, const_ctx)
        full_y[:, col] = var_ctx
    sol.y = full_y
    return sol