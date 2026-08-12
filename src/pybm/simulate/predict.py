import numpy as np
from pybm.estimate.int_scipy import minimal_var_ctx
from pybm.model import Choose, Context   
from scipy.integrate import solve_ivp

def simulate(model, t_eval, context : Context, **kwargs):
    """
    Simulate the model using the given context and time evaluation points.

    Parameters:
    - model: The model to simulate.
    - t_eval: Array of time points at which to store the computed solution.
    - context: A Context object containing the values of constants and variables. If context.vars is None, the initial values of the variables will be set to their default initial values.
    - kwargs: Additional keyword arguments to pass to the ODE solver (solve_ivp).

    Returns:
    - sol: The solution object returned by the ODE solver, containing the time points and
    """
    vars = model.get_endo_variables()
    const_ctx = context["consts"]
    initial_var_ctx = context["vars"]
    var_ctx_size = len(vars)
    def f(t, var_ctx):
            derivatives = np.zeros_like(var_ctx, dtype=float)
    
            for var in vars:
                if var.ode is None:
                    raise ValueError(f"Variable {var.name} does not have an ODE defined.")
                if isinstance(var.ode, Choose):
                    raise ValueError(
                        f"Variable {var.name} has a Choose object as ODE. First use model.induce() to get a list of models without Choose objects."
                    )
    
                new_ctx = Context(vars=var_ctx, consts=const_ctx)
                derivatives[var.index_in_ctx] = var.ode(t, new_ctx)
    
            return derivatives
    
        # collect initial values 
        #initial_values = minimal_var_ctx(*vars, set_initials=True, fixed_lenght=var_ctx_size)

    if initial_var_ctx is None:
        initial_values = minimal_var_ctx(*vars, set_initials=True, fixed_lenght=var_ctx_size)
    else:
        initial_values = initial_var_ctx

    # solve GRAD(vars) = F(t, ctx)
    sol = solve_ivp(fun=f, t_span=(t_eval[0], t_eval[-1]), y0=initial_values, t_eval=t_eval, **kwargs)
    return sol