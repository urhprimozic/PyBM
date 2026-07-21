from logging import warning
from typing import Any, cast
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pybm.model import Choose, Context, Model, Var

def minimal_var_ctx(*vars : Var, set_initials=True, fixed_lenght=None):
    """
    Create the minimal variable context, which is valid for a given set of variables. 
    Returns an array of zeros `var_ctx`, such that `var.index_in_ctx < len(var_ctx)` for all variables in `vars`. 
    
    If set_initials is True, the values of the context are set to the initial values of the variables.
    """
    assert all(isinstance(var.index_in_ctx, int) for var in vars), "All variables must have a valid index in context. Make sure to add the variables to the model first."
    
    max_index = max(cast(int, var.index_in_ctx) for var in vars)

    if fixed_lenght is not None:
        if fixed_lenght <= max_index:
            raise ValueError(f"Fixed length {fixed_lenght} is too small for the given variables. Max index is {max_index}.")
        max_index = fixed_lenght - 1

    ans : list[Any]= [0. for _ in range(max_index + 1)]
    if set_initials:
        for var in vars:
            index = cast(int, var.index_in_ctx)
            if var.initial is None:
                raise ValueError(f"Variable {var.name} does not have an initial value defined.")
            ans[index] = var.initial
    return ans

def simulate(*vars: Var, t_eval, const_ctx, var_ctx_size=None, **kwargs):
    """
    Solve ODE for given endogenous variables.

    Parameters
    ----------
    *vars : Var
        List of endogenous variables. Each variable `var : Var` must have: 
            - differential equation `var.ode(t, ctx)`  
            - inital value `var.initial : float|Any`  
            - Valid index in context `var.index_in_ctx : int`, obtained with adding the variable to the model
    t_eval : array-like, optional
        Time points to evaluate and compute the solution. 
    const_ctx : array-like
        The context containing the current values of all constants. Constants will remain unchanged. 
    var_ctx_size : int, optional
        The size of the context containing the endogenous variables. If None, it will be inferred from the variables. 
    **kwargs : dict
        Additional keyword arguments passed to `scipy.integrate.solve_ivp`. 


    Returns    
    -------
    TODO
    """

    # build a function F(t, var_ctx) that computes the derivatives of the variables at time t given the context ctx
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
    initial_values = minimal_var_ctx(*vars, set_initials=True, fixed_lenght=var_ctx_size)

    # solve GRAD(vars) = F(t, ctx)
    sol = solve_ivp(fun=f, t_span=(t_eval[0], t_eval[-1]), y0=initial_values, t_eval=t_eval)
    return sol

def get_data_matrix(*vars: Var, t_eval):
    """
    Returns a matrix of shape (n_vars, n_time_points) containing the data of the variables at the given time points.

    Parameters
    ----------
    *vars : Var
        List of endogenous variables. Each variable `var : Var` must have:
            - data `var.data : TimeSeries`, the observed data for the variable
    t_eval : array-like
        Time points to evaluate and compute the solution.

    Returns
    -------
    data_matrix : ndarray
        Shape (n_vars, n_time_points). data_matrix[i, j] is the data of variable i at time t_eval[j]
    """
    data_matrix = np.zeros((len(vars), len(t_eval)), dtype=float)

    for i, var in enumerate(vars):
        if var.data is None:
            raise ValueError(f"Variable {var.name} does not have data defined.")
        for j, t in enumerate(t_eval):
            data_matrix[i, j] = var.data(t)

    return data_matrix




def estimate(model : Model, t_eval, return_scipy=True):
    """
    Estimate the constants of the model based on the data. 

    Parameters
    ----------
    *vars : Var
        List of endogenous variables. Each variable `var : Var` must have: 
            - differential equation `var.ode(t, ctx)`  
            - inital value `var.initial : float|Any`  
            - Valid index in context `var.index_in_ctx : int`, obtained with adding the variable to the model
            - data `var.data : array-like`, the observed data for the variable
    t_eval : array-like, optional
        Time points to evaluate and compute the solution.
    """
    # collect vars 
    vars = model.get_endo_variables()
    # get the data, that we want to fit the model to
    data = get_data_matrix(*vars, t_eval=t_eval)
    # initial context:
    initial_ctx = model.get_initial_const_ctx()
    



    def residuals(const_ctx):
        # get predictions
        sol = simulate(*vars, t_eval=t_eval, const_ctx=const_ctx)
        pred = sol.y # of shape (n_vars, n_time_points)
        return (pred - data ).ravel()
    

    result = least_squares(
            fun=residuals,
            x0=np.asarray(initial_ctx, dtype=float),
            method="trf",
            jac="2-point"
        )

    if return_scipy:
        return result
    else:
        return result.x
