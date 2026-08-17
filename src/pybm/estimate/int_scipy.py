from types import SimpleNamespace
from scipy.optimize import NonlinearConstraint, minimize
from functools import lru_cache
from logging import warning
from typing import Any, Callable, Literal, cast
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pybm.model import Choose, Context, InducedModel, Var
from cma import fmin2


def set_consts(model : InducedModel, const_ctx : np.ndarray):
    """
    Set the values of the constants in the model based on the given context.
    """
    print("warning: TODO use .value for current value and initial_value just for initial value.")
    for const_name, const in model.consts.items():
        if const.index_in_ctx is None:
            raise ValueError(f"Constant {const.name} has no index in context.")
        const.initial_value = const_ctx[const.index_in_ctx]

def get_initial_const_ctx(model : InducedModel, default : float | None = 0.0):
        """
        Returns the initial context for the constants in the model.
        """
        const_ctx = np.zeros(len(model.consts), dtype=float)
        for const_name, const in model.consts.items():
            if const.index_in_ctx is None:
                raise ValueError(f"Constant {const.name} has no index in context.")
            if const.initial_value is None  or  const.index_in_ctx == np.nan:
                if default is not None:
                    const_ctx[const.index_in_ctx] = default
                else:
                    raise ValueError(f"Constant {const.name} has no initial value.")
            else:
                const_ctx[const.index_in_ctx] = const.initial_value
        return const_ctx

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

def var_ctx_from_time(*vars : Var, t, fixed_lenght=None):
    """
    Create the minimal variable context, which is valid for a given set of variables. 
    Returns an array of zeros `var_ctx`, such that `var.index_in_ctx < len(var_ctx)` for all variables in `vars`. 
    
    Values in the contex are retrieved from the data of the variables at time t. 
    """
    assert all(isinstance(var.index_in_ctx, int) for var in vars), "All variables must have a valid index in context. Make sure to add the variables to the model first."
    
    max_index = max(cast(int, var.index_in_ctx) for var in vars)

    if fixed_lenght is not None:
        if fixed_lenght <= max_index:
            raise ValueError(f"Fixed length {fixed_lenght} is too small for the given variables. Max index is {max_index}.")
        max_index = fixed_lenght - 1

    ans : list[Any]= [0. for _ in range(max_index + 1)]

    for var in vars:
        index = cast(int, var.index_in_ctx)
        if var.data is None:
            raise ValueError(f"Variable {var.name} does not have data defined.")
        ans[index] = var.data(t)
    return ans

def no_data(*vars : Var):
    no_data = False
    for var in vars:
        if var.data is  None:
            no_data = True
            break
    return no_data


def trajectory_offset(t, pred, *vars : Var):
    """
    The difference between the predicted trajectory and the observed data at time t.

    If no_data is True, the function returns the sum of squares of the predicted trajectory, which can be used to check for divergence.
    """
    # get true values at time t
    true_values = np.zeros(len(vars), dtype=float)


    if not no_data(*vars):
        for var in vars:
            assert var.data is not None, f"Variable {var.name} does not have data defined."
            true_values[var.index_in_ctx] = var.data(t)
    return np.sum((pred - true_values)**2)
    


def simulate(*vars: Var, t_eval, const_ctx, var_ctx_size=None, initial_var_ctx=None, max_offset=1e10, **kwargs):
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
    max_offset : float, optional
        The maximum allowed offset in simulation. If the offset exceeds this value, simulation will be stopped and a dummy random trajectory will be returned. 
    **kwargs : dict
        Additional keyword arguments passed to `scipy.integrate.solve_ivp`. 


    Returns    
    -------
    sol : OdeResult
        The solution of the ODE. `sol.y` is an array of shape (n_vars, n_time_points) containing the values of the variables at the given time points.
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
    #initial_values = minimal_var_ctx(*vars, set_initials=True, fixed_lenght=var_ctx_size)

    
    if initial_var_ctx is None:
        # Use the model's declared initial values by default.
        # If you want a data-driven seed for shooting, pass initial_var_ctx explicitly.
        initial_values = minimal_var_ctx(*vars, set_initials=True, fixed_lenght=var_ctx_size)
    else:
        initial_values = initial_var_ctx

    check_loss: Callable[[float, np.ndarray], float] | None
    if max_offset is not None:
        # check if the trajecory diverged 
        def check_loss_fn(t, y):
            offset = trajectory_offset(t, y, *vars)
            verbose = kwargs.get("verbose", 0)
            if offset > max_offset and verbose > 0:
                warning(f"Simulation stopped at time {t} due to exceeding maximal offset {max_offset}. Current offset: {offset}.")
            return max_offset - offset
        check_loss_fn.terminal = True
        check_loss_fn.direction = -1
        check_loss = check_loss_fn
    else:
        check_loss = None

    # solve GRAD(vars) = F(t, ctx)
    sol = solve_ivp(fun=f, t_span=(t_eval[0], t_eval[-1]), y0=initial_values, t_eval=t_eval, events=check_loss, **kwargs)
    return sol


def simulate_multishooting(
    *vars: Var,
    t_eval,
    params,
    n_consts: int,
    n_subintervals: int,
    max_offset=None,
    dummy_value=1e10,
    **kwargs,
):
    """
    Reconstruct a stitched multiple-shooting trajectory from a flat parameter vector.

    The parameter layout matches `estimate_scipy`:
        [consts..., s_0, s_1, ..., s_{K-1}]
    where each shooting state s_i is the initial state for subinterval i.

    Returns a lightweight object with `.t` and `.y`, similar to the SciPy result.
    """
    params = np.asarray(params, dtype=float)
    const_ctx = params[:n_consts]
    initials = params[n_consts:]
    sub_indices = np.linspace(0, len(t_eval) - 1, n_subintervals + 1, dtype=int)

    stitched: np.ndarray | None = None

    for i in range(n_subintervals):
        t_sub_eval = t_eval[sub_indices[i] : sub_indices[i + 1] + 1]
        initial_var_ctx = initials[i * len(vars) : (i + 1) * len(vars)]

        if max_offset is None:
            sol = simulate(
                *vars,
                t_eval=t_sub_eval,
                const_ctx=const_ctx,
                initial_var_ctx=initial_var_ctx,
                **kwargs,
            )
        else:
            sol = simulate(
                *vars,
                t_eval=t_sub_eval,
                const_ctx=const_ctx,
                initial_var_ctx=initial_var_ctx,
                max_offset=max_offset,
                **kwargs,
            )

        pred = sol.y
        expected_shape = (len(vars), sub_indices[i + 1] - sub_indices[i] + 1)
        if pred.shape != expected_shape:
            if kwargs.get("verbose", 0) == 2:
                warning(
                    f"Simulation returned unexpected shape {pred.shape}. Expected {expected_shape}. Returning large residuals."
                )
            pred = np.full(expected_shape, dummy_value)

        if stitched is None:
            stitched = pred
        else:
            stitched = np.hstack((stitched, pred[:, 1:]))

    if stitched is None:
        stitched = np.zeros((len(vars), len(t_eval)), dtype=float)

    return SimpleNamespace(t=np.asarray(t_eval, dtype=float), y=stitched, success=True, message="multiple-shooting reconstruction")

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
            value = var.data(t)
            if value is None:
                raise ValueError(f"Variable {var.name} returned no data at time {t}.")
            data_matrix[i, j] = float(value)

    return data_matrix

def estimate_scipy(model : InducedModel, t_eval, verbose=0, n_subintervals=1, dummy_value=1e10, method : Literal["least_squares", "constraints"]="constraints", max_iter=None, ignore_dim_warnings=True, gtol=1e-3, **kwargs):
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
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = print progress, 2 = print detailed information.
    n_subintervals : int, optional
        Number of subintervals to use for the integration. The time interval will be divided into `n_subintervals` subintervals, and the ODE will be solved separately on each subinterval. This can improve the accuracy of the solution, especially for stiff ODEs. 
        If n_subintervals=1, the ODE will be solved on the entire time interval.
    dummy_value : float, optional
        The value to return for the residuals if the simulation fails. This should be a large value to indicate that the current parameter guess is not valid. The default is 1e10.
    """
    # collect vars 
    vars = model.get_endo_variables()
    # get the data, that we want to fit the model to
    data = get_data_matrix(*vars, t_eval=t_eval)
    # initial context:
    initial_ctx = get_initial_const_ctx(model)
    n_consts = len(initial_ctx)
    
    # divide t_eval into subintervals
    sub_indices = np.linspace(0, len(t_eval) - 1, n_subintervals + 1, dtype=int)

    # TODO cache!! 
    def trajectory(params):
        # params are c1, c2, ...,cn, v1(t0), v2(t0), ..., vm(t0) , v1(t1), v2(t1), ..., vm(t1), ..., v1(tK), v2(tK), ..., vm(tK)
        # unpack 
        const_ctx = params[:n_consts]
        initials = params[n_consts:]

        # get predictions

        for i in range(n_subintervals):
            t_start = t_eval[sub_indices[i]]
            t_end = t_eval[sub_indices[i + 1]]
            t_sub_eval = t_eval[sub_indices[i]:sub_indices[i + 1] + 1]

            # get initial var ctx for this subinterval
            initial_var_ctx = initials[i * len(vars):(i + 1) * len(vars)]

            sol = simulate(*vars, t_eval=t_sub_eval, const_ctx=const_ctx, initial_var_ctx=initial_var_ctx, verbose=verbose, **kwargs)

            pred = sol.y # of shape (n_vars, n_time_points)

            if pred.shape != (len(vars), sub_indices[i + 1] - sub_indices[i] + 1):
                if verbose == 2 and not ignore_dim_warnings:
                    warning(f"Simulation returned unexpected shape {pred.shape}. Expected {(len(vars), sub_indices[i + 1] - sub_indices[i] + 1)}. Returning large residuals.")
                shape = sub_indices[i + 1] - sub_indices[i] + 1
                pred =  np.full((len(vars), shape), dummy_value)

            if i == 0:
                all_pred = pred
            else:
                all_pred = np.hstack((all_pred, pred[:, 1:]))  # skip the first point to avoid duplicates

        residuals = (all_pred - data ).ravel()
        residuals = np.linalg.norm(residuals)

        smoothness_penalty = 0
        for i in range(1, n_subintervals):
            # get the last point of the previous subinterval and the first point of the current subinterval
            last_point_prev = all_pred[:, sub_indices[i] - sub_indices[0]]
            first_point_curr = all_pred[:, sub_indices[i] - sub_indices[0] + 1]
            smoothness_penalty += np.linalg.norm((last_point_prev - first_point_curr)**2)
        return residuals, smoothness_penalty
    def residuals(params):
        res, _ = trajectory(params)
        return res
    def constraints(params):
        _, smoothness_penalty = trajectory(params)
        return smoothness_penalty

    # get params 
    initial_params = initial_ctx 
    for index in sub_indices[:-1]:
        initial_var_ctx = var_ctx_from_time(*vars, t=t_eval[index], fixed_lenght=None)
        initial_params = np.concatenate((initial_params, initial_var_ctx))

    if method == "least_squares":
        result = least_squares(
            fun=residuals,
            x0=np.asarray(initial_params, dtype=float),
            method="trf",
            jac="2-point",
            verbose=verbose,
            max_nfev=max_iter,
            gtol=gtol,
        )
    elif method == "constraints":
        result = minimize(
            fun=lambda params: residuals(params),
            x0=np.asarray(initial_params, dtype=float),
            constraints=NonlinearConstraint(constraints, 0, 0),
            method="trust-constr",
            options={"maxiter": max_iter, "verbose":verbose, "gtol" : gtol},
        )


    const_to_value = {const.name: const_ctx for const, const_ctx in zip(model.consts.values(), result.x)}


    return result

def estimate_cmaes(model : InducedModel, t_eval, return_old=False, verbose=0, n_gen=50, sigma=10):
    """
    Estimate the constants of the model based on the data using CMA-ES optimization. 

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
    initial_ctx = get_initial_const_ctx(model)

    def residuals(const_ctx):
        # get predictions
        sol = simulate(*vars, t_eval=t_eval, const_ctx=const_ctx, verbose=verbose)
        pred = sol.y # of shape (n_vars, n_time_points)
        return (pred - data ).ravel()

    def fitness(const_ctx):
        res = residuals(const_ctx)
        return np.sum(res**2)
    


def estimate(model : InducedModel, t_eval, return_old=False, verbose=0, method : Literal["least_squares","constraints", "cmaes"]="least_squares", n_subintervals=1, *args, **kwargs):
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
    method : str, optional
        The optimization method to use for estimating the constants. Options are "least_squares" or "cmaes". Default is "least_squares".
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = print progress, 2 = print detailed information.
    n_subintervals : int, optional
        Number of subintervals to use for the integration. The time interval will be divided into `n_subintervals` subintervals, and the ODE will be solved separately on each subinterval. This can improve the accuracy of the solution, especially for stiff ODEs.
    """
    if method == "least_squares":
        if verbose > 0:
            print("Estimating constants using least squares optimization...")
        return estimate_scipy(model, t_eval=t_eval, n_subintervals=n_subintervals, verbose=verbose, method="least_squares", *args, **kwargs)
    elif method == "constraints":
        if verbose > 0:
            print("Estimating constants using constraints optimization...")
        return estimate_scipy(model, t_eval=t_eval, n_subintervals=n_subintervals, verbose=verbose, method="constraints", *args, **kwargs)
    elif method == "cmaes":
        raise NotImplementedError("CMA-ES estimation is not implemented yet.")
    else:
        raise ValueError(f"Unknown estimation method: {method}")