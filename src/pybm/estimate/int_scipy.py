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


class Estimator:
    def __init__(self, f):
        pass

    def fit(self, *params):
        pass


class SingleShootingScipy(Estimator):
    def __init__(
        self,
        f,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    ):
        """
        Parameters
        ----------
        f : callable
            Function f(t, x, *params) returning dx/dt.
        method : str
            solve_ivp integration method.
        """
        self.f = f
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def simulate(self, t, x0, params):
        """
        Simulate the ODE.

        Parameters
        ----------
        t : (N,) array
            Time points.
        x0 : (n,) array
            Initial condition.
        params : iterable
            Model parameters.

        Returns
        -------
        ndarray
            Shape (N, n_states)
        """

        x0 = np.asarray(x0, dtype=float).reshape(-1)

        sol = solve_ivp(
            lambda tt, xx: self.f(tt, xx, *params),
            t_span=(t[0], t[-1]),
            y0=x0,
            t_eval=t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol.y.T

    def residuals(self, params, t, x):
        """
        Residual vector for least_squares.
        """

        xhat = self.simulate(
            t=t,
            x0=x[0],
            params=params,
        )

        return (xhat - x).ravel()

    def fit(
        self,
        t,
        x,
        initial_guess,
        bounds=(-np.inf, np.inf),
        loss="linear",
    ):
        """
        Estimate parameters.

        Parameters
        ----------
        t : (N,) array
        x : (N,) or (N,n)
        initial_guess : iterable
        bounds : tuple
        loss : str
            Passed to scipy.optimize.least_squares.

        Returns
        -------
        OptimizeResult
        """

        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)

        if t.ndim != 1:
            raise ValueError("t must be one-dimensional.")

        if x.ndim == 1:
            x = x[:, None]

        if len(t) != len(x):
            raise ValueError("t and x must have the same length.")

        result = least_squares(
            fun=self.residuals,
            x0=np.asarray(initial_guess, dtype=float),
            bounds=bounds,
            args=(t, x),
            method="trf",
            jac="2-point",
            loss=loss,
        )

        return result

    def predict(self, t, x0, params):
        """
        Simulate the model.

        Returns data in the same shape as the input state.
        """

        xhat = self.simulate(t, x0, params)

        if xhat.shape[1] == 1:
            return xhat[:, 0]

        return xhat
