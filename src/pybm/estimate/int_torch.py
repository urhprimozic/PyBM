from typing import Any, Literal
import warnings

import torch
import torchode as to
import numpy as np
from torchmin import minimize # TODO doesnt work....
from torch.optim import LBFGS
from pybm.model import Choose, Context, Model, Var


def adaptive_loss(pred, target, q=0.9, eps=1e-8):
    # 1) relativna napaka (za eksponente je to ključno)
    r = (pred - target) / (target.abs() + eps)

    # 2) adaptivni prag iz trenutnih residualov
    delta = torch.quantile(r.detach().abs(), q).clamp_min(1e-3)

    # 3) pseudo-Huber (gladek, ne eksplodira kot MSE)
    return (delta**2 * (torch.sqrt(1.0 + (r / delta)**2) - 1.0)).mean()



def get_initial_var_ctx(model: Model):
    """
    Creates a torch tensor context with initial values of the variables.
    """
    # initial variables are constants - no need for grads
    ans =  [0] * len(model.endo_index)#torch.zeros(len(model.endo_index))
    
    for var_name, index in model.endo_index.items():
        var = model.vars[var_name]
        if var.initial is None:
            raise ValueError(f"Variable {var_name} does not have an initial value defined.")
        ans[index] = var.initial
    ans = torch.tensor(ans, requires_grad=True)
    return ans.unsqueeze(1)

def simulate(model:Model, t_eval, const_ctx, initial_var_ctx, **kwargs):
    """
    Solve ODE for given endogenous variables using torchode.

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
        Constant context for the model. Must be of shape (n_constants,).        
    """
   # function f(t, var_ctx) for x'(t) = f(t, x(t)
    vars = model.get_endo_variables()

    def f(t, var_ctx):
        derivatives : list[Any] = [0] * len(vars)  # initialize derivatives tensor
        for var in vars:
            if var.ode is None:
                raise ValueError(f"Variable {var.name} does not have an ODE defined.")
            if isinstance(var.ode, Choose):
                raise ValueError(
                    f"Variable {var.name} has a Choose object as ODE. First use model.induce() to get a list of models without Choose objects."
                )             
            # compute and store new derivatives
            #new_ctx = Context(vars=var_ctx, consts=const_ctx)
            new_ctx = {
                "vars": var_ctx,
                "consts": const_ctx
            }
            derivatives[var.index_in_ctx] = var.ode(t, new_ctx)

        derivatives = torch.stack(derivatives)  # convert list to tensor
        return derivatives

    # prepare solver 
    term = to.ODETerm(f)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)

    # fix times dimensions
    if t_eval.dim() == 1:
        t_eval = t_eval.unsqueeze(0)

    # solve
    sol = solver.solve(to.InitialValueProblem(y0=initial_var_ctx, t_eval=t_eval))

    return sol


def get_data_matrix(*vars: Var, t_eval):
    """
    Returns data as shape (n_vars, n_time_points).
    """
    t_eval = t_eval.reshape(-1)
    data_matrix = torch.zeros((len(vars), len(t_eval)), dtype=torch.float64)

    for i, var in enumerate(vars):
        if var.data is None:
            raise ValueError(f"Variable {var.name} does not have data defined.")
        for j, t in enumerate(t_eval):
            data_matrix[i, j] = float(var.data(float(t)))

    return data_matrix


def get_initial_const_ctx(model: Model, default: float | None = 1.0):
    """
    Initial constant vector.
    """
    const_ctx = [0] * len(model.const_index)

    for const_name, const in model.consts.items():
        if const.index_in_ctx is None:
            raise ValueError(f"Constant {const.name} has no index in context.")
        if const.initial_value is None:
            if default is not None:
                const_ctx[const.index_in_ctx] = default
            else:
                raise ValueError(f"Constant {const.name} has no initial value.")
        else:
            const_ctx[const.index_in_ctx] = float(const.initial_value)

    const_ctx = torch.tensor(const_ctx, requires_grad=True)

    return const_ctx


def get_jit_estimate():
    """
    Estimate constants with L-BFGS. Just in time compiled.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
    return torch.compile(estimate)
     

def estimate(model: Model, t_eval, loss : Literal["sum", "mean", "adaptive"]="mean", **kwargs):
    """
    Estimate constants with L-BFGS.

    Method can fail due to overflows. In this case, try 
    using `loss="adaptive"` which is more robust to outliers and overflows.

    Parameters
    ----------
    model : Model
        Model with defined variables, constants, and ODEs.
    t_eval : array-like
        Time points to evaluate and compute the solution.
    loss : str, optional
        Loss function to use for optimization. Options are:
        - "sum": Sum of squared residuals.
        - "mean": Mean of squared residuals.
        - "adaptive": Adaptive loss function based on quantiles of residuals.
    """
    vars_ = model.get_endo_variables()
    t_eval = t_eval.reshape(-1)
    data = get_data_matrix(*vars_, t_eval=t_eval)
    # get initial values of variables
    initial_var_ctx = get_initial_var_ctx(model)
    # get initial values of constants
    initial_const_ctx = get_initial_const_ctx(model)

    # define objective function
    def objective(const_ctx):
        # calculate the whole trajectory
        sim = simulate(model, t_eval=t_eval, const_ctx=const_ctx, initial_var_ctx=initial_var_ctx)
        # extract 
        pred = sim.ys.squeeze(-1)
        # compute residuals
        residuals = torch.ravel(pred - data)
        # compute mean squared error
        if loss == "mean":
            mse = torch.mean(torch.pow(residuals, 2))
        elif loss == "sum":
            mse = torch.sum(torch.pow(residuals, 2))
        else:
            mse = adaptive_loss(pred, data)
        return mse

    # minimize
    if kwargs is None:
        kwargs = {"lr": 0.1}
    if "lr" not in kwargs:
        kwargs["lr"] = 0.1
    optimizer = LBFGS([initial_const_ctx], **kwargs)

    def closure():
        optimizer.zero_grad()
        loss = objective(initial_const_ctx)
        loss.backward()
        return loss

    optimizer.step(closure)


    loss = objective(initial_const_ctx).item()
    # return the estimated constants
    return initial_const_ctx, loss
        

    


