import torch 
import torchode as to
from typing import Any, cast
import numpy as np
from pybm.model import Choose, Context, Model, Var

def get_initial_var_ctx(model: Model):
    """
    Creates a torch tensor context with initial values of the variables.
    """
    ans = torch.zeros(len(model.endo_index))
    
    for var_name, index in model.endo_index.items():
        var = model.vars[var_name]
        if var.initial is None:
            raise ValueError(f"Variable {var_name} does not have an initial value defined.")
        ans[index] = var.initial
    return ans.unsqueeze(1)

def simulate(model:Model, t_eval, const_ctx, var_ctx_size=None, **kwargs):
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
        derivatives = torch.zeros_like(var_ctx)
        for var in vars:
            if var.ode is None:
                raise ValueError(f"Variable {var.name} does not have an ODE defined.")
            if isinstance(var.ode, Choose):
                raise ValueError(
                    f"Variable {var.name} has a Choose object as ODE. First use model.induce() to get a list of models without Choose objects."
                )             
            # compute and store new derivatives
            new_ctx = Context(vars=var_ctx, consts=const_ctx)
            derivatives[var.index_in_ctx] = var.ode(t, new_ctx)
        
        return derivatives

    # get initiač values
    initial_var_ctx = get_initial_var_ctx(model)

    # prepare solver 
    term = to.ODETerm(f)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    # fix times dimensions
    t_eval = torch.tensor(t_eval)
    if t_eval.dim() == 1:
        t_eval = t_eval.unsqueeze(0)

    # solve
    sol = jit_solver.solve(to.InitialValueProblem(y0=initial_var_ctx, t_eval=t_eval))

    return sol
        
