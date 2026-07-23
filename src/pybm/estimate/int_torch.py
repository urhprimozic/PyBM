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



def get_initial_var_ctx(model: Model, *, dtype: torch.dtype, device: torch.device):
    """
    Creates a torch tensor context with initial values of the variables.
    """
    # torchode expects (batch_size, state_size). Initial values are fixed, so
    # they must not be optimization parameters.
    ans = [0.0] * len(model.endo_index)
    
    for var_name, index in model.endo_index.items():
        var = model.vars[var_name]
        if var.initial is None:
            raise ValueError(f"Variable {var_name} does not have an initial value defined.")
        ans[index] = var.initial
    return torch.tensor(ans, dtype=dtype, device=device).unsqueeze(0)

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
        # torchode stores states as (batch, n_variables), whereas the PyBM
        # model indexes variables along axis 0. Adapt at this boundary.
        model_var_ctx = var_ctx.transpose(0, 1)
        derivatives: list[Any] = []

        # TimeSeries is NumPy-based. Evaluation times are inputs, not fitted
        # parameters, therefore a Python scalar is appropriate here. The
        # current PyBM variable interface supports one trajectory per solve.
        if t.numel() != 1:
            raise ValueError("Torch integration currently supports batch size 1.")
        ode_t = float(t.detach().reshape(-1)[0])

        for var in vars:
            if var.ode is None:
                raise ValueError(f"Variable {var.name} does not have an ODE defined.")
            if isinstance(var.ode, Choose):
                raise ValueError(
                    f"Variable {var.name} has a Choose object as ODE. First use model.induce() to get a list of models without Choose objects."
                )             
            new_ctx = {
                "vars": model_var_ctx,
                "consts": const_ctx
            }
            derivatives.append(var.ode(ode_t, new_ctx))

        # One derivative vector per batch element: (batch, n_variables).
        return torch.stack(derivatives, dim=-1)

    # prepare solver 
    term = to.ODETerm(f)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    # Do not let an unstable trial parameter vector consume unbounded memory
    # or time during optimizer line searches.
    solver = to.AutoDiffAdjoint(
        step_method, step_size_controller, max_steps=10_000
    )

    # fix times dimensions
    t_eval = t_eval.reshape(1, -1)  # shape (1, n_time_points)

    # solve
    sol = solver.solve(to.InitialValueProblem(y0=initial_var_ctx, t_eval=t_eval))

    if torch.any(sol.status != 0):
        raise RuntimeError(
            f"ODE solver failed with status {sol.status.detach().cpu().tolist()}."
        )
    return sol


def get_data_matrix(*vars: Var, t_eval):
    """
    Returns data as shape (n_vars, n_time_points).
    """
    t_eval = t_eval.reshape(-1)
    data_matrix = torch.empty(
        (len(vars), len(t_eval)), dtype=t_eval.dtype, device=t_eval.device
    )

    for i, var in enumerate(vars):
        if var.data is None:
            raise ValueError(f"Variable {var.name} does not have data defined.")
        for j, t in enumerate(t_eval):
            # Observations are stored in NumPy TimeSeries objects.
            data_matrix[i, j] = float(var.data(float(t.detach())))

    return data_matrix


def get_initial_const_ctx(
    model: Model,
    default: float | None = 1.0,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
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

    const_ctx = torch.tensor(
        const_ctx, dtype=dtype, device=device, requires_grad=True
    )

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
    if not isinstance(t_eval, torch.Tensor):
        t_eval = torch.as_tensor(t_eval, dtype=torch.get_default_dtype())
    if not torch.is_floating_point(t_eval):
        t_eval = t_eval.to(torch.get_default_dtype())

    vars_ = model.get_endo_variables()
    t_eval = t_eval.reshape(-1)
    data = get_data_matrix(*vars_, t_eval=t_eval)
    # get initial values of variables
    initial_var_ctx = get_initial_var_ctx(
        model, dtype=t_eval.dtype, device=t_eval.device
    )
    # get initial values of constants
    initial_const_ctx = get_initial_const_ctx(
        model, dtype=t_eval.dtype, device=t_eval.device
    )

    # define objective function
    def objective(const_ctx):
        # calculate the whole trajectory
        sim = simulate(model, t_eval=t_eval, const_ctx=const_ctx, initial_var_ctx=initial_var_ctx)
        # extract 
        # torchode returns (batch, n_times, n_variables); data is
        # (n_variables, n_times).
        pred = sim.ys.squeeze(0).transpose(0, 1)
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
    # Without a line search L-BFGS may propose huge rate constants. That makes
    # the ODE stiff and used to look like an infinite loop in a notebook.
    kwargs.setdefault("lr", 0.01)
    kwargs.setdefault("line_search_fn", "strong_wolfe")
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
        

    
