import torch
from torch import nn
import torchode as to
from pybm.model import Model


class ODEFunc(nn.Module):
    def __init__(self, f, initial_guess):
        super().__init__()

        self.f = f

        self.params = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(c))) for c in initial_guess]
        )

    def forward(self, t, x):
        return self.f(t, x, *self.params)


class SingleShootingEstimatorTorch:

    def __init__(
        self,
        f,
        rtol=1e-6,
        atol=1e-8,
    ):
        self.f = f
        self.rtol = rtol
        self.atol = atol

    def fit(
        self,
        t,
        x,
        initial_guess,
        max_iter=1000,
    ):

        t = torch.as_tensor(t, dtype=torch.float32)

        x = torch.as_tensor(x, dtype=torch.float32)

        if x.ndim == 1:
            x = x[:, None]

        model = ODEFunc(
            self.f,
            initial_guess,
        )

        term = to.ODETerm(model)

        step_method = to.Tsit5(term)

        controller = to.IntegralController(
            atol=self.atol,
            rtol=self.rtol,
            term=term,
        )

        solver = to.AutoDiffAdjoint(
            step_method,
            controller,
        )

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        x0 = x[:1]

        def closure():

            optimizer.zero_grad()

            problem = to.InitialValueProblem(
                y0=x0,
                t_eval=t.unsqueeze(0),
            )

            sol = solver.solve(problem)

            pred = sol.ys.squeeze(0)

            loss = ((pred - x) ** 2).mean()

            loss.backward()

            return loss

        optimizer.step(closure)

        params = [p.detach().item() for p in model.params]

        return params
