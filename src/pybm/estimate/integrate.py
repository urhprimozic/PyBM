import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


class SingleShootingEstimator:
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