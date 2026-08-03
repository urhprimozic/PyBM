"""
Implements multiple shooting method for parameter estimation in ODEs.

Let x(t) be (multivariable) at time t, given by the ODE
$$
x'(t) = f(t, x(t), c),
$$
where c is a vector of constants to be estimated from data 
$
(t_i, y_i = x_i + \epsilon_i)_{i=1}^N,
$ where $\epsilon_i$ is noise.

A time interval is divided into K subintervals. In addition to parameters c, a new set of
parameters s_1, ..., s_K is introduced.
k trajectories x_i(t) are simulated, each starting from s_i at the beginning of the i-th subinterval. 
We then minimize the MSE between the simulated trajectories and the data, 
subject to the constraint that the end of each trajectory matches the start of the next one. 

This is done by minimizing the loss function 
$$
L(x_1, ..., x_K, c) = \sum_{i=1}^N ||x(t_i) - y_i||^2.
$$
subject to the constraints
$$
h_j(q) = ||x_j(tau_{j+1}) - s_{j+1}||^2 = 0, j=0..K-1.
$$

Optimisation is done using 'trust-constr' method from scipy.optimize.minimize. 
Jacobian and Hessian are computed with adjont method using torch. 

TODO In addition, different selections of the initial values s_i can be used in parallel, and could be 
combined with a global optimisation method (e.g. CMA-ES) to find the best initial values s_i and parameters c.

Literature:
- https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf
- https://doi.org/10.1016/j.amc.2020.125644 


TODO Plan dela
---------------
- Konvencija: Uporabnik poda f(t, x, c) v torch operacijah!
- enakomeren? časoven razmik --> batching
- dtype=torch.float64 za boljšo natančnost
- rescape parameters?? c in s naj bodo istega veliksotnega reda 
-----------------

- 
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torchdiffeq import odeint_adjoint 

@dataclass
class MultipleShooting:
    """
    Mutliple shooting method for parameter estimation in ODEs.

    Implements multiple shooting method for parameter estimation in ODEs.
    -----------------
    Parameters
    ----------
    f : Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        The ODE function f(t, x, c) where t is time, x is the state vector, and c is the parameter vector to be estimated.
    ts : torch.Tensor
        The time points at which the data is observed.
    ys : torch.Tensor
        The observed data corresponding to the time points in ts.
    n_intervals : int
        The number of subintervals to divide the time interval into.


    Let x(t) be (multivariable) at time t, given by the ODE
    $$
    x'(t) = f(t, x(t), c),
    $$
    where c is a vector of constants to be estimated from data 
    $
    (t_i, y_i = x_i + \epsilon_i)_{i=1}^N,
    $ where $\epsilon_i$ is noise.

    A time interval is divided into K subintervals. In addition to parameters c, a new set of
    parameters s_1, ..., s_K is introduced.
    k trajectories x_i(t) are simulated, each starting from s_i at the beginning of the i-th subinterval. 
    We then minimize the MSE between the simulated trajectories and the data, 
    subject to the constraint that the end of each trajectory matches the start of the next one. 

    This is done by minimizing the loss function 
    $$
    L(x_1, ..., x_K, c) = \sum_{i=1}^N ||x(t_i) - y_i||^2.
    $$
    subject to the constraints
    $$
    h_j(q) = ||x_j(tau_{j+1}) - s_{j+1}||^2 = 0, j=0..K-1.
    $$

    Optimisation is done using 'trust-constr' method from scipy.optimize.minimize. 
    Jacobian and Hessian are computed with adjont method using torch. 

    TODO In addition, different selections of the initial values s_i can be used in parallel, and could be 
    combined with a global optimisation method (e.g. CMA-ES) to find the best initial values s_i and parameters c.

    Literature:
    - https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf
    - https://doi.org/10.1016/j.amc.2020.125644 

    """
    f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ts: torch.Tensor
    xs: torch.Tensor
    n_intervals: int
    atol: float = 1e-7
    rtol: float = 1e-6

    
    def integrate(self, t0 : torch.Tensor, x0 : torch.Tensor, t1 : torch.Tensor, c : torch.Tensor):
        """
            Integrate the ODE from t0 to t1 starting from x(t_0) = x_0 with parameters c.
        """
        sol = odeint_adjoint(self.f, x0, )


    