"""
Modified Multiple Shooting for ODE Parameter Estimation (Aydogmus & Tor, 2021)
recreated with torch autograd instead of hand-derived adjoint equations.

Key design choices (the "smart" part):

1. torch has no built-in constrained NLP solver (no SQP). We keep the paper's
   exact problem structure -- minimize L(q) subject to G_j(q)=0 -- and hand it
   to scipy.optimize.minimize(method='SLSQP'), which IS an SQP solver, just
   like the "active set (sqp)" solver the paper used in Matlab. torch is only
   used to compute L, its gradient, and the constraint Jacobians.

2. The paper's whole point (Section 3) was to avoid needing d backward solves
   per interval by turning the vector constraint G_j in R^d into a SCALAR
   constraint h_j = ||G_j||^2. We reproduce that exact trick here: even though
   autograd could differentiate the vector G_j directly, doing so would still
   require d separate backward() calls per interval (one per output
   component) to build the row-by-row Jacobian SLSQP needs. By constraining
   on the scalar h_j instead, ONE backward() call per interval gives the full
   gradient row. This is precisely the reduction from d^2 to d adjoint
   solves that the paper derives by hand in eq. (17) -- autograd doesn't
   remove the need for that reformulation, it just removes the need to
   hand-derive lambda(t).

3. The trajectory integration itself uses a plain fixed-step RK4 written in
   torch. Backpropagating through it is a *discrete* adjoint (backprop
   through solver steps), not the *continuous* adjoint of eq. (13). It's the
   natural thing to reach for first, and it's what most people mean by
   "autograd instead of hand derivation." A drop-in upgrade to the literal
   continuous adjoint of the paper is noted at the bottom (torchdiffeq).
"""

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# 1. The ODE system (eq. 18): Lotka-Volterra, as written in the paper
# ---------------------------------------------------------------------------
def lv_rhs(x, p):
    x1, x2 = x[..., 0], x[..., 1]
    p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
    dx1 = -p1 * x1 + p2 * x1 * x2
    dx2 = p3 * x2 - p4 * x1 * x2
    return torch.stack([dx1, dx2], dim=-1)


def rk4_integrate(x0, p, t0, t1, n_steps=20):
    """Fixed-step RK4 from t0 to t1. Differentiable end-to-end via autograd."""
    dt = (t1 - t0) / n_steps
    x = x0
    for _ in range(n_steps):
        k1 = lv_rhs(x, p)
        k2 = lv_rhs(x + 0.5 * dt * k1, p)
        k3 = lv_rhs(x + 0.5 * dt * k2, p)
        k4 = lv_rhs(x + dt * k3, p)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ---------------------------------------------------------------------------
# 2. Synthetic data, exactly as in Section 4
# ---------------------------------------------------------------------------
def make_data(seed=0, sigma=0.05):
    rng = np.random.default_rng(seed)
    p_true = torch.tensor([1.0, 1.0, 1.0, 1.0])
    x0 = torch.tensor([0.4, 1.0])
    K = 10
    taus = np.arange(0, K + 1, dtype=float)  # tau_j = j, j=0..10

    traj = [x0]
    x = x0
    for j in range(K):
        x = rk4_integrate(x, p_true, taus[j], taus[j + 1], n_steps=40)
        traj.append(x)
    traj = torch.stack(traj).detach().numpy()  # (K+1, 2), noiseless

    data = traj + rng.normal(0, sigma, traj.shape)
    return taus, data, sigma, p_true.numpy()


# ---------------------------------------------------------------------------
# 3. Packing / unpacking q = (s_0, ..., s_K, p) <-> flat numpy vector
# ---------------------------------------------------------------------------
K = 10
D = 2   # state dimension
M = 4   # number of parameters


def unpack(q_np):
    q = torch.tensor(q_np, requires_grad=True)
    S = q[: (K + 1) * D].reshape(K + 1, D)   # shooting nodes s_0..s_K
    p = q[(K + 1) * D:]                       # parameters
    return q, S, p


# ---------------------------------------------------------------------------
# 4. Objective L(q) -- depends only on s_j (Remark 2), so its gradient is
#    plain closed-form / cheap autograd, no adjoint needed at all.
# ---------------------------------------------------------------------------
def make_objective(taus, data, sigma):
    data_t = torch.tensor(data)

    def L_and_grad(q_np):
        q, S, p = unpack(q_np)
        resid = (S - data_t) / sigma
        L = (resid ** 2).sum()
        L.backward()
        return L.item(), q.grad.numpy().copy()

    return L_and_grad


# ---------------------------------------------------------------------------
# 5. Constraints h_j(q) = ||x_j(tau_{j+1}) - s_{j+1}||^2 = 0, j=0..K-1
#    ONE scalar per interval -> ONE backward() call gives the full gradient
#    row (this is the paper's d^2 -> d trick, done via autograd).
# ---------------------------------------------------------------------------
def make_constraints(taus, n_substeps=40):

    def h_j_and_grad(q_np, j):
        q, S, p = unpack(q_np)
        s_j = S[j]
        s_j1 = S[j + 1]
        x_end = rk4_integrate(s_j, p, taus[j], taus[j + 1], n_steps=n_substeps)
        G = x_end - s_j1
        h = (G ** 2).sum()
        h.backward()
        return h.item(), q.grad.numpy().copy()

    return h_j_and_grad


# ---------------------------------------------------------------------------
# 6. Assemble and solve with SLSQP (the SQP solver torch doesn't provide)
# ---------------------------------------------------------------------------
def solve(seed=0, p0=(0.5, 0.5, 0.5, -0.2), verbose=True):
    taus, data, sigma, p_true = make_data(seed=seed)
    L_and_grad = make_objective(taus, data, sigma)
    h_and_grad = make_constraints(taus)

    # initial guess: nodes at the data (a reasonable start, NOT a constraint),
    # parameters at the paper's deliberately bad guess p0 that breaks single
    # shooting (pole near t=3.3)
    s_init = data.flatten()
    q0 = np.concatenate([s_init, np.array(p0)])

    def fun(q_np):
        L, _ = L_and_grad(q_np)
        return L

    def jac(q_np):
        _, g = L_and_grad(q_np)
        return g

    constraints = []
    for j in range(K):
        constraints.append({
            "type": "eq",
            "fun": (lambda q_np, j=j: h_and_grad(q_np, j)[0]),
            "jac": (lambda q_np, j=j: h_and_grad(q_np, j)[1]),
        })

    res = minimize(
        fun, q0, jac=jac, constraints=constraints,
        method="SLSQP", options={"maxiter": 500, "ftol": 1e-12},
    )

    p_est = res.x[(K + 1) * D:]
    if verbose:
        print(f"success={res.success}  iters={res.nit}  L*={res.fun:.4g}")
        print(f"p_true = {p_true}")
        print(f"p_est  = {p_est}")
        print(f"max |G_j| at solution: "
              f"{max(abs(h_and_grad(res.x, j)[0]) ** 0.5 for j in range(K)):.2e}")
    return p_est, p_true, res


if __name__ == "__main__":
    solve()

# ---------------------------------------------------------------------------
# Upgrading to the literal continuous adjoint of the paper:
#
#   from torchdiffeq import odeint_adjoint as odeint
#   x_end = odeint(lambda t, x: lv_rhs(x, p), s_j, torch.tensor([taus[j], taus[j+1]]))[-1]
#
# odeint_adjoint solves the exact backward-time adjoint ODE (13) instead of
# backpropagating through the RK4 steps, giving O(1) memory in the number of
# integration steps -- the practical payoff the paper is chasing, now
# obtained by swapping one function instead of deriving lambda(t) by hand.
# The scalar-h_j reformulation above is still required either way.
# ---------------------------------------------------------------------------