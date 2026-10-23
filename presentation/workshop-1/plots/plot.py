from pybm.conditional import RelaxedIfElse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def plot_relaxed_conditional(filename, eps=0.1):
    """
    Plots the loss L(theta) of x(t) = int_0^t F dt against the true trajectory x(t) = t,
    for the hard conditional F and its relaxation, on theta in [-1, 1].
    """
    F = lambda x, t, theta: 1 if theta >= 0 else 0
    # If(d, .) fires for d <= 0, so If(-theta, .) is the relaxation of theta >= 0
    relaxedF = lambda x, t, theta: RelaxedIfElse(eps=eps).If(-theta, 1).Else(0)

    ts = np.linspace(0, 1, 100)
    x_true = ts

    # x(0) = 0, dx/dt = f(x, t, theta) on [0, 1]
    def trajectory(f, theta):
        sol = solve_ivp(lambda t, x: np.atleast_1d(f(x, t, theta)), (0, 1), [0.0], t_eval=ts)
        return sol.y[0]

    # discrepancy between xs and true trajectory
    def L(theta, f):
        return np.mean((trajectory(f, theta) - x_true) ** 2)

    thetas = np.linspace(-1, 1, 801)
    loss_hard = [L(theta, F) for theta in thetas]
    loss_relaxed = [L(theta, relaxedF) for theta in thetas]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(thetas, loss_hard, color="#2a78d6", lw=2, label=r"$x = \int_0^t F\,dt$")
    ax.plot(thetas, loss_relaxed, color="#eb6834", lw=2,
            label=rf"$x = \int_0^t \tilde F\,dt$ ($\varepsilon={eps}$)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\mathcal{L}(\theta)$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.02)
    return fig, ax

def plot_multishooting(filename, seed=0):
    rng = np.random.default_rng(seed)
    theta_true = 1
    def x(theta, t):
        return np.sin(theta * t)/t
    def F(theta, t):
        # d/dt sin(theta t)/t
        return - np.sin(theta*t)/t**2 + theta*np.cos(theta*t)/t
    n_data  =10
    t_end = 10
    ts = np.linspace(1, t_end, n_data)
    ys  = x(theta_true, ts) + rng.normal(0, 0.05, n_data)

    #try for theta = 0.7 and init points
    theta=0.7
    n_intervals = 5
    t_grid = np.linspace(1, t_end, n_intervals + 1)  # interval k is [t_grid[k], t_grid[k+1]]
    t_initials = t_grid[:-1]
    x_initials = x(theta, t_initials) + rng.uniform(-0.1, 0.1, n_intervals)

    # solve ode on every interval
    trajectory = []
    for (t0, t1, x0) in zip(t_initials, t_grid[1:], x_initials):
        t_eval = np.linspace(t0, t1, 50)
        sol = solve_ivp(lambda t, x: F(theta, t), (t0, t1), [x0], t_eval=t_eval)
        trajectory.append((sol.t, sol.y[0]))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    t_fine = np.linspace(1, t_end, 400)
    # plot ground truth in dashed gray
    ax.plot(t_fine, x(theta_true, t_fine), color="#929292", lw=1.5, ls="--", label="ground truth")
    # plot data points with black points
    ax.plot(ts, ys, "o", color="black", ms=4, label="data")
    # plot blue trajectory
    for k, (t_seg, x_seg) in enumerate(trajectory):
        ax.plot(t_seg, x_seg, color="#2a78d6", lw=2, label=rf"$x_\theta$, $\theta={theta}$" if k == 0 else None)
    # plot straight line on trajectory jumps (end of interval k -> initial value of interval k+1)
    for k in range(n_intervals - 1):
        ax.plot([t_grid[k + 1]] * 2, [trajectory[k][1][-1], x_initials[k + 1]],
                color="#e34948", lw=1, ls=":", label="gap" if k == 0 else None)
    # plot initial values of trajectories with red points
    ax.plot(t_initials, x_initials, "o", color="#e34948", ms=5, zorder=3, label="initial values")
    # add legend
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.02)
    return fig, ax


if __name__ == "__main__":
    plot_relaxed_conditional("relaxed_conditional.pdf")
    plot_multishooting("multishooting.pdf")
