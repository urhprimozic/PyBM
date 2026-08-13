"""Plain-assert checks that pybm.simulate.predict.simulate() correctly re-derives algebraic
(non-differential) endogenous variables at every time point, instead of leaving them frozen at
whatever scratch value they started with. Run directly: python -m pybm.tests.test_simulate_algebraic
"""

import numpy as np

from pybm.model import Var, Const, Model, Context
from pybm.simulate.predict import simulate


def test_algebraic_variable_tracks_state_not_frozen():
    # x: differential, dx/dt = -k * x * factor
    # factor: algebraic, factor = 1 / (1 + x)   <- depends on x's CURRENT value each step
    # As x decays toward 0, factor should rise toward 1 - if factor were incorrectly frozen at its
    # initial value (the bug this fixes), it would stay flat at factor(x0) for the whole simulation.
    x = Var("x", type="endo", initial=5.0)
    factor = Var("factor", type="endo", initial=0.0)  # scratch initial - must be re-derived, not used
    k = Const("k", initial_value=0.5)
    model = Model(x, factor, k)

    factor.algebraic = lambda t, ctx: 1.0 / (1.0 + x(t, ctx))
    x.ode = lambda t, ctx: -k(ctx) * x(t, ctx) * factor(t, ctx)

    ctx = Context(vars=[5.0, 0.0], consts=[0.5])
    t_eval = np.linspace(0, 10, 50)
    sol = simulate(model, t_eval=t_eval, context=ctx)

    assert sol.success
    x_traj = sol.y[x.index_in_ctx]
    factor_traj = sol.y[factor.index_in_ctx]

    # x decays monotonically (it's a positive decay process)
    assert np.all(np.diff(x_traj) <= 1e-9)

    # factor must match 1/(1+x) at EVERY point, not just t=0 - this is what would fail if factor
    # were left frozen at its initial scratch value instead of being re-derived each step
    expected_factor = 1.0 / (1.0 + x_traj)
    assert np.allclose(factor_traj, expected_factor, atol=1e-6), (factor_traj, expected_factor)

    # in particular, factor must have actually CHANGED between the first and last time point
    # (x moved a lot over this window, so factor should too - catches "frozen" regressions directly)
    assert abs(factor_traj[-1] - factor_traj[0]) > 0.05


def test_chained_algebraic_dependencies_settle_correctly():
    # a two-hop algebraic dependency chain: b depends on a, a depends on state x.
    x = Var("x", type="endo", initial=2.0)
    a = Var("a", type="endo", initial=0.0)
    b = Var("b", type="endo", initial=0.0)
    model = Model(x, a, b)

    a.algebraic = lambda t, ctx: 2.0 * x(t, ctx)
    b.algebraic = lambda t, ctx: a(t, ctx) + 1.0
    x.ode = lambda t, ctx: -0.1 * x(t, ctx)

    ctx = Context(vars=[2.0, 0.0, 0.0], consts=[])
    sol = simulate(model, t_eval=np.linspace(0, 5, 10), context=ctx)

    assert sol.success
    x_traj = sol.y[x.index_in_ctx]
    a_traj = sol.y[a.index_in_ctx]
    b_traj = sol.y[b.index_in_ctx]
    assert np.allclose(a_traj, 2.0 * x_traj, atol=1e-6)
    assert np.allclose(b_traj, 2.0 * x_traj + 1.0, atol=1e-6)


if __name__ == "__main__":
    test_algebraic_variable_tracks_state_not_frozen()
    test_chained_algebraic_dependencies_settle_correctly()
    print("ALL OK")
