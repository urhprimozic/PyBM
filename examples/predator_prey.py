"""A small process-based predator-prey example expressed with PyBM variables.

The individual process terms are kept separate below for readability.  The
final ODE of each population is their sum, which is the flattened form a
ProBMoT process model should produce.
"""

from typing_extensions import Literal

import numpy as np
from scipy.integrate import solve_ivp

from pybm.model_old import Const, Context, Model, Var


# Temperature is a dimensionless multiplier, not degrees Celsius.  These
# values keep it positive and make three deliberately slow cycles in 30 days.
TEMPERATURE_MEAN = 1.0
TEMPERATURE_AMPLITUDE = 0.2
TEMPERATURE_PERIOD = 10.0  # days

# Parameters used to generate the synthetic data.
GROWTH_RATE_PREY = 0.5
GROWTH_RATE_PREDATOR = -0.2  # temperature-dependent background mortality
PREDATION_RATE = 0.02
CONVERSION_EFFICIENCY = 0.1


def build_model(engine : Literal["scipy", "torch", "jax"] = "scipy") -> tuple[Model, dict[str, Var | Const]]:
    """Build the model and return its named components."""
    # create variables
    n_prey = Var("n_prey", type="endo", initial=40.0)
    n_predator = Var("n_predator", type="endo", initial=8.0)
    temperature = Var("temperature", type="exo")

    # create constants
    growth_rate_prey = Const("growth_rate_prey", GROWTH_RATE_PREY)
    growth_rate_predator = Const("growth_rate_predator", GROWTH_RATE_PREDATOR)
    predation_rate = Const("predation_rate", PREDATION_RATE)
    conversion_efficiency = Const("conversion_efficiency", CONVERSION_EFFICIENCY)

    # create a model
    model = Model(
        n_prey,
        n_predator,
        temperature,
        growth_rate_prey,
        growth_rate_predator,
        predation_rate,
        conversion_efficiency,
        engine=engine
    )

    # Process: temperature-dependent growth (or mortality for predator).
    def prey_growth(t: float, ctx: Context):
        return temperature(t, ctx) * growth_rate_prey(ctx) * n_prey(t, ctx)

    def predator_growth(t: float, ctx: Context):
        return temperature(t, ctx) * growth_rate_predator(ctx) * n_predator(t, ctx)

    # Process: a predator consumes prey and turns a fraction into predators.
    def predation(t: float, ctx: Context):
        return predation_rate(ctx) * n_prey(t, ctx) * n_predator(t, ctx)

    # add the ODEs to the variables
    n_prey.ode = lambda t, ctx: prey_growth(t, ctx) - predation(t, ctx)
    n_predator.ode = lambda t, ctx: (
        predator_growth(t, ctx) + conversion_efficiency(ctx) * predation(t, ctx)
    )

    return model, {
        "n_prey": n_prey,
        "n_predator": n_predator,
        "temperature": temperature,
        "growth_rate_prey": growth_rate_prey,
        "growth_rate_predator": growth_rate_predator,
        "predation_rate": predation_rate,
        "conversion_efficiency": conversion_efficiency,
    }


def generate_synthetic_data(seed: int = 7) -> tuple[Model, dict[str, Var | Const], np.ndarray]:
    """Simulate 30 days and attach slightly noisy observations with ``set_data``."""
    model, components = build_model()
    n_prey = components["n_prey"]
    n_predator = components["n_predator"]
    temperature = components["temperature"]
    assert isinstance(n_prey, Var)
    assert isinstance(n_predator, Var)
    assert isinstance(temperature, Var)

    times = np.linspace(0.0, 30.0, 1201)  # one observation every 0.025 days
    temperature_values = TEMPERATURE_MEAN + TEMPERATURE_AMPLITUDE * np.sin(
        2 * np.pi * times / TEMPERATURE_PERIOD
    )
    temperature.set_data(times, temperature_values)

    def derivatives(t: float, state: np.ndarray) -> list[float]:
        ctx: Context = {
            "vars": state,
            "consts": np.array(
                [
                    GROWTH_RATE_PREY,
                    GROWTH_RATE_PREDATOR,
                    PREDATION_RATE,
                    CONVERSION_EFFICIENCY,
                ]
            ),
        }
        return [n_prey.ode(t, ctx), n_predator.ode(t, ctx)]  # type: ignore[misc]

    solution = solve_ivp(
        derivatives,
        t_span=(times[0], times[-1]),
        y0=[n_prey.initial, n_predator.initial],
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    # Measurements need not be exact.  A 1% multiplicative noise keeps both
    # populations positive while making later parameter fitting meaningful.
    rng = np.random.default_rng(seed)
    prey_data = solution.y[0] * (1 + rng.normal(0, 0.01, size=times.size))
    predator_data = solution.y[1] * (1 + rng.normal(0, 0.01, size=times.size))

    n_prey.set_data(times, prey_data)
    n_predator.set_data(times, predator_data)

    # For an estimation run, the first observation is the known initial state.
    n_prey.initial = float(prey_data[0])
    n_predator.initial = float(predator_data[0])

    return model, components, times


if __name__ == "__main__":
    _, components, times = generate_synthetic_data()
    prey = components["n_prey"]
    predator = components["n_predator"]
    assert isinstance(prey, Var) and isinstance(predator, Var)
    print(f"Generated {len(times)} observations from t={times[0]} to t={times[-1]}.")
    print(f"Initial observations: prey={prey.data.x[0]:.3f}, predator={predator.data.x[0]:.3f}")
