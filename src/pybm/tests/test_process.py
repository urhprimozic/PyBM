"""Plain-assert checks for Process/ProcessTemplate. Run directly: python -m pybm.tests.test_process

No pytest dependency - matches the rest of the project, which has none either.
"""

from pybm.model import (
    Model, EntityTemp, VarTemp, ConstTemp,
    Process, ProcessTemplate, Ode,
)

# --- Entity template, à la ProBMoT's "Population" ---
PopulationTemp = EntityTemp(
    VarTemp("conc", type="endo", initial=1.0),
)

# --- A process template: exponential growth, local rate constant `rate` ---
def growth_build(roles, consts):
    pop = roles["pop"]
    rate = consts["rate"]
    return [
        Ode(pop["conc"], lambda t, ctx: rate(ctx) * pop["conc"](t, ctx)),
    ]

Growth = ProcessTemplate(
    ConstTemp("rate", initial_value=0.1),
    roles={"pop": PopulationTemp},
    build=growth_build,
)

# --- Inherit: growth with an extra multiplicative cap constant ---
def growth_capped_build(roles, consts):
    pop = roles["pop"]
    rate = consts["rate"]
    cap = consts["cap"]
    return [
        Ode(pop["conc"], lambda t, ctx: rate(ctx) * pop["conc"](t, ctx) * (1 - pop["conc"](t, ctx) / cap(ctx))),
    ]

GrowthCapped = ProcessTemplate.inherits(
    Growth, ConstTemp("cap", initial_value=100.0), build=growth_capped_build
)

# --- A nested process: growth composed of two sub-processes (birth, death) ---
def birth_build(roles, consts):
    pop = roles["pop"]
    b = consts["b"]
    return [Ode(pop["conc"], lambda t, ctx: b(ctx) * pop["conc"](t, ctx))]

Birth = ProcessTemplate(ConstTemp("b", initial_value=0.3), roles={"pop": PopulationTemp}, build=birth_build)

def death_build(roles, consts):
    pop = roles["pop"]
    d = consts["d"]
    return [Ode(pop["conc"], lambda t, ctx: -d(ctx) * pop["conc"](t, ctx))]

Death = ProcessTemplate(ConstTemp("d", initial_value=0.1), roles={"pop": PopulationTemp}, build=death_build)

def combined_build(roles, consts):
    pop = roles["pop"]
    return [
        Birth(name="birth", pop=pop),
        Death(name="death", pop=pop),
    ]

Combined = ProcessTemplate(roles={"pop": PopulationTemp}, build=combined_build)


def test_basic_growth():
    prey = PopulationTemp("prey")
    proc = Growth(name="growth1", pop=prey)
    model = Model(prey, proc)

    assert "growth1.rate" in model.consts, model.consts.keys()
    ctx = {"vars": [2.0], "consts": [0.1]}
    got = model.vars["prey.conc"].ode(0.0, ctx)
    assert abs(got - 0.2) < 1e-12, got


def test_inherits_overrides():
    prey = PopulationTemp("prey2")
    proc = GrowthCapped(name="capgrowth", pop=prey)
    model = Model(prey, proc)
    assert "capgrowth.rate" in model.consts
    assert "capgrowth.cap" in model.consts
    ctx = {"vars": [10.0], "consts": [0.1, 100.0]}
    got = model.vars["prey2.conc"].ode(0.0, ctx)
    expected = 0.1 * 10.0 * (1 - 10.0 / 100.0)
    assert abs(got - expected) < 1e-12, (got, expected)


def test_nested_process_namespacing_and_aggregation():
    prey = PopulationTemp("prey3")
    proc = Combined(name="combined", pop=prey)
    model = Model(prey, proc)

    # nested consts get dotted namespace: combined.birth.b / combined.death.d
    assert "combined.birth.b" in model.consts, model.consts.keys()
    assert "combined.death.d" in model.consts, model.consts.keys()
    assert model.context_size == 2

    ctx = {"vars": [5.0], "consts": [0.3, 0.1]}
    got = model.vars["prey3.conc"].ode(0.0, ctx)
    expected = 0.3 * 5.0 - 0.1 * 5.0
    assert abs(got - expected) < 1e-12, (got, expected)


def test_copy_no_double_aggregation():
    prey = PopulationTemp("prey4")
    proc = Combined(name="combined2", pop=prey)
    model = Model(prey, proc)

    model_copy = model.copy()
    # constants must be FRESH objects (not shared) but same values/positions
    orig_const = model.consts["combined2.birth.b"]
    copy_const = model_copy.consts["combined2.birth.b"]
    assert orig_const is not copy_const
    assert orig_const.index_in_ctx == copy_const.index_in_ctx

    ctx = {"vars": [5.0], "consts": [0.3, 0.1]}
    got_orig = model.vars["prey4.conc"].ode(0.0, ctx)
    got_copy = model_copy.vars["prey4.conc"].ode(0.0, ctx)
    assert abs(got_orig - got_copy) < 1e-12, (got_orig, got_copy)

    # n_processes must NOT have doubled from the copy (no double `append_ode`)
    assert model.vars["prey4.conc"].n_processes == 2, model.vars["prey4.conc"].n_processes
    assert model_copy.vars["prey4.conc"].n_processes == 2, model_copy.vars["prey4.conc"].n_processes


if __name__ == "__main__":
    test_basic_growth()
    test_inherits_overrides()
    test_nested_process_namespacing_and_aggregation()
    test_copy_no_double_aggregation()
    print("ALL OK")
