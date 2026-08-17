"""Plain-assert checks for ChooseProcess + Model.induce(). Run directly:
python -m pybm.tests.test_choose_process

No pytest dependency - matches the rest of the project, which has none either.
"""

from pybm.model_old import Model, Entity, Var, Const, Process, ChooseProcess, Ode


def make_growth_factory(name, entity_name, rate_value):
    """Returns a ChooseProcess candidate factory: builds a fresh Process against
    `model.entities[entity_name]` - the entity that's actually live in the resolving model/branch,
    not whatever object was around when the factory was defined."""

    def factory(model):
        pop = model.entities[entity_name]
        conc = pop["conc"]
        rate = Const("rate", initial_value=rate_value)
        return Process(
            Ode(conc, lambda t, ctx: rate(ctx) * conc(t, ctx)),
            rate,
            name=name,
        )

    return factory


def test_unresolved_model_has_no_equation_yet():
    pop = Entity(Var("conc", type="endo", initial=1.0), name="pop")
    fast = make_growth_factory("FastGrowth", "pop", 0.5)
    slow = make_growth_factory("SlowGrowth", "pop", 0.1)
    slot = ChooseProcess(fast, slow, name="growth")

    model = Model(pop, slot)
    # nothing compiled yet - the slot is unresolved
    assert model.vars["pop.conc"].ode is None
    assert "growth" in model.pending_process_choices
    assert "growth.rate" not in model.consts


def test_induce_resolves_each_candidate_into_its_own_model():
    pop = Entity(Var("conc", type="endo", initial=1.0), name="pop")
    fast = make_growth_factory("FastGrowth", "pop", 0.5)
    slow = make_growth_factory("SlowGrowth", "pop", 0.1)
    slot = ChooseProcess(fast, slow, name="growth")

    model = Model(pop, slot)
    induced = model.induce()
    assert len(induced) == 2

    ctx = {"vars": [2.0], "consts": [0.0]}
    results = set()
    for m in induced:
        assert m.pending_process_choices == {}
        assert "growth.rate" in m.consts
        rate_index = m.consts["growth.rate"].index_in_ctx
        this_ctx = {"vars": [2.0], "consts": [0.0] * (rate_index + 1)}
        this_ctx["consts"][rate_index] = m.consts["growth.rate"].initial_value
        got = m.vars["pop.conc"].ode(0.0, this_ctx)
        results.add(round(got, 6))

    assert results == {round(0.5 * 2.0, 6), round(0.1 * 2.0, 6)}, results


def test_choose_process_combines_with_var_level_choose():
    from pybm.model_old import Choose

    pop = Entity(
        Var("conc", type="endo", initial=1.0),
        Var("aux", type="endo", initial=0.0, algebraic=Choose(
            lambda t, ctx: 1.0, lambda t, ctx: 2.0
        )),
        name="pop2",
    )
    fast = make_growth_factory("FastGrowth2", "pop2", 0.5)
    slow = make_growth_factory("SlowGrowth2", "pop2", 0.1)
    slot = ChooseProcess(fast, slow, name="growth2")

    model = Model(pop, slot)
    induced = model.induce()
    # 2 (aux Choose options) x 2 (process candidates) = 4 fully resolved models
    assert len(induced) == 4
    for m in induced:
        assert m.pending_process_choices == {}
        assert not isinstance(m.vars["pop2.aux"].algebraic, Choose)


if __name__ == "__main__":
    test_unresolved_model_has_no_equation_yet()
    test_induce_resolves_each_candidate_into_its_own_model()
    test_choose_process_combines_with_var_level_choose()
    print("ALL OK")
