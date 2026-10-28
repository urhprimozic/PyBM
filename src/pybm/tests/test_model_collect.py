"""Plain-assert checks that a Model finds the entities its processes use. Run directly:
python -m pybm.tests.test_model_collect (or with pytest).
"""

import copy
import pickle

import numpy as np

from pybm.latex import text
from pybm.model import (Algebraic, Choose, ChooseProcess, Const, Context, Entity, LinearInterpolation, Model, Ode,
                        Process, Var)
from pybm.simulate.trajectory import simulate


def _raises(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def build():
    """Prey/predator with a temperature input, a ChooseProcess for grazing and a Choose in the
    prey's temperature limitation, which alone uses the temperature."""
    env = Entity(Var("temperature", type="exo", data=LinearInterpolation(np.array([0.0, 50.0]), np.array([0.5, 1.5]))),
                 name="env")
    prey = Entity(Var("conc", initial=1.0), Const("growthRate", initial_value=1.0, range=(0, 5)),
                  Var("tempLim", algebraic=Choose(1.0, env["temperature"])), name="prey")
    predator = Entity(Var("conc", initial=0.5), Const("mortality", initial_value=0.2, range=(0, 2)), name="predator")
    attack, efficiency = Const("attackRate", initial_value=1.0), Const("efficiency", initial_value=0.5)
    half = Const("half", initial_value=1.0)
    growth = Process(Ode(prey["conc"], prey["growthRate"] * prey["tempLim"] * prey["conc"]), name="growth")
    linear = Process(Ode(prey["conc"], -attack * prey["conc"] * predator["conc"]),
                     Ode(predator["conc"], efficiency * attack * prey["conc"] * predator["conc"]),
                     attack, efficiency, name="linear")
    holling = Process(Ode(prey["conc"], -attack * prey["conc"] / (half + prey["conc"]) * predator["conc"]),
                      Ode(predator["conc"], efficiency * attack * prey["conc"] / (half + prey["conc"]) * predator["conc"]),
                      attack, efficiency, half, name="holling")
    grazing = ChooseProcess(linear, holling, name="grazing")
    mortality = Process(Ode(predator["conc"], -predator["mortality"] * predator["conc"]), name="mortality")
    return (prey, predator, env), (growth, grazing, mortality)


def _fingerprint(model):
    t = np.linspace(0, 20, 41)
    out = []
    for m in model.induce():
        consts = np.array([c.initial_value for c in sorted(m.consts.values(), key=lambda c: c.index_in_ctx)], float)
        x = simulate(m, t, Context(vars=None, consts=consts)).y
        out.append(({n: v.index_in_ctx for n, v in m.vars.items()}, {n: c.index_in_ctx for n, c in m.consts.items()},
                    m.equations(), np.asarray(x)))
    return out


def test_entities_found_through_processes():
    entities, processes = build()
    explicit = Model(*entities, *processes)
    entities, processes = build()
    found = Model(*processes)
    # env is used only in an option of the Choose of prey.tempLim, predator first in a ChooseProcess option
    assert list(found.entities) == ["prey", "env", "predator"]
    assert list(found.vars) == ["prey.conc", "prey.tempLim", "env.temperature", "predator.conc"]
    assert set(found.consts) == set(explicit.consts)
    # same models once the entity order is the same
    entities, processes = build()
    prey, predator, env = entities
    reordered = Model(prey, env, predator, *processes)
    for a, b in zip(_fingerprint(reordered), _fingerprint(found)):
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and np.array_equal(a[3], b[3])
    assert len(found.induce()) == 4


def test_listing_an_entity_again_is_harmless():
    (prey, predator, env), processes = build()
    model = Model(*processes, predator, prey, env)
    assert list(model.entities) == ["prey", "env", "predator"]
    model.add(prey)
    assert len(model.entities) == 3


def test_entity_used_by_no_equation_must_be_listed():
    entities, processes = build()
    unused = Entity(Var("light", type="exo"), name="sun")
    assert "sun" not in Model(*processes).entities
    assert "sun.light" in Model(unused, *processes).vars


def test_var_or_const_of_an_entity_adds_the_entity():
    (prey, predator, env), _ = build()
    model = Model(prey["conc"], predator["mortality"])
    assert list(model.entities) == ["prey", "env", "predator"] and not model.free_vars and not model.free_consts  # env via prey.tempLim


def test_two_entities_with_one_name():
    (prey, predator, env), processes = build()
    other = Entity(Var("conc", initial=1.0), name="prey")
    _raises(ValueError, lambda: Model(other, *processes))


def test_equations_set_on_vars():
    """An equation set directly on a Var: found from its entity, also when set after Model()."""
    a = Entity(Var("x", initial=1.0), name="a")
    b = Entity(Var("y", initial=2.0), name="b")
    c = Entity(Var("z", initial=3.0), name="c")
    a["x"].ode = -b["y"] * a["x"]
    model = Model(a)
    assert list(model.entities) == ["a", "b"]
    b["y"].algebraic = c["z"] + 1.0
    c["z"].ode = 0.0 * c["z"]
    [m] = model.induce()
    assert list(m.vars) == ["a.x", "b.y", "c.z"]
    assert all(v.index_in_ctx is not None for v in m.vars.values())


def test_free_var_and_undeclared_const():
    """A Var without an entity is added under its own name; a Const in no entity or process (here in
    an equation set on the Var) is named at induce()."""
    x = Var("x", initial=1.0)
    rate = Const("rate", initial_value=0.3)
    x.ode = -rate * x
    model = Model(x)
    assert list(model.vars) == ["x"]
    [m] = model.induce()
    assert list(m.consts) == ["rate"] and m.consts["rate"].index_in_ctx == 0
    t = np.linspace(0, 2, 5)
    sim = simulate(m, t, Context(vars=None, consts=np.array([0.3])), rtol=1e-10, atol=1e-12)
    assert np.allclose(sim.y.ravel(), np.exp(-0.3 * t), rtol=1e-8)
    # a second, different Const with that name: an error, not a silent mix-up
    y = Var("y", initial=1.0)
    y.ode = Const("rate", initial_value=1.0) * y
    _raises(ValueError, Model(x, y).induce)


def test_process_constants_found_through_equations():
    prey = Entity(Var("conc", initial=1.0), Const("growthRate", initial_value=1.0), name="prey")
    predator = Entity(Var("conc", initial=0.5), Const("mortality", initial_value=0.2), name="predator")
    P, Z = prey["conc"], predator["conc"]
    attack, efficiency = Const("attackRate", initial_value=1.0), Const("efficiency", initial_value=0.5)
    implicit = Process(Ode(P, -attack * P * Z), Ode(Z, efficiency * attack * P * Z), name="grazing")
    # entity constants stay the entity's
    assert implicit.own_consts == {"attackRate": attack, "efficiency": efficiency}
    mortality = Process(Ode(Z, -predator["mortality"] * Z), name="mortality")
    assert mortality.own_consts == {}
    # the same model as with the constants passed
    explicit = Process(Ode(P, -attack * P * Z), Ode(Z, efficiency * attack * P * Z), attack, name="grazing")
    assert list(explicit.own_consts) == ["attackRate", "efficiency"]
    [a], [b] = Model(implicit, mortality).induce(), Model(explicit, mortality).induce()
    assert list(a.consts) == list(b.consts) == ["prey.growthRate", "predator.mortality", "grazing.attackRate",
                                                "grazing.efficiency"]
    assert a.equations() == b.equations()


def test_process_constants_nested_and_shared():
    x = Var("x", initial=1.0)
    inner_rate, outer_rate, shared = Const("r1"), Const("r2"), Const("s")
    inner = Process(Ode(x, -inner_rate * shared * x), name="inner")
    outer = Process(Ode(x, outer_rate * shared * x), inner, name="outer")
    assert list(inner.own_consts) == ["r1", "s"] and list(outer.own_consts) == ["r2"]
    other = Process(Ode(x, shared), name="other")  # s again, in another process
    [m] = Model(outer, other).induce()
    assert list(m.consts) == ["outer.r2", "outer.inner.r1", "outer.inner.s"]  # one name per constant
    assert sorted(c.index_in_ctx for c in m.consts.values()) == [0, 1, 2]
    _raises(ValueError, lambda: Process(Ode(x, Const("k") + Const("k")), name="twice"))


def test_back_references_in_induced_copies():
    entities, processes = build()
    for m in Model(*processes).induce():
        for name, var in m.vars.items():
            assert var.entity is m.entities[name.split(".")[0]]
            assert var.entity[var.name] is var
        for entity in m.entities.values():
            for obj in entity.values():
                assert obj.entity is entity


def test_entity_mapping_keeps_back_references():
    e = Entity(Var("x"), name="e")
    y = Var("y")
    e["y"] = y
    assert y.entity is e
    del e["y"]
    assert y.entity is None


def test_dataclass_behaviour_unchanged():
    (prey, predator, env), _ = build()
    twin = Entity(Var("conc", initial=1.0), name="twin")
    assert prey["conc"] == twin["conc"]  # equal fields (entity is not compared), as before
    assert "entity" not in repr(prey["conc"])
    _raises(TypeError, lambda: Var("x", entity=prey))


def test_copy_and_pickle():
    entities, processes = build()
    model = Model(*processes)
    for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert clone.vars["prey.conc"].entity is clone.entities["prey"]
        assert clone.vars["prey.conc"] is not model.vars["prey.conc"]
    m = model.induce()[0]
    clone = pickle.loads(pickle.dumps(m))
    assert clone.vars["predator.conc"].entity is clone.entities["predator"]


def test_text_names():
    (prey, predator, env), (growth, grazing, mortality) = build()
    expr = 1.5 * prey["conc"] * predator["conc"]
    assert text(expr) == "((1.5 * prey.conc) * predator.conc)"
    assert text(expr, Model(growth)) == "((1.5 * prey.conc) * predator.conc)"
    assert text(Var("x") * Const("k")) == "(x * k)"
    m = Model(growth, grazing, mortality).induce()[1]
    assert text(m.vars["predator.conc"].ode, m).count("grazing.attackRate") == 1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
