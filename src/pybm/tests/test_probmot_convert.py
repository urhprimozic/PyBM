"""Plain-assert checks for the ProBMoT .pbl/.pbm converter. Run directly:
python -m pybm.tests.test_probmot_convert

Depends on the real ProBMoT example files living at ../probmot/examples/aquatic relative to this
repo (see convert/probmot.py's module docstring for the converter's scope/limitations) - skips
itself if that directory isn't present (e.g. a checkout without the sibling probmot repo).
"""

import os

import numpy as np

from pybm.convert.probmot import load_library, load_model
from pybm.model import Context

_AQUATIC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "probmot", "examples", "aquatic"
)
_LIB_PATH = os.path.join(_AQUATIC_DIR, "AquaticEcosystem.pbl")
_SKIP = not os.path.exists(_LIB_PATH)


def test_complete_model_builds_and_computes_correctly():
    lib = load_library(_LIB_PATH)
    model = load_model(os.path.join(_AQUATIC_DIR, "BledComplete.pbm"), lib)

    assert set(model.entities.keys()) == {"ortp", "no", "silica", "phyto", "daph", "env"}
    assert model.vars["phyto.conc"].initial is not None  # complete model - no nulls anywhere
    assert all(c.initial_value is not None for c in model.consts.values())

    # hand-check: phyto.nutrientLim aggregates (product) 3 MonodNutrientLim terms, one per nutrient
    exo = {"ortp.conc": 3.0, "no.conc": 0.02, "silica.conc": 4.0}
    for name, val in exo.items():
        model.vars[name].set_data(np.array([0.0, 100.0]), np.array([val, val]))
    half = {n: model.consts[f"{n}.halfSaturation"].initial_value for n in ("ortp", "no", "silica")}
    expected = 1.0
    for n in ("ortp", "no", "silica"):
        expected *= exo[f"{n}.conc"] / (exo[f"{n}.conc"] + half[n])

    c = np.zeros(len(model.consts))
    for cst in model.consts.values():
        c[cst.index_in_ctx] = cst.initial_value
    ctx = Context(vars=np.ones(len(model.vars)), consts=c)
    got = model.vars["phyto.nutrientLim"].algebraic(0.0, ctx)
    assert abs(got - expected) < 1e-9, (got, expected)


def test_incomplete_parameters_are_null():
    lib = load_library(_LIB_PATH)
    model = load_model(os.path.join(_AQUATIC_DIR, "BledIncompleteParameters.pbm"), lib)

    null_consts = {name for name, c in model.consts.items() if c.initial_value is None}
    # exactly matches the 9 `= null;` constant assignments in the source file
    assert len(null_consts) == 9, null_consts
    assert model.vars["phyto.conc"].initial is None  # the file's one null initial value


def test_structural_incompleteness_produces_expected_choose_process_slots():
    lib = load_library(_LIB_PATH)
    model = load_model(os.path.join(_AQUATIC_DIR, "BledIncompleteStructure.pbm"), lib)

    # every abstract reference in the .pbm becomes its own ChooseProcess slot (see convert/probmot.py
    # module docstring for why nested abstract references get *flattened* to the model's top level)
    candidate_counts = {name: len(cp.options) for name, cp in model.pending_process_choices.items()}
    assert sum(candidate_counts.values()) > 0

    induced = model.induce()
    expected_total = 1
    for n in candidate_counts.values():
        expected_total *= n
    assert len(induced) == expected_total
    assert all(m.pending_process_choices == {} for m in induced)

    # at least one resolved branch picked a Summation-based PhytoLim candidate (RHS-iterator =>
    # implicit sum via aggregation, not a target-iterated equation)
    assert any(m.vars["daph.phytoSum"].algebraic is not None for m in induced)
    # at least one resolved branch picked a LightInfluence candidate with its own local constant,
    # namespaced under the flattened slot's own name (not the winning leaf template's name)
    assert any(any(k.startswith("LightInfluence") for k in m.consts) for m in induced)


if __name__ == "__main__":
    if _SKIP:
        print(f"SKIPPED: {_LIB_PATH} not found (no sibling probmot/examples checkout)")
    else:
        test_complete_model_builds_and_computes_correctly()
        test_incomplete_parameters_are_null()
        test_structural_incompleteness_produces_expected_choose_process_slots()
        print("ALL OK")
