"""
Reads a ProBMoT `Models.out` file (the output of an `exhaustive_search`/`induce`/`estimate` task,
e.g. `<writeDir>/Models.out`) back into pybm, as a `pybm.estimate.estimate.FullEstimationResults` -
the same return type `pybm.estimate.estimate.estimate_model` produces, so a ProBMoT run and a pybm
run can be compared/plotted with the same downstream code.

`Models.out` is a sequence of one or more blocks, each a complete, already-resolved model in the
same `.pbm` grammar `pybm.convert.probmot` already parses (`model <name> : <library>; ...`) -
ProBMoT re-dumps the model text once per structural candidate it evaluated, each ending with a
`//Train Error :<Metric> = <value>` comment holding that candidate's own error. This module reuses
`pybm.convert.probmot`'s existing parser/builder unchanged - it only splits the file into those
blocks and reads off each one's trailing error.
"""

from __future__ import annotations

import re

import numpy as np
from tqdm.asyncio import tqdm

from pybm.convert.probmot import Library, ModelBuilder, parse
from pybm.estimate.estimate import FullEstimationResults
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel

_MODEL_HEADER = re.compile(r"^// Model #(\d+) for dataset (\S+)", re.MULTILINE)
_TRAIN_ERROR = re.compile(r"//\s*Train Error\s*:\s*(\S+?)\s*=\s*([-+0-9.eE]+)\s*$", re.MULTILINE)


def _split_model_blocks(text: str) -> "list[str]":
    """Splits a `Models.out` file into one text block per `// Model #N for dataset ...` entry - or,
    if the file has no such header (a bare single-model dump), the whole text as one block."""
    starts = [m.start() for m in _MODEL_HEADER.finditer(text)]
    if not starts:
        return [text]
    starts.append(len(text))
    return [text[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


def _parse_train_error(block: str, block_index: int) -> float:
    match = _TRAIN_ERROR.search(block)
    if match is None:
        raise ValueError(
            f"Model block #{block_index} has no trailing '//Train Error :<Metric> = <value>' line - "
            "not a ProBMoT Models.out block, or the file was truncated."
        )
    return float(match.group(2))


def _build_result(block: str, block_index: int, library: Library) -> ParamEstimationResults:
    induced = ModelBuilder(library, parse(block)).build().induce()
    if len(induced) != 1:
        raise ValueError(
            f"Model block #{block_index} induced to {len(induced)} candidates, expected exactly 1 - "
            "a ProBMoT Models.out entry should already be a fully-resolved model (no Choose/"
            "ChooseProcess left)."
        )
    model: InducedModel = induced[0]

    # A const still `null` here is one ProBMoT never wrote a value back for - in practice because
    # the resolved structure's own equations never read it (e.g. a NoNutrientLim branch that
    # ignores its entity's alpha/halfSaturation, or minTemp/optTemp on a LinearTempGrowthLim whose
    # equation only uses refTemp), not a missing fit. It stays `None` on the `Const` itself (same
    # as any other unresolved `.pbm` - `pybm.simulate.initials.get_ctx` already knows how to fill
    # that in), and shows as `nan` in the flat arrays below since those need a plain float.
    consts = np.full(len(model.consts), np.nan, dtype=float)
    const_by_name: "dict[str, float]" = {}
    for name, const in model.consts.items():
        value = float(const.initial_value) if const.initial_value is not None else float("nan")
        consts[const.index_in_ctx] = value
        const_by_name[name] = value

    loss = _parse_train_error(block, block_index)
    return ParamEstimationResults(model=model, consts=consts, const_by_name=const_by_name, loss=loss)


def load_probmot_results(models_out_path: str, library: Library) -> FullEstimationResults:
    """
    Reads a ProBMoT `Models.out` file into a `FullEstimationResults`, ranked by each candidate's own
    `Train Error` (lower is better - `min()`, same convention `estimate_model` uses for its own
    `_singleshooting_loss`).

    Parameters
    ----------
    models_out_path : str
        Path to a ProBMoT `Models.out` file (one or more `// Model #N for dataset ...` blocks).
    library : Library
        The same `pybm.convert.probmot.Library` (`load_library(...)`) the ProBMoT task's `.pbl`
        library file corresponds to - every block's process/entity templates are resolved against
        it, exactly like `pybm.convert.probmot.load_model`.

    Returns
    -------
    FullEstimationResults
        `best_model`/`best_consts`/`best_const_by_name`: the lowest-`Train-Error` candidate.
        `best_loss`: that candidate's own `Train Error` value, AS WRITTEN in the file - this is
        whatever metric ProBMoT's task config reported (commonly `RMSEMultiDataset`, i.e. RMSE, not
        pybm's own MSE convention elsewhere in this codebase). Ranking (`min`) is valid regardless
        of which one it is, but don't compare this number directly against a pybm `.loss`
        (`_singleshooting_loss`, always MSE) without checking the metric name in `Models.out` first.
        `all_results`: one `ParamEstimationResults` per block, in file order, each `.loss` set the
        same way.

    Note
    ----
    Exogenous variables' `data` is NOT attached here (ProBMoT's own dataset paths in the `// Model
    #N for dataset ...` header aren't resolved to a pybm `TimeSeries`) - attach it the same way as
    any other converted model, e.g. `result.best_model.vars["phyto.conc"].data = TimeSeries(...)`,
    before simulating.
    """
    text = open(models_out_path).read()
    blocks = _split_model_blocks(text)
    all_results = [_build_result(block, i, library) for i, block in tqdm(enumerate(blocks), total=len(blocks))]

    best_result = min(all_results, key=lambda r: r.loss)
    return FullEstimationResults(
        best_consts=best_result.consts,
        best_const_by_name=best_result.const_by_name,
        best_loss=best_result.loss,
        all_results=all_results,
        best_model=best_result.model,
    )
