"""
Per-block gradient-matching structural search - a faster alternative to `estimate_model`'s
brute-force search over the full Cartesian product of every open structural choice.

See `pybm.model_graph`'s module docstring for *why* this is possible (gradient matching never
simulates the ODE forward, so a state variable's own constant fit is independent of any
structurally-unrelated variable's). `build_blocks` there partitions a model's state variables into
independent groups; `estimate_model_decomposed` below fits each group's own, much smaller,
structural search separately, then combines the winners into one final model.

Design choice: simplicity over maximum speed
----------------------------------------------
Each block's own candidates are resolved from the *whole* model (`Model._resolve`), with every
slot *outside* this block pinned to its first option, rather than from a specially pruned model
containing only this block's own entities/constants. That keeps this module small and lets it
reuse `Model._resolve`/`estimate_gradient_matching` completely unmodified, at the cost of some
wasted work: `estimate_gradient_matching` still ends up jointly fitting the (already correctly
fit, from an earlier block) or (still arbitrarily pinned, from a not-yet-reached block) constants
of every *other* state variable too, alongside this block's own. This does not change the
resulting constants for this block - the least-squares Jacobian across independent blocks is
block-diagonal, so their optimal constants do not affect each other - it only means each
candidate's own fit is somewhat larger (more state variables, more constants) than the strict
minimum. The combinatorial win (evaluating a sum of small per-block candidate counts instead of
their product) is unaffected either way, and is normally by far the dominant cost.
"""

from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
from tqdm import tqdm

from pybm.estimate.estimate import FullEstimationResults, _singleshooting_loss
from pybm.estimate.gradient_matching import estimate_gradient_matching, fit_gps
from pybm.estimate.results import ParamEstimationResults
from pybm.model import InducedModel, Model
from pybm.model_graph import Block, build_blocks


def _switch_engine(induced: InducedModel, engine: Literal["torch", "scipy"], device) -> None:
    """
    Switches `induced` onto `engine`, passing `device` along only for `engine="torch"` (the only
    engine `InducedModel.switch_engine` accepts a `device` for).

    Parameters
    ----------
    induced : InducedModel
        The model to switch, in place.
    engine : "torch" | "scipy"
        The compute engine to switch to.
    device : torch.device | str | None
        Where to place `torch`-engine data (ignored for `engine="scipy"`).
    """
    if engine == "torch":
        induced.switch_engine(engine, device=device)
    else:
        induced.switch_engine(engine)


def _block_option_counts(
    model: Model, block: Block, var_choices: "list[tuple[str | None, str, str]]", process_slot_names: "list[str]"
) -> "tuple[list[int], list[int], list[int], list[int]]":
    """
    Reads off how many candidates each of a block's own slots has - the shape of that block's own,
    much smaller, Cartesian product.

    Parameters
    ----------
    model : Model
        The incomplete model `block` belongs to.
    block : Block
        The block to look up slot sizes for.
    var_choices : list[tuple[str | None, str, str]]
        `model._open_var_choices()`, in the same order used to build `block`.
    process_slot_names : list[str]
        `list(model.pending_process_choices.keys())`, in the same order used to build `block`.

    Returns
    -------
    tuple[list[int], list[int], list[int], list[int]]
        `(var_slot_order, process_slot_order, var_option_counts, process_option_counts)`:
        `block`'s own slot indices, sorted (a fixed order to zip candidate-index tuples against),
        and how many options each one has, in that same order.
    """
    var_slot_order = sorted(block.var_slot_indices)
    process_slot_order = sorted(block.process_slot_indices)

    var_option_counts = [
        len(getattr(model._lookup_var(var_choices[i][0], var_choices[i][1]), var_choices[i][2]).options)
        for i in var_slot_order
    ]
    process_option_counts = [
        len(model.pending_process_choices[process_slot_names[i]].options) for i in process_slot_order
    ]
    return var_slot_order, process_slot_order, var_option_counts, process_option_counts


def _apply_combo(base_combo: "list[int]", slot_order: "list[int]", choice_indices: "tuple[int, ...]") -> "list[int]":
    """
    Overlays a block's own chosen candidate indices onto a full-length combo, leaving every other
    slot untouched.

    Parameters
    ----------
    base_combo : list[int]
        A full-length (one entry per slot in the whole model) list of chosen option indices to
        start from - typically the winning combo built up so far by earlier blocks, with every
        not-yet-decided slot still at its default of `0`.
    slot_order : list[int]
        The (sorted) global slot indices this block owns - as returned by `_block_option_counts`.
    choice_indices : tuple[int, ...]
        One chosen option index per entry of `slot_order`, in the same order.

    Returns
    -------
    list[int]
        A copy of `base_combo` with `base_combo[slot_order[k]]` set to `choice_indices[k]` for
        every `k` - every slot outside `slot_order` is left exactly as it was in `base_combo`.
    """
    combo = list(base_combo)
    for slot_index, choice_index in zip(slot_order, choice_indices):
        combo[slot_index] = choice_index
    return combo


def estimate_model_decomposed(
    incomplete_model: Model,
    t_eval,
    engine: Literal["torch", "scipy"] = "torch",
    device=None,
    collocation_times=None,
    max_iter_loss: int = 100,
    verbose: int = 0,
    max_gp_iter: int = 100,
    max_gp_points: int = 300,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: "int | None" = 1000,
) -> FullEstimationResults:
    """
    Estimates an incomplete model's best structure and constants by solving each independent
    block's own (much smaller) structural search separately, via gradient matching - see the
    module docstring for the reasoning and its cost trade-off, and `pybm.model_graph.build_blocks`
    for how blocks are found.

    Parameters
    ----------
    incomplete_model : Model
        The incomplete model to estimate. Every variable that should be fit against data must
        already have that data attached (`incomplete_model.vars[name].set_data(...)`) before
        calling this - matching `estimate_model`'s own precondition for `recepie="gp"`.
    t_eval : array-like
        Time points to fit against - forwarded to `estimate_gradient_matching` and to the final
        trajectory-loss check (`_singleshooting_loss`).
    engine : "torch" | "scipy", optional
        Compute engine for every induced candidate model. `estimate_gradient_matching` itself
        requires `"torch"`; use `"scipy"` only if `t_eval`'s trajectory-loss check is all you need
        `engine` for in some other context. Default `"torch"`.
    device : torch.device | str, optional
        Where `engine="torch"` computation runs - see `InducedModel.switch_engine`. Default: CPU.
    collocation_times : array-like, optional
        Where each candidate's GP-implied state/derivative are evaluated for the constant fit -
        see `estimate_gradient_matching`. Defaults to `t_eval`.
    max_iter_loss : int, optional
        Passed to the final `_singleshooting_loss` call (the same trajectory-MSE ranking metric
        `estimate_model` reports as `best_loss`). Default 100.
    verbose : int, optional
        `0` for silent; `>=1` also shows a `tqdm` progress bar over each block's own candidates and
        prints one summary line per block once that block's own search is done. Forwarded to
        `fit_gps`/`estimate_gradient_matching` for their own verbosity too.
    max_gp_iter, max_gp_points : int, optional
        Passed to `fit_gps` - see `pybm.estimate.gradient_matching.fit_gps`. The GP fit itself is
        computed once, up front, shared by every block and every candidate within it - it depends
        only on each variable's own observed data, never on model structure.
    ftol, xtol, gtol, max_nfev : optional
        Passed to every candidate's `estimate_gradient_matching` call (its
        `scipy.optimize.least_squares` constant fit) - see that function's docstring.

    Returns
    -------
    FullEstimationResults
        `best_consts`/`best_const_by_name`: the winning constants, assembled from every block's own
        winning fit. `best_loss`: the *trajectory* loss (`_singleshooting_loss`) of the final,
        fully-assembled model - computed once, at the end, not per candidate, since (unlike
        gradient matching's own objective) it integrates the ODE and so is not block-separable.
        `all_results`: one `ParamEstimationResults` per block (not per raw candidate) - each
        holding that block's own winning constants and its own gradient-matching cost as `.loss`.
    """
    blocks = build_blocks(incomplete_model)
    if not blocks:
        raise ValueError("incomplete_model has no differential (state) variables to estimate.")

    var_choices = incomplete_model._open_var_choices()
    process_slot_names = list(incomplete_model.pending_process_choices.keys())

    vars_with_data = {name: var for name, var in incomplete_model.vars.items() if var.data is not None}
    gps = fit_gps(vars_with_data, max_gp_points=max_gp_points, max_gp_iter=max_gp_iter, verbose=verbose)

    # The combo of chosen option indices for every slot in the whole model, built up one block at
    # a time. Any slot not owned by any block (never read by any state variable's equation) simply
    # keeps its default of 0 - it has no effect on any block's fit either way.
    winning_var_combo = [0] * len(var_choices)
    winning_process_combo = [0] * len(process_slot_names)
    winning_const_by_name: "dict[str, float]" = {}
    block_results: "list[ParamEstimationResults]" = []

    for block in blocks:
        var_slot_order, process_slot_order, var_option_counts, process_option_counts = _block_option_counts(
            incomplete_model, block, var_choices, process_slot_names
        )
        var_choice_combos = list(product(*(range(n) for n in var_option_counts))) if var_option_counts else [()]
        process_choice_combos = (
            list(product(*(range(n) for n in process_option_counts))) if process_option_counts else [()]
        )

        best_cost = None
        best_result = None
        best_var_choice_indices: "tuple[int, ...]" = ()
        best_process_choice_indices: "tuple[int, ...]" = ()

        block_desc = f"[decompose] block ({', '.join(var.name for var in block.state_vars)})"
        block_candidates = list(product(var_choice_combos, process_choice_combos))
        for var_choice_indices, process_choice_indices in tqdm(
            block_candidates, desc=block_desc, disable=verbose == 0
        ):
            full_var_combo = _apply_combo(winning_var_combo, var_slot_order, var_choice_indices)
            full_process_combo = _apply_combo(winning_process_combo, process_slot_order, process_choice_indices)

            candidate = incomplete_model._resolve(
                var_choices, tuple(full_var_combo), process_slot_names, tuple(full_process_combo)
            )
            _switch_engine(candidate, engine, device)

            result = estimate_gradient_matching(
                candidate, t_eval, collocation_times=collocation_times, gps=gps, device=device,
                verbose=verbose, ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
            )
            cost = float(result.least_squares_result.cost)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_result = result
                best_var_choice_indices = var_choice_indices
                best_process_choice_indices = process_choice_indices

        winning_var_combo = _apply_combo(winning_var_combo, var_slot_order, best_var_choice_indices)
        winning_process_combo = _apply_combo(winning_process_combo, process_slot_order, best_process_choice_indices)
        winning_const_by_name.update(best_result.const_by_name)
        block_results.append(
            ParamEstimationResults(
                model=best_result.model, consts=best_result.consts,
                const_by_name=best_result.const_by_name, loss=best_cost,
            )
        )
        if verbose:
            var_names = ", ".join(var.name for var in block.state_vars)
            print(f"[decompose] block ({var_names}): {len(var_choice_combos) * len(process_choice_combos)} "
                  f"candidate(s), best cost={best_cost:.4g}")

    if verbose:
        print(f"[decompose] final model: {len(winning_var_combo)} var slots, "
              f"{len(winning_process_combo)} process slots, "
              f"{len(winning_const_by_name)} constants")
    final_model = incomplete_model._resolve(
        var_choices, tuple(winning_var_combo), process_slot_names, tuple(winning_process_combo)
    )
    _switch_engine(final_model, engine, device)

    # Assemble the final constants array from every block's own winning fit, by name (each block
    # only ever contributed values for its own constants, so there is no risk of one block
    # overwriting another's) - any constant belonging to no block at all (an orphan slot, pinned
    # to its first option, never read by any state variable) falls back to its own template
    # default, if it has one.
    final_consts = np.zeros(len(final_model.consts))
    for name, const in tqdm(final_model.consts.items(), desc="Assembling final constants", disable=verbose == 0):
        if name in winning_const_by_name:
            final_consts[const.index_in_ctx] = winning_const_by_name[name]
        elif const.initial_value is not None:
            final_consts[const.index_in_ctx] = const.initial_value

    if verbose:
        print(f"Integrating final model..")
    trajectory_loss = _singleshooting_loss(
        final_model, t_eval, final_consts, method=engine, max_iter=max_iter_loss, device=device, verbose=verbose
    )

    return FullEstimationResults(
        best_consts=final_consts,
        best_const_by_name=winning_const_by_name,
        best_loss=trajectory_loss,
        all_results=block_results,
        best_model=final_model,
    )
