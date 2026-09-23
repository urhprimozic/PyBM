# full model estimation
import numpy as np
import torch
from tqdm import tqdm

from odestimate.gp.regressor import GP
from pybm.estimate.gradient_matching import gradient_matching
from pybm.model import InducedModel, _get_data_tensor


def _check_shared_state_vars(models: "list[InducedModel]") -> None:
    """
    Every candidate must have the SAME state variables, in the SAME order, to safely share one GP
    across all of them (see `_fit_shared_gp`'s own docstring for why this is normally true across
    a structural search - state/algebraic/exogenous ROLE is a property of the entity/variable
    declaration itself, only the RHS *equation* choice varies). Raises with a specific message
    naming which candidate differs, rather than silently misaligning one candidate's state vector
    against another's interpolant (columns would line up by POSITION, not name).
    """
    def state_var_names(model: InducedModel) -> "list[str]":
        qualified = {id(v): name for name, v in model.vars.items()}
        state_vars, _, _ = model.split_endo_vars()
        return [qualified[id(v)] for v in state_vars]

    reference = state_var_names(models[0])
    for i, model in enumerate(models[1:], start=1):
        current = state_var_names(model)
        if current != reference:
            raise ValueError(
                f"Candidate model {i} has different state variables than model 0 "
                f"({current} vs {reference}) - can't share one GP across candidates with "
                f"different state-variable sets or ordering. Fit them separately instead."
            )


def _fit_shared_gp(model: InducedModel, t_eval, n_points: int, device: "torch.device | None" = None) -> GP:
    """
    Fits ONE `odestimate` GP, shared across every candidate that has the SAME state variables (in
    the same order) as `model` - hoisted OUT of `estimate_models`' own loop, because gradient
    matching's interpolant never depends on which candidate RHS is chosen, only on each state
    variable's own observed data (see the chat this followed).

    Evenly-SPACED index subsampling down to `n_points` (not random - same convention PyBM's old
    `_fit_gp_1d` used) keeps the GP FIT ITSELF cheap, not just the later least-squares step - the
    whole point of using a small `n_samples_1` for the first, cheap screening pass (a smaller
    `uniform_trust_region(n_samples=...)` alone would still leave the GP fit itself, an O(n^3)
    Cholesky factorization over the FULL `t_eval`, as the dominant cost).
    """
    device = device or torch.device("cpu")
    model.switch_engine("torch", device=device)
    t_eval = np.asarray(t_eval, dtype=float)
    state_vars, _, _ = model.split_endo_vars()

    idx = np.linspace(0, len(t_eval) - 1, min(n_points, len(t_eval))).astype(int)
    t_sub = t_eval[idx]
    x_obs = _get_data_tensor(state_vars, t_sub, device, torch.float64).cpu().numpy()  # (n_vars, n_points)
    return GP(t_sub, x_obs)


def _prune_by_trust_region(results: list) -> list:
    """
    Keeps only candidates whose confidence interval's LOWER bound doesn't exceed the best ACHIEVED
    upper bound among all candidates - i.e. drops a candidate only once its best-case performance
    is provably worse than some other candidate's worst-case (no interval overlap). See
    `notes/gradient-matching-pruning.md` SS5 (PyBM repo) for the derivation. Always keeps at least
    one candidate (whichever achieves the best upper bound survives trivially, against itself).
    """
    best_upper = min(r.interval[1] for r in results)
    return [r for r in results if r.interval[0] <= best_upper]


def estimate_models(models, t_eval, n_samples, gp: GP, **kwargs):
    """
    Estimate every candidate in `models` using gradient matching against the SAME, already-fit
    `gp` (see `_fit_shared_gp`) - only the per-candidate right-hand side changes; the interpolant
    doesn't.
    """
    results = []
    for index, model in tqdm(enumerate(models), desc="Estimating models", total=len(models)):
        results.append(gradient_matching(model, t_eval, gp=gp, n_samples=n_samples, **kwargs))
    # find best results
    best_result = min(results, key=lambda r: r.cost)
    return { "best_result": best_result , "all_results": results }

def two_step_estimation(incomplete_model, t_eval, n_samples_1, n_samples_2, **kwargs):
    """
    Two-step estimation over every structural candidate `incomplete_model.induce()` produces - see
    `notes/gradient-matching-pruning.md` (PyBM repo) for the full derivation.

    Step 1 (cheap screen): fit ONE shared GP on only `n_samples_1` (evenly-subsampled) points - a
    small `n_samples_1` keeps the GP FIT ITSELF cheap, not just the later least-squares step (see
    `_fit_shared_gp`). Run gradient matching + a `uniform_trust_region` confidence interval for
    every candidate against that SAME shared interpolant, then drop every candidate whose
    best-case loss is still provably worse than the best ACHIEVED worst-case (`_prune_by_trust_
    region`) - a safe (conservative) elimination, not a heuristic cutoff.

    Step 2 (precise re-fit): fit a SECOND, more precise shared GP on `n_samples_2` (> n_samples_1)
    points, and re-run gradient matching - now only on the survivors.

    Parameters
    ----------
    incomplete_model : Model
        Not yet induced - every open `Choose()`/`ChooseProcess()` slot still unresolved.
    t_eval : array-like
        The full observed time grid.
    n_samples_1, n_samples_2 : int
        Point budgets for step 1 (cheap screen, e.g. 50-100) and step 2 (precise re-fit, larger) -
        each used for BOTH the shared GP's own fit (subsampled from `t_eval`, `_fit_shared_gp`)
        and `uniform_trust_region`'s own randomly-sampled test points at that stage.
    **kwargs :
        Forwarded to `gradient_matching` (and from there to `uniform_trust_region`, e.g.
        `p_value`).

    Returns
    -------
    dict
        `"best_result"` (step 2's winner), `"step1_results"`, `"step2_results"` (every
        candidate's own result at each stage - survivors only for step 2), `"n_pruned"` (how many
        candidates step 1 eliminated).
    """
    models = incomplete_model.induce()
    _check_shared_state_vars(models)

    # Step 1: cheap screen, against a cheap (few-point) shared interpolant.
    gp_1 = _fit_shared_gp(models[0], t_eval, n_samples_1)
    step1 = estimate_models(models, t_eval, n_samples_1, gp_1, **kwargs)
    survivors = _prune_by_trust_region(step1["all_results"])

    # Step 2: precise re-fit, against a more precise shared interpolant - survivors only.
    survivor_models = [r.model for r in survivors]
    gp_2 = _fit_shared_gp(survivor_models[0], t_eval, n_samples_2)
    step2 = estimate_models(survivor_models, t_eval, n_samples_2, gp_2, **kwargs)

    return {
        "best_result": step2["best_result"],
        "step1_results": step1["all_results"],
        "step2_results": step2["all_results"],
        "n_pruned": len(models) - len(survivors),
    } 