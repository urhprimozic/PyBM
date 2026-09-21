# test different methods of estimation and compare results
from attr import dataclass
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional

from pybm.estimate.results import ParamEstimationResults
from pybm.model import Model


Dataset = pd.DataFrame

@dataclass
class CVResults:
    """
    Class to store the results of cross-validation for a single estimation method.
    """
    best_method: str
    results : Dict[str, ParamEstimationResults]

def test_cv(
    datasets: List[Dataset],
    set_data: Callable[[pd.DataFrame, Model], None],
    model_factory: Callable[[], Model],
    estimation_methods: Dict[str, Callable],
    get_t_eval: Callable[[pd.DataFrame], np.ndarray],
    kwargs: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> CVResults:
    """
    Tests different estimation methods with cross-validation on a list of datasets.

    Runs len(datasets)-fold cross-validation. Each time, one dataset is used for testing and the
    rest for training.

    Parameters
    ------------
    datasets : List[Dataset]
        List of datasets to use for cross-validation. Each dataset should be a pandas DataFrame.
    set_data : Callable[[pd.DataFrame, Model], None]
        Function to set the data for the model. set_data(df, model) should set the data of the
        model to the given DataFrame.
    model_factory : Callable[[], Model]
        Builds a FRESH, still-incomplete model instance (no data attached yet) - called once per
        fold, so no state (attached data, induced structure, ...) leaks between folds.
    estimation_methods : Dict[str, Callable]
        Dictionary of estimation methods to test. Each must return a `ParamEstimationResults`
        (or subclass) with `.loss` already set to a metric that's comparable ACROSS methods (e.g.
        a real trajectory loss - the different methods' own internal fit costs, like a
        gradient-matching residual vs. a least-squares cost, are not directly comparable to each
        other, only within their own method).
    get_t_eval : Callable[[pd.DataFrame], np.ndarray]
        Derives the time points to fit/evaluate against from a fold's own training data (e.g. its
        "t" column) - computed per fold since different folds train on different data.
    kwargs : Dict[str, Dict], optional
        Keyword arguments for each estimation method, keyed by the same name used in
        `estimation_methods`. A method `e` named `name` is called as
        `e(model=current_model, t_eval=t_eval, **kwargs.get(name, {}))`.
    verbose : bool
        Whether to print verbose output.

    Returns
    ---------
    CVResults
        Results of the cross-validation. Keys:
        - best_method : str
            The name of the best estimation method (lowest average `.loss` across folds).
        - results : Dict[str, ParamEstimationResults]
            Dictionary of results for each estimation method, averaged across folds. Each value is
            a ParamEstimationResults object.
    """
    kwargs = kwargs or {}
    results = {}
    for method_name, method in estimation_methods.items():
        if verbose:
            print(f"Testing method: {method_name}")
        method_results = []
        for i, test_dataset in enumerate(datasets):
            # Prepare training data by excluding the test dataset
            train_datasets = datasets[:i] + datasets[i+1:]
            train_data = pd.concat(train_datasets, ignore_index=True)

            # Initialize model and set training data
            current_model = model_factory()
            set_data(train_data, current_model)
            t_eval = get_t_eval(train_data)

            # Estimate parameters using the current method
            estimation_result = method(model=current_model, t_eval=t_eval, **kwargs.get(method_name, {}))
            method_results.append(estimation_result)

        # Aggregate results for the current method
        losses = [res.loss for res in method_results if res.loss is not None]
        if len(losses) != len(method_results):
            raise ValueError(
                f"Method {method_name!r} returned a result with loss=None on at least one fold - "
                "every estimation_methods callable must set .loss to a cross-method-comparable metric."
            )
        avg_loss = np.mean(losses)
        results[method_name] = ParamEstimationResults(
            model=current_model,
            consts=np.mean([res.consts for res in method_results], axis=0),
            const_by_name={name: np.mean([res.const_by_name[name] for res in method_results]) for name in method_results[0].const_by_name},
            loss=avg_loss
        )

    # Determine the best method based on average loss
    best_method = min(results.keys(), key=lambda k: results[k].loss)

    return CVResults(best_method=best_method, results=results)
