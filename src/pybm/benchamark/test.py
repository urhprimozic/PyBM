# test different methods of estimation and compare results
from attr import dataclass
import numpy as np
import pandas as pd
from typing import Callable, Dict, List

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
   
def test_cv(datasets : List[Dataset],set_data : Callable[[pd.DataFrame, Model], None], model=Model, estimation_methods : Dict[str, Callable], kwargs = Dict[str, Dict],verbose : bool = True) -> CVResults:
    """
    Tests different estimation methods with cross-validation on a list of datasets. 

    Runs len(datasets) -fold cross-validation. Each time, one dataset is used for testing and the rest for training. 

    Parameters
    ------------
    datasets : List[Dataset]
        List of datasets to use for cross-validation. Each dataset should be a pandas DataFrame.
    set_data : Callable[[pd.DataFrame, Model], None]
        Function to set the data for the model. set_data(df, model) should set the data of the model to the given DataFrame.
    model : Model
        The model to use for estimation.
    estimation_methods : Dict[str, Callable]
        Dictionary of estimation methods to test.
    kwargs : Dict[str, Dict]
        Dictionary of keyword arguments for each estimation method. A method e will be called as e(model=model, t_eval=t_eval, kwargs[e.__name__])
    verbose : bool
        Whether to print verbose output.
    
    Returns
    ---------
    CVResults
        Results of the cross-validation. Keys:
        - best_method : str
            The name of the best estimation method.
        - results : Dict[str, ParamEstimationResults]
            Dictionary of results for each estimation method. Each value is a ParamEstimationResults object.
    """
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
            current_model = model()
            set_data(train_data, current_model)
            
            # Estimate parameters using the current method
            estimation_result = method(model=current_model, **kwargs.get(method_name, {}))
            method_results.append(estimation_result)
        
        # Aggregate results for the current method
        avg_loss = np.mean([res.loss for res in method_results if res.loss is not None])
        results[method_name] = ParamEstimationResults(
            model=current_model,
            consts=np.mean([res.consts for res in method_results], axis=0),
            const_by_name={name: np.mean([res.const_by_name[name] for res in method_results]) for name in method_results[0].const_by_name},
            loss=avg_loss
        )
    
    # Determine the best method based on average loss
    best_method = min(results.keys(), key=lambda k: results[k].loss if results[k].loss is not None else float('inf'))
    
    return CVResults(best_method=best_method, results=results)
