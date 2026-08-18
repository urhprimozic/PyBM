from typing import Any

from attr import dataclass

from pybm.model import InducedModel
import numpy as np
import torch

@dataclass
class ParamEstimationResults:
    """
    Results of parameter estimation for a given model. Contains index with optimal constants 
    and dictionary with constant values by name. 
    Optionally, it can also contain the loss value of the estimation.
    """
    model: InducedModel
    consts: (
        np.ndarray | torch.Tensor | Any
    )  # (n_consts,), ordered by const.index_in_ctx
    const_by_name: dict[str, float]
    loss : float  | np.ndarray | torch.Tensor | None = None
