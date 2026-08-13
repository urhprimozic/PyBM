# convert ProBMoT libary to a Python module
import os
from pybm.convert.probmot import load_library, load_model

def get_library():
    """
    Load the ProBMoT library and model for the Bled example.

    Returns:
        library: The loaded ProBMoT library.
        model: The loaded ProBMoT model.
    """
    # Get the current working directory
    cwd = os.getcwd()
    
    # Load the ProBMoT library and model
    library = load_library(os.path.join(cwd, "probmot", "AquaticEcosystem.pbl"))
    model = load_model(os.path.join(cwd, "probmot", "BledComplete.pbm"), library)
    
    return library, model
# library = load_library("../probmot/AquaticEcosystem.pbl")
# model = load_model("../probmot/BledComplete.pbm", library)



