# Proccess-based modeling formalisation in Python. 
# Types are used, but not enforced. 
from dataclasses import dataclass
from typing import Literal
import warnings

@dataclass
class Var:
    """
    Represents a variable in the model. 
    It has a name and a type, which can be either 'endo' (endogenous) or 'exo' (exogenous).

    Endogenous variables are given by a formula (either ODE or algebraic). A model for an exogenus variable is not provided. 
    """
    name: str | None = None
    type : Literal['endo', 'exo', 'not_set'] = 'not_set'


    def __str__(self):
        return self.name

@dataclass
class Const:
    """
    Represents a constant parameter in the model. 
    
    """
    name: str
    value: float | None = None

    def __str__(self):
        return self.name

class Entity:
    """
    Represents an entity in the model. 
    An entity is a collection of variables and constants. 
    """
    def __init__(self, *args, name: str | None = None):
        """
        Parameters
        ----------
        *args : Var or Const
             A variable or constant to be added to the entity.
        name : str | None
             The name of the entity.
        """
        # collect variables and constants
        self.vars: list[Var] = []
        self.consts: list[Const] = []
        for arg in args:
            if isinstance(arg, Var):
                self.vars.append(arg)
            elif isinstance(arg, Const):
                self.consts.append(arg)
            else:
                warnings.warn(f"Argument {arg} is not a Var or Const")
        
        # set name
        self.name = name if name is not None else f"Entity_{id(self)}"

    def add_var(self, var: Var):
        self.vars.append(var)

    def add_const(self, const: Const):
        self.consts.append(const)

    def __str__(self):
        return self.name

@dataclass
class VarTemplate:
    aggregation : Literal['sum', 'mean', 'max', 'min']  = 'sum'
    unit : str | None = None
    range : tuple[float, float] | None = None
    
    def __call__(self, name: str | None = None):
        """
        Create a Var instance from the template.
        """
        return Var(name=name)


class EntityTemplate:
    def __init__(self, *args, name: str | None = None):
        pass