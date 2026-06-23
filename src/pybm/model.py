# Proccess-based modeling formalisation in Python.
# Types are used, but not enforced.
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Literal
import warnings

VarType = Literal["endo", "exo", "not_set"]
Aggregation = Literal["sum", "product", "mean", "max", "min"]


@dataclass
class Var:
    """
    Represents a variable in the model.
    It has a name and a type, which can be either 'endo' (endogenous) or 'exo' (exogenous).

    Endogenous variables are given by a formula (either ODE or algebraic). A model for an exogenus variable is not provided.
    """

    name: str
    type: VarType = "not_set"
    initial: float | None = None
    value: float | Any = None
    range: tuple[float, float] | None = None
    aggregation: Aggregation | None = None
    unit: str | None = None

    def __str__(self):
        return self.name

@dataclass(frozen=True)
class VarTemp:
    """
    Variable specification DSL. Used to define a variable type in the entity template.
    """
    name: str
    type: VarType = "not_set"
    initial: float = 0.0
    range: tuple[float, float] | None = None
    aggregation : Aggregation | None = None
    unit: str | None = None

    def create(self):
        return Var(name=self.name, type=self.type, initial=self.initial, range=self.range, aggregation=self.aggregation, unit=self.unit)


@dataclass
class Const:
    """
    Represents a constant parameter in the model.

    """

    name: str
    value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None

    def __str__(self):
        return self.name

@dataclass(frozen=True)
class ConstTemp:
    """
    Constant parameter specification DSL. Used to define a constant type in the entity template.
    """
    name: str
    value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None

    def create(self):
        return Const(name=self.name, value=self.value, range=self.range, unit=self.unit)


class Entity(MutableMapping):
    """
    Represents an entity in the model.
    An entity is a collection of variables and constants.

    This implementation also allows for other types of objects to be stored in the entity, if they have a name attribute.
    This is not recommended and may lead to unexpected behavior.

    Usage
    -----
    >>> v1 = Var("x")
    >>> v2 = Var("y")
    >>> c1 = Const("mass", value=0.5)
    >>> particle = Entity(v1, v2, c1)
    >>> particle["x"]
    Var(name='x', type='not_set', initial=None, value=None)
    """

    def __init__(self, *args : Var | Const | Any, name: str | None = None, template: "EntityTemp | None" = None):
        """ """
        self._data = {}
        self._vars = set()
        self._consts = set()
        self.name: str | None = name
        self.template: "EntityTemp | None" = template

        for arg in args:
            if isinstance(arg, Var):
                self._vars.add(arg.name)
                self._data[arg.name] = arg
            elif isinstance(arg, Const):
                self._consts.add(arg.name)
                self._data[arg.name] = arg
            else:
                warnings.warn(f"Argument {arg} is not a Var or Const.")
                try:
                    self._data[arg.name] = arg
                except AttributeError:
                    raise ValueError(f"Argument {arg} does not have a name attribute.")

    def __getitem__(self, name):
        # get by name
        return self._data.get(name)

    def __setitem__(self, name, value):
        self._data[name] = value

        if isinstance(value, Var):
            self._vars.add(name)
        elif isinstance(value, Const):
            self._consts.add(name)
        else:
            warnings.warn(f" {value} is not a Var or Const.")

    def __delitem__(self, name):
        del self._data[name]
        if name in self._vars:
            self._vars.remove(name)
        elif name in self._consts:
            self._consts.remove(name)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return f"Entity({', '.join(self._data.keys())})"
    def add(self, obj: Var | Const | Any):
        """
        Adds a variable or constant to the entity.
        """
        if isinstance(obj, Var):
            self._vars.add(obj.name)
            self._data[obj.name] = obj
        elif isinstance(obj, Const):
            self._consts.add(obj.name)
            self._data[obj.name] = obj
        else:
            warnings.warn(f"Argument {obj} is not a Var or Const.")
            try:
                self._data[obj.name] = obj
            except AttributeError:
                raise ValueError(f"Argument {obj} does not have a name attribute.")



class EntityTemp():
    """
    Entity template. 
    """
    def __init__(self, *args: VarTemp | ConstTemp):
        self._data = args 
    def __call__(self, name: str | None = None):
        """
        Creates an Entity instance from the template. 
        """
        # generate new variables and constants from the template
        generated_data = [obj.create() for obj in self._data]
        # pack them into a new Entity. Remember to set self as a parent template
        entity = Entity(*generated_data, name=name, template=self)
        return entity
    def add(self, obj: VarTemp | ConstTemp):
        """
        Adds a variable or constant to the entity template.
        """
        if isinstance(obj, VarTemp) or isinstance(obj, ConstTemp):
            self._data += (obj,)
        else:
            raise ValueError(f"Argument {obj} is not a VarTemp or ConstTemp.")
    




if __name__ == "__main__":
    # Example usage
    v1 = Var("x")
    v2 = Var("y")
    c1 = Const("mass", value=0.5)

    particle = Entity(v1, v2, c1)

    print(particle["x"])  # Output: Var(name='x', type='endo', initial=0.0, value=None)
    print(particle["y"])  # Output: Var(name='y', type='exo', initial=1.0, value=None)
    print(particle["k"])  # Output: Const(name='k', value=0.5)

    # Create an entity template
    particle_template = EntityTemp(
        VarTemp("x", type="endo", initial=0.0),
        VarTemp("y", type="exo", initial=1.0),
        ConstTemp("k", value=0.5)
    )
    # create one instance of the template
    neon_particle = particle_template(name="neon_particle")

    # create another and add a new variable to it
    higgs_boson = particle_template(name="higgs_boson")
    higgs_boson.add(Var("spin"))

    # every elementary particle has its own spin --> create a new template for elementary particles based on the old one 
    elementary_particle_template = particle_template
    elementary_particle_template.add(VarTemp("spin"))

    # now create a new instance of the elementary particle template - it will automaticly have the spin variable
    electron = elementary_particle_template(name="electron")