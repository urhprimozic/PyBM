# Proccess-based modeling formalisation in Python.
# Types are used, but not enforced.
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, List, Literal
from unicodedata import name
import warnings

VarType = Literal["endo", "exo", "not_set"]
Aggregation = Literal["sum", "product", "mean", "max", "min"]


class Model:
    """
    Represents different model configuration. Each model has its own set of variables and constants. 

    Entities can communitate with a model. 
    """

    def __init__(self, *args: "Entity | Var | Const"):
        self.entities : "dict[str, Entity]" = {}
        self.vars : "dict[str, Var]" = {}
        self.consts : "dict[str, Const]" = {}
        self.elements : "dict[str, Entity | Var | Const]" = {}

        # collect data from args
        for arg in args:
            if isinstance(arg, Entity):
                # store entity AND its variables and constants
                # check for name conflicts
                if arg.name in self.entities:
                    raise ValueError(f"Entity {arg.name} already exists in the model.")
                self.entities[arg.name] = arg
                if arg.name in self.elements:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
                self.elements[arg.name] = arg

                for key, value in arg._data.items():
                    new_name = f"{arg.name}.{key}"
                    if isinstance(value, Var):
                        if new_name in self.vars:
                            raise ValueError(f"Variable {new_name} already exists in the model.")
                        self.vars[new_name] = value
                        if new_name in self.elements:
                            raise ValueError(f"Element {new_name} already exists in the model.")
                        self.elements[new_name] = value
                    elif isinstance(value, Const):
                        if new_name in self.consts:
                            raise ValueError(f"Constant {new_name} already exists in the model.")
                        self.consts[new_name] = value
                        if new_name in self.elements:
                            raise ValueError(f"Element {new_name} already exists in the model.")
                        self.elements[new_name] = value
                    else:
                        raise ValueError(f"Entity {arg.name} contains an invalid model component: {value}")
                    
            elif isinstance(arg, Var):
                if arg.name in self.vars:
                    raise ValueError(f"Variable {arg.name} already exists in the model.")
                self.vars[arg.name] = arg
                if arg.name in self.elements:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
                self.elements[arg.name] = arg
            elif isinstance(arg, Const):
                if arg.name in self.consts:
                    raise ValueError(f"Constant {arg.name} already exists in the model.")
                self.consts[arg.name] = arg
                if arg.name in self.elements:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
                self.elements[arg.name] = arg
            else:
                raise ValueError(f"Argument {arg} is not a valid model component.")

EMPTY_MODEL = Model()

        

class DataContainer:
    """ 
    Data container with additional attributes. This is a base class for Var and Const.
    """

    def __init__(self, data: Any | None = None):
        self.data = data


@dataclass
class Var(DataContainer):
    """
    Represents a variable in the model.
    It has a name and a type, which can be either 'endo' (endogenous) or 'exo' (exogenous).

    Endogenous variables are given by a formula (either ODE or algebraic). A model for an exogenus variable is not provided.
    """

    name: str
    type: VarType = "not_set"
    initial: float | None = None
    range: tuple[float, float] | None = None
    aggregation: Aggregation | None = None
    unit: str | None = None
    ode: Callable[[Any], Any] | "Choose" | None = None
    algebraic: Callable[[Any], Any] | "Choose" | None = None

    def __post_init__(self):
        # add data
        super().__init__()

    def __str__(self):
        return self.name

    def copy(self):
        """
        Returns a copy of the variable.
        """
        return Var(
            name=self.name,
            type=self.type,
            initial=self.initial,
            range=self.range,
            aggregation=self.aggregation,
            unit=self.unit,
            ode=self.ode,
            algebraic=self.algebraic,
        )

    def unpack(self):
        """
        Returns a tuple of all possible realisations of equations.
        """
        raise NotImplementedError("TODO!!! Use Model class!")
        # check if ode and algebraic are both set
        if self.ode is not None and self.algebraic is not None:
            raise ValueError(
                f"Variable {self.name} has both an ODE and an algebraic equation. Only one can be set."
            )
        # check if neither is set
        if self.ode is None and self.algebraic is None:
            return [self.copy()]

        if isinstance(self.ode, Choose):
            ans = []
            for option in self.ode.options:
                var_copy = self.copy()
                var_copy.ode = option
                ans.append(var_copy)
            return ans
        if isinstance(self.algebraic, Choose):
            ans = []
            for option in self.algebraic.options:
                var_copy = self.copy()
                var_copy.algebraic = option
                ans.append(var_copy)
            return ans
        # if neither is a Choose, return the variable as is
        return [self.copy()]

@dataclass(frozen=True)
class VarTemp:
    """
    Variable specification DSL. Used to define a variable type in the entity template.
    """

    name: str
    type: VarType = "not_set"
    initial: float = 0.0
    range: tuple[float, float] | None = None
    aggregation: Aggregation | None = None
    unit: str | None = None

    def create(self):
        return Var(
            name=self.name,
            type=self.type,
            initial=self.initial,
            range=self.range,
            aggregation=self.aggregation,
            unit=self.unit,
        )


@dataclass
class Const(DataContainer):
    """
    Represents a constant parameter in the model.

    """

    name: str
    value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None

    # add data
    def __post_init__(self):
        super().__init__()

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

    def __init__(
        self,
        *args: Var | Const | Any,
        name: str | None = None,
        template: "EntityTemp | None" = None,
        active_model: "Model | None" = None
    ):
        """ """

        # self _data should be DEPRECATED!!!
        self._data: dict[str, Var | Const] = {}
        self._vars : set[str] = set()
        self._consts : set[str] = set()
        self.template: "EntityTemp | None" = template

        if name is None:
            warnings.warn("Entity name is not set. Default name was used. In the future, this will raise an error.")
            name = f"Entity_{id(self)}"
        self.name: str = name

        # set actrive model - container of vars and consts. 
        if active_model is None:    
            active_model = EMPTY_MODEL
        self.active_model: "Model" = active_model


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
        """
        Returns the value of the variable or constant with the given name.
        """
        # get variable data  from the model 
        var_or_const = self.active_model.elements.get(name)
        if not isinstance(var_or_const, (Var, Const)):
            raise KeyError(f"{name} is not a Var or Const in the active model.")
        return var_or_const.data

    def get(self, name):
        """
        Returns the variable or constant with the given name.
        """
        return self.active_model.elements.get(name)

    def __setitem__(self, name, value):
        """
        Sets the value of the variable or constant with the given name.
        Fails if the name is not a variable or constant in the entity.

        This metod is used to directly override the data of the variable/constant. To set a new variable or constant under a certain name, use the 'set' method.
        """
        raise NotImplementedError("This method is not implemented yet. Use the 'set' method instead.")

    def set(self, name, value: Var | Const | Any):
        """
        Replaces the variable or constant with the given name.
        """
        raise NotImplementedError("This method is not implemented yet. Use the 'add' method instead.")

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


class EntityTemp:
    """
    Entity template.
    """

    def __init__(self, *args: VarTemp | ConstTemp, parent: EntityTemp | None = None):
        self._data = args
        self.parent = parent

    def __call__(self, name: str | None = None):
        """
        Creates an Entity instance from the template.
        """
        # generate new variables and constants from the template
        generated_data = [obj.create() for obj in self._data]
        # pack them into a new Entity. Remember to set self as a parent template
        entity = Entity(*generated_data, name=name, template=self)
        return entity

    def variables(self):
        """
        Returns a list of variable names in the template.
        """
        return [obj.name for obj in self._data if isinstance(obj, VarTemp)]

    def constants(self):
        """
        Returns a list of constant names in the template.
        """
        return [obj.name for obj in self._data if isinstance(obj, ConstTemp)]

    def params(self):
        """
        Returns a list of parameter names in the template.
        """
        return [obj.name for obj in self._data]

    def add(self, obj: VarTemp | ConstTemp):
        """
        Adds a variable or constant to the entity template.
        """
        if isinstance(obj, VarTemp) or isinstance(obj, ConstTemp):
            self._data += (obj,)
        else:
            raise ValueError(f"Argument {obj} is not a VarTemp or ConstTemp.")

    @staticmethod
    def inherits(entity_template: EntityTemp):
        """
        Creates an entity template from an existing template
        All the data is copied, but the template is not linked to the entity.
        """
        new_template = EntityTemp(*entity_template._data, parent=entity_template)
        return new_template


class Process:
    def __init(self, *args):
        raise NotImplementedError("Process class is not implemented yet.")


class ProcessTemplate:
    def __init__(
        self,
        odes: List[Callable[[Any], Any]] | None = None,
        aes: List[Callable[[Any], Any]] | None = None,
        processes: List[ProcessTemplate | Process] | None = None,
    ):
        """
        Creates a new process template with the given ODEs and algebraic equations.
        The ODEs and algebraic equations are functions that take the entity as an argument and return the time derivative of the variable or the value of the algebraic equation.
        """
        self.odes = odes or []
        self.aes = aes or []

        # collect data
        # collect EntityTemplates from this...

        # not implemented!
        raise NotImplementedError("ProcessTemplate class is not implemented yet.")


class Choose:
    """
    Represents a choice between different functions/processes.
    """

    def __init__(self, *args:  Callable[[Any], Any] | Any):
        self.options = args


def generate_models(
    *args: Var | Const | ProcessTemplate | Process | EntityTemp | Entity,
):
    """
    Generates a model from the given variables, constants and processes.
    """
    # type checking
    for arg in args:
        if not isinstance(
            arg, (Var, Const, ProcessTemplate, Process, EntityTemp, Entity)
        ):
            raise ValueError(f"Argument {arg} is not a valid model component.")

        # only vars and consts are allowed for now
        if not isinstance(arg, (Var, Const)):
            raise NotImplementedError(
                "ProcessTemplate and Process are not implemented yet."
            )

    models = []
    raise NotImplementedError("Model generation is not implemented yet.")


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
        ConstTemp("k", value=0.5),
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
