# Proccess-based modeling formalisation in Python.
# Types are used, but not enforced.
from collections import deque
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, List, Literal, TypedDict
from unicodedata import name
import warnings
import numpy as np

VarType = Literal["endo", "exo", "not_set"]
Aggregation = Literal["sum", "product", "mean", "max", "min"]


class Model:
    """
    Represents different model configuration. Each model has its own set of variables and constants.

    Entities can communitate with a model.
    """

    def __init__(self, *args: "Entity | Var | Const"):
        self.entities: "dict[str, Entity]" = {}
        self.vars: "dict[str, Var]" = {}
        self.consts: "dict[str, Const]" = {}
        self.elements: "dict[str, Entity | Var | Const]" = {}
        # parents - parent entities or None
        self.parents: "dict[str, str | None]" = {}
        # dict between endo names and index in ctx
        self.endo_index: "dict[str, int]" = {}
        # dict between const names and index in ctx
        self.const_index: "dict[str, int]" = {}
        self.context_size : int = 0

        # collect data from args
        for arg in args:
            self.add(arg)

    def add(self, arg: "Entity | Var | Const", ignore_conflicts: bool = False, set_as_active=True):
        # add directly - no parent
        self.parents[arg.name] = None
        if isinstance(arg, Entity):
            # store entity AND its variables and constants
            # check for name conflicts
            if arg.name in self.entities:
                if not ignore_conflicts:  # else just skip!
                    raise ValueError(f"Entity {arg.name} already exists in the model.")
            self.entities[arg.name] = arg
            if arg.name in self.elements:
                if not ignore_conflicts:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
            self.elements[arg.name] = arg
            # set as active model 
            if set_as_active:  
                arg.active_model = self
            for key, value in arg._data.items():
                new_name = f"{arg.name}.{key}"
                if isinstance(value, Var):
                    # check for name conflicts
                    if new_name in self.vars:
                        if not ignore_conflicts:
                            raise ValueError(
                                f"Variable {new_name} already exists in the model."
                            )
                    self.vars[new_name] = value
                    if new_name in self.elements:
                        if not ignore_conflicts:
                            raise ValueError(
                                f"Element {new_name} already exists in the model."
                            )
                    self.elements[new_name] = value
                    # set as active model
                    if set_as_active:
                        value.active_model = self

                    # assing a place in context for the variable
                    if value.type == "endo":
                        value.index_in_ctx = len(self.endo_index)
                        self.endo_index[new_name] = len(self.endo_index)
                elif isinstance(value, Const):
                    if new_name in self.consts:
                        if not ignore_conflicts:
                            raise ValueError(
                                f"Constant {new_name} already exists in the model."
                            )
                    self.consts[new_name] = value
                    if new_name in self.elements:
                        if not ignore_conflicts:
                            raise ValueError(
                                f"Element {new_name} already exists in the model."
                            )
                    self.elements[new_name] = value
                    # set as active model
                    if set_as_active:
                        value.active_model = self
                    value.index_in_ctx = len(self.const_index)
                    self.const_index[new_name] = len(self.const_index)
                    self.context_size += 1
                else:
                    raise ValueError(
                        f"Entity {arg.name} contains an invalid model component: {value}"
                    )
                # store  parent
                self.parents[new_name] = arg.name

        elif isinstance(arg, Var):
            if arg.name in self.vars:
                if not ignore_conflicts:
                    raise ValueError(
                        f"Variable {arg.name} already exists in the model."
                    )
            self.vars[arg.name] = arg
            if arg.name in self.elements:
                if not ignore_conflicts:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
            self.elements[arg.name] = arg
            # set as active model
            if set_as_active:
                arg.active_model = self
            if arg.type == "endo":
                arg.index_in_ctx = len(self.endo_index)
                self.endo_index[arg.name] = len(self.endo_index)
        elif isinstance(arg, Const):
            if arg.name in self.consts:
                if not ignore_conflicts:
                    raise ValueError(
                        f"Constant {arg.name} already exists in the model."
                    )
            self.consts[arg.name] = arg
            if arg.name in self.elements:
                if not ignore_conflicts:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
            self.elements[arg.name] = arg
            # set as active model
            if set_as_active:
                arg.active_model = self
            arg.index_in_ctx = len(self.const_index)
            self.const_index[arg.name] = len(self.const_index)
            self.context_size += 1
        else:
            raise ValueError(f"Argument {arg} is not a valid model component.")

    def copy(self):
        """
        Returns a copy of the model.
        """
        new_model = Model()

        # make sure to first add the entities! If you first add all the variables,
        # the entities will not be properly initialized?? TODO try
        for entity in self.entities.values():
            entity_copy = entity.copy(active_model=new_model)
            new_model.add(entity_copy, ignore_conflicts=True)
        for obj_name, obj in self.elements.items():
            if self.parents[obj_name] is not None:
                # this object is already added as part of an entity, skip it
                continue
            if isinstance(obj, Entity):
                continue
            else:
                obj_copy = obj.copy() if hasattr(obj, "copy") else obj
            new_model.add(obj_copy, ignore_conflicts=True)
        return new_model

    def induce(self):
        """
        Returns a list of all the possible models, induced from the current model.
        """
        # bfs through the model tree, looking for any choices. If a choice is found, it creates a new model for each option and adds it to the list of models to explore.
        # alternatively a dfs could be used

        models = deque([self.copy()])
        finished_models = []

        while models:
            current_model = models.popleft()

            # check for any choices
            finished = True
            for var_name, var in current_model.vars.items():
                if var.ode is not None:
                    attr = "ode"
                elif var.algebraic is not None:
                    attr = "algebraic"
                else:
                    continue
                expr = getattr(var, attr)

                if isinstance(expr, Choose):
                    # append new models
                    for option in expr.options:
                        new_model = current_model.copy()
                        setattr(new_model.vars[var_name], attr, option)
                        models.append(new_model)
                    # this model is not finished
                    finished = False
            # no new models were created, so this model has no more choices and is finished
            if finished:
                finished_models.append(current_model)
        return finished_models

    def __str__(self):
        return f"Model(Entities: {list(self.entities.keys())}, Vars: {list(self.vars.keys())}, Consts: {list(self.consts.keys())})"

    def __repr__(self):
        return f"Model(Entities: {list(self.entities.values())}, Vars: {list(self.vars.values())}, Consts: {list(self.consts.values())})"


    def set_as_active(self):
        """
        Sets this model as the active model for all its entities, variables and constants.
        """
        for entity in self.entities.values():
            entity.active_model = self
            for var in entity._vars:
                entity._data[var].active_model = self
            for const in entity._consts:
                entity._data[const].active_model = self
        for var in self.vars.values():
            var.active_model = self
        for const in self.consts.values():
            const.active_model = self
    def get_endo_variables(self):
        """
        Returns a list of all endogenous variables in the model.
        """
        return [var for var in self.vars.values() if var.type == "endo"]


EMPTY_MODEL = Model()


@dataclass
class EvaluationContext:
    """
    Evaluation context for the model.
    """

    t: float | None = None
    x: np.ndarray | None = None


class TimeSeries:
    def __init__(self, t: np.ndarray | Any, x: np.ndarray | Any):
        self.t = t
        self.x = x
        self.index = 0

    def interpolate(self, index, t):
        """
        Interpolates the value of the variable at the given index.
        """
        # out of bounds --> return the closest value
        if index < 0:
            return self.x[0]
        elif index >= len(self.t):
            return self.x[-1]
        
        # linear interpolation
        if index == len(self.t) - 1:
            return self.x[index]
        else:
            t0 = self.t[index]
            t1 = self.t[index + 1]
            x0 = self.x[index]
            x1 = self.x[index + 1]
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0)

    def bisect(self, t, i0, i1):
        """
        Finds the index of the closest time point to t in self.t[i0:i1] using bisection.
        Interpolates linearly between the two closest points.
        """
        self.index = np.searchsorted(self.t[i0:i1], t) + i0 - 1
        # linear interpolation
        return self.interpolate(self.index, t)

    def __call__(self, t, bisection=False, numpy=True):
        if numpy:
            return np.interp(t, self.t, self.x)
        else:
            if self.index == 0 and t < self.t[0]:
                warnings.warn(
                    f"Exogenous variable  is called with t={t} < {self.t[0]}. Returning the first value."
                )
                return self.x[0]
            if bisection:
                return self.bisect(t, 0, len(self.t) - 1)
            else:
                # if the index points on this time, do linear interpolation between the two points
                if self.t[self.index] <= t and t <= self.t[self.index + 1]:
                    # linear interpolation
                    return self.interpolate(self.index, t)
                elif t < self.t[self.index]:
                    # search the left side of the array with bisection
                    return self.bisect(t, 0, self.index)
                elif t > self.t[self.index + 1]:
                    # search the right side of the array with bisection
                    return self.bisect(t, self.index + 1, len(self.t) - 1)


class Context(TypedDict):
    vars : Any
    consts : Any

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
    range: tuple[float, float] | None = None
    aggregation: Aggregation | None = None
    unit: str | None = None
    ode: Callable[[Any, Any], Any] | "Choose" | None = None
    algebraic: Callable[[Any], Any] | "Choose" | None = None
    active_model: "Model | None" = None
    data: TimeSeries | None = None
    index_in_ctx: int | None = None

    def __post_init__(self):
        # add data
        super().__init__()

    def __str__(self):
        return self.name

    def get_exo(self, t):
        if self.type != "exo":
            raise ValueError(f"Variable {self.name} is not exogenous.")
        if self.data is None:
            raise ValueError(f"Variable {self.name} has no data.")
        return self.data(t)

    def __call__(self, t, ctx: Context):
        if self.active_model is None:
            raise ValueError("Active model is not set for variable.")

        if self.type == "not_set":
            raise ValueError(f"Variable {self.name} has no type set.")
        if self.type == "exo":
            # read from data
            return self.get_exo(t)
        if self.type == "endo":
            
            # compute - read from context
            index = self.index_in_ctx
            if index is None:
                raise ValueError(f"Variable {self.name} has no index in context.")
            
            # read from context 
            return ctx["vars"][index]

    def set_data(self, t, x):
        """
        Sets the data for the variable.
        """
        self.data = TimeSeries(t, x)

    def set_model(self, model: "Model"):
        self.active_model = model

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
            index_in_ctx=self.index_in_ctx,
        )

    def update(self, **kwargs):
        raise NotImplementedError()


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
class Const:
    """
    Represents a constant parameter in the model.

    """

    name: str
    initial_value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None
    active_model: "Model | None" = None
    index_in_ctx: int | None = None

    # add data
    def __post_init__(self):
        super().__init__()

    def set_model(self, model: "Model"):
        self.active_model = model

    def __call__(self, ctx: Context):
        if self.active_model is None:
            raise ValueError(f"Active model is not set for constant {self.name}.")
        index = self.index_in_ctx
        if index is None:
            raise ValueError(f"Constant {self.name} has no index in context.")
        return ctx["consts"][index]

    def __str__(self):
        return self.name

    def copy(self):
        """
        Returns a copy of the constant.
        """
        return Const(
            name=self.name,
            initial_value=self.initial_value,
            range=self.range,
            unit=self.unit,
            index_in_ctx=self.index_in_ctx,
        )


@dataclass(frozen=True)
class ConstTemp:
    """
    Constant parameter specification DSL. Used to define a constant type in the entity template.
    """

    name: str
    initial_value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None

    def create(self):
        return Const(name=self.name, initial_value=self.initial_value, range=self.range, unit=self.unit)


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
        active_model: "Model | None" = None,
    ):
        """ """

        # self _data should be DEPRECATED!!!
        self._data: dict[str, Var | Const] = {}
        self._vars: set[str] = set()
        self._consts: set[str] = set()
        self.template: "EntityTemp | None" = template

        if name is None:
            warnings.warn(
                "Entity name is not set. Default name was used. In the future, this will raise an error."
            )
            name = f"Entity_{id(self)}"
        self.name: str = name

        # set actrive model - container of vars and consts.
        if active_model is None:
            active_model = Model()
        self.active_model: "Model" = active_model

        for arg in args:

            # add to entity
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

        # add the entity to the active model
        self.active_model.add(self)

    def parse_name(self, name: str):
        return self.name + "." + name

    def __getitem__(self, name):
        """
        Returns the  the variable or constant with the given name.
        """
        # get variable data  from the model
        var_or_const = self.active_model.elements.get(self.parse_name(name))
        if not isinstance(var_or_const, (Var, Const)):
            raise KeyError(
                f"{self.parse_name(name)} is not a Var or Const in the active model."
            )
        return var_or_const

    def add_ode(self, var_name: str, ode: Callable[[Any], Any]):
        """
        Adds an ODE to the variable with the given name.
        """
        var = self.get(var_name)
        if not isinstance(var, Var):
            raise KeyError(
                f"{self.parse_name(var_name)} is not a Var in the active model."
            )
        var.ode = ode

    def get(self, name):
        """
        Returns the variable or constant with the given name.
        """
        return self.active_model.elements.get(self.parse_name(name))

    def __setitem__(self, name, value):
        """
        Sets the value of the variable or constant with the given name.
        Fails if the name is not a variable or constant in the entity.

        This metod is used to directly override the data of the variable/constant. To set a new variable or constant under a certain name, use the 'set' method.
        """
        raise NotImplementedError("Deprecated. Usage of .data is deprecated")
        var_or_const = self.active_model.elements.get(self.parse_name(name))
        if not isinstance(var_or_const, (Var, Const)):
            raise KeyError(f"{name} is not a Var or Const in the active model.")
        var_or_const.data = value

    def set(self, name, value: Var | Const | Any):
        """
        Replaces the variable or constant with the given name.
        """
        raise NotImplementedError("TODO")

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

    def copy(self, active_model: Model | None = None):
        """
        Returns a copy of the entity.
        """
        if active_model is None:
            active_model = self.active_model

        args = [
            obj.copy() if hasattr(obj, "copy") else obj for obj in self._data.values()
        ]
        new_entity = Entity(
            *args, name=self.name, template=self.template, active_model=active_model
        )
        return new_entity


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
    def __init__(self, *args):
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

    def __init__(self, *args: Callable[[Any], Any] | Any):
        self.options = args

