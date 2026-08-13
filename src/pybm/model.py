# Proccess-based modeling formalisation in Python.
# Types are used, but not enforced.
from __future__ import annotations
from collections import deque
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, List, Literal, TypedDict
from unicodedata import name
import warnings
import numpy as np
import torch
from pybm.utils import torch_interp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybm.templates import EntityTemp, ProcessTemplate

VarType = Literal["endo", "exo", "not_set"]
Aggregation = Literal["sum", "product", "mean", "max", "min"]


class Model:
    """
    Represents different model configuration. Each model has its own set of variables and constants.

    Entities can communitate with a model.
    """

    def __init__(
        self,
        *args: "Entity | Var | Const | Process | ChooseProcess",
        engine: Literal["scipy", "torch", "jax"] = "scipy",
    ):
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
        self.context_size: int = 0
        self.engine: Literal["scipy", "torch", "jax"] = engine
        # ChooseProcess slots added to the model but not yet resolved by induce() - see ChooseProcess
        self.pending_process_choices: "dict[str, ChooseProcess]" = {}

        # collect data from args
        for arg in args:
            self.add(arg)

    def add(
        self,
        arg: "Entity | Var | Const | Process | ChooseProcess",
        ignore_conflicts: bool = False,
        set_as_active=True,
    ):
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
        elif isinstance(arg, Process):
            if arg.name in self.elements:
                if not ignore_conflicts:
                    raise ValueError(f"Element {arg.name} already exists in the model.")
            self.elements[arg.name] = arg
            self._register_process(arg, arg.name, ignore_conflicts, set_as_active)
            # apply the process's equations to the affected variables (no-op if already compiled)
            arg.compile()
        elif isinstance(arg, ChooseProcess):
            # structure is still undetermined - don't register consts/compile yet, just remember the
            # slot. Model.induce() resolves it into a concrete Process later.
            if arg.name in self.pending_process_choices:
                if not ignore_conflicts:
                    raise ValueError(
                        f"ChooseProcess {arg.name} already exists in the model."
                    )
            self.pending_process_choices[arg.name] = arg
        else:
            raise ValueError(f"Argument {arg} is not a valid model component.")

    def _register_process(
        self,
        process: "Process",
        full_name: str,
        ignore_conflicts: bool,
        set_as_active: bool,
    ):
        """
        Registers a process's own local constants under `full_name.const_name`, then recurses into its
        sub-processes using `full_name.sub_process_name` as their own prefix.
        """
        for const_name, const in process.own_consts.items():
            new_name = f"{full_name}.{const_name}"
            if new_name in self.consts:
                if not ignore_conflicts:
                    raise ValueError(
                        f"Constant {new_name} already exists in the model."
                    )
            self.consts[new_name] = const
            if new_name in self.elements:
                if not ignore_conflicts:
                    raise ValueError(f"Element {new_name} already exists in the model.")
            self.elements[new_name] = const
            if set_as_active:
                const.active_model = self
            const.index_in_ctx = len(self.const_index)
            self.const_index[new_name] = len(self.const_index)
            self.context_size += 1
            self.parents[new_name] = full_name
        for sub_process in process.processes:
            self._register_process(
                sub_process,
                f"{full_name}.{sub_process.name}",
                ignore_conflicts,
                set_as_active,
            )

    def copy(self):
        """
        Returns a copy of the model.
        """
        new_model = Model()

        # make sure to first add the entities! If you first add all the variables,
        # the entities will not be properly initialized?? TODO try
        for entity in self.entities.values():
            # Entity.__init__ already self-registers into `active_model` (see its last line) - do
            # NOT also call new_model.add() here, that would register it a second time and silently
            # corrupt index_in_ctx for its vars/consts (add() reassigns the index unconditionally,
            # even when ignore_conflicts=True suppresses the "already exists" error).
            entity.copy(active_model=new_model)
        for obj_name, obj in self.elements.items():
            if self.parents[obj_name] is not None:
                # this object is already added as part of an entity, skip it
                continue
            if isinstance(obj, Entity):
                continue
            else:
                obj_copy = obj.copy() if hasattr(obj, "copy") else obj
            new_model.add(obj_copy, ignore_conflicts=True)
        for choose_process in self.pending_process_choices.values():
            new_model.add(choose_process.copy(), ignore_conflicts=True)
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

            # Resolve exactly ONE ambiguity per step (then requeue the branches for the next pass),
            # rather than every ambiguity found in this model at once: branching on N independent
            # choices simultaneously produces the same fully-resolved combination via multiple
            # different orderings, duplicating entries in finished_models. One-at-a-time avoids that.
            resolved_choice = False

            for var_name, var in current_model.vars.items():
                if var.ode is not None:
                    attr = "ode"
                elif var.algebraic is not None:
                    attr = "algebraic"
                else:
                    continue
                expr = getattr(var, attr)

                if isinstance(expr, Choose):
                    for option in expr.options:
                        new_model = current_model.copy()
                        setattr(new_model.vars[var_name], attr, option)
                        models.append(new_model)
                    resolved_choice = True
                    break

            if not resolved_choice:
                for slot_name, choose_process in current_model.pending_process_choices.items():
                    for factory in choose_process.options:
                        new_model = current_model.copy()
                        del new_model.pending_process_choices[slot_name]
                        # build the candidate fresh, against new_model's OWN (already-copied)
                        # entities - see ChooseProcess's docstring for why a pre-built Process can't
                        # be reused here.
                        resolved = factory(new_model)
                        # registered under the SLOT's name, not whatever name the factory gave it -
                        # the slot name is what's stable across candidates (e.g. matches a `.pbm`
                        # process instance name), regardless of which one wins.
                        resolved.name = slot_name
                        new_model.add(resolved)
                        models.append(new_model)
                    resolved_choice = True
                    break

            # no ambiguity left to resolve - this model is finished
            if not resolved_choice:
                finished_models.append(current_model)
        return finished_models

    def __str__(self):
        return f"Model(Entities: {list(self.entities.keys())},\n Vars: {list(self.vars.keys())},\n Consts: {list(self.consts.keys())})"

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

    def to_torch(self):
        """
        Converts the time series to torch tensors.
        """
        self.t = torch.tensor(self.t)
        self.x = torch.tensor(self.x)

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
        if torch.is_tensor(t):
            return torch_interp(t, self.t, self.x)

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
    vars: Any
    consts: Any


@dataclass
class Var:
    """
    Represents a variable in the model.
    It has a name and a type, which can be either 'endo' (endogenous) or 'exo' (exogenous).

    Endogenous variables are given by a formula (either ODE or algebraic). A model for an exogenus variable is not provided.

    Parameters
    ----------
    name : str
        The name of the variable.
    type : VarType
        The type of the variable. Can be either 'endo'  for endogenous variables (equation exists) or 'exo' for exogenous variables (not modelled).
    initial : float | None
        The initial value of the variable. Only used for endogenous variables.
    range : tuple[float, float] | None
        The range of the variable.
    aggregation : Aggregation
        The aggregation method for the endogenous variable. Can be either 'sum', 'product', 'mean', 'max' or 'min'. If process-based formalism is used,
        aggregation method is used to combine the results of multiple processes that affect the same variable. For example, if two processes affect the same variable, and the aggregation method is 'sum', then the value of the variable will be the sum of the values from both processes.
    data : TimeSeries | None
        The data for the variable.
    unit : str | None
        The unit of the variable.
    """

    name: str
    type: VarType = "endo"
    initial: float | None = None
    range: tuple[float, float] | None = None
    aggregation: Aggregation = "sum"
    unit: str | None = None
    ode: Callable[[Any, Any], Any] | "Choose" | None = None
    algebraic: Callable[[Any, Any], Any] | "Choose" | None = None
    active_model: "Model | None" = None
    data: TimeSeries | None = None
    index_in_ctx: int | None = None

    def __post_init__(self):
        # add data
        super().__init__()
        self.n_processes = (
            0  # number of processes that affect this variable. Used for aggregation.
        )

    def append_ae(self, ae: Callable[[Any, Any], Any]):
        """
        Adds new algebraic equation to the variable and agregates it with the existing one using the aggregation method.
        """
        if self.algebraic is None:
            self.algebraic = ae
        else:
            if self.aggregation == "sum":
                old_ae = self.algebraic
                self.algebraic = lambda t, ctx: old_ae(t, ctx) + ae(t, ctx)
            elif self.aggregation == "product":
                old_ae = self.algebraic
                self.algebraic = lambda t, ctx: old_ae(t, ctx) * ae(t, ctx)
            elif self.aggregation == "mean":
                old_ae = self.algebraic
                n = self.n_processes + 1
                self.algebraic = (
                    lambda t, ctx: (old_ae(t, ctx) * (n - 1) + ae(t, ctx)) / n
                )
            elif self.aggregation == "max":
                old_ae = self.algebraic
                self.algebraic = lambda t, ctx: max(old_ae(t, ctx), ae(t, ctx))
            elif self.aggregation == "min":
                old_ae = self.algebraic
                self.algebraic = lambda t, ctx: min(old_ae(t, ctx), ae(t, ctx))
            else:
                raise ValueError(f"Unknown aggregation method {self.aggregation}.")
        # increment the number of processes that affect this variable
        self.n_processes += 1

    def append_ode(self, ode: Callable[[Any, Any], Any]):
        """
        Adds new ode to the variable and agregates it with the existing one using the aggregation method.
        """
        if self.ode is None:
            self.ode = ode
        else:
            if self.aggregation == "sum":
                old_ode = self.ode
                self.ode = lambda t, ctx: old_ode(t, ctx) + ode(t, ctx)
            elif self.aggregation == "product":
                old_ode = self.ode
                self.ode = lambda t, ctx: old_ode(t, ctx) * ode(t, ctx)
            elif self.aggregation == "mean":
                old_ode = self.ode
                n = self.n_processes + 1
                self.ode = lambda t, ctx: (old_ode(t, ctx) * (n - 1) + ode(t, ctx)) / n
            elif self.aggregation == "max":
                old_ode = self.ode
                self.ode = lambda t, ctx: max(old_ode(t, ctx), ode(t, ctx))
            elif self.aggregation == "min":
                old_ode = self.ode
                self.ode = lambda t, ctx: min(old_ode(t, ctx), ode(t, ctx))
            else:
                raise ValueError(f"Unknown aggregation method {self.aggregation}.")
        # increment the number of processes that affect this variable
        self.n_processes += 1

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
        new_var = Var(
            name=self.name,
            type=self.type,
            initial=self.initial,
            range=self.range,
            aggregation=self.aggregation,
            unit=self.unit,
            ode=self.ode,
            algebraic=self.algebraic,
            data=self.data,
            index_in_ctx=self.index_in_ctx,
        )
        # n_processes isn't a dataclass field, so __post_init__ would otherwise reset it to 0 on every
        # copy, desyncing "mean" aggregation's running count from how many terms are actually in `ode`/
        # `algebraic`. Found via Process + Model.copy()/induce(), but it's a general Var.copy() gap.
        new_var.n_processes = self.n_processes
        return new_var

    def update(self, **kwargs):
        raise NotImplementedError()


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


@dataclass
class Ode:
    """
    Represents an ordinary differential equation (ODE) in the model.
    """

    var: Var
    func: Callable[[Any, Any], Any]

    def __iter__(self):
        yield self.var
        yield self.func


@dataclass
class Algebraic:
    """
    Represents an algebraic equation in the model.
    """

    var: Var
    func: Callable[[Any, Any], Any]

    def __iter__(self):
        yield self.var
        yield self.func


class Process:
    def __init__(
        self,
        *args: "Ode | Algebraic | Process | Const",
        name: str | None = None,
    ):
        """
        Creates a new process with the given ODEs, algebraic equations, sub-processes and local constants.
        A process is meant to be handed to `Model`/`Model.add` directly - the model takes care of registering
        the process's local constants and applying its equations to the affected variables; there is no need
        to call `compile()` manually.
        """
        self.odes: list[Ode] = []
        self.aes: list[Algebraic] = []
        self.processes: list[Process] = []
        self.own_consts: dict[str, Const] = {}
        self._compiled = False

        for arg in args:
            if isinstance(arg, Ode):
                self.odes.append(arg)
            elif isinstance(arg, Algebraic):
                self.aes.append(arg)
            elif isinstance(arg, Const):
                self.own_consts[arg.name] = arg
            elif isinstance(arg, Process):
                self.processes.append(arg)
                # recursively collect equations from the sub-process. Local constants are NOT flattened here -
                # they stay scoped to their own process to avoid name collisions between sibling sub-processes
                # (e.g. two different reactions both having a local constant called "k"); Model.add() namespaces
                # them per process instead.
                self.odes.extend(arg.odes)
                self.aes.extend(arg.aes)
            else:
                warnings.warn(
                    f"Argument {arg} is not a valid process component. It will be ignored."
                )

        if name is None:
            warnings.warn(
                "Process name is not set. Default name was used. In the future, this will raise an error."
            )
            name = f"Process_{id(self)}"
        self.name = name

    def compile(self):
        """
        Adds this process's equations to the affected variables. Idempotent - safe to call more than once
        (e.g. when a model containing this process is copied/re-added), equations are only applied once.
        """
        if self._compiled:
            return
        for var, func in self.odes:
            var.append_ode(func)
        for var, func in self.aes:
            var.append_ae(func)
        self._compiled = True

    def copy(self):
        """
        Returns a copy of the process, with fresh copies of its own local constants (so a copied model can
        assign them a new context index without mutating the original). The ODEs/AEs and sub-processes are
        shared as-is - equation evaluation reads values by context index, not by object identity, exactly
        like `Var.copy()` already relies on, so re-applying them is unnecessary and (per `compile()`) skipped.
        """
        new_consts = [const.copy() for const in self.own_consts.values()]
        new_subprocesses = [process.copy() for process in self.processes]
        new_process = Process(
            *self.odes, *self.aes, *new_consts, *new_subprocesses, name=self.name
        )
        new_process._compiled = True
        return new_process


class Choose:
    """
    Represents a choice between different functions/processes.
    """

    def __init__(self, *args: Callable[[Any], Any] | Any):
        self.options = args

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Choose class is not meant to be called directly. Use model.induce() to generate all possible models with the different choices first."
        )


class ChooseProcess:
    """
    Represents a structural choice between several candidate processes that could fill the same
    named slot (e.g. several concrete process templates that all satisfy the same abstract process
    template in a ProBMoT-style library). Deliberately separate from `Choose`, which stays scoped to
    equation-level alternatives on `Var.ode`/`Var.algebraic` - a process is a whole bag of
    equations/consts/sub-processes, not a single callable, so it needs its own container.

    Candidates are given as *factories* - `Callable[[Model], Process]`, not already-built `Process`
    objects. This matters: `Model.induce()` resolves a slot by copying the model (possibly more than
    once, across branches), and a pre-built `Process`'s `Ode`/`Algebraic` entries and local `Const`s
    would keep referencing the ORIGINAL (pre-copy) `Var`/`Const` objects - `Process.compile()` would
    then mutate those stale objects instead of the ones actually living in the model being resolved,
    silently leaving that model's variables without their equations. A factory sidesteps this by
    building a brand new `Process` from scratch, each time, against the resolving model's own
    `model.entities`/`model.elements` - exactly the entities that are actually live in that branch.

    Add a `ChooseProcess` to a `Model` like any other component (`Model(entity, ChooseProcess(...))`);
    it is *not* compiled/registered immediately (its structure is still undetermined) - instead
    `Model.induce()` branches into one model per candidate, calling its factory with that branch's own
    model and adding the resulting `Process` under this slot's name. Models with unresolved
    `ChooseProcess`es are never "finished" for `induce()`'s purposes.
    """

    def __init__(self, *process_factories: "Callable[[Model], Process]", name: str | None = None):
        self.options = process_factories
        if name is None:
            warnings.warn(
                "ChooseProcess name is not set. Default name was used. In the future, this will raise an error."
            )
            name = f"ChooseProcess_{id(self)}"
        self.name = name

    def copy(self):
        """
        Factories are plain, stateless functions - safe to share across model copies as-is, unlike
        `Process`/`Const`, which carry model-specific mutable state (`index_in_ctx` etc).
        """
        return ChooseProcess(*self.options, name=self.name)
