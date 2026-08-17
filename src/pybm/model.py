"""
Process-based model definition and induction for PyBM.

Building a model happens in two clearly separated phases:

1. INCOMPLETE MODEL (`Entity`, `Process`, `ChooseProcess`, `Choose`, `Model` below). You describe
   *structure*: which entities exist, which processes act on them, and - if the model is not
   fully specified yet (ProBMoT calls this "incomplete") - which alternative equations/processes
   are still open choices (`Choose` for a single equation, `ChooseProcess` for a whole process).
   Nothing here has an `index_in_ctx` yet, because there is no single, unambiguous model to
   index - a variable's `index_in_ctx` would be different depending on which choices end up
   being made elsewhere in the model.

2. INDUCTION (`Model.induce()`). Resolves every open choice at once, producing one *complete*
   `InducedModel` per combination of choices - independent choices don't interact, so this is a
   plain Cartesian product. Only once a model is complete does it make sense to combine every
   process's contribution to a variable into one equation and hand out `index_in_ctx` slots in a
   flat context array - so that is exactly where it happens, in one pass, per resulting model.
   Simulation code should only ever run against an `InducedModel`.

Splitting it this way means the "incomplete model" side never has to worry about context/index
bookkeeping (no stale indices, no partial re-indexing as choices get resolved one at a time) - and
`induce()` never has to worry about which Var/Const objects to reuse vs. copy, because it takes
exactly one `copy.deepcopy()` of the whole structure per resulting combination, which - by
Python's own deepcopy semantics - automatically keeps every shared reference (e.g. the same `conc`
Var read from both its owning Entity and from a process's equation) consistent within that copy.
"""

from __future__ import annotations

import copy as _copy
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, TypedDict

import numpy as np
import torch

from pybm.expr import Expr, MaxExpr, MinExpr, Num, wrap
from pybm.utils import torch_interp

VarType = Literal["endo", "exo", "not_set"]
Aggregation = Literal["sum", "product", "mean", "max", "min"]


class Context(TypedDict):
    """Flat, indexable storage an induced model's equations read from at evaluation time."""

    vars: Any
    consts: Any


class TimeSeries:
    """Time-indexed data for an exogenous variable, with linear interpolation between samples.
    """

    def __init__(self, t: np.ndarray | Any, x: np.ndarray | Any):
        """
        Parameters
        ----------
        t : np.ndarray | Any
            Time points (1D array, increasing).
        x : np.ndarray | Any
            Values at each time point (1D array, same length as `t`).
        """
        self.t = t
        self.x = x
        self.index = 0

    def to_torch(self):
        """Converts the time series to torch tensors."""
        self.t = torch.tensor(self.t)
        self.x = torch.tensor(self.x)

    def to_jax(self):
        """Converts the time series to jax arrays."""
        import jax.numpy as jnp
        if torch.is_tensor(self.t):
            self.t, self.x = self.t.detach().cpu().numpy(), self.x.detach().cpu().numpy()
        self.t = jnp.asarray(self.t)
        self.x = jnp.asarray(self.x)

    def to_numpy(self):
        """Converts the time series back to numpy arrays (undoes `to_torch()`/`to_jax()`)."""
        if torch.is_tensor(self.t):
            self.t, self.x = self.t.detach().cpu().numpy(), self.x.detach().cpu().numpy()
        self.t = np.asarray(self.t)
        self.x = np.asarray(self.x)

    def interpolate(self, index, t):
        """Interpolates the value of the variable at the given index."""
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
        return self.interpolate(self.index, t)

    def __call__(self, t, bisection=False, numpy=True):
        if torch.is_tensor(t):
            return torch_interp(t, self.t, self.x)
        if type(t).__module__.startswith("jax"):
            import jax.numpy as jnp
            return jnp.interp(t, self.t, self.x)

        if numpy:
            return np.interp(t, self.t, self.x)
        else:
            if self.index == 0 and t < self.t[0]:
                warnings.warn(
                    f"Exogenous variable is called with t={t} < {self.t[0]}. Returning the first value."
                )
                return self.x[0]
            if bisection:
                return self.bisect(t, 0, len(self.t) - 1)
            else:
                if self.t[self.index] <= t and t <= self.t[self.index + 1]:
                    return self.interpolate(self.index, t)
                elif t < self.t[self.index]:
                    return self.bisect(t, 0, self.index)
                elif t > self.t[self.index + 1]:
                    return self.bisect(t, self.index + 1, len(self.t) - 1)


@dataclass
class Var(Expr):
    """
    A variable in the model - endogenous (governed by an equation) or exogenous (read from
    `data`). `Var` subclasses `Expr`, so it can be used directly inside equations (`rate * conc`)
    and, once a model is induced, evaluated as `var(t, ctx)` to read its *current value* - not to
    be confused with `var.ode`/`var.algebraic`, which are the Exprs that *define* that value.

    Parameters
    ----------
    name : str
        The name of the variable.
    type : VarType
        'endo' if this variable is governed by an equation, 'exo' if it is read from `data`.
    initial : float | None
        Initial value, used for endogenous variables when simulating.
    range : tuple[float, float] | None
        Optional bounds on the variable's value.
    aggregation : Aggregation
        How to combine multiple processes' contributions to this variable's equation ('sum',
        'product', 'mean', 'max' or 'min'). Only relevant for endogenous variables driven by more
        than one Process.
    unit : str | None
        The unit of the variable, for documentation purposes only.
    data : TimeSeries | None
        Time series backing an exogenous variable's value.
    ode : Expr | Choose | None
        The differential equation for this variable (`d(var)/dt = ode`). Set directly, or filled
        in by `Model.induce()` once assigned from a `Process`'s `Ode` entries.
    algebraic : Expr | Choose | None
        The algebraic equation for this variable (`var = algebraic`, re-derived at every time
        point instead of integrated).
    index_in_ctx : int | None
        This variable's slot in an induced model's flat `ctx["vars"]` array. `None` until the
        model has been induced - see the module docstring.
    """

    name: str
    type: VarType = "endo"
    initial: float | None = None
    range: tuple[float, float] | None = None
    aggregation: Aggregation = "sum"
    unit: str | None = None
    data: TimeSeries | None = None
    ode: "Expr | Choose | None" = None
    algebraic: "Expr | Choose | None" = None
    index_in_ctx: int | None = None

    def __call__(self, t, ctx: Context):
        """Reads this variable's current value at time `t` - exogenous vars from `data`,
        endogenous vars from `ctx["vars"]` at `index_in_ctx`."""
        if self.type == "exo":
            if self.data is None:
                raise ValueError(f"Exogenous variable {self.name} has no data - call set_data() first.")
            return self.data(t)
        if self.index_in_ctx is None:
            raise ValueError(
                f"Variable {self.name} has no index in context yet - has the model been induce()d?"
            )
        return ctx["vars"][self.index_in_ctx]

    def set_data(self, t, x):
        """Sets the data backing an exogenous variable."""
        self.data = TimeSeries(t, x)

    def __str__(self):
        return self.name


@dataclass
class Const(Expr):
    """
    A constant parameter. Subclasses `Expr` like `Var`, so it can be used directly inside
    equations (`rate * conc`); `Const.__call__` ignores `t` (constants don't change over time)
    but still accepts it, to match every other `Expr`'s `(t, ctx)` signature.
    """

    name: str
    initial_value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None
    index_in_ctx: int | None = None

    def __call__(self, t, ctx: Context):
        if self.index_in_ctx is None:
            raise ValueError(
                f"Constant {self.name} has no index in context yet - has the model been induce()d?"
            )
        return ctx["consts"][self.index_in_ctx]

    def __str__(self):
        return self.name


class Choose:
    """
    An open choice between several alternative Exprs for the same equation (a `Var`'s `ode` or
    `algebraic`). Not itself evaluable - `Model.induce()` must replace it with exactly one of its
    options before the model is complete.
    """

    def __init__(self, *options: "Expr | float"):
        self.options: tuple[Expr, ...] = tuple(wrap(option) for option in options)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Choose is not evaluable directly - call Model.induce() first to resolve it into "
            "concrete models."
        )


@dataclass
class Ode:
    """One process's contribution to a variable's differential equation (`d(var)/dt`)."""

    var: Var
    expr: "Expr | Choose"

    def __post_init__(self):
        if not isinstance(self.expr, Choose):
            self.expr = wrap(self.expr)

    def __iter__(self):
        yield self.var
        yield self.expr


@dataclass
class Algebraic:
    """One process's contribution to a variable's algebraic equation (`var = ...`)."""

    var: Var
    expr: "Expr | Choose"

    def __post_init__(self):
        if not isinstance(self.expr, Choose):
            self.expr = wrap(self.expr)

    def __iter__(self):
        yield self.var
        yield self.expr


class Entity(MutableMapping):
    """
    A named bag of Vars/Consts (e.g. a population, a compartment). Supports `entity["conc"]` to
    look up one of its variables/constants by name, and dict-like iteration/`in`/`.get()` via
    `MutableMapping`.

    Usage
    -----
    >>> conc = Var("conc", type="endo", initial=1.0)
    >>> rate = Const("rate", initial_value=0.1)
    >>> pop = Entity(conc, rate, name="prey")
    >>> pop["conc"] is conc
    True
    """

    def __init__(self, *args: "Var | Const", name: str | None = None):
        self._data: dict[str, "Var | Const"] = {}
        if name is None:
            warnings.warn("Entity name is not set. Using a default name.")
            name = f"Entity_{id(self)}"
        self.name = name
        for arg in args:
            self._data[arg.name] = arg

    def __getitem__(self, name):
        return self._data[name]

    def __setitem__(self, name, value):
        self._data[name] = value

    def __delitem__(self, name):
        del self._data[name]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return f"Entity({', '.join(self._data)})"

    def vars(self) -> "list[Var]":
        return [obj for obj in self._data.values() if isinstance(obj, Var)]

    def consts(self) -> "list[Const]":
        return [obj for obj in self._data.values() if isinstance(obj, Const)]


class Process:
    """
    A named bundle of equations (`Ode`/`Algebraic`), local constants and nested sub-processes.
    Built once, against concrete Vars/Consts (directly, or via a `ProcessTemplate` - see
    templates.py) - a `Process` carries no context/index state of its own, so the same `Process`
    object can be shared freely while a model is still incomplete; only `Model.induce()` (via
    `copy.deepcopy`) ever needs an independent copy of one, once per resolved combination.
    """

    def __init__(self, *args: "Ode | Algebraic | Process | Const", name: str | None = None):
        self.odes: list[Ode] = []
        self.aes: list[Algebraic] = []
        self.processes: list[Process] = []
        self.own_consts: dict[str, Const] = {}

        for arg in args:
            if isinstance(arg, Ode):
                self.odes.append(arg)
            elif isinstance(arg, Algebraic):
                self.aes.append(arg)
            elif isinstance(arg, Const):
                self.own_consts[arg.name] = arg
            elif isinstance(arg, Process):
                self.processes.append(arg)
            else:
                warnings.warn(f"Argument {arg} is not a valid process component. It will be ignored.")

        if name is None:
            warnings.warn("Process name is not set. Using a default name.")
            name = f"Process_{id(self)}"
        self.name = name

    def all_odes(self) -> "list[Ode]":
        """This process's own Ode entries, plus every nested sub-process's, recursively."""
        odes = list(self.odes)
        for sub_process in self.processes:
            odes.extend(sub_process.all_odes())
        return odes

    def all_aes(self) -> "list[Algebraic]":
        """This process's own Algebraic entries, plus every nested sub-process's, recursively."""
        aes = list(self.aes)
        for sub_process in self.processes:
            aes.extend(sub_process.all_aes())
        return aes

    def all_consts(self) -> "list[Const]":
        """This process's own local constants, plus every nested sub-process's, recursively."""
        consts = list(self.own_consts.values())
        for sub_process in self.processes:
            consts.extend(sub_process.all_consts())
        return consts


class ChooseProcess:
    """
    An open choice between several candidate Processes that could fill the same named slot (e.g.
    several concrete process templates that all satisfy one abstract process template in a
    ProBMoT-style library). Candidates are plain, already-built `Process` objects - no factory
    indirection needed: resolving a `ChooseProcess` (see `Model.induce()`) never touches context/
    index state, it only ever picks one of the pre-built Processes. Indexing happens later, in one
    pass, only once a model has no more open choices left.
    """

    def __init__(self, *options: Process, name: str | None = None):
        self.options = options
        if name is None:
            warnings.warn("ChooseProcess name is not set. Using a default name.")
            name = f"ChooseProcess_{id(self)}"
        self.name = name


def _aggregate(exprs: "list[Expr]", aggregation: Aggregation) -> Expr:
    """Combines several processes' contributions to the same variable into one Expr, using the
    variable's aggregation method."""
    if len(exprs) == 1:
        return exprs[0]
    if aggregation == "sum":
        result = exprs[0]
        for expr in exprs[1:]:
            result = result + expr
        return result
    if aggregation == "product":
        result = exprs[0]
        for expr in exprs[1:]:
            result = result * expr
        return result
    if aggregation == "mean":
        result = exprs[0]
        for expr in exprs[1:]:
            result = result + expr
        return result / Num(len(exprs))
    if aggregation == "max":
        return MaxExpr(list(exprs))
    if aggregation == "min":
        return MinExpr(list(exprs))
    raise ValueError(f"Unknown aggregation method {aggregation!r}.")


class Model:
    """
    An "incomplete" (possibly ambiguous) process-based model: a set of Entities plus Processes
    (some of which may be `ChooseProcess` slots) that act on them, and/or bare top-level Vars/
    Consts. No Var/Const has an `index_in_ctx` yet - see the module docstring. Call `induce()` to
    resolve every open choice and get back one fully indexed, simulatable `InducedModel` per
    combination - even a model with no ambiguity at all must go through `induce()` (it will just
    return a single-element list), since that is the only place indices are assigned.
    """

    def __init__(
        self,
        *args: "Entity | Process | ChooseProcess | Var | Const",
        engine: Literal["scipy", "torch", "jax"] = "scipy",
    ):
        self.entities: dict[str, Entity] = {}
        self.processes: dict[str, Process] = {}
        self.pending_process_choices: dict[str, ChooseProcess] = {}
        self.free_vars: dict[str, Var] = {}
        self.free_consts: dict[str, Const] = {}
        self.engine = engine

        for arg in args:
            self.add(arg)

    def add(self, arg: "Entity | Process | ChooseProcess | Var | Const"):
        if isinstance(arg, Entity):
            if arg.name in self.entities:
                raise ValueError(f"Entity {arg.name} already exists in the model.")
            self.entities[arg.name] = arg
        elif isinstance(arg, ChooseProcess):
            if arg.name in self.pending_process_choices:
                raise ValueError(f"ChooseProcess {arg.name} already exists in the model.")
            self.pending_process_choices[arg.name] = arg
        elif isinstance(arg, Process):
            if arg.name in self.processes:
                raise ValueError(f"Process {arg.name} already exists in the model.")
            self.processes[arg.name] = arg
        elif isinstance(arg, Var):
            if arg.name in self.free_vars:
                raise ValueError(f"Variable {arg.name} already exists in the model.")
            self.free_vars[arg.name] = arg
        elif isinstance(arg, Const):
            if arg.name in self.free_consts:
                raise ValueError(f"Constant {arg.name} already exists in the model.")
            self.free_consts[arg.name] = arg
        else:
            raise ValueError(f"Argument {arg} is not a valid model component.")

    def _lookup_var(self, entity_name: "str | None", var_name: str) -> Var:
        """Finds a Var by (entity_name, var_name) - entity_name is None for a bare top-level Var."""
        if entity_name is None:
            return self.free_vars[var_name]
        return self.entities[entity_name][var_name]

    def _open_var_choices(self) -> "list[tuple[str | None, str, str]]":
        """Every (entity_name, var_name, 'ode'|'algebraic') whose equation is still a `Choose`."""
        choices = []
        for entity_name, entity in self.entities.items():
            for var in entity.vars():
                for attr in ("ode", "algebraic"):
                    if isinstance(getattr(var, attr), Choose):
                        choices.append((entity_name, var.name, attr))
        for var in self.free_vars.values():
            for attr in ("ode", "algebraic"):
                if isinstance(getattr(var, attr), Choose):
                    choices.append((None, var.name, attr))
        return choices

    def induce(self) -> "list[InducedModel]":
        """
        Resolves every open choice (`Choose` on a variable's equation, `ChooseProcess` slots)
        into one `InducedModel` per combination. Choices are independent of each other, so this
        is a plain Cartesian product over each choice's option indices.
        """
        var_choices = self._open_var_choices()
        var_option_counts = [
            len(getattr(self._lookup_var(entity_name, var_name), attr).options)
            for entity_name, var_name, attr in var_choices
        ]

        process_slot_names = list(self.pending_process_choices.keys())
        process_option_counts = [
            len(self.pending_process_choices[slot_name].options) for slot_name in process_slot_names
        ]

        # both materialized as lists (not left as `product()` iterators): the nested loop below
        # walks `process_combos` once per `var_combo`, which would silently yield nothing after
        # the first pass if it were a single-use iterator.
        var_combos = list(product(*(range(n) for n in var_option_counts))) if var_option_counts else [()]
        process_combos = (
            list(product(*(range(n) for n in process_option_counts))) if process_option_counts else [()]
        )

        induced_models = []
        for var_combo in var_combos:
            for process_combo in process_combos:
                induced_models.append(
                    self._resolve(var_choices, var_combo, process_slot_names, process_combo)
                )
        return induced_models

    def _resolve(
        self,
        var_choices: "list[tuple[str | None, str, str]]",
        var_combo: "tuple[int, ...]",
        process_slot_names: "list[str]",
        process_combo: "tuple[int, ...]",
    ) -> "InducedModel":
        """
        Materializes one combination of choices: takes a single, independent `copy.deepcopy` of
        the whole model (so this combination can never mutate/be mutated by any other), picks the
        chosen option for each open choice within that copy, then hands off to `_induce_single`
        for aggregation + indexing.
        """
        model_copy: Model = _copy.deepcopy(self)

        for (entity_name, var_name, attr), option_index in zip(var_choices, var_combo):
            var = model_copy._lookup_var(entity_name, var_name)
            choose: Choose = getattr(var, attr)
            setattr(var, attr, choose.options[option_index])

        for slot_name, option_index in zip(process_slot_names, process_combo):
            chosen_process = model_copy.pending_process_choices[slot_name].options[option_index]
            # namespaced under the SLOT's name, not whatever name the candidate itself carries -
            # the slot name is what's stable across candidates (e.g. matches a `.pbm` process
            # instance name), regardless of which one wins.
            chosen_process.name = slot_name
            model_copy.processes[slot_name] = chosen_process
        model_copy.pending_process_choices = {}

        return _induce_single(model_copy)

    def __str__(self):
        return (
            f"Model(Entities: {list(self.entities.keys())}, "
            f"Processes: {list(self.processes.keys())}, "
            f"Pending choices: {list(self.pending_process_choices.keys())})"
        )


def _induce_single(model: Model) -> "InducedModel":
    """
    Turns one fully-resolved incomplete `Model` (no `Choose`/`ChooseProcess` left) into a
    simulatable `InducedModel`, in a single non-incremental pass: combine every process's
    contribution to each variable into one Expr, then assign every endogenous Var/Const a slot in
    a flat context array.
    """
    assert not model.pending_process_choices, "model still has unresolved ChooseProcess slots"

    # --- collect every var/const, under the dotted names used throughout the rest of PyBM ---
    named_vars: dict[str, Var] = dict(model.free_vars)
    named_consts: dict[str, Const] = dict(model.free_consts)
    for entity_name, entity in model.entities.items():
        for var in entity.vars():
            named_vars[f"{entity_name}.{var.name}"] = var
        for const in entity.consts():
            named_consts[f"{entity_name}.{const.name}"] = const
    def collect_process_consts(process: Process, path: str):
        for const_name, const in process.own_consts.items():
            named_consts[f"{path}.{const_name}"] = const
        for sub_process in process.processes:
            collect_process_consts(sub_process, f"{path}.{sub_process.name}")

    for process in model.processes.values():
        collect_process_consts(process, process.name)

    # --- aggregate every process's contribution to each variable into one Expr ---
    # keyed by id(var) rather than var itself: Var is an unhashable dataclass (equality/hash are
    # not meaningful for it - two distinct variables could easily compare equal by field values).
    ode_contributions: dict[int, tuple[Var, list[Expr]]] = {}
    ae_contributions: dict[int, tuple[Var, list[Expr]]] = {}
    for process in model.processes.values():
        for ode in process.all_odes():
            _, exprs = ode_contributions.setdefault(id(ode.var), (ode.var, []))
            exprs.append(ode.expr)
        for algebraic in process.all_aes():
            _, exprs = ae_contributions.setdefault(id(algebraic.var), (algebraic.var, []))
            exprs.append(algebraic.expr)
    for var, exprs in ode_contributions.values():
        var.ode = _aggregate(exprs, var.aggregation)
    for var, exprs in ae_contributions.values():
        var.algebraic = _aggregate(exprs, var.aggregation)

    # --- assign a slot in the flat context arrays - only now, since the model is complete ---
    endo_vars = [var for var in named_vars.values() if var.type == "endo"]
    for index, var in enumerate(endo_vars):
        var.index_in_ctx = index
    for index, const in enumerate(named_consts.values()):
        const.index_in_ctx = index

    return InducedModel(entities=model.entities, vars=named_vars, consts=named_consts, engine=model.engine)


class InducedModel:
    """
    A fully resolved, indexed model - the output of `Model.induce()`. Every endogenous `Var` has
    an `index_in_ctx` into a flat `ctx["vars"]` array, and every `Const` one into a flat
    `ctx["consts"]` array; equations are ready to evaluate as `var.ode(t, ctx)` /
    `var.algebraic(t, ctx)`.
    """

    def __init__(
        self,
        entities: "dict[str, Entity]",
        vars: "dict[str, Var]",
        consts: "dict[str, Const]",
        engine: Literal["scipy", "torch", "jax"],
    ):
        self.entities = entities
        self.vars = vars
        self.consts = consts
        self.engine = engine
        self.context_size = len(consts)

    def get_endo_variables(self) -> "list[Var]":
        """Returns every endogenous variable in the model."""
        return [var for var in self.vars.values() if var.type == "endo"]

    def switch_engine(self, engine: Literal["scipy", "torch", "jax"]) -> "InducedModel":
        """
        Switches this model's compute engine in place: updates `self.engine`, and converts every
        variable's attached data (see `TimeSeries`) to match - so exogenous reads (`Var.__call__`)
        return values of the type the new engine's equations expect. Equations themselves (`Expr`
        trees) never need converting - they already dispatch on whatever type of value they're
        actually handed (see expr.py/func.py), regardless of `engine`. Returns `self`, for chaining.
        """
        self.engine = engine
        converter = {"torch": "to_torch", "jax": "to_jax", "scipy": "to_numpy"}[engine]
        for var in self.vars.values():
            if var.data is not None:
                getattr(var.data, converter)()
        return self

    def __str__(self):
        return (
            f"InducedModel(Entities: {list(self.entities.keys())},\n"
            f" Vars: {list(self.vars.keys())},\n"
            f" Consts: {list(self.consts.keys())})"
        )

    def __repr__(self):
        return (
            f"InducedModel(Entities: {list(self.entities.values())}, "
            f"Vars: {list(self.vars.values())}, Consts: {list(self.consts.values())})"
        )
