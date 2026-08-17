"""
Declarative DSL for defining entity types and process types once, then instantiating them
against concrete entities/roles - mirrors ProBMoT's template -> instance pattern.

`EntityTemp` (built from `VarTemp`/`ConstTemp`) describes an entity *type* (e.g. "a population
has a concentration and nothing else yet"); calling it creates a concrete `Entity`.
`ProcessTemplate` describes a process *type* (local constants + equations over named entity
"roles", e.g. "growth needs a population and a rate"); calling it with concrete entities bound to
its roles creates a concrete `Process` (see model.py), built out of `Expr`s the same way you would
write them by hand (`rate * pop["conc"]`) - no separate compilation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pybm.model import Aggregation, Algebraic, Const, Entity, Ode, Process, VarType, Var


@dataclass(frozen=True)
class VarTemp:
    """Variable specification. Describes a Var's shape without creating one - `create()` does."""

    name: str
    type: VarType = "not_set"
    initial: float = 0.0
    range: tuple[float, float] | None = None
    aggregation: Aggregation = "sum"
    unit: str | None = None

    def create(self) -> Var:
        return Var(
            name=self.name,
            type=self.type,
            initial=self.initial,
            range=self.range,
            aggregation=self.aggregation,
            unit=self.unit,
        )


@dataclass(frozen=True)
class ConstTemp:
    """Constant specification. Describes a Const's shape without creating one - `create()` does."""

    name: str
    initial_value: float | Any = None
    range: tuple[float, float] | None = None
    unit: str | None = None

    def create(self) -> Const:
        return Const(
            name=self.name,
            initial_value=self.initial_value,
            range=self.range,
            unit=self.unit,
        )


class EntityTemp:
    """Entity type: a fixed set of `VarTemp`/`ConstTemp`. Calling it creates one concrete `Entity`."""

    def __init__(self, *args: "VarTemp | ConstTemp", parent: "EntityTemp | None" = None):
        self._data = args
        self.parent = parent

    def __call__(self, name: str | None = None) -> Entity:
        """Creates a concrete Entity from the template."""
        return Entity(*(obj.create() for obj in self._data), name=name)

    def variables(self) -> list[str]:
        """Returns the names of the variables declared by this template."""
        return [obj.name for obj in self._data if isinstance(obj, VarTemp)]

    def constants(self) -> list[str]:
        """Returns the names of the constants declared by this template."""
        return [obj.name for obj in self._data if isinstance(obj, ConstTemp)]

    def params(self) -> list[str]:
        """Returns the names of everything declared by this template."""
        return [obj.name for obj in self._data]

    def add(self, obj: "VarTemp | ConstTemp"):
        """Adds a variable or constant specification to the template."""
        if not isinstance(obj, (VarTemp, ConstTemp)):
            raise ValueError(f"Argument {obj} is not a VarTemp or ConstTemp.")
        self._data += (obj,)

    @staticmethod
    def inherits(entity_template: "EntityTemp") -> "EntityTemp":
        """Creates a new template starting from an existing one's Var/ConstTemps - the new
        template is free to `add()` more without affecting the parent."""
        return EntityTemp(*entity_template._data, parent=entity_template)


class ProcessTemplate:
    """
    Process type: local constants + equations over named entity "roles", without committing to
    concrete entities. Calling the template with concrete entities bound to its roles builds a
    `Process` (see model.py) ready to hand to `Model`/`ChooseProcess`.
    """

    def __init__(
        self,
        *local_consts: ConstTemp,
        roles: "dict[str, EntityTemp]",
        build: "Callable[[dict[str, Entity | list[Entity]], dict[str, Const]], list[Ode | Algebraic | Process]] | None" = None,
        parent: "ProcessTemplate | None" = None,
    ):
        self.local_consts = local_consts
        self.roles = roles
        self.build = build
        self.parent = parent

    def __call__(self, name: str | None = None, **role_bindings: "Entity | list[Entity]") -> Process:
        """
        Binds concrete entities to this template's roles and builds a `Process`. `build` receives
        the bound roles and the instantiated local constants, and returns a flat list mixing
        `Ode`/`Algebraic` equations (built with plain operators, e.g. `Ode(pop["conc"], rate *
        pop["conc"])`) and already-instantiated sub-`Process`es (from calling other
        `ProcessTemplate`s).
        """
        consts = {const_temp.name: const_temp.create() for const_temp in self.local_consts}
        contributions = self.build(role_bindings, consts) if self.build is not None else []
        return Process(*contributions, *consts.values(), name=name)

    @staticmethod
    def inherits(
        process_template: "ProcessTemplate",
        *extra_consts: ConstTemp,
        roles: "dict[str, EntityTemp] | None" = None,
        build: "Callable[[dict[str, Entity | list[Entity]], dict[str, Const]], list[Ode | Algebraic | Process]] | None" = None,
    ) -> "ProcessTemplate":
        """
        Creates a process template that inherits from an existing one: local constants are
        combined, and `roles`/`build` fall back to the parent's when not overridden.
        """
        return ProcessTemplate(
            *process_template.local_consts,
            *extra_consts,
            roles=roles if roles is not None else process_template.roles,
            build=build if build is not None else process_template.build,
            parent=process_template,
        )
