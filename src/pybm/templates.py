
from dataclasses import dataclass
from pybm.model import *


@dataclass(frozen=True)
class VarTemp:
    """
    Variable specification DSL. Used to define a variable type in the entity template.
    """

    name: str
    type: VarType = "not_set"
    initial: float = 0.0
    range: tuple[float, float] | None = None
    aggregation: Aggregation = "sum"
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
        return Const(
            name=self.name,
            initial_value=self.initial_value,
            range=self.range,
            unit=self.unit,
        )
class ProcessTemplate:
    """
    Process specification DSL. Used to define a process type - a set of local constants and equations
    (possibly built out of sub-processes) over a set of named entity "roles" - without committing to
    concrete entities. Calling the template with concrete entities bound to its roles produces a `Process`.
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
        Binds concrete entities to this template's roles and creates a `Process` instance.
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
        Creates a process template that inherits from an existing one: local constants are combined, and
        `roles`/`build` fall back to the parent's when not overridden.
        """
        return ProcessTemplate(
            *process_template.local_consts,
            *extra_consts,
            roles=roles if roles is not None else process_template.roles,
            build=build if build is not None else process_template.build,
            parent=process_template,
        )

@dataclass
class OdeTemp:
    """
    Represents an ordinary differential equation (ODE) in the model.
    """

    var_temp: VarTemp
    func_temp: Callable[[Any, Any], Any]

    def __iter__(self):
        yield self.var_temp
        yield self.func_temp


@dataclass
class AlgebraicTemp:
    """
    Represents an algebraic equation in the model.
    """

    var: VarTemp
    func: Callable[[Any, Any], Any]

    def __iter__(self):
        yield self.var
        yield self.func


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
