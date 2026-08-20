"""ProBMoT .pbl (library) / .pbm (model) -> PyBM converter.

Scope of this pass (see the memory/conversation this was built from for the full design
discussion): fully supports entity templates and process templates with inheritance,
cardinality-typed roles, nested sub-processes, `<x:roleset>` iterators, and the expression
language (+,-,*,/, unary minus, exp/pow/sin/cos/sign/min/max/log/log10, `inf`). Supports both
fully-specified (.pbm `model`) and parameter-incomplete (`null` values) files directly.

Structural incompleteness (a `.pbm` `processes:` entry naming an *abstract* process template,
meaning "any leaf descendant is a candidate") is supported via `ChooseProcess`. `ChooseProcess`
(see model.py) is a `Model`-level primitive, not something that nests inside a `Process` - but
ProBMoT lets an abstract reference appear *nested* inside another process's own sub-process list
(e.g. `TempRespInfluence`, declared only via `Temp1RespirationPP`/`Temp2RespirationPP`'s shared
`TempRespirationPP` ancestor, never via the sibling `ExponentialRespirationPP`). An anonymous
nested reference (one the `.pbm` never explicitly names) is therefore *cross-product-expanded
into its enclosing candidate* rather than flattened out as its own top-level slot - see
`ModelBuilder._build_leaf_candidates`/`_build_bare_candidates` for the full reasoning. Flattening
it to a sibling top-level slot instead (an earlier version of this converter did exactly that) is
a real correctness bug, not a harmless simplification: `Model.induce()` would then vary that
nested slot independently of whether its own enclosing leaf (e.g. `Temp1RespirationPP`) is even
the one `RespirationPP` picks, producing structurally duplicate induced models whenever a sibling
leaf without that nesting (e.g. `ExponentialRespirationPP`) is chosen instead - confirmed on the
real Bled library, where it inflated 4032 truly distinct structures into 5184 induced ones. A
reference the `.pbm` DOES explicitly name (in some process's own `processes:` list) is still
flattened to a top-level `ChooseProcess` when ambiguous - that is a genuinely independent,
author-declared structural choice, not a conditionally-relevant nested one, and constant
namespacing for it still ends up under the synthesized top-level name, not `parent.slotname...`.

NOT covered: compartments, incomplete process *arguments* (the `[[],[...]]` bound syntax).
"""

from __future__ import annotations

import itertools as _itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lark import Lark, Token, Tree

from pybm.expr import Expr, FuncExpr
from pybm.expr import Num as ExprNum
from pybm.func import _dispatched_binary, _dispatched_math
from pybm.model import Const, Entity, Model, Process, ChooseProcess, Ode, Algebraic, Var

_GRAMMAR_PATH = Path(__file__).parent / "probmot_grammar.lark"
_parser = Lark(_GRAMMAR_PATH.read_text(), parser="earley", start="start")


# ============================================================ expression AST ====

@dataclass
class Num:
    value: float


@dataclass
class InfLit:
    sign: float = 1.0


@dataclass
class FuncCall:
    name: str
    args: "list[Any]"


@dataclass
class Dotted:
    ref: str  # a role name, or an iterator variable name
    attr: str
    iter_roleset: "str | None" = None  # set if `ref` came from a `<ref:iter_roleset>` marker


@dataclass
class NameRef:
    name: str  # a local (process-own) constant


@dataclass
class Neg:
    operand: Any


@dataclass
class BinOp:
    op: str  # "+" "-" "*" "/"
    left: Any
    right: Any


def _sign(x: Any) -> Any:
    # comparison + subtraction already dispatch generically across numpy/torch/jax - and, being a
    # plain module-level function rather than a lambda, this stays picklable (see
    # pybm.func._apply_dispatched_math's note on why that matters).
    return (x > 0) - (x < 0)


_FUNCS: "dict[str, Callable]" = {
    # dispatched by the runtime type of the (possibly batched, torch/jax) value they're actually
    # called with - see pybm.func._dispatched_math/_dispatched_binary - not plain `math.*`, which
    # only accepts a single Python float and breaks the moment an equation is evaluated under the
    # torch estimation pipeline (a batched tensor, not a scalar).
    "exp": _dispatched_math("exp"),
    "pow": _dispatched_binary("power", torch_name="pow"),
    "sin": _dispatched_math("sin"),
    "cos": _dispatched_math("cos"),
    "sign": _sign,
    "min": _dispatched_binary("minimum"),
    "max": _dispatched_binary("maximum"),
    "log": _dispatched_math("log"),
    "log10": _dispatched_math("log10"),
}


def compile_expr(node, resolve_dotted, resolve_name) -> Expr:
    """Compiles a parsed expression AST node (the small dataclasses above - NOT to be confused
    with `pybm.expr.Expr`, the *runtime* expression tree this function builds) into an `Expr`.

    `resolve_dotted(ref, attr)` and `resolve_name(name)` return the concrete `Var`/`Const` for a
    given reference - see `_build_equations` for how those are supplied. Both `Var` and `Const`
    are themselves `Expr`s (see model.py/expr.py), so combining them with `+ - * /` below just
    builds more `Expr` tree via plain Python operators - no closures needed.
    """
    if isinstance(node, Num):
        return ExprNum(node.value)
    if isinstance(node, InfLit):
        return ExprNum(node.sign * math.inf)
    if isinstance(node, NameRef):
        return resolve_name(node.name)
    if isinstance(node, Dotted):
        return resolve_dotted(node.ref, node.attr)
    if isinstance(node, Neg):
        return -compile_expr(node.operand, resolve_dotted, resolve_name)
    if isinstance(node, BinOp):
        left = compile_expr(node.left, resolve_dotted, resolve_name)
        right = compile_expr(node.right, resolve_dotted, resolve_name)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        raise ValueError(f"Unknown operator {node.op}")
    if isinstance(node, FuncCall):
        func = _FUNCS[node.name]
        args = [compile_expr(a, resolve_dotted, resolve_name) for a in node.args]
        return FuncExpr(func, args, name=node.name)
    raise TypeError(f"Unknown expression node {node!r}")


# ============================================================ template/instance specs ====


@dataclass
class VarSpec:
    name: str
    attrs: dict


@dataclass
class ConstSpec:
    name: str
    attrs: dict


@dataclass
class EntityTemplateSpec:
    name: str
    parent: "str | None"
    vars: "list[VarSpec]"
    consts: "list[ConstSpec]"


@dataclass
class Param:
    name: str
    type_name: str
    cardinality: "tuple[float, float] | None"  # None means exactly 1


@dataclass
class SubProcessRef:
    template_name: str
    call_args: "list[tuple[str, str] | tuple[str, str, str]]"
    # ("name", role) for a plain role reference, ("iter", iter_var, roleset) for <iter_var:roleset>


@dataclass
class EquationSpec:
    is_diff: bool
    target_ref: "str | None"  # role/iterator-var name for the target, or None if target_iter used
    target_iter: "tuple[str, str] | None"  # (iter_var, roleset) if the LHS/target is iterated
    target_attr: str
    expr: Any


@dataclass
class ProcessTemplateSpec:
    name: str
    parent: "str | None"
    params: "list[Param] | None"  # None means "inherit from nearest ancestor"
    consts: "list[ConstSpec]"
    sub_processes: "list[SubProcessRef]"
    equations: "list[EquationSpec]"


@dataclass
class EntityInstanceSpec:
    name: str
    template: str
    vars: "dict[str, dict]"
    consts: "dict[str, dict]"


@dataclass
class ProcessInstanceSpec:
    name: str
    template: str
    args: "list[list[str]]"  # each positional arg is a list of entity names (len 1 unless bracketed)
    consts: "dict[str, dict]"
    sub_processes: "list[str]"


@dataclass
class ParsedFile:
    kind: str  # "library" | "model" | "incomplete_model"
    name: str
    library_ref: "str | None"  # for model files: the library they depend on
    entity_templates: "list[EntityTemplateSpec]"
    process_templates: "list[ProcessTemplateSpec]"
    entity_instances: "list[EntityInstanceSpec]"
    process_instances: "list[ProcessInstanceSpec]"


# ============================================================ tree -> spec transformer ====


def _text(tok) -> str:
    return str(tok)


def _attr_value(node):
    """attr_value node -> a plain python value (str | (lo,hi) tuple | float | None)."""
    if isinstance(node, Tree):
        if node.data == "range":
            return _bound(node.children[0]), _bound(node.children[1])
        if node.data == "num_value":
            return float(node.children[0])
        if node.data == "null_value":
            return None
    if isinstance(node, Token):
        text = str(node)
        if text.startswith('"'):
            return text[1:-1]
        return text
    raise TypeError(f"Unexpected attr_value node {node!r}")


def _bound(node):
    if isinstance(node, Tree):
        if node.data == "inf_bound":
            return math.inf
        if node.data == "neg_inf_bound":
            return -math.inf
    return float(node)


def _attr_block(tree: "Tree | None") -> dict:
    if tree is None:
        return {}
    out = {}
    for attr in tree.children:
        name_tok, value_node = attr.children
        out[_text(name_tok)] = _attr_value(value_node)
    return out


def _decl_list(tree: "Tree | None", spec_cls):
    if tree is None:
        return []
    out = []
    for decl in tree.children:
        name_tok = decl.children[0]
        attr_block = decl.children[1] if len(decl.children) > 1 else None
        out.append(spec_cls(_text(name_tok), _attr_block(attr_block)))
    return out


def _find(tree_children, rule_name):
    for c in tree_children:
        if isinstance(c, Tree) and c.data == rule_name:
            return c
    return None


def _expr(node):
    if isinstance(node, Token):
        return NameRef(_text(node))
    assert isinstance(node, Tree)
    if node.data == "number":
        return Num(float(node.children[0]))
    if node.data == "inf_":
        return InfLit(1.0)
    if node.data == "name":
        return NameRef(_text(node.children[0]))
    if node.data == "dotted":
        ref_node, attr_tok = node.children
        if ref_node.data == "ref_iterator":
            iter_var, roleset = _iterator_pair(ref_node.children[0])
            return Dotted(iter_var, _text(attr_tok), iter_roleset=roleset)
        return Dotted(_ref(ref_node), _text(attr_tok))
    if node.data == "func_call":
        name_tok, *arg_nodes = node.children
        return FuncCall(_text(name_tok), [_expr(a) for a in arg_nodes])
    if node.data == "neg":
        return Neg(_expr(node.children[0]))
    if node.data == "add":
        return BinOp("+", _expr(node.children[0]), _expr(node.children[1]))
    if node.data == "sub":
        return BinOp("-", _expr(node.children[0]), _expr(node.children[1]))
    if node.data == "mul":
        return BinOp("*", _expr(node.children[0]), _expr(node.children[1]))
    if node.data == "div":
        return BinOp("/", _expr(node.children[0]), _expr(node.children[1]))
    if node.data == "paren":
        return _expr(node.children[0])
    raise TypeError(f"Unknown expr node {node.data}")


def _ref(node: Tree) -> str:
    """ref_name / ref_iterator -> the name to use as the "role" key (plain role name, or the
    iterator's bound variable name for `<n:ns>`)."""
    if node.data == "ref_name":
        return _text(node.children[0])
    if node.data == "ref_iterator":
        iter_var, _roleset = node.children[0].children
        return _text(iter_var)
    raise TypeError(node.data)


def _iterator_pair(node: Tree) -> "tuple[str, str]":
    iter_var, roleset = node.children
    return _text(iter_var), _text(roleset)


def _call_args(tree: "Tree | None"):
    out = []
    if tree is None:
        return out
    for arg in tree.children:
        if arg.data == "call_arg_name":
            out.append(("name", _text(arg.children[0])))
        else:
            iter_var, roleset = _iterator_pair(arg.children[0])
            out.append(("iter", iter_var, roleset))
    return out


def _parse_entity_template(tree: Tree) -> EntityTemplateSpec:
    children = tree.children
    name = _text(children[0])
    parent = _text(children[1]) if len(children) > 1 and isinstance(children[1], Token) else None
    body = children[-1]
    vars_section = _find(body.children, "vars_temp_section")
    consts_section = _find(body.children, "consts_temp_section")
    return EntityTemplateSpec(
        name, parent,
        _decl_list(vars_section, VarSpec),
        _decl_list(consts_section, ConstSpec),
    )


def _parse_param_list(tree: "Tree | None"):
    if tree is None:
        return None
    out = []
    for param in tree.children:
        name_tok, type_tok, *card = param.children
        cardinality = None
        if card:
            bounds = card[0].children
            if len(bounds) == 1:
                lo = hi = _bound(bounds[0])
            else:
                lo, hi = _bound(bounds[0]), _bound(bounds[1])
            cardinality = (lo, hi)
        out.append(Param(_text(name_tok), _text(type_tok), cardinality))
    return out


def _parse_sub_processes_temp(tree: "Tree | None"):
    if tree is None:
        return []
    out = []
    for ref in tree.children:
        name_tok, *arg_nodes = ref.children
        call_args_tree = None
        args = []
        for a in ref.children[1:]:
            if a.data == "call_arg_name":
                args.append(("name", _text(a.children[0])))
            else:
                iter_var, roleset = _iterator_pair(a.children[0])
                args.append(("iter", iter_var, roleset))
        out.append(SubProcessRef(_text(name_tok), args))
    return out


def _find_expr_iterator(node) -> "tuple[str, str] | None":
    """Finds a `<x:roleset>` marker anywhere inside an already-built expression AST - needed because
    ProBMoT allows the iterator to appear in the RHS instead of the LHS target (e.g.
    `zoo.phytoSum = <pp:pps>.conc;` sums pp.conc over pps via aggregation, without `phytoSum` itself
    being iterated - see the module-level design notes)."""
    if isinstance(node, Dotted):
        return (node.ref, node.iter_roleset) if node.iter_roleset is not None else None
    if isinstance(node, Neg):
        return _find_expr_iterator(node.operand)
    if isinstance(node, BinOp):
        return _find_expr_iterator(node.left) or _find_expr_iterator(node.right)
    if isinstance(node, FuncCall):
        for a in node.args:
            found = _find_expr_iterator(a)
            if found is not None:
                return found
    return None


def _parse_equations(tree: "Tree | None"):
    if tree is None:
        return []
    out = []
    for eq in tree.children:
        is_diff = eq.data == "diff_equation"
        target_tree, expr_tree = eq.children
        ref_node, attr_tok = target_tree.children
        expr = _expr(expr_tree)
        if ref_node.data == "ref_iterator":
            iter_var, roleset = _iterator_pair(ref_node.children[0])
            out.append(EquationSpec(is_diff, None, (iter_var, roleset), _text(attr_tok), expr))
        else:
            target_ref = _text(ref_node.children[0])
            out.append(EquationSpec(is_diff, target_ref, _find_expr_iterator(expr), _text(attr_tok), expr))
    return out


def _parse_process_template(tree: Tree) -> ProcessTemplateSpec:
    children = tree.children
    idx = 1
    name = _text(children[0])
    params = None
    if idx < len(children) and isinstance(children[idx], Tree) and children[idx].data == "param_list":
        params = _parse_param_list(children[idx])
        idx += 1
    parent = None
    if idx < len(children) and isinstance(children[idx], Token):
        parent = _text(children[idx])
        idx += 1
    body = children[idx]
    consts_section = _find(body.children, "consts_temp_section")
    sub_processes_section = _find(body.children, "sub_processes_temp_section")
    equations_section = _find(body.children, "equations_section")
    return ProcessTemplateSpec(
        name, parent, params,
        _decl_list(consts_section, ConstSpec),
        _parse_sub_processes_temp(sub_processes_section),
        _parse_equations(equations_section),
    )


def _parse_entity_instance(tree: Tree) -> EntityInstanceSpec:
    name_tok, template_tok, body = tree.children
    vars_section = _find(body.children, "vars_inst_section")
    consts_section = _find(body.children, "consts_inst_section")
    var_values = {}
    if vars_section is not None:
        for decl in vars_section.children:
            var_name = _text(decl.children[0])
            attrs = _attr_block(decl.children[1] if len(decl.children) > 1 else None)
            var_values[var_name] = attrs
    const_values = {}
    if consts_section is not None:
        for decl in consts_section.children:
            const_name = _text(decl.children[0])
            rest = decl.children[1:]
            attrs = {}
            value = None
            has_value = False
            for r in rest:
                if isinstance(r, Tree) and r.data == "attr_block":
                    attrs = _attr_block(r)
                else:
                    value = _attr_value(r)
                    has_value = True
            if has_value:
                attrs["value"] = value
            const_values[const_name] = attrs
    return EntityInstanceSpec(_text(name_tok), _text(template_tok), var_values, const_values)


def _parse_process_instance(tree: Tree) -> ProcessInstanceSpec:
    children = tree.children
    name_tok = children[0]
    idx = 1
    args = []
    while idx < len(children) and isinstance(children[idx], Tree) and children[idx].data.startswith("inst_arg"):
        arg_tree = children[idx]
        if arg_tree.data == "inst_arg_single":
            args.append([_text(arg_tree.children[0])])
        else:
            args.append([_text(t) for t in arg_tree.children])
        idx += 1
    template_tok = children[idx]
    idx += 1
    body = children[idx]
    consts_section = _find(body.children, "consts_inst_section")
    sub_processes_section = _find(body.children, "sub_processes_inst_section")
    const_values = {}
    if consts_section is not None:
        for decl in consts_section.children:
            const_name = _text(decl.children[0])
            rest = decl.children[1:]
            attrs = {}
            value = None
            has_value = False
            for r in rest:
                if isinstance(r, Tree) and r.data == "attr_block":
                    attrs = _attr_block(r)
                else:
                    value = _attr_value(r)
                    has_value = True
            if has_value:
                attrs["value"] = value
            const_values[const_name] = attrs
    sub_processes = []
    if sub_processes_section is not None:
        sub_processes = [_text(t) for t in sub_processes_section.children]
    return ProcessInstanceSpec(_text(name_tok), _text(template_tok), args, const_values, sub_processes)


def parse(text: str) -> ParsedFile:
    tree = _parser.parse(text)
    header, *statements = tree.children
    kind = header.data.replace("_header", "")
    if kind == "library":
        (name,) = header.children
        name, library_ref = _text(name), None
    else:
        model_name, lib_name = header.children
        name, library_ref = _text(model_name), _text(lib_name)

    entity_templates, process_templates, entity_instances, process_instances = [], [], [], []
    for stmt in statements:
        inner = stmt.children[0]
        if inner.data == "entity_template":
            entity_templates.append(_parse_entity_template(inner))
        elif inner.data == "process_template":
            process_templates.append(_parse_process_template(inner))
        elif inner.data == "entity_instance":
            entity_instances.append(_parse_entity_instance(inner))
        elif inner.data == "process_instance":
            process_instances.append(_parse_process_instance(inner))
    return ParsedFile(kind, name, library_ref, entity_templates, process_templates, entity_instances, process_instances)


# ============================================================ Library (inheritance-resolved) ====


class Library:
    """A parsed `.pbl` file, with entity/process template inheritance chains and process-template
    leaf-descendant sets (for structural search) resolved and ready to query."""

    def __init__(self, spec: ParsedFile):
        assert spec.kind == "library", f"expected a library file, got {spec.kind}"
        self.name = spec.name
        self.entity_templates: "dict[str, EntityTemplateSpec]" = {t.name: t for t in spec.entity_templates}
        self.process_templates: "dict[str, ProcessTemplateSpec]" = {t.name: t for t in spec.process_templates}
        self._process_children: "dict[str, list[str]]" = {}
        for t in spec.process_templates:
            if t.parent:
                self._process_children.setdefault(t.parent, []).append(t.name)

    def _entity_chain(self, name: str) -> "list[EntityTemplateSpec]":
        chain = []
        cur = name
        while cur is not None:
            spec = self.entity_templates[cur]
            chain.append(spec)
            cur = spec.parent
        return list(reversed(chain))

    def entity_vars_consts(self, name: str) -> "tuple[list[VarSpec], list[ConstSpec]]":
        vars_, consts_ = [], []
        for spec in self._entity_chain(name):
            vars_.extend(spec.vars)
            consts_.extend(spec.consts)
        return vars_, consts_

    def _process_chain(self, name: str) -> "list[ProcessTemplateSpec]":
        chain = []
        cur = name
        while cur is not None:
            spec = self.process_templates[cur]
            chain.append(spec)
            cur = spec.parent
        return list(reversed(chain))

    def process_params(self, name: str) -> "list[Param]":
        for spec in reversed(self._process_chain(name)):
            if spec.params is not None:
                return spec.params
        return []

    def process_consts(self, name: str) -> "list[ConstSpec]":
        return [c for spec in self._process_chain(name) for c in spec.consts]

    def process_sub_processes(self, name: str) -> "list[SubProcessRef]":
        return [s for spec in self._process_chain(name) for s in spec.sub_processes]

    def process_equations(self, name: str) -> "list[EquationSpec]":
        return [e for spec in self._process_chain(name) for e in spec.equations]

    def is_leaf(self, name: str) -> bool:
        return not self._process_children.get(name)

    def leaf_descendants(self, name: str) -> "list[str]":
        """All descendant templates (at any depth) that are themselves leaves - the candidate set
        for a structural choice at `name`, per ProBMoT's `specifications(P)` (thesis 5.3.2)."""
        if self.is_leaf(name):
            return [name]
        out = []
        for child in self._process_children.get(name, []):
            out.extend(self.leaf_descendants(child))
        return out


# ============================================================ Model builder (.pbm -> pybm.Model) ====

_ROLE_MAP = {"endogenous": "endo", "exogenous": "exo"}


def _build_entity(library: Library, spec: EntityInstanceSpec) -> Entity:
    var_specs, const_specs = library.entity_vars_consts(spec.template)
    parts: "list[Var | Const]" = []
    for vs in var_specs:
        inst_attrs = spec.vars.get(vs.name, {})
        var_type = _ROLE_MAP.get(inst_attrs.get("role"), "exo")
        parts.append(Var(
            vs.name, type=var_type, initial=inst_attrs.get("initial"),
            aggregation=vs.attrs.get("aggregation", "sum"), unit=vs.attrs.get("unit"),
        ))
    for cs in const_specs:
        inst_attrs = spec.consts.get(cs.name, {})
        parts.append(Const(
            cs.name, initial_value=inst_attrs.get("value"),
            range=cs.attrs.get("range"), unit=cs.attrs.get("unit"),
        ))
    return Entity(*parts, name=spec.name)


RoleBindings = "dict[str, str | list[str]]"  # role name -> entity name, or list of entity names


class ModelBuilder:
    """Builds a `pybm.model.Model` from a parsed `.pbm` file against a `Library`.

    Role bindings are threaded through as entity NAME strings (not `Entity` objects), resolved to
    the actual `Entity`/`Var`/`Const` only at the point of use (`_lookup`) - this mirrors
    ProBMoT's own role syntax (roles are always referenced by name) and keeps role-binding dicts
    small and printable for debugging.

    `ChooseProcess` candidates are plain, already-built `Process` objects, built directly against
    `self.entities` - there is exactly one `entities` dict for the whole file, built once up
    front. `Model.induce()` (see model.py) takes care of giving each resolved combination its own
    independent copy internally (one `copy.deepcopy` per combination), so the converter never
    needs to defer building a candidate until resolution time the way an earlier design did.
    """

    def __init__(self, library: Library, pbm: ParsedFile):
        assert pbm.kind in ("model", "incomplete_model"), f"expected a model file, got {pbm.kind}"
        self.library = library
        self.pbm = pbm
        self.model = Model()
        self.entities: "dict[str, Entity]" = {}
        self.named_processes: "dict[str, ProcessInstanceSpec]" = {p.name: p for p in pbm.process_instances}
        self._built: "dict[str, Process | ChooseProcess]" = {}
        self._anon_counter = 0

    def build(self) -> Model:
        for espec in self.pbm.entity_instances:
            entity = _build_entity(self.library, espec)
            self.entities[espec.name] = entity
            self.model.add(entity)

        referenced_as_nested = {name for p in self.pbm.process_instances for name in p.sub_processes}
        for pspec in self.pbm.process_instances:
            if pspec.name in referenced_as_nested:
                continue  # only built lazily, as part of whichever process nests it
            result = self._resolve_named_instance(pspec.name)
            if result is not None:  # None => already flattened & added to the model itself
                self.model.add(result)
        return self.model

    # ---- name-based role binding helpers ----

    def _role_bindings_from_args(self, params: "list[Param]", args: "list[list[str]]") -> RoleBindings:
        bindings: RoleBindings = {}
        for param, names in zip(params, args):
            bindings[param.name] = names if param.cardinality is not None else names[0]
        return bindings

    def _lookup(self, ref: str, attr: str, role_bindings: RoleBindings) -> "Var | Const":
        target_name = role_bindings[ref]
        if isinstance(target_name, list):
            raise ValueError(f"'{ref}.{attr}' refers to a set-valued role outside of an iterator")
        return self.entities[target_name][attr]

    def _anon_name(self, template_name: str) -> str:
        self._anon_counter += 1
        return f"{template_name}_{self._anon_counter}"

    # ---- equations / sub-process slots ----

    def _build_equations(self, template_name: str, role_bindings: RoleBindings, consts: "dict[str, Const]"):
        out = []
        for eq in self.library.process_equations(template_name):
            resolve_dotted = lambda ref, attr, rb=role_bindings: self._lookup(ref, attr, rb)
            resolve_name = lambda name, c=consts: c[name]
            if eq.target_iter is None:
                var = self._lookup(eq.target_ref, eq.target_attr, role_bindings)
                expr = compile_expr(eq.expr, resolve_dotted, resolve_name)
                out.append((Ode if eq.is_diff else Algebraic)(var, expr))
            else:
                iter_var, roleset = eq.target_iter
                for entity_name in role_bindings[roleset]:
                    rb = dict(role_bindings)
                    rb[iter_var] = entity_name
                    # target_ref set => the LHS itself is a fixed, non-iterated var (e.g.
                    # `zoo.phytoSum = <pp:pps>.conc;`) - all N expansions target the SAME var,
                    # relying on its own `aggregation` (here `sum`) to combine them, exactly as
                    # ProBMoT does. target_ref is None => the LHS itself was `<n:ns>.attr`, so each
                    # expansion targets a DIFFERENT var (one per entity).
                    if eq.target_ref is not None:
                        var = self._lookup(eq.target_ref, eq.target_attr, role_bindings)
                    else:
                        var = self.entities[entity_name][eq.target_attr]
                    rd = lambda ref, attr, rb=rb: self._lookup(ref, attr, rb)
                    expr = compile_expr(eq.expr, rd, resolve_name)
                    out.append((Ode if eq.is_diff else Algebraic)(var, expr))
        return out

    def _collect_slots(self, template_name: str, role_bindings: RoleBindings):
        """Returns [(sub_template_name, slot_role_bindings), ...] in positional order, with any
        iterated sub-process reference expanded into one slot per bound entity."""
        slots = []
        for ref in self.library.process_sub_processes(template_name):
            sub_params = self.library.process_params(ref.template_name)
            iter_arg = next((a for a in ref.call_args if a[0] == "iter"), None)
            if iter_arg is None:
                slot_roles = {sp.name: role_bindings[ca[1]] for sp, ca in zip(sub_params, ref.call_args)}
                slots.append((ref.template_name, slot_roles))
            else:
                _, iter_var, roleset = iter_arg
                for entity_name in role_bindings[roleset]:
                    slot_roles = {}
                    for sp, ca in zip(sub_params, ref.call_args):
                        slot_roles[sp.name] = entity_name if ca[0] == "iter" else role_bindings[ca[1]]
                    slots.append((ref.template_name, slot_roles))
        return slots

    # ---- building a concrete (leaf) process, expanding its own nested ambiguity ----

    def _build_leaf_candidates(
        self, template_name: str, role_bindings: RoleBindings, consts_override: dict,
        sub_process_overrides: "list[str] | None", name_hint: str,
    ) -> "list[Process]":
        """Builds every fully-resolved candidate `Process` for ONE already-chosen leaf template.

        Returns exactly one Process, UNLESS `sub_process_overrides` is `None` (no .pbm-given name
        for this leaf's own slots - i.e. this leaf's nested sub-processes, if any, were never
        explicitly named/pinned in the `.pbm`) AND at least one of those own slots is itself
        structurally ambiguous - in which case this returns one Process per combination of those
        slots' own candidates (a full cross product), each with a *concrete* choice already baked
        in for every nested slot.

        This is what keeps a slot like `TempRespInfluence` - declared only via
        `Temp1RespirationPP`/`Temp2RespirationPP`'s shared `TempRespirationPP` ancestor, never via
        `ExponentialRespirationPP` - from ending up as an independent top-level `ChooseProcess`
        that `Model.induce()` would vary regardless of whether `Temp1`/`Temp2RespirationPP` is even
        the sibling `RespirationPP` picks (which produced 5184 induced Bled models, of which only
        4032 were actually structurally distinct - the other 1152 varied a nested slot whose value
        no equation in that branch ever reads). Instead, the caller (`_build_bare_candidates`,
        recursively) bakes the cross product straight into `RespirationPP`'s own candidate list -
        7 candidates (1 Exponential + 3 Temp1 + 3 Temp2), not 3, with no separate slot left over.
        """
        const_specs = self.library.process_consts(template_name)

        def fresh_consts() -> "dict[str, Const]":
            return {
                cs.name: Const(
                    cs.name, initial_value=consts_override.get(cs.name, {}).get("value"),
                    range=cs.attrs.get("range"), unit=cs.attrs.get("unit"),
                )
                for cs in const_specs
            }

        slots = self._collect_slots(template_name, role_bindings)

        if sub_process_overrides is not None:
            assert len(slots) == len(sub_process_overrides), (
                f"{name_hint}: {template_name} declares {len(slots)} sub-process slot(s), "
                f"but the .pbm instance lists {len(sub_process_overrides)}: {sub_process_overrides}"
            )
            consts = fresh_consts()
            equations = self._build_equations(template_name, role_bindings, consts)
            nested: "list[Process]" = []
            for (slot_template, slot_roles), given_name in zip(slots, sub_process_overrides):
                resolved = self._resolve_sub_process_slot(given_name, slot_template, slot_roles)
                if resolved is not None:
                    nested.append(resolved)
            return [Process(*equations, *nested, *consts.values(), name=name_hint)]

        # No .pbm-given names for this leaf's own slots - cross product over each slot's own
        # (possibly single-item) candidate list, so every returned Process is fully concrete.
        slot_candidate_lists = [
            self._build_bare_candidates(slot_template, slot_roles) for slot_template, slot_roles in slots
        ]
        out: "list[Process]" = []
        combos = _itertools.product(*slot_candidate_lists) if slot_candidate_lists else [()]
        for combo in combos:
            consts = fresh_consts()
            equations = self._build_equations(template_name, role_bindings, consts)
            out.append(Process(*equations, *combo, *consts.values(), name=self._anon_name(template_name)))
        return out

    def _build_bare_candidates(self, template_name: str, role_bindings: RoleBindings) -> "list[Process]":
        """Every fully-resolved candidate `Process` for an ANONYMOUS reference (no .pbm-given name
        at all) to `template_name`, which may itself be abstract - the Cartesian product of "which
        leaf template" and each chosen leaf's own nested-slot candidates, recursively (via
        `_build_leaf_candidates`, which calls back into this function for each of ITS OWN un-named
        slots). Never returns a `ChooseProcess` - ambiguity is always resolved into the length of
        the returned list, so a caller can decide for itself whether that multiplicity should
        become an independent top-level choice (`_resolve_sub_process_slot`, for a name the `.pbm`
        author explicitly listed) or stay baked into an enclosing candidate
        (`_build_leaf_candidates`'s own cross-product branch, for everything nested beneath that).
        """
        leaves = [template_name] if self.library.is_leaf(template_name) else self.library.leaf_descendants(template_name)
        out: "list[Process]" = []
        for leaf in leaves:
            out.extend(self._build_leaf_candidates(leaf, role_bindings, {}, None, leaf))
        return out

    def _resolve_sub_process_slot(self, given_name: str, slot_template: str, slot_roles: RoleBindings) -> "Process | None":
        if given_name in self.named_processes:
            result = self._resolve_named_instance(given_name)
            if isinstance(result, ChooseProcess):
                self.model.add(result)  # flatten: register at the model level instead of nesting
                return None
            return result
        candidates = self._build_bare_candidates(given_name, slot_roles)
        if len(candidates) == 1:
            return candidates[0]
        # `given_name` was explicitly listed in the .pbm's OWN `processes:` list for a top-level
        # (or explicitly-nested) instance - a genuinely independent, author-declared structural
        # choice, unlike the anonymous, conditionally-relevant nesting `_build_leaf_candidates`
        # handles above - so flattening it to the model level here is correct.
        self.model.add(ChooseProcess(*candidates, name=self._anon_name(given_name)))
        return None

    # ---- named, top-level (or explicitly-declared nested) .pbm process instances ----

    def _resolve_named_instance(self, name: str) -> "Process | ChooseProcess | None":
        if name in self._built:
            return self._built[name]
        pspec = self.named_processes[name]
        params = self.library.process_params(pspec.template)
        role_bindings = self._role_bindings_from_args(params, pspec.args)

        leaves = [pspec.template] if self.library.is_leaf(pspec.template) else self.library.leaf_descendants(pspec.template)
        candidates: "list[Process]" = []
        for leaf in leaves:
            leaf_name = pspec.name if len(leaves) == 1 else self._anon_name(leaf)
            candidates.extend(
                self._build_leaf_candidates(leaf, role_bindings, pspec.consts, pspec.sub_processes, leaf_name)
            )
        result = candidates[0] if len(candidates) == 1 else ChooseProcess(*candidates, name=pspec.name)
        self._built[name] = result
        return result


def load_library(path: str) -> Library:
    return Library(parse(Path(path).read_text()))


def load_model(path: str, library: Library) -> Model:
    return ModelBuilder(library, parse(Path(path).read_text())).build()
