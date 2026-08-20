"""
Dependency-graph analysis over an incomplete `Model` - finds groups of differential ("state")
variables whose structural search can be solved independently of each other.

Why this is possible
---------------------
Gradient matching (see `pybm.estimate.gradient_matching`) never simulates the ODE forward: every
variable's "current value" at a collocation point always comes from a *fixed* Gaussian-process fit
of its own observed data, never from evaluating some other slot's chosen candidate equation. That
means the constant-fit for one state variable's own equation is provably independent of a
structurally-unrelated variable's own equation, as long as neither shares a constant nor a
structural choice with it. Where that independence holds, a brute-force search over the full
Cartesian product of every open choice (`Model.induce()`) can instead be solved one much smaller
"block" at a time - see `build_blocks` below, and `pybm.estimate.decompose` for how a block is
actually fit.

What counts as "independent"
-----------------------------
Two state variables end up in the same block if resolving one choice could possibly change the
other's residual - concretely, if they (transitively, through the algebraic variables and
processes standing between them) share a `Choose`/`ChooseProcess` slot or a `Const` object. A
state variable that another state variable's equation merely *reads the value of* (e.g. a
predator's growth rate reading the prey's concentration) does *not* create a shared block: in
gradient matching that value always comes from the read variable's own GP fit, never from its
own choice of equation, so the two variables' own structural searches stay independent regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pybm.expr import BinOp, Expr, FuncExpr, MaxExpr, MinExpr, Neg, Num
from pybm.func import RelaxedIfExpr
from pybm.model import Choose, Const, Model, Var

# A variable's identity within an (incomplete) Model: the owning entity's name (None for a
# free-standing top-level variable) together with the variable's own name - the same convention
# `Model._lookup_var` uses.
VarKey = "tuple[str | None, str]"


# =====================================================================================
# Expr tree walking
# =====================================================================================


def collect_leaves(expr: Expr) -> "tuple[dict[int, Var], dict[int, Const]]":
    """
    Finds every `Var` and `Const` referenced anywhere inside an `Expr` tree.

    Recurses through every kind of node the equation-building layer (`pybm.expr`/`pybm.func`) can
    produce. Deliberately does *not* look inside a `Var`'s own `.ode`/`.algebraic` - knowing that a
    variable is *read* by an equation never depends on how that variable's *own* value is defined,
    which is exactly the property that lets `build_blocks` stop at another state variable instead
    of recursing into it (see the module docstring). A `Choose` never appears as a node inside an
    `Expr` tree (it is not an `Expr` subclass and has no operator overloads), so it needs no case
    here - callers that walk a `Var`'s equation deal with `Choose` themselves (see `_var_sources`).

    Parameters
    ----------
    expr : Expr
        The root of the expression tree to walk (e.g. `some_var.ode`, once it is a plain `Expr`
        and not a `Choose`).

    Returns
    -------
    tuple[dict[int, Var], dict[int, Const]]
        Every `Var`/`Const` found, keyed by `id(...)` (both are plain dataclasses without a
        meaningful `__eq__`/`__hash__` for this purpose, so identity is the only safe key).
    """
    if isinstance(expr, Var):
        return {id(expr): expr}, {}
    if isinstance(expr, Const):
        return {}, {id(expr): expr}
    if isinstance(expr, Num):
        return {}, {}
    if isinstance(expr, Neg):
        return collect_leaves(expr.operand)
    if isinstance(expr, BinOp):
        return _collect_from_many([expr.left, expr.right])
    if isinstance(expr, FuncExpr):
        return _collect_from_many(expr.args)
    if isinstance(expr, (MaxExpr, MinExpr)):
        return _collect_from_many(expr.operands)
    if isinstance(expr, RelaxedIfExpr):
        conditions_and_values = [part for branch in expr.branches for part in branch]
        return _collect_from_many([expr.default, *conditions_and_values])
    raise TypeError(
        f"collect_leaves does not know how to walk an Expr of type {type(expr).__name__!r}. "
        "Add a case for it above - silently skipping an unknown node type would under-count "
        "dependencies and could make build_blocks merge two structures that are not actually "
        "independent."
    )


def _collect_from_many(exprs: "list[Expr]") -> "tuple[dict[int, Var], dict[int, Const]]":
    """
    Runs `collect_leaves` over several `Expr`s and unions the results - the shared helper behind
    every `collect_leaves` case that has more than one child to walk.

    Parameters
    ----------
    exprs : list[Expr]
        The child expressions to walk.

    Returns
    -------
    tuple[dict[int, Var], dict[int, Const]]
        The union of `collect_leaves(e)` over every `e` in `exprs`, keyed by `id(...)`.
    """
    all_vars: "dict[int, Var]" = {}
    all_consts: "dict[int, Const]" = {}
    for expr in exprs:
        found_vars, found_consts = collect_leaves(expr)
        all_vars.update(found_vars)
        all_consts.update(found_consts)
    return all_vars, all_consts


# =====================================================================================
# Blocks
# =====================================================================================


@dataclass
class Block:
    """
    One group of state variables whose structural search is independent of every other block's.

    Parameters
    ----------
    state_vars : list[Var]
        The differential ("state") variables that belong to this block.
    external_state_vars : list[Var]
        State variables belonging to a *different* block that some equation in this block reads
        the value of (e.g. a predator's growth term reading the prey's concentration). Gradient
        matching already reads every state variable's value from its own GP fit rather than from
        simulating its equation, so these need no special handling when fitting this block - they
        are recorded here purely for inspection/debugging.
    var_slot_indices : set[int]
        Indices into the model's `_open_var_choices()` list: which top-level `Choose` slots this
        block's own structural search ranges over.
    process_slot_indices : set[int]
        Indices into the model's `pending_process_choices` (in `dict` iteration order): which
        top-level `ChooseProcess` slots this block's own structural search ranges over.
    const_ids : set[int]
        `id(...)` of every `Const` referenced anywhere in this block's own equations - used only
        to detect when two blocks should be merged (see `build_blocks`), never for indexing.
    """

    state_vars: "list[Var]"
    external_state_vars: "list[Var]" = field(default_factory=list)
    var_slot_indices: "set[int]" = field(default_factory=set)
    process_slot_indices: "set[int]" = field(default_factory=set)
    const_ids: "set[int]" = field(default_factory=set)


def _all_named_vars(model: Model) -> "dict[VarKey, Var]":
    """
    Collects every `Var` reachable from an incomplete `Model`, keyed by its `(entity_name,
    var_name)` identity.

    Parameters
    ----------
    model : Model
        The incomplete model to walk.

    Returns
    -------
    dict[VarKey, Var]
        Every entity-owned and free-standing `Var` in `model`, keyed the same way
        `Model._lookup_var` addresses them.
    """
    named: "dict[VarKey, Var]" = {}
    for entity_name, entity in model.entities.items():
        for var in entity.vars():
            named[(entity_name, var.name)] = var
    for var in model.free_vars.values():
        named[(None, var.name)] = var
    return named


def _is_state_var(model: Model, var: Var) -> bool:
    """
    Decides whether `var` is a differential ("state") variable - one that will end up with
    `var.ode is not None` once the model is induced.

    A variable counts as a state variable if it already has a differential equation set directly
    (an `Expr`, or a not-yet-resolved `Choose`), or if any process - fixed (`model.processes`) or
    still ambiguous (any candidate of a `model.pending_process_choices` slot) - targets it with an
    `Ode`. This mirrors `pybm.estimate.multishooting_torch._split_endo_vars`'s post-induction
    classification, computed here *before* induction (see the module docstring).

    Parameters
    ----------
    model : Model
        The incomplete model `var` belongs to.
    var : Var
        The variable to classify.

    Returns
    -------
    bool
        `True` if `var` is (or will become) a state variable.
    """
    if var.ode is not None:
        return True
    for process in model.processes.values():
        if any(ode.var is var for ode in process.all_odes()):
            return True
    for choose_process in model.pending_process_choices.values():
        for candidate in choose_process.options:
            if any(ode.var is var for ode in candidate.all_odes()):
                return True
    return False


def _var_sources(
    model: Model,
    entity_name: "str | None",
    var_name: str,
    attr: str,
    var_choices: "list[tuple[str | None, str, str]]",
    process_slot_names: "list[str]",
) -> "tuple[list[Expr], set[int], set[int]]":
    """
    Finds every way a variable's `attr` equation (`"ode"` or `"algebraic"`) could end up being
    defined, across every currently-open structural choice.

    A process-based contribution always wins over a value set directly on the variable, exactly as
    `Model._induce_single` resolves it at induction time (a variable's own `.ode`/`.algebraic` is
    only kept as-is when *no* process contributes to it). When several processes/candidates
    contribute (e.g. one growth process and one mortality process both writing to the same
    variable's derivative, or several leaf candidates of the same still-open `ChooseProcess`), this
    returns *all* of them - a conservative union, so a slot that could possibly affect this
    equation is never missed, even for a candidate that will not end up being chosen. A choice that
    is never actually selected can still force two blocks to merge that way; that is safe, it only
    ever makes the resulting blocks a little larger (never smaller) than strictly necessary.

    Parameters
    ----------
    model : Model
        The incomplete model `var_name` belongs to.
    entity_name : str | None
        The owning entity's name, or `None` for a free-standing top-level variable.
    var_name : str
        The variable's own name.
    attr : str
        Which equation to look up - `"ode"` or `"algebraic"`.
    var_choices : list[tuple[str | None, str, str]]
        `model._open_var_choices()`, computed once by the caller and reused - the source of truth
        for `Choose`-slot indices.
    process_slot_names : list[str]
        `list(model.pending_process_choices.keys())`, computed once by the caller and reused - the
        source of truth for `ChooseProcess`-slot indices.

    Returns
    -------
    tuple[list[Expr], set[int], set[int]]
        `(exprs, var_slot_indices, process_slot_indices)`: every candidate `Expr` that could define
        this equation, and the indices (into `var_choices`/`process_slot_names`) of every slot
        whose resolution could change which of them is picked.
    """
    var = model._lookup_var(entity_name, var_name)
    exprs: "list[Expr]" = []
    process_slot_indices: "set[int]" = set()

    for process in model.processes.values():
        entries = process.all_odes() if attr == "ode" else process.all_aes()
        exprs.extend(entry.expr for entry in entries if entry.var is var)

    for slot_index, slot_name in enumerate(process_slot_names):
        for candidate in model.pending_process_choices[slot_name].options:
            entries = candidate.all_odes() if attr == "ode" else candidate.all_aes()
            for entry in entries:
                if entry.var is var:
                    exprs.append(entry.expr)
                    process_slot_indices.add(slot_index)

    if exprs:
        return exprs, set(), process_slot_indices

    # No process contributes to this equation at all - fall back to whatever is set directly.
    equation = getattr(var, attr)
    if equation is None:
        return [], set(), set()
    if isinstance(equation, Choose):
        slot_index = var_choices.index((entity_name, var_name, attr))
        return list(equation.options), {slot_index}, set()
    return [equation], set(), set()


class _BlockBuilder:
    """
    Builds one `Block` by walking a single state variable's dependency graph.

    Shared, read-only context (the model and its two global slot orderings) is stored once on the
    instance; `build()` resets the per-call accumulators and can be invoked once per state
    variable. Kept as a small class rather than a bag of parameters threaded through a recursive
    function, purely for readability.
    """

    def __init__(
        self,
        model: Model,
        var_choices: "list[tuple[str | None, str, str]]",
        process_slot_names: "list[str]",
        state_var_keys: "set[VarKey]",
        var_id_to_key: "dict[int, VarKey]",
    ):
        """
        Parameters
        ----------
        model : Model
            The incomplete model being analyzed.
        var_choices : list[tuple[str | None, str, str]]
            `model._open_var_choices()`, shared across every block this builder is used for.
        process_slot_names : list[str]
            `list(model.pending_process_choices.keys())`, shared across every block.
        state_var_keys : set[VarKey]
            `(entity_name, var_name)` of every state variable in `model` - used to decide whether
            a referenced variable should be recursed into (it is algebraic) or recorded as
            external and left alone (it is itself a state variable, see the module docstring).
        var_id_to_key : dict[int, VarKey]
            `id(var) -> (entity_name, var_name)` for every named variable in `model` - lets a `Var`
            object found by `collect_leaves` be looked back up by name. A variable *not* present
            here is exogenous (it lives outside `model.entities`/`model.free_vars`' bookkeeping, or
            has no name of its own) and is simply ignored.
        """
        self._model = model
        self._var_choices = var_choices
        self._process_slot_names = process_slot_names
        self._state_var_keys = state_var_keys
        self._var_id_to_key = var_id_to_key

    def build(self, entity_name: "str | None", var_name: str) -> Block:
        """
        Builds the `Block` for one state variable.

        Parameters
        ----------
        entity_name : str | None
            The state variable's owning entity, or `None` if it is free-standing.
        var_name : str
            The state variable's own name.

        Returns
        -------
        Block
            A block containing exactly this one state variable (before any merging - see
            `build_blocks`), with every slot/const its own equation depends on recorded.
        """
        self._focal_key: VarKey = (entity_name, var_name)
        self._visited: "set[tuple[str | None, str, str]]" = set()
        self._var_slot_indices: "set[int]" = set()
        self._process_slot_indices: "set[int]" = set()
        self._const_ids: "set[int]" = set()
        self._external: "list[Var]" = []

        self._walk(entity_name, var_name, "ode")

        return Block(
            state_vars=[self._model._lookup_var(entity_name, var_name)],
            external_state_vars=self._external,
            var_slot_indices=self._var_slot_indices,
            process_slot_indices=self._process_slot_indices,
            const_ids=self._const_ids,
        )

    def _walk(self, entity_name: "str | None", var_name: str, attr: str) -> None:
        """
        Visits one variable's equation, records the slots/consts it depends on, and recurses into
        every algebraic variable it reads (stopping at exogenous variables and at other state
        variables - see the module docstring). Idempotent: visiting the same `(entity, var, attr)`
        twice (e.g. two different equations both reading the same algebraic variable) is a no-op
        the second time.

        Parameters
        ----------
        entity_name : str | None
            The variable's owning entity, or `None` if free-standing.
        var_name : str
            The variable's own name.
        attr : str
            Which equation to walk - `"ode"` (only ever the entry point, for the focal state
            variable itself) or `"algebraic"` (every variable reached by recursion).
        """
        key = (entity_name, var_name, attr)
        if key in self._visited:
            return
        self._visited.add(key)

        exprs, var_slots, process_slots = _var_sources(
            self._model, entity_name, var_name, attr, self._var_choices, self._process_slot_names
        )
        self._var_slot_indices.update(var_slots)
        self._process_slot_indices.update(process_slots)

        referenced_vars, referenced_consts = _collect_from_many(exprs)
        self._const_ids.update(referenced_consts.keys())

        for other_var in referenced_vars.values():
            other_key = self._var_id_to_key.get(id(other_var))
            if other_key is None or other_key == self._focal_key:
                continue  # exogenous, or this equation referencing its own variable (e.g. -k*x)
            if other_key in self._state_var_keys:
                if not any(existing is other_var for existing in self._external):
                    self._external.append(other_var)
                continue
            self._walk(other_key[0], other_key[1], "algebraic")


def _blocks_share_anything(a: Block, b: Block) -> bool:
    """
    Decides whether two blocks must be merged: they do if their structural searches range over a
    common slot, or their equations reference a common constant (a shared constant is not itself a
    "slot", but jointly fitting it would still couple the two blocks' least-squares problems, so it
    has to be caught the same way).

    Parameters
    ----------
    a, b : Block
        The two blocks to compare.

    Returns
    -------
    bool
        `True` if `a` and `b` share a `Choose` slot, a `ChooseProcess` slot, or a `Const`.
    """
    return bool(
        a.var_slot_indices & b.var_slot_indices
        or a.process_slot_indices & b.process_slot_indices
        or a.const_ids & b.const_ids
    )


def _merge_two_blocks(a: Block, b: Block) -> Block:
    """
    Combines two blocks that `_blocks_share_anything` found to be connected into one.

    Parameters
    ----------
    a, b : Block
        The two blocks to merge.

    Returns
    -------
    Block
        A single block containing the union of both blocks' state variables, slots and constants.
        Any variable that was "external" to one of them but is now one of the merged block's own
        state variables is dropped from `external_state_vars` (see `Block`'s docstring) - it no
        longer counts as external once it is inside the same block.
    """
    state_vars = a.state_vars + b.state_vars
    state_var_ids = {id(var) for var in state_vars}

    merged_external: "list[Var]" = []
    seen_external_ids: "set[int]" = set()
    for var in a.external_state_vars + b.external_state_vars:
        if id(var) in state_var_ids or id(var) in seen_external_ids:
            continue
        seen_external_ids.add(id(var))
        merged_external.append(var)

    return Block(
        state_vars=state_vars,
        external_state_vars=merged_external,
        var_slot_indices=a.var_slot_indices | b.var_slot_indices,
        process_slot_indices=a.process_slot_indices | b.process_slot_indices,
        const_ids=a.const_ids | b.const_ids,
    )


def _merge_overlapping_blocks(blocks: "list[Block]") -> "list[Block]":
    """
    Repeatedly merges any two blocks that `_blocks_share_anything` until no more merges are
    possible - the connected-components step of `build_blocks`.

    Parameters
    ----------
    blocks : list[Block]
        One block per state variable, as built independently by `_BlockBuilder`.

    Returns
    -------
    list[Block]
        The same blocks, with every pair that shares a slot or constant merged together
        (transitively - if `a` shares with `b` and `b` shares with `c`, all three end up in one
        block, even if `a` and `c` share nothing directly).
    """
    blocks = list(blocks)
    merged_this_pass = True
    while merged_this_pass:
        merged_this_pass = False
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                if _blocks_share_anything(blocks[i], blocks[j]):
                    blocks[i] = _merge_two_blocks(blocks[i], blocks[j])
                    del blocks[j]
                    merged_this_pass = True
                    break
            if merged_this_pass:
                break
    return blocks


def build_blocks(model: Model) -> "list[Block]":
    """
    Partitions every state variable in an incomplete model into independent blocks - the entry
    point of this module (see the module docstring for why this partition is valid).

    Algorithm: build one block per state variable in isolation (walking its own dependency graph
    via `_BlockBuilder`), then repeatedly merge any two blocks that turn out to share a structural
    slot or a constant (`_merge_overlapping_blocks`) - the same "build small, then merge on shared
    resources" shape as a standard connected-components computation.

    Parameters
    ----------
    model : Model
        The incomplete model to analyze. Not modified.

    Returns
    -------
    list[Block]
        One block per group of mutually-dependent state variables. Two different blocks' own
        structural searches can be solved (and scored) in any order, or in parallel, with no loss
        of accuracy relative to searching the whole model's Cartesian product at once - see
        `pybm.estimate.decompose.estimate_model_decomposed`.
    """
    var_choices = model._open_var_choices()
    process_slot_names = list(model.pending_process_choices.keys())
    named_vars = _all_named_vars(model)
    var_id_to_key = {id(var): key for key, var in named_vars.items()}

    state_var_keys = {key for key, var in named_vars.items() if _is_state_var(model, var)}

    builder = _BlockBuilder(model, var_choices, process_slot_names, state_var_keys, var_id_to_key)
    blocks = [builder.build(entity_name, var_name) for entity_name, var_name in state_var_keys]

    return _merge_overlapping_blocks(blocks)
