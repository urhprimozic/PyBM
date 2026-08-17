"""
Named math functions and a lazy if/elif/.../else expression, both built on top of `pybm.expr`.

Function wrappers (`exp`, `sin`, `cos`, `tan`, `arcsin`, `tanh`) turn a plain backend function
into an Expr-producing one: `exp(rate * conc)` is an Expr, not a number - it only calls the real
`exp` once actual values are read from a ctx (see `pybm.expr`'s module docstring), and picks the
right backend (numpy, torch, or jax.numpy) from the runtime type of its argument at that point -
the same "write the equation once, run it on whatever engine the values turn out to be" trick
`Expr`'s own `+ - * /` already get for free from Python's operator dispatch. Composes like any
other Expr, e.g. `Ode(v, exp(var1 + var2) + var3)`.

`If`/`Elif`/`Else` build a lazy, Expr-tree version of `pybm.conditional.RelaxedIfElse` - so an
equation can express a smooth if/elif/.../else the same way it expresses anything else (every
branch's condition and value are Exprs, evaluated only once `t`/`ctx` are known), instead of
`RelaxedIfElse` itself, which expects already-computed numbers up front. Usage:

    tempLim = If(c - env["temperature"])(expr1).Elif(d - env["temperature"])(expr2).Else(expr3)

is the lazy equivalent of `RelaxedIfElse().If(c - t, expr1).Elif(d - t, expr2).Else(expr3)`,
usable directly inside an `Ode`/`Algebraic`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch

from pybm.conditional import RelaxedIfElse
from pybm.expr import Expr, FuncExpr, wrap


# ============================================================ backend-dispatched math functions ====


def infer_engine(value: Any) -> Literal["scipy", "torch", "jax"]:
    """
    Picks a `RelaxedIfElse`-style engine name from the runtime type of an already-evaluated
    value. jax is checked first, and only by its module name - a jax array under jit/grad/vmap is
    a tracer, and letting anything else (numpy, `torch.is_tensor`) touch it first can silently
    break tracing (see `RelaxedIfElse._weight`'s own note on this).
    """
    if type(value).__module__.startswith("jax"):
        return "jax"
    if torch.is_tensor(value):
        return "torch"
    return "scipy"


def _dispatched_math(name: str) -> Callable[[Any], Any]:
    """Returns a scalar math function that picks `torch.<name>` / `jax.numpy.<name>` /
    `numpy.<name>` via `infer_engine`, so the same Expr works unmodified on any engine."""

    def dispatched(x):
        engine = infer_engine(x)
        if engine == "jax":
            import jax.numpy as jnp
            return getattr(jnp, name)(x)
        if engine == "torch":
            return getattr(torch, name)(x)
        return getattr(np, name)(x)

    dispatched.__name__ = name
    return dispatched


def _dispatched_binary(name: str, torch_name: "str | None" = None) -> Callable[[Any, Any], Any]:
    """Two-argument analogue of `_dispatched_math` (e.g. for `pow`/`min`/`max`). `torch_name`
    covers the rare case where torch's name differs from numpy/jax's (e.g. `pow` vs `power`)."""
    torch_name = torch_name or name

    def dispatched(a, b):
        engine = infer_engine(a)
        if engine == "jax":
            import jax.numpy as jnp
            return getattr(jnp, name)(a, b)
        if engine == "torch":
            return getattr(torch, torch_name)(a, b)
        return getattr(np, name)(a, b)

    return dispatched


def _make_unary_math(name: str) -> "Callable[[Expr | float], FuncExpr]":
    """Builds an Expr-producing wrapper for a scalar math function, e.g. `exp = _make_unary_math
    ("exp")` gives `exp(rate * conc)` an Expr."""
    func = _dispatched_math(name)

    def make_expr(x: "Expr | float") -> FuncExpr:
        return FuncExpr(func, [wrap(x)], name=name)

    return make_expr


exp = _make_unary_math("exp")
sin = _make_unary_math("sin")
cos = _make_unary_math("cos")
tan = _make_unary_math("tan")
arcsin = _make_unary_math("arcsin")
tanh = _make_unary_math("tanh")


# ============================================================ lazy relaxed if/elif/.../else ====


@dataclass
class RelaxedIfExpr(Expr):
    """
    Lazy Expr version of `RelaxedIfElse`: `branches` is an ordered list of (condition, value)
    Exprs, `default` is the else-branch Expr. Built via `If`/`Elif`/`Else` below, not directly.
    """

    branches: "list[tuple[Expr, Expr]]"
    default: Expr
    eps: float = 0.01
    method: str = "tanh"

    def __call__(self, t, ctx):
        # engine is inferred from the first evaluated value (any branch, or the default if there
        # are no branches) - RelaxedIfElse itself needs an explicit engine string internally, it
        # doesn't dispatch by type the way the plain math functions above do.
        probe = self.branches[0][0](t, ctx) if self.branches else self.default(t, ctx)
        relaxed = RelaxedIfElse(eps=self.eps, method=self.method, engine=infer_engine(probe))
        for condition, value in self.branches:
            relaxed = relaxed.If(condition(t, ctx), value(t, ctx))
        return relaxed.Else(self.default(t, ctx))

    def __repr__(self):
        branches = " ".join(f".Elif({c!r})({v!r})" for c, v in self.branches)
        return f"If{branches}.Else({self.default!r})"


class _PendingBranch:
    """Returned by `If(cond)` / `.Elif(cond)` - call it with that branch's value Expr to continue
    the chain, e.g. `If(cond)(value)`."""

    def __init__(self, chain: "_IfChain", condition: "Expr | float"):
        self._chain = chain
        self._condition = wrap(condition)

    def __call__(self, value: "Expr | float") -> "_IfChain":
        self._chain.branches.append((self._condition, wrap(value)))
        return self._chain


@dataclass
class _IfChain:
    """Accumulates branches for one `If(...)( ... ).Elif(...)( ... ).Else(...)` chain."""

    eps: float
    method: str
    branches: "list[tuple[Expr, Expr]]" = field(default_factory=list)

    def Elif(self, condition: "Expr | float") -> _PendingBranch:
        return _PendingBranch(self, condition)

    def Else(self, value: "Expr | float") -> RelaxedIfExpr:
        return RelaxedIfExpr(list(self.branches), wrap(value), eps=self.eps, method=self.method)


def If(condition: "Expr | float", eps: float = 0.01, method: str = "tanh") -> _PendingBranch:
    """
    Starts a lazy, smooth if/elif/.../else Expr - the Expr equivalent of
    `RelaxedIfElse(eps, method).If(...)`. Chain `.Elif(condition)(value)` for more branches, and
    finish with `.Else(value)`:

        If(c - v)(expr1).Elif(d - v)(expr2).Else(expr3)

    Every `condition`/`value` is an Expr (or a plain number, auto-wrapped) - nothing is computed
    until the resulting Expr itself is called with `(t, ctx)`.
    """
    return _PendingBranch(_IfChain(eps=eps, method=method), condition)
