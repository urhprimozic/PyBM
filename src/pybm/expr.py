"""
Symbolic expression layer for PyBM.

An `Expr` is a small, composable node in an expression tree - the right-hand side of an
equation (e.g. the `rate * conc` in `d(conc)/dt = rate * conc`). Expr trees are built with plain
Python operators (`+ - * / ** -`) instead of hand-written `lambda t, ctx: ...` closures, so an
equation can be written as `rate * conc` directly.

Building an Expr tree never touches a Model or a context - it is pure structure, and can be done
freely while a model is still "incomplete" (see model.py). A tree only produces a number once it
is *evaluated*: `expr(t, ctx)` recursively asks each subexpression for its value at time `t`,
given a `ctx` holding the model's current variable/constant values. `Var`/`Const` (defined in
model.py, both subclass `Expr`) are the leaves of this tree - they read their value from `ctx` by
`index_in_ctx`, which is only assigned once a model is complete (`Model.induce()`). So: building/
combining expressions works at any time, but *evaluating* one requires a complete, induced model.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable


class Expr:
    """
    Base class for anything that can appear on the right-hand side of an equation: a single
    Var/Const, or a combination of them built with `+ - * / **`. Every subclass implements
    `__call__(t, ctx)` to compute its value, recursing into its children with the same `t`/`ctx` -
    this is what lets a whole tree be evaluated by calling only its root.
    """

    def __call__(self, t: Any, ctx: Any) -> Any:
        raise NotImplementedError

    def __add__(self, other: "Expr | float") -> "Expr":
        return BinOp(self, wrap(other), operator.add, "+")

    def __radd__(self, other: "Expr | float") -> "Expr":
        return BinOp(wrap(other), self, operator.add, "+")

    def __sub__(self, other: "Expr | float") -> "Expr":
        return BinOp(self, wrap(other), operator.sub, "-")

    def __rsub__(self, other: "Expr | float") -> "Expr":
        return BinOp(wrap(other), self, operator.sub, "-")

    def __mul__(self, other: "Expr | float") -> "Expr":
        return BinOp(self, wrap(other), operator.mul, "*")

    def __rmul__(self, other: "Expr | float") -> "Expr":
        return BinOp(wrap(other), self, operator.mul, "*")

    def __truediv__(self, other: "Expr | float") -> "Expr":
        return BinOp(self, wrap(other), operator.truediv, "/")

    def __rtruediv__(self, other: "Expr | float") -> "Expr":
        return BinOp(wrap(other), self, operator.truediv, "/")

    def __pow__(self, other: "Expr | float") -> "Expr":
        return BinOp(self, wrap(other), operator.pow, "**")

    def __neg__(self) -> "Expr":
        return Neg(self)


def wrap(value: "Expr | float") -> Expr:
    """Wraps a plain Python number as a constant-valued Expr leaf, so e.g. `var + 2.0` works."""
    return value if isinstance(value, Expr) else Num(value)


@dataclass
class Num(Expr):
    """A plain numeric literal used inside an expression, e.g. the `2.0` in `var * 2.0`."""

    value: float

    def __call__(self, t, ctx):
        return self.value

    def __repr__(self):
        return repr(self.value)


@dataclass
class Neg(Expr):
    """Unary minus: `-operand`."""

    operand: Expr

    def __call__(self, t, ctx):
        return -self.operand(t, ctx)

    def __repr__(self):
        return f"-{self.operand!r}"


@dataclass
class BinOp(Expr):
    """A binary operation between two Exprs, e.g. `left + right`. `symbol` is only used for
    `__repr__` - the actual computation is `op(left(t, ctx), right(t, ctx))`."""

    left: Expr
    right: Expr
    op: Callable[[Any, Any], Any]
    symbol: str

    def __call__(self, t, ctx):
        return self.op(self.left(t, ctx), self.right(t, ctx))

    def __repr__(self):
        return f"({self.left!r} {self.symbol} {self.right!r})"


@dataclass
class FuncExpr(Expr):
    """A named scalar function applied to one or more Exprs, e.g. `exp(rate * t)`. `name` is only
    used for `__repr__` - the actual computation is `func(*(arg(t, ctx) for arg in args))`."""

    func: Callable[..., Any]
    args: "list[Expr]"
    name: str = ""

    def __call__(self, t, ctx):
        return self.func(*(arg(t, ctx) for arg in self.args))

    def __repr__(self):
        label = self.name or getattr(self.func, "__name__", "func")
        return f"{label}({', '.join(repr(a) for a in self.args)})"


@dataclass
class MaxExpr(Expr):
    """Element-wise max of several Exprs. Used for variables with `aggregation="max"`."""

    operands: list[Expr]

    def __call__(self, t, ctx):
        return max(operand(t, ctx) for operand in self.operands)

    def __repr__(self):
        return f"max({', '.join(repr(o) for o in self.operands)})"


@dataclass
class MinExpr(Expr):
    """Element-wise min of several Exprs. Used for variables with `aggregation="min"`."""

    operands: list[Expr]

    def __call__(self, t, ctx):
        return min(operand(t, ctx) for operand in self.operands)

    def __repr__(self):
        return f"min({', '.join(repr(o) for o in self.operands)})"
