"""
LaTeX for the equations of an induced model.

    model.equations()                                  # constants as symbols
    model.equations_estimated(result.const_by_name)    # constants replaced by estimated values

and plain text of a single expression, `text(expr, model)`.

Both return one `aligned` block: first every state variable's ODE, then every algebraic variable
in dependency order. In Jupyter: `IPython.display.Math(model.equations())`.

A smooth if/elif/else (`pybm.func.If`) is written as an ordinary `cases` block, with the condition
of each branch as `If(d)` means it: d < 0. The relaxation width `eps` is not shown.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pybm.expr import BinOp, Expr, FuncExpr, MaxExpr, MinExpr, Neg, Num
from pybm.func import RelaxedIfExpr
from pybm.model import Choose, Const, InducedModel, Var, _algebraic_order

# operator precedence of a rendered expression, to decide on parentheses. _SIGNED: a term with a
# leading minus (-a, -a · b): fine as the first factor of a product, not as a later one.
_ADD, _SIGNED, _MUL, _FRAC, _POW, _ATOM = 1, 2, 3, 4, 5, 6

_FUNCTIONS = {
    "exp": r"\exp", "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "tanh": r"\tanh", "arcsin": r"\arcsin",
    "log": r"\ln", "log10": r"\log_{10}", "sign": r"\operatorname{sign}",
    "max": r"\max", "maximum": r"\max", "min": r"\min", "minimum": r"\min",
    "relaxed_max": r"\max_{\varepsilon}", "relaxed_min": r"\min_{\varepsilon}",
}


def default_labels(model: InducedModel) -> "dict[str, str]":
    """`{var name: v_{name}}` for every variable and `{const name: \\theta_{name}}` for every constant."""
    labels = {name: rf"v_{{\text{{{_escape(name)}}}}}" for name in model.vars}
    labels.update({name: rf"\theta_{{\text{{{_escape(name)}}}}}" for name in model.consts})
    return labels


def equations(model: InducedModel, labels: "dict[str, str] | None" = None) -> str:
    """
    LaTeX of the model's equations, constants as symbols.

    Parameters
    ----------
    model : InducedModel
    labels : dict, optional
        Variable or constant name -> LaTeX symbol, e.g. `{"electron.mass": "m_{e}"}`. Names not
        in it get the default from `default_labels`.
    """
    return _equations(model, labels, values=None, digits=None)


def equations_estimated(model: InducedModel, consts, labels: "dict[str, str] | None" = None, round: int = 2) -> str:
    """
    LaTeX of the model's equations with the constants replaced by their estimated values.

    Parameters
    ----------
    model : InducedModel
    consts : dict, array-like or estimation result
        `{const name: value}` (e.g. `result.const_by_name`), an array in `index_in_ctx` order
        (e.g. `result.consts`), or a result object with `.const_by_name`.
    labels : dict, optional
        As in `equations` (only the variables' labels are used).
    round : int
        Decimals of the estimated values. Numbers written into the equations themselves are not
        rounded - they are part of the model.
    """
    if hasattr(consts, "const_by_name"):
        consts = consts.const_by_name
    if isinstance(consts, dict):
        values = {name: float(consts[name]) for name in model.consts if name in consts}
    else:
        consts = np.asarray(consts, dtype=float)
        values = {name: float(consts[c.index_in_ctx]) for name, c in model.consts.items()}
    return _equations(model, labels, values=values, digits=round)


def text(expr: Expr, model=None) -> str:
    """
    An expression as plain text, e.g. "((1.5 * prey.conc) * predator.conc)" - every binary operation
    in parentheses, so the text shows the tree. Variables and constants are named as in `model.vars`
    / `model.consts` (a Model or an InducedModel); without a model, or for one it doesn't list (e.g. a
    process constant before induce()), as entity.name, or by their own name outside an entity.
    """
    names = {} if model is None else {id(obj): name for name, obj in {**model.vars, **model.consts}.items()}

    def render(e) -> str:
        if isinstance(e, (Var, Const)):
            return names.get(id(e), e.name if e.entity is None else f"{e.entity.name}.{e.name}")
        if isinstance(e, Num):
            return repr(e.value)
        if isinstance(e, Neg):
            return f"-{render(e.operand)}"
        if isinstance(e, BinOp):
            return f"({render(e.left)} {e.symbol} {render(e.right)})"
        if isinstance(e, FuncExpr):
            return f"{e.name}({', '.join(render(a) for a in e.args)})"
        if isinstance(e, (MaxExpr, MinExpr)):
            return f"{'max' if isinstance(e, MaxExpr) else 'min'}({', '.join(render(o) for o in e.operands)})"
        if isinstance(e, RelaxedIfExpr):
            chain = "".join(f".{'If' if i == 0 else 'Elif'}({render(c)})({render(v)})" for i, (c, v) in enumerate(e.branches))
            return f"{chain[1:]}.Else({render(e.default)})"
        if isinstance(e, Choose):
            return f"Choose({', '.join(render(o) for o in e.options)})"
        raise TypeError(f"no text for an Expr of type {type(e).__name__!r}.")

    return render(expr)

def _equations(model, labels, values, digits) -> str:
    if not isinstance(model, InducedModel):
        raise TypeError("equations need an InducedModel: call induce() first, the processes' contributions are combined there.")
    labels = {**default_labels(model), **(labels or {})}
    names = {id(v): n for n, v in model.vars.items()} | {id(c): n for n, c in model.consts.items()}
    render = _Renderer(names, labels, values, digits)

    state_vars, algebraic_vars, _ = model.split_endo_vars()
    lines = [rf"\frac{{\mathrm{{d}}{labels[names[id(v)]]}}}{{\mathrm{{d}}t}} &= {render(v.ode)}" for v in state_vars]
    lines += [rf"{labels[names[id(v)]]} &= {render(v.algebraic)}" for v in _algebraic_order(algebraic_vars) or algebraic_vars]
    return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"


class _Renderer:
    def __init__(self, names: "dict[int, str]", labels: "dict[str, str]", values: "dict[str, float] | None", digits):
        self.names, self.labels, self.values, self.digits = names, labels, values, digits

    def __call__(self, expr: Expr) -> str:
        return self.render(expr)[0]

    def wrap(self, expr: Expr, at_least: int) -> str:
        """expr, in parentheses if it binds weaker than `at_least`."""
        text, precedence = self.render(expr)
        return text if precedence >= at_least else rf"\left({text}\right)"

    def render(self, expr: Any) -> "tuple[str, int]":
        if isinstance(expr, Const):
            name = self.names[id(expr)]
            if self.values is not None and name in self.values:
                value = self.values[name]
                return f"{value:.{self.digits}f}", _ATOM if value >= 0 else _SIGNED
            return self.labels[name], _ATOM
        if isinstance(expr, Var):
            return self.labels[self.names[id(expr)]], _ATOM
        if isinstance(expr, Num):
            return _number(expr.value), _ATOM if expr.value >= 0 else _SIGNED
        if isinstance(expr, Neg):
            return "-" + self.wrap(expr.operand, _MUL), _SIGNED
        if isinstance(expr, BinOp):
            return self.binop(expr.symbol, expr.left, expr.right)
        if isinstance(expr, FuncExpr):
            if expr.name in ("pow", "power"):
                return self.binop("**", *expr.args)
            command = _FUNCTIONS.get(expr.name, rf"\operatorname{{{_escape(expr.name)}}}")
            return rf"{command}\left({', '.join(self(a) for a in expr.args)}\right)", _ATOM
        if isinstance(expr, (MaxExpr, MinExpr)):
            command = r"\max" if isinstance(expr, MaxExpr) else r"\min"
            return rf"{command}\left({', '.join(self(o) for o in expr.operands)}\right)", _ATOM
        if isinstance(expr, RelaxedIfExpr):
            rows = [rf"{self(value)} & \text{{{'if' if i == 0 else 'else if'} }} {self.condition(cond)}"
                    for i, (cond, value) in enumerate(expr.branches)]
            rows.append(rf"{self(expr.default)} & \text{{otherwise}}")
            return "\\begin{cases} " + r" \\ ".join(rows) + " \\end{cases}", _ATOM
        if isinstance(expr, Choose):
            raise ValueError("unresolved Choose in an equation - call induce() first.")
        raise TypeError(f"no LaTeX for an Expr of type {type(expr).__name__!r}.")

    def binop(self, symbol: str, left: Expr, right: Expr) -> "tuple[str, int]":
        if symbol == "+":
            text = self(right)
            # a + (-b ...) reads as a - b ...: a leading minus only ever belongs to the first term
            return (f"{self(left)} - {text[1:]}" if text.startswith("-") else f"{self(left)} + {text}"), _ADD
        if symbol == "-":
            return f"{self(left)} - {self.wrap(right, _MUL)}", _ADD
        if symbol == "*":
            for one, other in ((left, right), (right, left)):  # 1 · x is x (also when "1" is 1 · 1)
                if self(one) == "1":
                    return self.render(other)
            text = rf"{self.wrap(left, _SIGNED)} \cdot {self.wrap(right, _MUL)}"  # (-a) · b reads as -a · b
            return text, _SIGNED if text.startswith("-") else _MUL
        if symbol == "/":
            return rf"\frac{{{self(left)}}}{{{self(right)}}}", _FRAC
        if symbol == "**":
            return f"{{{self.wrap(left, _ATOM)}}}^{{{self(right)}}}", _POW
        raise ValueError(f"unknown operator {symbol!r}")

    def condition(self, cond: Expr) -> str:
        """If(d) takes its branch where d < 0; d = a - b reads as a < b."""
        if isinstance(cond, BinOp) and cond.symbol == "-":
            return f"{self(cond.left)} < {self(cond.right)}"
        return f"{self(cond)} < 0"


def _number(value: float) -> str:
    if np.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if 1e-4 <= abs(value) < 1e6:
        return np.format_float_positional(value, trim="-")
    mantissa, exponent = f"{value:.6g}".split("e")
    power = rf"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else ("-" + power if mantissa == "-1" else rf"{mantissa} \cdot {power}")


_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
    "{": r"\{", "}": r"\}", "^": r"\^{}", "~": r"\~{}",
})


def _escape(text: str) -> str:
    """Escapes LaTeX's special characters for use inside \\text{} (in one pass, so the braces of
    one replacement are not escaped again)."""
    return text.translate(_ESCAPES)
