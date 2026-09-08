"""A safe arithmetic tool — no eval() of arbitrary strings."""

from __future__ import annotations

import ast
import operator
from typing import ClassVar

from pydantic import BaseModel, Field

from agent_toolkit.tools.base import BaseTool, ToolError

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ..., description="A pure arithmetic expression, e.g. '(12 + 4) * 3 / 2'"
    )


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate a pure arithmetic expression (add, subtract, multiply, "
        "divide, power, modulo). Use this instead of doing math yourself."
    )
    args_model: ClassVar[type[BaseModel]] = CalculatorArgs

    def run(self, expression: str) -> dict:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _safe_eval(tree)
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero") from exc
        except (SyntaxError, ValueError) as exc:
            raise ToolError(f"Could not parse expression: {exc}") from exc
        return {"expression": expression, "result": value}
