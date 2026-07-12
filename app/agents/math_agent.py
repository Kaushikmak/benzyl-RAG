"""Specialized MathAgent for secure arithmetic AST evaluation without eval()."""

import ast
import math
import re
from typing import Any, Optional


class SafeMathEvaluator(ast.NodeVisitor):
    """Secure recursive AST node visitor for arithmetic expressions.

    Permitted AST node types (Strict Whitelist):
    - ast.Expression: Root expression wrapper
    - ast.Constant: Numeric constants (int, float only; rejects str, bytes, complex)
    - ast.BinOp: Binary arithmetic operators (+, -, *, /, **, %)
    - ast.UnaryOp: Unary signs (+, -)
    - ast.Call: Function calls restricted strictly to SAFE_FUNCTIONS ('sqrt', 'abs', 'round', 'pow')
    Any other AST node (Attribute, Import, Lambda, Name outside whitelist, Subscript, etc.) raises ValueError.
    """

    SAFE_FUNCTIONS = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pow": math.pow,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Disallowed constant type: {type(node.value)}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left ** right
        elif isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError(f"Disallowed binary operator: {type(node.op)}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Disallowed unary operator: {type(node.op)}")

    def visit_Call(self, node: ast.Call) -> float:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.SAFE_FUNCTIONS:
                args = [self.visit(arg) for arg in node.args]
                return float(self.SAFE_FUNCTIONS[func_name](*args))
        raise ValueError("Disallowed function call in math expression")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def eval_math_ast(expression: str) -> float:
    """Safely parse and evaluate pure arithmetic expressions."""
    tree = ast.parse(expression, mode="eval")
    evaluator = SafeMathEvaluator()
    return evaluator.visit(tree)


class MathAgent:
    """Specialized MathAgent handling arithmetic evaluation."""

    def __init__(self):
        self._math_allowed_words = {"sqrt", "abs", "round", "pow"}

    def is_pure_math_expression(self, query: str) -> bool:
        cleaned = query.strip()
        if not re.search(r"\d", cleaned):
            return False

        words = set(re.findall(r"[a-zA-Z]+", cleaned.lower()))
        if words - self._math_allowed_words:
            return False

        if not re.search(r"[\+\-\*\/\^\%]", cleaned) and not words:
            return False

        return True

    def evaluate(self, query: str) -> Optional[str]:
        if not self.is_pure_math_expression(query):
            return None
        try:
            expr = query.replace("^", "**")
            result = eval_math_ast(expr)
            if result.is_integer():
                formatted_result = str(int(result))
            else:
                formatted_result = f"{result:.4f}".rstrip("0").rstrip(".")
            return (
                "### Instant Calculator Evaluation\n\n"
                f"**Expression**: `{query}`\n\n**Result**: `{formatted_result}`"
            )
        except Exception:
            return None
