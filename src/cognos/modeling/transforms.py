"""Safe, target-hidden execution of LLM-authored feature transforms.

The reasoning layer proposes feature engineering as expressions over *existing feature columns*; the
engine evaluates them under two guarantees (ADR-0002):

1. **Target-hidden** — transforms are evaluated against a features-only frame (`X`); the target is
   never in scope, so a transform physically cannot leak the answer (even onto the labelled holdout).
2. **Safe** — expressions are AST-whitelisted (only column names, numeric literals, arithmetic, and a
   fixed set of ``np.*`` functions; no imports, builtins, attributes, dunders, or calls outside the
   whitelist). This is the "restricted DSL" mode of ADR-0002; it round-trips verbatim to an IMPACT
   derived field (IMPACT evaluates the same ``np``-qualified expression), so train-time and serve-time
   feature logic are identical.

(The ADR's "arbitrary code under OS-level isolation" remains the eventual default; this expression
executor is the safe, IMPACT-compatible mechanism that ships first.)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Whitelisted numpy functions available as np.<name> inside a transform expression.
SAFE_NP_FUNCS = {
    "log", "log1p", "log10", "sqrt", "cbrt", "exp", "expm1", "abs", "sign", "square",
    "clip", "where", "minimum", "maximum", "tanh", "floor", "ceil", "round",
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Attribute, ast.Compare, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
)


@dataclass
class TransformSpec:
    """A named feature transform: ``name = <expr over existing columns and np.* funcs>``."""

    name: str
    expr: str

    def to_impact_field(self) -> dict:
        """Emit as an IMPACT derived field (same expression evaluated at serve time)."""
        return {"name": self.name, "derived": self.expr, "dtype": "float64"}

    def to_dict(self) -> dict:
        return {"name": self.name, "expr": self.expr}


class UnsafeExpressionError(ValueError):
    pass


def validate_expr(expr: str, allowed_columns: set[str]) -> None:
    """Raise UnsafeExpressionError unless ``expr`` uses only whitelisted constructs."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpressionError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            # only np.<whitelisted_func>
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"
                    and node.attr in SAFE_NP_FUNCS):
                raise UnsafeExpressionError(f"disallowed attribute: {ast.dump(node)}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np" and node.func.attr in SAFE_NP_FUNCS):
                raise UnsafeExpressionError("only np.<whitelisted> calls are allowed")
        if isinstance(node, ast.Name):
            if "__" in node.id:
                raise UnsafeExpressionError("dunder names are not allowed")
            if node.id != "np" and node.id not in allowed_columns:
                raise UnsafeExpressionError(f"unknown name '{node.id}' (not a feature column)")


def evaluate_expr(expr: str, X: pd.DataFrame) -> np.ndarray:
    """Evaluate a validated expression against a features-only frame. Target is never in scope."""
    validate_expr(expr, set(X.columns))
    namespace = {col: X[col].to_numpy() for col in X.columns}
    namespace["np"] = _NpProxy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307 - sandboxed by validate_expr
    return np.asarray(result, dtype=float)


class _NpProxy:
    """A numpy facade exposing only the whitelisted functions."""

    def __getattr__(self, name: str):
        if name in SAFE_NP_FUNCS:
            return getattr(np, name)
        raise UnsafeExpressionError(f"np.{name} is not whitelisted")


def apply_transforms(
    X: pd.DataFrame, specs: list[TransformSpec]
) -> tuple[pd.DataFrame, list[TransformSpec], list[tuple[TransformSpec, str]]]:
    """Apply transforms to a features-only frame.

    Returns (augmented_X, applied, rejected). A transform is rejected (and skipped, never fatal) if it
    is unsafe, references the target/unknown columns, errors, or yields non-finite values — the
    accept-if-better ratchet then simply never benefits from a bad transform.
    """
    out = X.copy()
    applied: list[TransformSpec] = []
    rejected: list[tuple[TransformSpec, str]] = []
    for spec in specs:
        if spec.name in X.columns:
            rejected.append((spec, "name collides with an existing column"))
            continue
        try:
            values = evaluate_expr(spec.expr, X)
        except UnsafeExpressionError as exc:
            rejected.append((spec, f"unsafe: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 - any eval failure just drops the transform
            rejected.append((spec, f"error: {type(exc).__name__}: {exc}"))
            continue
        if values.shape[0] != len(X) or not np.all(np.isfinite(values)):
            rejected.append((spec, "non-finite or wrong-length result"))
            continue
        out[spec.name] = values
        applied.append(spec)
    return out, applied, rejected
