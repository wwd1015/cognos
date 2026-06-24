"""Safe, target-hidden feature-transform execution."""

from __future__ import annotations

import pandas as pd
import pytest

from cognos.modeling.transforms import (
    TransformSpec,
    UnsafeExpressionError,
    apply_transforms,
    evaluate_expr,
    validate_expr,
)


def _X() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [10.0, 20, 30, 40]})


def test_valid_expressions_apply():
    X = _X()
    out, applied, rejected = apply_transforms(
        X, [TransformSpec("ratio", "a / b"), TransformSpec("la", "np.log1p(a)")])
    assert "ratio" in out.columns and "la" in out.columns
    assert len(applied) == 2 and not rejected
    assert evaluate_expr("a + b", X).tolist() == [11, 22, 33, 44]


def test_target_hidden_unknown_name_rejected():
    # The target column is not in the features-only frame, so a transform cannot reference it.
    X = _X()
    out, applied, rejected = apply_transforms(X, [TransformSpec("leak", "target * 1.0")])
    assert not applied and rejected and "leak" not in out.columns


@pytest.mark.parametrize("expr", [
    "__import__('os')", "a.__class__", "open('x')", "np.foo(a)", "np.linalg(a)",
    "(0).__class__.__bases__", "lambda x: x",
])
def test_unsafe_constructs_rejected(expr):
    with pytest.raises(UnsafeExpressionError):
        validate_expr(expr, {"a", "b"})


def test_to_impact_field_roundtrip():
    f = TransformSpec("z", "np.log1p(a)").to_impact_field()
    assert f == {"name": "z", "derived": "np.log1p(a)", "dtype": "float64"}


def test_collision_and_nonfinite_rejected():
    X = _X()
    out, applied, rejected = apply_transforms(
        X, [TransformSpec("a", "a * 2"), TransformSpec("div0", "a / (a - a)")])
    names = {spec.name for spec, _ in rejected}
    assert "a" in names  # collides with existing column
    assert "div0" in names  # non-finite (0/0)
    assert not applied
