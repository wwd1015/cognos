"""Config, artifacts, and feature-selection contracts."""

from __future__ import annotations

import pandas as pd

from cognos.artifacts import Finding, RunSummary, Severity, StageResult, Verdict
from cognos.config import CognosConfig, Direction, TaskType
from cognos.datautil import coerce_target, select_features


def test_verdict_semantics():
    assert Verdict.PASS.ok and Verdict.WARN.ok and Verdict.SKIP.ok
    assert not Verdict.BLOCK.ok and not Verdict.FAIL.ok
    assert Verdict.BLOCK.blocking and Verdict.ERROR.blocking
    assert not Verdict.PASS.blocking


def test_severity_rank_order():
    assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.MEDIUM.rank > Severity.LOW.rank > Severity.INFO.rank


def test_stage_result_tokens():
    r = StageResult(stage="model", verdict=Verdict.WARN)
    r.add_finding(Finding(id="x", severity=Severity.HIGH, category="c", message="m"))
    assert "model" in r.token_line() and "WARN" in r.token_line()
    assert r.max_severity() == Severity.HIGH


def test_run_summary_block():
    s = RunSummary(run_id="r1", mode="autonomous", project="p", stages_run=["explore"],
                   verdicts={"explore": "PASS"})
    block = s.token_block()
    assert "run_id: r1" in block and "verdict.explore: PASS" in block


def test_config_auto_metric_classification():
    cfg = CognosConfig.from_dict({
        "name": "c", "task": "classification",
        "data": {"target": "y"}, "metric": {"name": "auto"},
    })
    assert cfg.metric.name == "roc_auc"
    assert cfg.metric.direction == Direction.MAXIMIZE


def test_config_auto_metric_regression():
    cfg = CognosConfig.from_dict({"name": "r", "task": "regression", "data": {"target": "y"}})
    assert cfg.metric.name == "rmse"
    assert cfg.metric.direction == Direction.MINIMIZE
    assert cfg.task == TaskType.REGRESSION


def test_select_features_excludes_protected_and_target():
    cfg = CognosConfig.from_dict({
        "name": "c", "task": "classification",
        "data": {"target": "default", "protected_attributes": ["group"], "datetime_col": "date"},
    })
    df = pd.DataFrame({"x1": [1, 2], "group": ["A", "B"], "date": [1, 2], "default": [0, 1]})
    feats = select_features(df, cfg)
    assert feats == ["x1"]  # target, protected, datetime all excluded


def test_coerce_target_classification_int():
    cfg = CognosConfig.from_dict({"name": "c", "task": "classification", "data": {"target": "y"}})
    df = pd.DataFrame({"y": [0.0, 1.0, 0.0]})
    y = coerce_target(df, cfg)
    assert y.dtype.kind in ("i", "u") and list(y) == [0, 1, 0]


def test_config_yaml_roundtrip(tmp_path):
    cfg = CognosConfig.from_dict({"name": "rt", "task": "regression", "data": {"target": "y"}})
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    cfg2 = CognosConfig.from_yaml(p)
    assert cfg2.name == "rt" and cfg2.task == TaskType.REGRESSION
