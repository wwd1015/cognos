"""Shared test fixtures: synthetic datasets + small/fast CognosConfig builders."""

from __future__ import annotations

import pytest

from cognos import synth
from cognos.config import CognosConfig


def _write_csv(tmp_path, df, name) -> str:
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def _offline_agents(monkeypatch):
    """Tests never need a key or network: pin the deterministic (heuristic) agents."""
    monkeypatch.setenv("COGNOS_PROVIDER", "heuristic")


@pytest.fixture
def replay_dir(tmp_path, monkeypatch):
    """Factory: replay_dir({agent: output | [outputs...]}) -> a recorded-output directory, with the
    replay provider selected (agents without a recording fall back to the heuristic agent)."""
    import json

    def _make(recordings: dict) -> str:
        d = tmp_path / "recorded"
        d.mkdir(exist_ok=True)
        for agent, out in recordings.items():
            (d / f"{agent}.json").write_text(json.dumps(out), encoding="utf-8")
        monkeypatch.setenv("COGNOS_PROVIDER", "replay")
        return str(d)

    return _make


@pytest.fixture
def runs_dir(tmp_path) -> str:
    return str(tmp_path / "runs")


@pytest.fixture
def make_config(tmp_path):
    """Factory: make_config('regression'|'classification'|'timeseries'|'credit') -> CognosConfig.

    Uses small search budgets so the full pipeline runs in a couple of seconds.
    """

    def _make(task: str = "regression", **overrides) -> CognosConfig:
        presets = {
            "regression": dict(gen="regression", task="regression", target="target", metric="rmse"),
            "classification": dict(gen="classification", task="classification", target="target", metric="roc_auc"),
            "timeseries": dict(gen="timeseries", task="timeseries", target="target", metric="rmse",
                               datetime_col="date"),
            "credit": dict(gen="credit", task="classification", target="default", metric="roc_auc",
                           protected=["group"], fair_lending=True),
            "commercial": dict(gen="commercial", task="classification", target="default",
                               metric="roc_auc", datetime_col="vintage"),
        }
        p = presets[task]
        # Credit needs a larger sample so the injected group disparity is statistically present
        # in the holdout (the fair-lending disparate-impact gate operates on the sealed holdout).
        n = 800 if task in ("credit", "commercial") else 240
        df = synth.GENERATORS[p["gen"]](n=n)
        csv = _write_csv(tmp_path, df, task)
        raw = {
            "name": f"test_{task}",
            "description": f"test {task} model",
            "task": p["task"],
            "data": {
                "path": csv, "format": "csv", "target": p["target"],
                "datetime_col": p.get("datetime_col"),
                "protected_attributes": p.get("protected", []),
            },
            "metric": {"name": p["metric"]},
            "search": {"max_candidates": 8, "cv_folds": 3, "holdout_fraction": 0.2, "ensemble": True},
            "backtest": {"n_splits": 4},
            "compliance": {"fair_lending": p.get("fair_lending", False),
                           "jurisdictions": ["US", "EU"],
                           "risk_tier": "high" if task == "credit" else "medium"},
        }
        raw.update(overrides)
        return CognosConfig.from_dict(raw)

    return _make


def _fill_brief(cfg):
    """Fill the four sponsor decisions a package cannot be sealed without."""
    cfg.design.use_case = cfg.design.use_case or "origination underwriting"
    cfg.design.horizon = cfg.design.horizon or "12-month"
    cfg.design.default_definition = cfg.design.default_definition or "90+ DPD or nonaccrual"
    cfg.design.segment = cfg.design.segment or "test book"
    return cfg


@pytest.fixture
def apply_brief():
    return _fill_brief


@pytest.fixture
def leak_config(tmp_path):
    """A regression dataset with a deliberately leaking feature (near-copy of the target).

    The champion will use it, `explore` flags it as a leakage suspect, and `validate` must BLOCK —
    the real (and now only) hard-BLOCK condition after compliance was demoted (ADR-0006).
    """
    import numpy as np

    df = synth.make_regression_dataset(n=240)
    df["leaky"] = df["target"] + np.random.default_rng(0).normal(0, 1e-3, len(df))
    csv = tmp_path / "leak.csv"
    df.to_csv(csv, index=False)
    return CognosConfig.from_dict({
        "name": "test_leak", "task": "regression",
        "data": {"path": str(csv), "target": "target"},
        "metric": {"name": "rmse"},
        "search": {"max_candidates": 8, "cv_folds": 3},
    })
