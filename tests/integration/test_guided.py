"""LLM-guided search (reasoning proposes, engine disposes) driven by a deterministic scripted brain."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cognos import synth
from cognos.brains import ScriptedBrain
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator, run_pipeline
from cognos.runtime.score import load_scorer


def _quad_config(tmp_path) -> CognosConfig:
    """A dataset whose real signal is quadratic in x1, so a squared transform genuinely helps."""
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-3, 3, 400)
    x2 = rng.normal(0, 1, 400)
    y = 2.0 * x1**2 + 0.5 * x2 + rng.normal(0, 0.5, 400)
    csv = tmp_path / "quad.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "target": y}).to_csv(csv, index=False)
    return CognosConfig.from_dict({
        "name": "guided", "task": "regression",
        "data": {"path": str(csv), "target": "target"},
        "metric": {"name": "rmse"},
        "search": {"max_candidates": 6, "cv_folds": 3, "guided": True, "guided_rounds": 3},
        "stages": {"enabled": ["explore", "ideate", "model"]},
    })


def test_scripted_brain_replays():
    b = ScriptedBrain(["one", "two"])
    assert b.available
    assert [b.generate("p"), b.generate("p"), b.generate("p")] == ["one", "two", "one"]


def test_guided_search_keeps_verified_transform(tmp_path, runs_dir):
    cfg = _quad_config(tmp_path)
    proposal = json.dumps({"family": "ols",
                           "transforms": [{"name": "x1_sq", "expr": "np.square(x1)"}],
                           "rationale": "quadratic signal"})
    orch = Orchestrator(cfg, runs_root=runs_dir, brain=ScriptedBrain([proposal]))
    orch.run()
    mp = orch.ctx.require("model").payload

    # The engine independently verified the LLM-proposed transform improves the metric, then kept it.
    assert mp["guided"]["improved"] is True
    assert any(t["name"] == "x1_sq" for t in mp["transforms"])
    assert "x1_sq" in mp["champion"]["features"]
    # The reasoning trajectory is recorded for replay/audit (ADR-0003).
    assert (orch.ctx.reasoning_dir / "transcript.jsonl").exists()
    # The deployed scorer re-applies the transform target-hidden on RAW base features.
    load_scorer.cache_clear()
    bundle = load_scorer(mp["scorer_path"])
    preds = bundle.score_frame(pd.DataFrame({"x1": [1.0, -2.0], "x2": [0.0, 0.5]}))
    assert preds.shape == (2,) and np.all(np.isfinite(preds))


def test_guided_off_by_default_without_brain(make_config, runs_dir):
    # Default config has guided=False and the heuristic brain => no guided phase, no transforms.
    cfg = make_config("regression")
    ctx, _ = run_pipeline(cfg, runs_root=runs_dir)
    mp = ctx.require("model").payload
    assert mp["guided"] is None
    assert mp["transforms"] == []


def test_guided_rejects_unsafe_proposal(tmp_path, runs_dir):
    # An unsafe / target-referencing proposal is dropped by the engine; the run still completes.
    df = synth.make_regression_dataset(300)
    csv = tmp_path / "lin.csv"
    df.to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "g", "task": "regression",
        "data": {"path": str(csv), "target": "target"},
        "metric": {"name": "rmse"},
        "search": {"max_candidates": 6, "cv_folds": 3, "guided": True, "guided_rounds": 2},
        "stages": {"enabled": ["explore", "ideate", "model"]},
    })
    bad = json.dumps({"family": "ols", "transforms": [{"name": "leak", "expr": "target * 2"}]})
    orch = Orchestrator(cfg, runs_root=runs_dir, brain=ScriptedBrain([bad]))
    orch.run()
    mp = orch.ctx.require("model").payload
    assert mp["guided"]["rounds"] >= 1
    assert all(t["name"] != "leak" for t in mp["transforms"])  # unsafe transform never kept
