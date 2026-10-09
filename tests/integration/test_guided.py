"""Agent-guided search (the modeler proposes, the engine disposes), driven by recorded outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognos import synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator, run_pipeline
from cognos.runtime.score import load_scorer


def _quad_config(tmp_path, replay: str | None = None) -> CognosConfig:
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
        "agents": {"replay_dir": replay},
    })


def test_guided_search_keeps_verified_transform(tmp_path, runs_dir, replay_dir):
    proposal = {"stop": False, "family": "ols",
                "transforms": [{"name": "x1_sq", "expr": "np.square(x1)",
                                "rationale": "quadratic signal"}],
                "rationale": "quadratic signal"}
    cfg = _quad_config(tmp_path, replay_dir({"experiment": [proposal, {"stop": True}]}))
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run()
    mp = orch.ctx.require("model").payload

    # The engine independently verified the LLM-proposed transform improves the metric, then kept it.
    assert mp["guided"]["improved"] is True
    assert any(t["name"] == "x1_sq" for t in mp["transforms"])
    assert "x1_sq" in mp["champion"]["features"]
    # The guided candidate entered the admissible set and the (heuristic) modeler picked it.
    assert mp["champion_id"] == "g1"
    # Every agent call is audited with its raw input/output (ADR-0003).
    audit = (orch.ctx.agents_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert '"agent": "experiment"' in audit
    # The deployed scorer re-applies the transform target-hidden on RAW base features.
    load_scorer.cache_clear()
    bundle = load_scorer(mp["scorer_path"])
    preds = bundle.score_frame(pd.DataFrame({"x1": [1.0, -2.0], "x2": [0.0, 0.5]}))
    assert preds.shape == (2,) and np.all(np.isfinite(preds))


def test_guided_off_with_heuristic_agents(make_config, runs_dir):
    # Default config has guided=False and heuristic agents => no guided phase, no transforms.
    cfg = make_config("regression")
    ctx, _ = run_pipeline(cfg, runs_root=runs_dir)
    mp = ctx.require("model").payload
    assert mp["guided"] is None
    assert mp["transforms"] == []


def test_guided_rejects_unsafe_proposal(tmp_path, runs_dir, replay_dir):
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
        "agents": {"replay_dir": replay_dir({"experiment": {
            "stop": False, "family": "ols",
            "transforms": [{"name": "leak", "expr": "target * 2", "rationale": "x"}]}})},
    })
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run()
    mp = orch.ctx.require("model").payload
    assert mp["guided"]["rounds"] >= 1
    assert all(t["name"] != "leak" for t in mp["transforms"])  # unsafe transform never kept
