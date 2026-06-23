"""Per-stage integration: run the full regression pipeline once, assert each stage's contract."""

from __future__ import annotations

import pytest

from cognos import synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator


@pytest.fixture(scope="module")
def reg_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("regrun")
    csv = d / "reg.csv"
    synth.make_regression_dataset(n=240).to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "stages_reg", "task": "regression",
        "data": {"path": str(csv), "target": "target"},
        "search": {"max_candidates": 8, "cv_folds": 3},
        "backtest": {"n_splits": 4},
        "compliance": {"jurisdictions": ["US", "EU"]},
    })
    orch = Orchestrator(cfg, runs_root=str(d / "runs"))
    summary = orch.run()
    return orch.ctx, summary


def test_all_stages_ran(reg_run):
    ctx, summary = reg_run
    assert summary.stages_run == ["explore", "ideate", "model", "backtest", "validate",
                                  "comply", "document", "review"]
    assert summary.final_verdict.value != "ERROR"


def test_explore_payload(reg_run):
    ctx, _ = reg_run
    p = ctx.require("explore").payload
    assert "x1" in p["features"] and p["n_rows"] == 240
    assert "leakage_suspects" in p


def test_model_payload_and_scorer(reg_run):
    ctx, _ = reg_run
    p = ctx.require("model").payload
    assert p["champion"]["family"] in ("ols", "ridge", "lasso", "elasticnet")
    assert p["cv_mean"] > 0 and p["holdout_metric"] is not None
    assert "diagnostics" in p and p["diagnostics"]["n_run"] > 0
    from pathlib import Path
    assert Path(p["scorer_path"]).exists()
    assert ctx.resolve(p["oof_perf_path"]).exists()


def test_backtest_used_impact_and_pbo(reg_run):
    ctx, _ = reg_run
    p = ctx.require("backtest").payload
    assert "used_impact" in p and "score" not in p  # payload holds metrics, not the table
    assert p["oos_metric"] is not None
    assert p["pbo"] is not None and "pbo" in p["pbo"]
    assert (ctx.run_dir / "stages/backtest/scored.parquet").exists()


def test_validate_rubric(reg_run):
    ctx, _ = reg_run
    p = ctx.require("validate").payload
    assert set(p["rubric"]) >= {"leakage", "overfitting", "stability", "diagnostics", "significance"}
    assert 0.0 <= p["overall_score"] <= 1.0
    assert p["rubric"]["leakage"] == 1.0  # synthetic data has no leakage


def test_comply_sr117_and_inventory(reg_run):
    ctx, _ = reg_run
    p = ctx.require("comply").payload
    assert set(p["sr11_7"]) == {"conceptual_soundness", "ongoing_monitoring", "outcomes_analysis"}
    assert p["inventory"]["model_id"] == "stages_reg"
    assert p["inventory"].get("annex_iv_required") is True  # EU jurisdiction


def test_document_okf_bundle(reg_run):
    ctx, _ = reg_run
    p = ctx.require("document").payload
    assert (ctx.docs_dir / "index.md").exists()
    assert (ctx.docs_dir / "model_card.md").exists()
    assert (ctx.docs_dir / "annex_iv.md").exists()  # EU
    assert p["has_code_links"] and len(p["code_links"]) >= 1


def test_review_no_drift(reg_run):
    ctx, _ = reg_run
    p = ctx.require("review").payload
    # document emitted real code anchors, so there must be no stale code references
    assert p["missing_code_paths"] == []
    assert p["missing_symbols"] == []
    assert p["n_code_anchors"] >= 1
