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
    assert summary.stages_run == ["intake", "explore", "ideate", "model", "backtest", "validate",
                                  "comply", "document", "review"]
    assert summary.final_verdict.value != "ERROR"


def test_explore_payload(reg_run):
    ctx, _ = reg_run
    p = ctx.require("explore").payload
    assert "x1" in p["features"] and p["n_rows"] == 240
    assert "leakage_suspects" in p


def test_model_payload_and_scorer(reg_run):
    ctx, _ = reg_run
    stage = ctx.require("model").payload
    assert "champion" not in stage and stage["pool"]  # the search ends on a pool
    assert stage["shortlist"] and stage["screen"] and stage["paths"]
    p = ctx.champion()  # the pool member the outcomes analysis pointed to
    assert p["champion"]["family"] in ("ols", "ridge", "lasso", "elasticnet")
    assert p["champion_id"] == ctx.require("backtest").payload["champion_id"]
    assert p["cv_mean"] > 0 and p["test_metric"] is not None
    assert p["holdout_metric"] is None and p["n_holdout"] == 0  # 240 rows: no third sample
    assert "diagnostics" in p and p["diagnostics"]["n_run"] > 0
    from pathlib import Path
    assert Path(p["scorer_path"]).exists()
    assert ctx.resolve(p["oof_perf_path"]).exists()


def test_backtest_oos_and_walkforward(reg_run):
    ctx, _ = reg_run
    p = ctx.require("backtest").payload
    assert "used_impact" in p and "score" not in p  # payload holds metrics, not the table
    assert p["oos_metric"] is not None
    assert p["walk_forward"] is not None and "mean" in p["walk_forward"]
    # regression with no returns column: outcomes analysis (credit) and trading metrics are absent
    assert p["outcomes_analysis"] is None
    assert p["trading"] is None
    assert (ctx.run_dir / "stages/backtest/scored.parquet").exists()


def test_validate_rubric(reg_run):
    ctx, _ = reg_run
    p = ctx.require("validate").payload
    assert set(p["rubric"]) >= {"leakage", "overfitting", "stability", "diagnostics", "significance"}
    assert 0.0 <= p["overall_score"] <= 1.0
    assert p["rubric"]["leakage"] == 1.0  # synthetic data has no leakage


def test_a_profile_naming_sr_11_7_reads_as_sr_26_2():
    from cognos.config import ComplianceConfig

    assert ComplianceConfig().regimes == ["SR26-2", "NIST-AI-RMF"]
    assert ComplianceConfig(regimes=["SR11-7", "SR26-2", "EU-AI-Act"]).regimes == [
        "SR26-2", "EU-AI-Act"]


def test_comply_sr262_and_inventory(reg_run):
    ctx, _ = reg_run
    p = ctx.require("comply").payload
    assert set(p["sr26_2"]) == {"conceptual_soundness", "ongoing_monitoring", "outcomes_analysis"}
    assert p["inventory"]["model_id"] == "stages_reg"
    assert p["inventory"].get("annex_iv_required") is True  # EU jurisdiction


def test_comply_is_nongating_report(reg_run):
    # ADR-0006: comply is a non-gating report — never a verdict, never a BLOCK.
    from cognos.stages.base import make_stage

    ctx, _ = reg_run
    assert make_stage("comply").is_gate is False
    comply = ctx.require("comply")
    assert comply.verdict.value == "PASS"  # "report produced", not a compliance judgement
    assert comply.payload.get("report_only") is True
    assert comply.payload.get("outstanding_human_steps")
    # ongoing monitoring is never auto-passed
    assert comply.payload["sr26_2"]["ongoing_monitoring"]["status"] != "pass"


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
    assert "repo_root" not in p  # the result travels in the export: no machine's path


def test_white_paper_backtest_names_the_comparison_sample(reg_run):
    """Outside trading mode the backtest section names the sample the pool was compared on and
    prints neither a scheme nor PBO / the deflated Sharpe ratio (ADR-0005)."""
    ctx, _ = reg_run
    bp = ctx.require("backtest").payload
    assert bp["trading"] is None and bp["evaluation_sample"] == "test"
    doc = (ctx.docs_dir / "backtest.md").read_text(encoding="utf-8")
    assert "- **Compared on:** the test sample" in doc
    for absent in ("Scheme", "PBO", "Deflated Sharpe", "n/a\n\n\n"):
        assert absent not in doc
    paper = (ctx.run_dir / "stages/document/whitepaper.md").read_text(encoding="utf-8")
    assert "Scheme" not in paper and "Deflated Sharpe" not in paper


def test_trading_mode_reaches_validation_and_the_white_paper(tmp_path):
    """PBO and the deflated Sharpe ratio are computed only when the profile names a returns
    column (ADR-0005). What the backtest records under ``trading`` is what validation weighs
    and what the white paper prints."""
    import numpy as np

    from cognos.engine import Engine

    df = synth.make_classification_dataset(n=400)
    df["ret"] = np.random.default_rng(0).normal(0, 0.02, len(df))
    df.to_csv(tmp_path / "d.csv", index=False)
    cfg = CognosConfig.from_dict({
        "name": "trading", "task": "classification",
        "data": {"path": str(tmp_path / "d.csv"), "target": "target", "drop_columns": ["ret"]},
        "metric": {"name": "roc_auc"}, "search": {"max_candidates": 8, "cv_folds": 3},
        "backtest": {"returns_column": "ret"}, "runs_dir": str(tmp_path / "runs")})
    eng = Engine(cfg, mode="autonomous", provider="heuristic")
    eng.run_until_idle()
    res = eng.results()
    trading = res["backtest"].payload["trading"]
    pbo, dsr = trading["pbo"]["pbo"], trading["deflated_sharpe"]["deflated_sharpe"]
    assert 0.0 < pbo < 1.0
    # the overfitting score is capped by 1 - PBO (the gap to the test sample may lower it more)
    assert res["validate"].payload["rubric"]["overfitting"] <= 1.0 - pbo + 1e-9
    doc = (eng.run_dir / "docs" / "backtest.md").read_text(encoding="utf-8")
    assert f"**PBO (prob. of backtest overfitting):** {pbo:.4f}" in doc
    assert f"**Deflated Sharpe ratio:** {dsr:.4f}" in doc
    assert "- **Compared on:** the test sample" in doc and "Scheme" not in doc
