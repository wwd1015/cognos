"""Pipeline-level behaviour: autonomous, gate-blocking, stage-by-stage, resume, interactive."""

from __future__ import annotations

from cognos.artifacts import Verdict
from cognos.orchestrator import Orchestrator, run_pipeline


def test_autonomous_regression_completes(make_config, runs_dir):
    cfg = make_config("regression")
    ctx, summary = run_pipeline(cfg, runs_root=runs_dir)
    assert len(summary.stages_run) == 8
    assert summary.final_verdict != Verdict.ERROR
    assert (ctx.run_dir / "summary.json").exists()
    assert summary.champion_metric is not None


def test_credit_completes_with_readiness_report(make_config, runs_dir):
    # ADR-0006: compliance no longer gates. A credit run completes all 8 stages and comply emits a
    # non-gating model-risk readiness report (never a BLOCK).
    cfg = make_config("credit")
    ctx, summary = run_pipeline(cfg, runs_root=runs_dir)
    assert len(summary.stages_run) == 8
    comply = ctx.require("comply")
    assert comply.verdict == Verdict.PASS
    assert comply.payload.get("report_only") is True
    assert comply.payload.get("outstanding_human_steps")


def test_commercial_backtest_outcomes_analysis(make_config, runs_dir):
    # ADR-0005: a commercial PD model gets credit outcomes analysis on an out-of-time sample.
    cfg = make_config("commercial")
    ctx, summary = run_pipeline(cfg, runs_root=runs_dir)
    bt = ctx.require("backtest").payload
    assert bt["evaluation_sample"] == "out_of_time"
    oa = bt["outcomes_analysis"]
    assert oa is not None
    assert set(oa) >= {"gini", "ks", "expected_calibration_error", "calibration_table", "psi"}
    assert bt["trading"] is None  # no returns column => trading metrics off


def test_leakage_blocks_at_validate(leak_config, runs_dir):
    # The real (and only) hard-BLOCK: confirmed target leakage stopped by the validate gate.
    ctx, summary = run_pipeline(leak_config, runs_root=runs_dir)
    assert ctx.require("validate").verdict == Verdict.BLOCK
    assert summary.final_verdict == Verdict.BLOCK
    assert "document" not in summary.stages_run


def test_stage_by_stage(make_config, runs_dir):
    cfg = make_config("classification")
    orch = Orchestrator(cfg, runs_root=runs_dir)
    for stage in ("explore", "ideate", "model"):
        r = orch.run_stage(stage)
        assert r.verdict != Verdict.ERROR
    assert orch.ctx.has("model")
    # backtest depends on model; runs individually now that model exists
    r = orch.run_stage("backtest")
    assert r.verdict != Verdict.ERROR
    assert orch.ctx.require("model").payload["champion"]["family"]


def test_run_stage_missing_dependency_errors(make_config, runs_dir):
    cfg = make_config("regression")
    orch = Orchestrator(cfg, runs_root=runs_dir)
    # model requires explore; running it first should yield ERROR (dependency missing)
    r = orch.run_stage("model")
    assert r.verdict == Verdict.ERROR


def test_resume_reuses_checkpoints(make_config, runs_dir):
    cfg = make_config("regression")
    orch1 = Orchestrator(cfg, runs_root=runs_dir)
    s1 = orch1.run()
    run_id = orch1.ctx.run_id
    # second orchestrator on the SAME run id; force=False should reuse persisted results
    orch2 = Orchestrator(cfg, runs_root=runs_dir, run_id=run_id)
    s2 = orch2.run(force=False)
    assert s2.verdicts == s1.verdicts
    assert orch2.ctx.run_id == run_id


def test_interactive_reject_halts(leak_config, runs_dir):
    orch = Orchestrator(leak_config, runs_root=runs_dir)
    summary = orch.run(interactive=True, gate_handler=lambda r: "reject")
    # rejecting the validate gate (leakage BLOCK) halts the run before documentation
    assert "document" not in summary.stages_run


def test_interactive_approve_overrides_block(leak_config, runs_dir):
    orch = Orchestrator(leak_config, runs_root=runs_dir)
    summary = orch.run(interactive=True, gate_handler=lambda r: "approve")
    # approving overrides the validate leakage BLOCK and continues to documentation/review
    assert "document" in summary.stages_run and "review" in summary.stages_run
