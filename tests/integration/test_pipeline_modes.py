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


def test_credit_blocks_at_comply_autonomous(make_config, runs_dir):
    cfg = make_config("credit")  # injected group disparity => disparate impact
    ctx, summary = run_pipeline(cfg, runs_root=runs_dir)
    assert ctx.require("comply").verdict == Verdict.BLOCK
    assert summary.final_verdict == Verdict.BLOCK
    # halt_on_block stops before documentation
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


def test_interactive_reject_halts(make_config, runs_dir):
    cfg = make_config("credit")
    orch = Orchestrator(cfg, runs_root=runs_dir)
    summary = orch.run(interactive=True, gate_handler=lambda r: "reject")
    # rejecting the first non-OK gate halts the run before documentation
    assert "document" not in summary.stages_run


def test_interactive_approve_overrides_block(make_config, runs_dir):
    cfg = make_config("credit")
    orch = Orchestrator(cfg, runs_root=runs_dir)
    summary = orch.run(interactive=True, gate_handler=lambda r: "approve")
    # approving overrides the disparate-impact BLOCK and continues to documentation/review
    assert "document" in summary.stages_run and "review" in summary.stages_run
