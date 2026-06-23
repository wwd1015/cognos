"""The mechanical orchestrator.

Like deputy's slash-command orchestrator, this is deliberately *mechanical*: it sequences stages in
the legal lifecycle order, persists each StageResult to disk (checkpoint), honours gate verdicts, and
escalates — judgment lives in the stage agents, determinism lives here. One control flow serves both
modes: autonomous (gates auto-approve; BLOCK optionally halts) and interactive/step-by-step (gates
pause for a human approve/reject via ``gate_handler``). Because every stage checkpoints, a failed run
resumes from where it stopped instead of restarting.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .artifacts import RunSummary, StageResult, Verdict
from .config import CognosConfig, Mode
from .context import RunContext

GateHandler = Callable[[StageResult], str]  # returns "approve" | "reject"

_SEVERITY_ORDER = {
    Verdict.PASS: 0, Verdict.SKIP: 0, Verdict.WARN: 1,
    Verdict.OPEN_QUESTIONS: 2, Verdict.FAIL: 3, Verdict.BLOCK: 4, Verdict.ERROR: 5,
}


def _import_stages() -> None:
    """Ensure all stage modules are imported so the registry is populated."""
    from . import stages  # noqa: F401

    if hasattr(stages, "load_all"):
        stages.load_all()


class Orchestrator:
    def __init__(self, config: CognosConfig, runs_root: str | None = None,
                 run_id: str | None = None, brain=None) -> None:
        _import_stages()
        self.config = config
        self.ctx = RunContext(config, run_id=run_id, runs_root=runs_root, brain=brain)

    # --- full pipeline -----------------------------------------------------------
    def run(
        self,
        stages: list[str] | None = None,
        *,
        interactive: bool | None = None,
        gate_handler: GateHandler | None = None,
        force: bool = False,
    ) -> RunSummary:
        from .stages.base import make_stage

        cfg = self.config
        interactive = (cfg.mode == Mode.INTERACTIVE) if interactive is None else interactive
        plan = stages or cfg.stages.enabled
        summary = RunSummary(run_id=self.ctx.run_id, mode="interactive" if interactive else "autonomous",
                             project=cfg.name)
        worst = Verdict.PASS

        for name in plan:
            if not force and self.ctx.has(name):
                result = self.ctx.get(name)  # resume: reuse checkpoint
            else:
                result = make_stage(name).run_guarded(self.ctx)
                self.ctx.record(result)
            summary.stages_run.append(name)
            summary.verdicts[name] = result.verdict.value
            if _SEVERITY_ORDER[result.verdict] > _SEVERITY_ORDER[worst]:
                worst = result.verdict
            if name == "model":
                summary.champion_metric = result.metrics.get("cv_mean")
                summary.champion_metric_name = cfg.metric.name

            decision = self._handle_gate(name, result, interactive, gate_handler)
            if decision == "halt":
                break

        summary.final_verdict = worst
        summary.n_findings = sum(
            len(self.ctx.get(s).findings) for s in summary.stages_run if self.ctx.get(s)
        )
        summary.ended_at = datetime.now(UTC).isoformat()
        self.ctx.save_json("summary.json", summary.model_dump(mode="json"))
        self.ctx.save_text("summary.txt", summary.token_block())
        return summary

    def _handle_gate(self, name: str, result: StageResult, interactive: bool,
                     gate_handler: GateHandler | None) -> str:
        cfg = self.config
        if result.verdict == Verdict.ERROR:
            return "halt"
        is_gate = name in cfg.stages.gates
        if not is_gate or result.verdict.ok:
            return "continue"
        # Gate raised a non-OK verdict.
        if interactive and gate_handler is not None:
            return "continue" if gate_handler(result) == "approve" else "halt"
        # Autonomous: halt only on BLOCK (when configured); FAIL/OPEN_QUESTIONS are recorded but the
        # prototype run proceeds so the user still gets docs/consistency output.
        if result.verdict == Verdict.BLOCK and cfg.stages.halt_on_block:
            return "halt"
        return "continue"

    # --- single stage (stage-by-stage / human-in-the-loop) -----------------------
    def run_stage(self, name: str, *, force: bool = True) -> StageResult:
        from .stages.base import make_stage

        if not force and self.ctx.has(name):
            return self.ctx.get(name)
        result = make_stage(name).run_guarded(self.ctx)
        self.ctx.record(result)
        return result


def run_pipeline(
    config: CognosConfig,
    *,
    runs_root: str | None = None,
    run_id: str | None = None,
    brain=None,
    interactive: bool = False,
    gate_handler: GateHandler | None = None,
    stages: list[str] | None = None,
) -> tuple[RunContext, RunSummary]:
    """Convenience: build an orchestrator, run the pipeline, return (context, summary)."""
    orch = Orchestrator(config, runs_root=runs_root, run_id=run_id, brain=brain)
    summary = orch.run(stages, interactive=interactive, gate_handler=gate_handler)
    return orch.ctx, summary
