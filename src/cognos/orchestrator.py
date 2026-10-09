"""Compatibility wrapper over the v1 workflow engine (``cognos.engine``).

v0.x code drove an ``Orchestrator`` that sequenced the eight stages and paused only at the two
verdict gates. It still works: the wrapper runs the engine with the human review gates auto-accepted
(the agents' recommendations stand) and, when ``interactive`` with a ``gate_handler``, asks the
handler about a non-OK ``validate``/``review`` verdict. v1 semantics apply to that answer: a FAIL or
WARN may be approved, a **BLOCK cannot** — approving it halts the run exactly like a rejection
(CLAUDE.md non-negotiable #6). New code should use :class:`cognos.engine.Engine` or
:mod:`cognos.service`, which expose the full gate actions (edit, override, send back).
"""

from __future__ import annotations

from collections.abc import Callable

from .artifacts import RunSummary, StageResult, Verdict
from .config import CognosConfig
from .context import RunContext
from .engine import Engine
from .engine.graph import GATE_OF_STAGE

GateHandler = Callable[[StageResult], str]  # returns "approve" | "reject"


def _import_stages() -> None:
    from . import stages

    stages.load_all()


class Orchestrator:
    def __init__(self, config: CognosConfig, runs_root: str | None = None,
                 run_id: str | None = None, provider: str | None = None, runner=None) -> None:
        _import_stages()
        self.config = config
        self.engine = Engine(config, run_id=run_id, runs_root=runs_root, provider=provider,
                             mode="autonomous", runner=runner)

    @property
    def ctx(self) -> RunContext:
        return self.engine.context()

    def run(self, stages: list[str] | None = None, *, interactive: bool | None = None,
            gate_handler: GateHandler | None = None, force: bool = False) -> RunSummary:
        eng = self.engine
        if force:
            eng.reset()
        if stages:
            with eng.lock:
                state = eng.state
                for s in state.steps:
                    base = s if s in GATE_OF_STAGE else next(
                        (st for st, g in GATE_OF_STAGE.items() if g == s), s)
                    if base not in stages and state.status_of(s) in ("pending", "stale"):
                        state.set_step(s, "skipped", "not in the requested stage list")
                eng._save(state)
        if not (interactive and gate_handler is not None):
            eng.run_until_idle()
            return eng.summary()
        # Legacy interactive mode: step manually so a verdict gate can halt what follows it.
        for _ in range(200):
            ready = eng.ready_steps()
            if not ready:
                break
            step = ready[0]
            eng._execute(step)
            if step not in self.config.stages.gates:
                continue
            res = eng.results().get(step)
            if res is None or res.verdict.ok:
                continue
            if gate_handler(res) != "approve" or res.verdict == Verdict.BLOCK:
                with eng.lock:  # rejected, or a BLOCK (never overridable): halt the run
                    state = eng.state
                    for s, st in state.steps.items():
                        if st.status in ("pending", "stale"):
                            state.set_step(s, "skipped", f"halted at the {step} gate")
                    state.status = "rejected"
                    state.halted_reason = f"halted at the {step} gate ({res.verdict.value})"
                    state.save(eng.run_dir)
                break
        eng._write_summary()
        return eng.summary()

    def run_stage(self, name: str, *, force: bool = True) -> StageResult:
        ctx = self.ctx
        if not force and ctx.has(name):
            return ctx.get(name)
        return self.engine.run_stage(name)


def run_pipeline(
    config: CognosConfig,
    *,
    runs_root: str | None = None,
    run_id: str | None = None,
    provider: str | None = None,
    interactive: bool = False,
    gate_handler: GateHandler | None = None,
    stages: list[str] | None = None,
) -> tuple[RunContext, RunSummary]:
    """Convenience: build an orchestrator, run the pipeline, return (context, summary)."""
    orch = Orchestrator(config, runs_root=runs_root, run_id=run_id, provider=provider)
    summary = orch.run(stages, interactive=interactive, gate_handler=gate_handler)
    return orch.ctx, summary
