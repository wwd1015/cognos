"""The deterministic workflow engine.

Mechanical by design (CLAUDE.md non-negotiable #2): it walks the step graph, runs ready steps,
checkpoints, pauses at human gates, applies gate decisions, routes challenges, syncs gaps, marks
stale work, and escalates. It never judges — judgment lives in the agents (recommend) and the
humans (decide); every number lives in the stages (the engine core).

Two drivers share one step executor: ``run_until_idle()`` (synchronous — CLI, tests, autonomous
runs) and ``start()``/``advance()`` (background threads — the UI). Every read-modify-write of
``state.json`` happens under the run's lock after re-loading from disk.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..artifacts import RunSummary, StageResult, Verdict
from ..config import CognosConfig, Mode
from ..context import RunContext, new_run_id
from . import events, gates
from .graph import GATE_OF_STAGE, GATES, LABELS, STAGE_OF_GATE, STAGES, STEPS, descendants
from .state import Challenge, Gap, GateDecision, RunState, run_lock

_SEVERITY_ORDER = {
    Verdict.PASS: 0, Verdict.SKIP: 0, Verdict.WARN: 1,
    Verdict.OPEN_QUESTIONS: 2, Verdict.FAIL: 3, Verdict.BLOCK: 4, Verdict.ERROR: 5,
}
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cognos-step")


class Engine:
    def __init__(self, config: CognosConfig, *, run_id: str | None = None,
                 runs_root: str | Path | None = None, provider: str | None = None,
                 mode: str | None = None, runner=None) -> None:
        from ..stages import load_all

        load_all()
        self.config = config
        self.runs_root = Path(runs_root or config.runs_dir)
        self.run_id = run_id or new_run_id()
        self.run_dir = self.runs_root / self.run_id
        self.lock = run_lock(self.run_dir)
        self._runner = runner  # injected (tests); otherwise built from the run's provider
        self._inflight: set[str] = set()
        self._cv = threading.Condition()
        with self.lock:
            if RunState.exists(self.run_dir):
                state = RunState.load(self.run_dir)
                if provider and provider != state.provider:
                    state.provider = self._resolve_provider(provider)
                    state.save(self.run_dir)
            else:
                state = self._new_state(provider, mode)
                self.run_dir.mkdir(parents=True, exist_ok=True)
                config.to_yaml(self.run_dir / "config.yaml")
                state.save(self.run_dir)
                events.publish(self.run_dir, "run_created",
                               f"Run created ({state.mode}, agents: {state.provider}).")

    # --- construction ------------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: str | Path, *, runner=None) -> Engine:
        run_dir = Path(run_dir)
        cfg = CognosConfig.from_yaml(run_dir / "config.yaml")
        return cls(cfg, run_id=run_dir.name, runs_root=run_dir.parent, runner=runner)

    def _resolve_provider(self, requested: str | None) -> str:
        from ..agents import providers

        if self._runner is not None:
            return self._runner.provider["id"]
        return providers.resolve(requested or self.config.agents.provider,
                                 self.config.agents.model)["id"]

    def _new_state(self, provider: str | None, mode: str | None) -> RunState:
        cfg = self.config
        mode = mode or cfg.mode.value
        state = RunState(run_id=self.run_id, project=cfg.name,
                         mode="interactive" if mode in ("interactive", Mode.INTERACTIVE) else "autonomous",
                         provider=self._resolve_provider(provider))
        enabled = set(cfg.stages.enabled)
        for stage in STAGES:
            if stage not in enabled:
                state.set_step(stage, "skipped", "disabled in this profile")
        for gate in GATES:
            if gate not in cfg.workflow.gates or STAGE_OF_GATE[gate] not in enabled:
                state.set_step(gate, "skipped", "gate disabled")
        return state

    # --- state helpers -----------------------------------------------------------------
    @property
    def state(self) -> RunState:
        return RunState.load(self.run_dir)

    def _save(self, state: RunState) -> None:
        self._refresh_status(state)
        state.save(self.run_dir)

    def context(self) -> RunContext:
        return RunContext(self.config, run_id=self.run_id, runs_root=self.runs_root,
                          runner=self.runner())

    def runner(self):
        if self._runner is not None:
            return self._runner
        from ..agents import providers
        from ..agents.runner import AgentRunner

        a = self.config.agents
        prov = providers.get(self.state.provider, a.model)
        return AgentRunner(prov, self.run_dir, max_retries=a.max_retries,
                           time_limit_s=a.time_limit_s, budget_usd=a.budget_usd,
                           replay_dir=a.replay_dir)

    def results(self) -> dict[str, StageResult]:
        ctx = RunContext(self.config, run_id=self.run_id, runs_root=self.runs_root)
        return {s: r for s in STAGES if (r := ctx.get(s)) is not None}

    # --- scheduling --------------------------------------------------------------------
    def _gate_ready(self, state: RunState, gate: str) -> bool:
        stage_status = state.status_of(STAGE_OF_GATE[gate])
        # A blocked stage still reaches its gate in interactive mode: the human must send the work
        # back or reject — the only ways past a BLOCK.
        return stage_status == "done" or (stage_status == "blocked" and state.mode == "interactive")

    def ready_steps(self, state: RunState | None = None) -> list[str]:
        state = state or self.state
        if state.status in ("rejected", "approved"):
            return []
        out = []
        for step in STEPS:
            if state.status_of(step) not in ("pending", "stale") or step in self._inflight:
                continue
            if step in GATES:
                if self._gate_ready(state, step):
                    out.append(step)
            elif state.deps_satisfied(step):
                out.append(step)
        return out

    def run_until_idle(self, max_steps: int = 200) -> RunState:
        """Run every ready step synchronously until the run waits, halts, or completes."""
        for _ in range(max_steps):
            ready = self.ready_steps()
            if not ready:
                break
            self._execute(ready[0])
        state = self.state
        self._write_summary()
        return state

    def start(self) -> None:
        """Advance in the background (the UI path); returns immediately."""
        self.advance()

    def advance(self) -> None:
        with self.lock:
            ready = self.ready_steps()
            for step in ready:
                self._inflight.add(step)
        for step in ready:
            _executor.submit(self._worker, step)

    def _worker(self, step: str) -> None:
        try:
            self._execute(step)
        finally:
            # Schedule successors *before* releasing this step, so a waiter never observes an
            # idle engine between two steps.
            try:
                self.advance()
            finally:
                with self._cv:
                    self._inflight.discard(step)
                    idle = not self._inflight
                    self._cv.notify_all()
                if idle:
                    self._write_summary()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until no step is in flight (UI tests)."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._inflight, timeout=timeout)

    @property
    def busy(self) -> bool:
        return bool(self._inflight)

    # --- executing one step ------------------------------------------------------------
    def _execute(self, step: str) -> None:
        with self.lock:
            state = self.state
            if state.status_of(step) not in ("pending", "stale"):
                return
            open_ids = [c.id for c in state.open_challenges(step)]
            state.set_step(step, "running")
            self._save(state)
        events.publish(self.run_dir, "step_start", f"{LABELS[step]} started.", step=step)
        if step in GATES:
            try:
                self._open_gate(step)
            except Exception as exc:  # never leave a gate stuck in 'running'
                with self.lock:
                    state = self.state
                    state.set_step(step, "failed", f"{type(exc).__name__}: {exc}")
                    self._save(state)
            return
        try:
            result = self._run_stage(step)
        except Exception as exc:  # the stage framework already converts crashes to ERROR
            result = StageResult(stage=step, verdict=Verdict.ERROR,
                                 summary=f"{type(exc).__name__}: {exc}")
        self._finish_stage(step, result, open_ids)

    def _run_stage(self, stage: str) -> StageResult:
        from ..stages.base import make_stage

        ctx = self.context()
        result = make_stage(stage).run_guarded(ctx)
        ctx.record(result)
        return result

    def _finish_stage(self, stage: str, result: StageResult, open_ids: list[str]) -> None:
        cfg = self.config
        with self.lock:
            state = self.state
            still_running = state.status_of(stage) == "running"
            v = result.verdict
            if still_running:
                if v == Verdict.ERROR:
                    state.set_step(stage, "failed", result.summary[:500], verdict=v.value)
                elif (v == Verdict.BLOCK and stage in cfg.stages.gates and cfg.stages.halt_on_block):
                    state.set_step(stage, "blocked", result.summary[:500], verdict=v.value)
                else:
                    state.set_step(stage, "done", result.summary[:500], verdict=v.value)
            if v != Verdict.ERROR:
                self._sync_challenges(state, stage, result, open_ids)
                self._sync_gaps(state, stage, result)
                if still_running and stage == "validate" and v != Verdict.BLOCK:
                    self._route_validator_findings(state, result)
            try:
                state.spend_usd = round(self.runner().spent_usd(), 4)
            except Exception:
                pass
            self._save(state)
            status = state.status_of(stage)
        kind = {"failed": "step_failed", "blocked": "step_blocked"}.get(status, "step_done")
        events.publish(self.run_dir, kind, f"{LABELS[stage]}: {result.summary}", step=stage,
                       verdict=result.verdict.value)

    def _sync_challenges(self, state: RunState, stage: str, result: StageResult,
                         open_ids: list[str]) -> None:
        output = ((result.payload or {}).get("recommendation") or {}).get("output") or {}
        responses = {r["challenge_id"]: r for r in output.get("responses_to_challenges", [])}
        for c in state.challenges:
            if c.id in open_ids and c.status == "open":
                r = responses.get(c.id)
                c.status = "answered"
                c.response = r["response"] if r else "No response recorded."
                c.changed_recommendation = r["changed_recommendation"] if r else None

    def _sync_gaps(self, state: RunState, stage: str, result: StageResult) -> None:
        raised = (result.payload or {}).get("questions") or []
        raised_ids = {q["id"] for q in raised}
        state.gaps = [g for g in state.gaps
                      if not (g.stage == stage and g.status == "open" and g.id not in raised_ids)]
        known = {g.id for g in state.gaps}
        for q in raised:
            if q["id"] not in known:
                state.gaps.append(Gap(id=q["id"], question=q["question"],
                                      category=q.get("category", "design"),
                                      source=q.get("source", "engine"), stage=stage,
                                      design_field=q.get("design_field"),
                                      reentry=q.get("reentry", stage)))

    def _route_validator_findings(self, state: RunState, result: StageResult) -> None:
        routed = (result.payload or {}).get("routed_findings") or []
        limit = self.config.workflow.auto_challenge_loops
        loops = state.loops.get("validator", 0)
        if not routed or loops >= limit:
            if routed:
                events.publish(self.run_dir, "progress",
                               f"{len(routed)} high-severity validator finding(s) remain after "
                               f"{loops} loop(s); they go to the human at the validation gate.",
                               step="validate")
            return
        state.loops["validator"] = loops + 1
        targets = []
        for f in routed:
            state.challenges.append(Challenge(
                source="validator", target_stage=f["target_stage"], severity="high",
                message=f["message"], evidence=f.get("evidence", []), remedy=f.get("remedy")))
            targets.append(f["target_stage"])
        first = min(targets, key=STEPS.index)
        state.invalidate([first])
        events.publish(self.run_dir, "challenge",
                       f"Validator sent {len(routed)} finding(s) back to {', '.join(sorted(set(targets)))} "
                       f"(loop {loops + 1} of {limit}).", step="validate")

    # --- gates -------------------------------------------------------------------------
    def _open_gate(self, gate: str) -> None:
        with self.lock:
            state = self.state
            if state.status_of(gate) != "running":
                return
            stage = STAGE_OF_GATE[gate]
            if state.mode == "autonomous":
                if state.status_of(stage) == "blocked":
                    state.set_step(gate, "blocked", "the reviewed stage BLOCKed")
                    self._save(state)
                    return
                action = "accept"
                try:
                    invalidate, done = gates.handle(state, gate, action, {}, "autonomous mode",
                                                    self.results())
                except gates.GateError as exc:
                    state.set_step(gate, "failed", str(exc))
                    self._save(state)
                    return
                state.decisions.append(GateDecision(gate=gate, action=action, actor="auto",
                                                    reason="autonomous mode"))
                state.set_step(gate, "done", "auto-accepted (autonomous mode)")
                state.invalidate([s for s in invalidate if s != gate])
                self._save(state)
                events.publish(self.run_dir, "gate_decision", f"{LABELS[gate]}: auto-accepted.",
                               step=gate)
                return
            state.set_step(gate, "awaiting", "waiting for your decision")
            self._save(state)
        events.publish(self.run_dir, "gate_waiting", f"{LABELS[gate]}: waiting for your decision.",
                       step=gate)

    def submit_gate(self, gate: str, action: str, payload: dict[str, Any] | None = None,
                    reason: str = "") -> RunState:
        if gate not in GATES:
            raise gates.GateError(f"unknown gate {gate!r}")
        with self.lock:
            state = self.state
            if state.status_of(gate) != "awaiting":
                raise gates.GateError(f"{gate} is not awaiting a decision "
                                      f"(it is {state.status_of(gate)})")
            invalidate, done = gates.handle(state, gate, action, payload or {}, reason,
                                            self.results())
            state.decisions.append(GateDecision(gate=gate, action=action, actor="human",
                                                reason=reason, payload=payload or {}))
            if done:
                state.set_step(gate, "done", f"{action} by reviewer")
            state.invalidate([s for s in invalidate if s != gate] if done else invalidate)
            if not done and state.status_of(gate) == "awaiting":
                state.set_step(gate, "stale", "re-opens after the re-run")
            self._save(state)
        events.publish(self.run_dir, "gate_decision",
                       f"{LABELS[gate]}: {action.replace('_', ' ')}" + (f" — {reason}" if reason else ""),
                       step=gate, action=action)
        return self.state

    def reopen(self, gate: str) -> RunState:
        """Re-open a decided gate so the human can revise it; a changed decision marks the work
        downstream stale."""
        with self.lock:
            state = self.state
            if gate not in GATES or state.status_of(gate) != "done":
                raise gates.GateError(f"{gate} is {state.status_of(gate)}; only a decided gate "
                                      "can be re-opened")
            if state.status in ("approved", "rejected"):
                state.status = "running"
            state.set_step(gate, "awaiting", "re-opened for revision")
            self._save(state)
        events.publish(self.run_dir, "gate_waiting", f"{LABELS[gate]}: re-opened for revision.",
                       step=gate)
        return self.state

    def answer_gap(self, gap_id: str, answer: str, *, assume: bool = False) -> RunState:
        """Answer (or accept as an assumption) an open question at any time."""
        with self.lock:
            state = self.state
            rerun = gates.apply_answers(state, {gap_id: answer}, assume=assume)
            ran = [s for s in rerun if state.status_of(s) not in ("pending", "skipped")]
            state.invalidate(ran)
            self._save(state)
        events.publish(self.run_dir, "gap_answered",
                       f"Question {gap_id} {'accepted as an assumption' if assume else 'answered'}.")
        return self.state

    def retry(self, step: str) -> RunState:
        with self.lock:
            state = self.state
            if state.status_of(step) not in ("failed", "blocked"):
                raise gates.GateError(f"{step} is {state.status_of(step)}, not failed")
            state.set_step(step, "pending", "retry requested")
            self._save(state)
        return self.state

    def reset(self) -> RunState:
        """Force a full re-run (compat: ``Orchestrator.run(force=True)``)."""
        with self.lock:
            state = self.state
            for step in STEPS:
                if state.status_of(step) != "skipped":
                    state.set_step(step, "pending")
            state.status = "created"
            state.halted_reason = None
            self._save(state)
        return self.state

    # --- single stage (stage-by-stage mode, no gates) -----------------------------------
    def run_stage(self, stage: str) -> StageResult:
        with self.lock:
            state = self.state
            open_ids = [c.id for c in state.open_challenges(stage)]
            state.set_step(stage, "running")
            self._save(state)
        result = self._run_stage(stage)
        self._finish_stage(stage, result, open_ids)
        return result

    # --- status + summary ----------------------------------------------------------------
    def _refresh_status(self, state: RunState) -> None:
        if state.status in ("rejected", "approved"):
            return
        statuses = {s: state.status_of(s) for s in STEPS}
        vals = set(statuses.values())
        if "running" in vals:
            state.status = "running"
        elif "awaiting" in vals:
            state.status = "awaiting"
        elif "failed" in vals:
            state.status = "failed"
        elif "blocked" in vals:
            state.status = "blocked"
            blocked = next(s for s in STEPS if statuses[s] == "blocked")
            state.halted_reason = f"{blocked} BLOCKed"
        elif vals <= {"done", "skipped"}:
            state.status = "completed"
        else:
            state.status = "running" if vals & {"done"} else "created"

    def summary(self) -> RunSummary:
        state = self.state
        results = self.results()
        ran = [s for s in STAGES if s in results and state.status_of(s) in ("done", "blocked", "failed")]
        summ = RunSummary(run_id=self.run_id, mode=state.mode, project=self.config.name)
        worst = Verdict.PASS
        for s in ran:
            v = results[s].verdict
            summ.stages_run.append(s)
            summ.verdicts[s] = v.value
            if _SEVERITY_ORDER[v] > _SEVERITY_ORDER[worst]:
                worst = v
        if "model" in results:
            summ.champion_metric = results["model"].metrics.get("cv_mean")
            summ.champion_metric_name = self.config.metric.name
        summ.final_verdict = worst
        summ.n_findings = sum(len(results[s].findings) for s in ran)
        summ.ended_at = state.updated_at
        return summ

    def _write_summary(self) -> None:
        try:
            summ = self.summary()
            ctx = RunContext(self.config, run_id=self.run_id, runs_root=self.runs_root)
            ctx.save_json("summary.json", summ.model_dump(mode="json"))
            ctx.save_text("summary.txt", summ.token_block())
        except Exception:  # a summary is a convenience; never let it break a run
            pass


def gate_for(stage: str) -> str | None:
    return GATE_OF_STAGE.get(stage)


def downstream(step: str) -> list[str]:
    return descendants(step)
