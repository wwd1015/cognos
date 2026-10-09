"""Run state: the durable record of where a run is and what humans and agents decided.

``runs/<id>/state.json`` holds step statuses, gate decisions, challenges (send-backs and validator
findings routed to a stage's agent), information gaps (sponsor questions), the human overrides that
shape the effective config, and loop counters. Every read-modify-write happens under the run's lock
and re-loads from disk first, so background step threads and UI callbacks never clobber each other.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..fsutil import atomic_write
from .graph import DEPS, INVALIDATABLE, SATISFIED, STEPS, descendants

StepStatus = Literal["pending", "running", "done", "awaiting", "stale", "failed", "blocked", "skipped"]
RunStatus = Literal["created", "running", "awaiting", "blocked", "failed", "completed", "approved",
                    "rejected"]


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def short_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


class StepState(BaseModel):
    status: StepStatus = "pending"
    message: str | None = None
    verdict: str | None = None  # a stage's verdict (PASS/WARN/FAIL/BLOCK/ERROR) once it finishes
    runs: int = 0
    # Why this step was last invalidated (the decision, answer or challenge that made it stale).
    # Kept through the re-run, so "why did this change?" still has an answer once it is done.
    rerun_reason: str | None = None
    updated_at: str = Field(default_factory=utcnow)


class GateDecision(BaseModel):
    id: str = Field(default_factory=lambda: short_id("gd"))
    gate: str
    action: str  # accept | edit | override | send_back | approve | reject
    actor: Literal["human", "auto"] = "human"
    # developer | reviewer | approver | express. Empty on decisions recorded before seats.
    seat: str = ""
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow)


class PackageSeal(BaseModel):
    """The live pointer at the newest sealed package. The file under ``packages/`` is never
    rewritten; a later change only flips this pointer to ``superseded``."""

    version: int
    digest: str
    sealed_at: str = Field(default_factory=utcnow)
    seat: str = "approver"
    preparation: Literal["human", "express"] = "human"
    status: Literal["sealed", "superseded"] = "sealed"
    superseded_because: str = ""
    path: str = ""  # packages/vN.json, relative to the run directory


class Challenge(BaseModel):
    """Pushback routed to a stage's agent: a human send-back or a validator finding."""

    id: str = Field(default_factory=lambda: short_id("ch"))
    source: Literal["human", "validator"] = "human"
    target_stage: str
    severity: Literal["low", "medium", "high"] = "high"
    message: str
    evidence: list[str] = Field(default_factory=list)
    remedy: str | None = None
    status: Literal["open", "answered", "closed"] = "open"
    response: str | None = None
    changed_recommendation: bool | None = None
    created_at: str = Field(default_factory=utcnow)


class Gap(BaseModel):
    """An open question the design must not silently assume (MD triangulation)."""

    id: str
    question: str
    category: Literal["design", "data"] = "design"
    source: str = "engine"  # engine | agent
    stage: str = "ideate"  # the stage that raised it
    design_field: str | None = None  # DesignConfig field an answer fills, if any
    reentry: str = "ideate"  # stage that re-runs when the gap is answered
    status: Literal["open", "answered", "assumed"] = "open"
    answer: str | None = None
    answered_at: str | None = None


class Overrides(BaseModel):
    """Human decisions that shape the effective config (the profile YAML is never edited)."""

    exclude_columns: list[str] = Field(default_factory=list)
    design: dict[str, str] = Field(default_factory=dict)
    slate: list[dict[str, Any]] | None = None
    champion: str | None = None  # candidate label chosen at gate_champion


class RunState(BaseModel):
    run_id: str
    project: str = ""
    mode: Literal["autonomous", "interactive"] = "autonomous"
    provider: str = "heuristic"
    status: RunStatus = "created"
    halted_reason: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    version: int = 0  # bumped on every save; the UI re-renders when it changes
    steps: dict[str, StepState] = Field(default_factory=lambda: {s: StepState() for s in STEPS})
    decisions: list[GateDecision] = Field(default_factory=list)
    challenges: list[Challenge] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    overrides: Overrides = Field(default_factory=Overrides)
    package: PackageSeal | None = None  # set only by an approver's approve
    loops: dict[str, int] = Field(default_factory=dict)
    spend_usd: float = 0.0

    # --- persistence -----------------------------------------------------------------
    @staticmethod
    def path(run_dir: Path) -> Path:
        return Path(run_dir) / "state.json"

    @classmethod
    def exists(cls, run_dir: Path) -> bool:
        return cls.path(run_dir).exists()

    @classmethod
    def load(cls, run_dir: Path) -> RunState:
        return cls.model_validate_json(cls.path(run_dir).read_text(encoding="utf-8"))

    def save(self, run_dir: Path) -> None:
        self.version += 1
        self.updated_at = utcnow()
        path = self.path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, self.model_dump_json(indent=1))

    # --- steps -----------------------------------------------------------------------
    def status_of(self, step: str) -> str:
        st = self.steps.get(step)
        return st.status if st else "pending"

    def set_step(self, step: str, status: str, message: str | None = None, *,
                 verdict: str | None = None, rerun_reason: str | None = None) -> None:
        prev = self.steps.get(step) or StepState()
        self.steps[step] = StepState(
            status=status, message=message,
            rerun_reason=rerun_reason if rerun_reason is not None else prev.rerun_reason,
            verdict=verdict if verdict is not None else (None if status in ("pending", "stale")
                                                         else prev.verdict),
            runs=prev.runs + (1 if status == "running" else 0),
        )

    def deps_satisfied(self, step: str) -> bool:
        return all(self.status_of(d) in SATISFIED for d in DEPS[step])

    def invalidate(self, steps: list[str], why: str = "an earlier input changed") -> list[str]:
        """Mark steps and everything downstream stale (only those that produced output). ``why``
        names the cause — a decision, an answer, a challenge — and stays on each step."""
        marked: list[str] = []
        for s in steps:
            for d in [s, *descendants(s)]:
                if d not in marked and self.status_of(d) in INVALIDATABLE:
                    self.set_step(d, "stale", why, rerun_reason=why)
                    marked.append(d)
        if marked:
            from .process import supersede
            supersede(self, why)
        return marked

    # --- challenges & gaps ----------------------------------------------------------
    def open_challenges(self, stage: str) -> list[Challenge]:
        return [c for c in self.challenges if c.target_stage == stage and c.status == "open"]

    def gap(self, gap_id: str) -> Gap | None:
        return next((g for g in self.gaps if g.id == gap_id), None)

    def open_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.status == "open"]

    def last_decision(self, gate: str) -> GateDecision | None:
        return next((d for d in reversed(self.decisions) if d.gate == gate), None)


_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
_locks_guard = threading.Lock()


def run_lock(run_dir: Path) -> threading.RLock:
    key = str(Path(run_dir).resolve())
    with _locks_guard:
        return _locks[key]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default
