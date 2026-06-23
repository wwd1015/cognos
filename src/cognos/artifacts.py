"""Core data contracts that flow between COGNOS stages.

Every stage consumes typed inputs from the :class:`~cognos.context.RunContext` and returns a
:class:`StageResult`. These objects are the *load-bearing artifacts* of the pipeline (in the
spirit of deputy's ``PLAN.md`` / ``QA_PASSED``): they are persisted to disk as JSON so that a
stage run in a fresh process — or a fresh agent context — can pick up exactly where the previous
one left off.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Verdict(str, Enum):
    """Greppable verdict tokens emitted by every stage (deputy-style control signal)."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    BLOCK = "BLOCK"  # a gate stage refuses to let the pipeline proceed
    OPEN_QUESTIONS = "OPEN_QUESTIONS"  # needs human input before continuing
    ERROR = "ERROR"  # the stage itself crashed
    SKIP = "SKIP"

    @property
    def ok(self) -> bool:
        """Whether the pipeline may proceed past this verdict without human intervention."""
        return self in (Verdict.PASS, Verdict.WARN, Verdict.SKIP)

    @property
    def blocking(self) -> bool:
        return self in (Verdict.FAIL, Verdict.BLOCK, Verdict.OPEN_QUESTIONS, Verdict.ERROR)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}[self.value]


class Finding(BaseModel):
    """A single issue/observation raised by a stage (validation, compliance, review, ...)."""

    id: str
    severity: Severity = Severity.INFO
    category: str = "general"
    message: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    location: str | None = None  # e.g. a code symbol, OKF concept, or feature name
    suggestion: str | None = None
    reviewed: bool = False  # consistency-review state persistence (avoid re-prompting)

    def line(self) -> str:
        loc = f" [{self.location}]" if self.location else ""
        return f"{self.severity.value} ({self.category}){loc}: {self.message}"


class ArtifactRef(BaseModel):
    """A lightweight reference to a heavy artifact stored on disk (passed by reference)."""

    name: str
    kind: str  # json | table | model | figure | okf | text | tsv
    path: str  # relative to the run directory
    description: str = ""


class StageResult(BaseModel):
    """The single return type of every stage. Persisted as ``stages/<stage>/result.json``."""

    stage: str
    verdict: Verdict
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)  # small structured outputs
    findings: list[Finding] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    started_at: str = Field(default_factory=_utcnow)
    ended_at: str | None = None
    duration_s: float | None = None
    error: str | None = None

    # --- convenience ---------------------------------------------------------------
    def add_finding(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    def add_artifact(self, ref: ArtifactRef) -> ArtifactRef:
        self.artifacts.append(ref)
        return ref

    def finalize(self, started: datetime) -> StageResult:
        end = datetime.now(UTC)
        self.ended_at = end.isoformat()
        self.duration_s = (end - started).total_seconds()
        return self

    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.INFO
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    def token_line(self) -> str:
        """The deputy-style one-line, machine-parseable summary token."""
        return f"COGNOS_STAGE: {self.stage} VERDICT: {self.verdict.value} FINDINGS: {len(self.findings)}"


class RunSummary(BaseModel):
    """Aggregate machine-readable summary of a full pipeline run (parsed by the eval harness)."""

    run_id: str
    mode: str
    project: str
    stages_run: list[str] = Field(default_factory=list)
    verdicts: dict[str, str] = Field(default_factory=dict)
    final_verdict: Verdict = Verdict.PASS
    champion_metric: float | None = None
    champion_metric_name: str | None = None
    n_findings: int = 0
    started_at: str = Field(default_factory=_utcnow)
    ended_at: str | None = None

    def token_block(self) -> str:
        lines = [
            "=== COGNOS RUN SUMMARY ===",
            f"run_id: {self.run_id}",
            f"project: {self.project}",
            f"mode: {self.mode}",
            f"final_verdict: {self.final_verdict.value}",
            f"stages_run: {' '.join(self.stages_run)}",
            f"champion_metric: {self.champion_metric_name}={self.champion_metric}",
            f"n_findings: {self.n_findings}",
        ]
        for stage, v in self.verdicts.items():
            lines.append(f"verdict.{stage}: {v}")
        return "\n".join(lines)
