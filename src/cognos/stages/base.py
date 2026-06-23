"""Stage base class + registry.

Every COGNOS agent is a Stage with a uniform interface ``run(ctx) -> StageResult``. This is what
makes each stage (a) orchestrator-composable and (b) independently invocable from the CLI
(``cognos run-stage <name>``) — the "individually invocable" requirement. Stages are deliberately
single-responsibility with declared dependencies, mirroring deputy's narrow-tool agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..artifacts import StageResult, Verdict

if TYPE_CHECKING:
    from ..context import RunContext

STAGE_REGISTRY: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    if not cls.name:
        raise ValueError(f"Stage {cls.__name__} must define a non-empty `name`.")
    STAGE_REGISTRY[cls.name] = cls
    return cls


class Stage(ABC):
    name: str = ""
    requires: tuple[str, ...] = ()
    is_gate: bool = False  # gates may BLOCK; in interactive mode they are human pause points
    description: str = ""

    @abstractmethod
    def run(self, ctx: RunContext) -> StageResult:
        """Execute the stage against the run context, returning a StageResult."""

    # --- orchestrator entry point (timing + crash safety) ------------------------
    def run_guarded(self, ctx: RunContext) -> StageResult:
        started = datetime.now(UTC)
        try:
            # Verify prerequisites are present (deputy-style: read load-bearing artifacts).
            for req in self.requires:
                ctx.require(req)
            result = self.run(ctx)
        except Exception as exc:  # a stage crash is a verdict, not a process crash
            import traceback

            result = StageResult(
                stage=self.name,
                verdict=Verdict.ERROR,
                summary=f"{type(exc).__name__}: {exc}",
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
            )
        return result.finalize(started)


def make_stage(name: str) -> Stage:
    if name not in STAGE_REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Known: {sorted(STAGE_REGISTRY)}")
    return STAGE_REGISTRY[name]()


def all_stage_names() -> list[str]:
    return list(STAGE_REGISTRY)
