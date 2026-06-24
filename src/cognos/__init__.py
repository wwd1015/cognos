"""COGNOS — an autonomous multi-agent system for end-to-end model development.

Public API is kept import-light here; the orchestrator/stage machinery is imported lazily so that
``import cognos`` is cheap and side-effect free.
"""

from __future__ import annotations

from .artifacts import Finding, RunSummary, Severity, StageResult, Verdict
from .config import CognosConfig
from .context import RunContext

__version__ = "0.2.0"

__all__ = [
    "CognosConfig",
    "RunContext",
    "StageResult",
    "RunSummary",
    "Verdict",
    "Severity",
    "Finding",
    "run_pipeline",
    "Orchestrator",
    "__version__",
]


def __getattr__(name: str):  # lazy re-export of the orchestrator (avoids importing stages eagerly)
    if name in ("run_pipeline", "Orchestrator"):
        from .orchestrator import Orchestrator, run_pipeline

        return {"run_pipeline": run_pipeline, "Orchestrator": Orchestrator}[name]
    raise AttributeError(f"module 'cognos' has no attribute {name!r}")
