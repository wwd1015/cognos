"""The workflow graph: every step (stage or human gate), its dependencies, and invalidation.

Deliberately data, not logic: the engine walks ``DEPS`` mechanically. Stages are the eight engine
stages (each consults its agent between prepare and finalize); gates are human decision points that
pause in interactive mode and auto-accept in autonomous mode.
"""

from __future__ import annotations

STAGES = ["explore", "ideate", "model", "backtest", "validate", "comply", "document", "review"]
GATES = ["gate_data", "gate_design", "gate_champion", "gate_validation", "gate_signoff"]

STEPS = [
    "explore", "gate_data", "ideate", "gate_design", "model", "gate_champion", "backtest",
    "validate", "gate_validation", "comply", "document", "review", "gate_signoff",
]

DEPS: dict[str, list[str]] = {
    "explore": [],
    "gate_data": ["explore"],
    "ideate": ["gate_data"],
    "gate_design": ["ideate"],
    "model": ["gate_design"],
    "gate_champion": ["model"],
    "backtest": ["gate_champion"],
    "validate": ["backtest"],
    "gate_validation": ["validate"],
    "comply": ["gate_validation"],
    "document": ["comply"],
    "review": ["document"],
    "gate_signoff": ["review"],
}

# The stage each gate reviews (its recommendation is what the human accepts or pushes back on).
GATE_OF_STAGE = {"explore": "gate_data", "ideate": "gate_design", "model": "gate_champion",
                 "validate": "gate_validation", "review": "gate_signoff"}
STAGE_OF_GATE = {g: s for s, g in GATE_OF_STAGE.items()}

# Stages a human (or the validator) may send work back to.
SEND_BACK_TARGETS = ["explore", "ideate", "model"]

LABELS = {
    "explore": "Explore data",
    "gate_data": "Review data decisions",
    "ideate": "Design the model",
    "gate_design": "Review the design",
    "model": "Search & select champion",
    "gate_champion": "Review the champion",
    "backtest": "Outcomes analysis",
    "validate": "Independent validation",
    "gate_validation": "Review validation",
    "comply": "Model-risk readiness",
    "document": "Write the white paper",
    "review": "Docs ↔ code review",
    "gate_signoff": "Sign off",
}

# Statuses that satisfy a dependency.
SATISFIED = {"done", "skipped"}
# Statuses a step can be invalidated from (it has produced, or is producing, output).
INVALIDATABLE = {"done", "awaiting", "failed", "running", "blocked"}


def is_gate(step: str) -> bool:
    return step in GATES


def descendants(step: str) -> list[str]:
    """Every step downstream of ``step``, in workflow order."""
    out: set[str] = set()
    frontier = [step]
    while frontier:
        cur = frontier.pop()
        for s, deps in DEPS.items():
            if cur in deps and s not in out:
                out.add(s)
                frontier.append(s)
    return [s for s in STEPS if s in out]
