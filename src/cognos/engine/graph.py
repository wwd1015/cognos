"""The workflow graph: every step (stage or human gate), its dependencies, and invalidation.

Deliberately data, not logic: the engine walks ``DEPS`` mechanically. Stages are the nine engine
stages (each consults its agent between prepare and finalize); gates are human decision points that
pause in interactive mode and auto-accept in autonomous mode. ``gate_source`` is the one gate that
reviews no stage: the developer gives the data once the intent is confirmed (ADR-0018).
"""

from __future__ import annotations

STAGES = ["intake", "explore", "ideate", "model", "backtest", "validate", "comply", "document",
          "review"]
GATES = ["gate_intent", "gate_source", "gate_data", "gate_design", "gate_candidates", "gate_champion",
         "gate_validation", "gate_signoff"]

STEPS = [
    "intake", "gate_intent", "gate_source",
    "explore", "gate_data", "ideate", "gate_design", "model", "gate_candidates", "backtest",
    "gate_champion", "validate", "gate_validation", "comply", "document", "review",
    "gate_signoff",
]

DEPS: dict[str, list[str]] = {
    "intake": [],
    "gate_intent": ["intake"],
    "gate_source": ["gate_intent"],
    "explore": ["gate_source"],
    "gate_data": ["explore"],
    "ideate": ["gate_data"],
    "gate_design": ["ideate"],
    "model": ["gate_design"],
    "gate_candidates": ["model"],
    "backtest": ["gate_candidates"],
    "gate_champion": ["backtest"],
    "validate": ["gate_champion"],
    "gate_validation": ["validate"],
    "comply": ["gate_validation"],
    "document": ["comply"],
    "review": ["document"],
    "gate_signoff": ["review"],
}

# The stage each gate reviews (its recommendation is what the human accepts or pushes back on).
GATE_OF_STAGE = {"intake": "gate_intent", "explore": "gate_data", "ideate": "gate_design",
                 "model": "gate_candidates", "backtest": "gate_champion",
                 "validate": "gate_validation", "review": "gate_signoff"}
STAGE_OF_GATE = {g: s for s, g in GATE_OF_STAGE.items()}
# A gate that reviews no stage, and the stage it comes before: the data is given here, and the
# workbench shows the decision on that stage's page.
GATE_BEFORE_STAGE = {"gate_source": "explore"}
STAGE_AFTER_GATE = {s: g for g, s in GATE_BEFORE_STAGE.items()}

# Stages a human (or the validator) may send work back to.
SEND_BACK_TARGETS = ["intake", "explore", "ideate", "model"]

LABELS = {
    "intake": "Understand the intent",
    "gate_intent": "Confirm the intent",
    "gate_source": "Provide the data",
    "explore": "Explore data",
    "gate_data": "Review data decisions",
    "ideate": "Design the model",
    "gate_design": "Review the design",
    "model": "Search & build the candidate pool",
    "gate_candidates": "Review the candidate pool",
    "backtest": "Outcomes analysis of the pool",
    "gate_champion": "Choose the champion",
    "validate": "Independent validation",
    "gate_validation": "Review validation",
    "comply": "Model-risk readiness",
    "document": "Write the white paper",
    "review": "Docs ↔ code review",
    "gate_signoff": "Sign off",
}

# A trial run (trials.py) stops before this step: the sealed holdout is scored there, and only
# the main run of a model may open it.
TRIAL_STOPS_BEFORE = "validate"

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
