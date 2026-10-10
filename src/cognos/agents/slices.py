"""Context slices: exactly what each agent may see.

A slice = project brief (with the accepted engagement brief) + the stage's evidence (passed in by the stage) + ``facts`` restricted to the
agent's independence scope + the sponsor's answers to open questions + the open ``challenges``
routed to its stage. Scope rules:

- The **modeler** never sees model/backtest/validate facts: it chooses the champion blind to the
  sealed holdout (frozen substrate) and to any stale result of a previous model run.
- ``prior.`` facts (what an earlier COGNOS run recorded about the model being updated, its holdout
  result included) reach only the agents that read results: outcomes analyst, validator, model-risk
  analyst and writer. The design lead and the modeler learn the incumbent's family, not its scores.
- The **validator** sees engine artifacts and metrics, never the modeler's rationale (SR 11-7
  independence): it re-derives risk from evidence, not from the developer's story.
"""

from __future__ import annotations

from typing import Any

from . import facts as facts_mod
from .contracts import AGENT_STAGE

UPSTREAM = ("intake.", "prior.", "explore.", "ideate.", "model.", "backtest.", "validate.",
            "comply.")
FACT_SCOPE: dict[str, tuple[str, ...]] = {
    "intake_analyst": ("intake.",),
    "data_analyst": ("intake.", "explore."),
    "design_lead": ("intake.", "explore.", "ideate."),
    "modeler": ("intake.", "explore.", "ideate."),
    "experiment": ("intake.", "explore.", "ideate."),
    "outcomes_analyst": ("intake.", "prior.", "explore.", "ideate.", "model.", "backtest."),
    "validator": ("intake.", "prior.", "explore.", "ideate.", "model.", "backtest.", "validate."),
    "risk_analyst": UPSTREAM,
    "writer": UPSTREAM,
}


def engagement_brief(ctx) -> dict[str, Any] | None:
    """The intake result every later agent works from: the development mode, the goal, what the
    sponsor decided, and (for an update) the change request. No scores of the prior model."""
    res = ctx.get("intake")
    p = (res.payload if res is not None else None) or {}
    if not p:
        return None
    return {
        "kind": p.get("kind"),
        "objective": p.get("objective"),
        "clarity": p.get("clarity"),
        "brief": {e["field"]: e["value"] for e in p.get("brief", []) if e.get("value")},
        "open_questions": [q["question"] for q in p.get("questions", [])],
        "update": p.get("update"),
    }


def project_brief(ctx) -> dict[str, Any]:
    cfg = ctx.config
    return {
        "name": cfg.name,
        "description": cfg.description,
        "task": cfg.task.value,
        "target": cfg.data.target,
        "metric": cfg.metric.name,
        "metric_direction": cfg.metric.direction.value if cfg.metric.direction else None,
        "design": cfg.design.model_dump(),
        "risk_tier": cfg.compliance.risk_tier,
        "intended_use": cfg.compliance.intended_use,
        "development_mode": cfg.engagement.kind,
    }


def build(ctx, agent: str, data: dict[str, Any], fresh: dict | None = None) -> dict[str, Any]:
    stage = AGENT_STAGE[agent]
    challenges = [] if agent == "experiment" else [
        {"id": c.id, "source": c.source, "severity": c.severity, "message": c.message,
         "evidence": c.evidence, "remedy": c.remedy}
        for c in ctx.challenges_for(stage)
    ]
    project = project_brief(ctx)
    if agent != "intake_analyst":  # the intake analyst writes the brief; it is not handed its own
        project["engagement"] = engagement_brief(ctx)
    return {
        "agent": agent,
        "stage": stage,
        "project": project,
        **data,
        "facts": facts_mod.collect(ctx, fresh=fresh, prefixes=FACT_SCOPE[agent]),
        "sponsor_answers": ctx.sponsor_answers(),
        "challenges": challenges,
    }
