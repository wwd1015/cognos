"""Context slices: exactly what each agent may see.

A slice = project brief + the stage's evidence (passed in by the stage) + ``facts`` restricted to the
agent's independence scope + the sponsor's answers to open questions + the open ``challenges``
routed to its stage. Scope rules:

- The **modeler** never sees model/backtest/validate facts: it chooses the champion blind to the
  sealed holdout (frozen substrate) and to any stale result of a previous model run.
- The **validator** sees engine artifacts and metrics, never the modeler's rationale (SR 11-7
  independence): it re-derives risk from evidence, not from the developer's story.
"""

from __future__ import annotations

from typing import Any

from . import facts as facts_mod
from .contracts import AGENT_STAGE

UPSTREAM = ("explore.", "ideate.", "model.", "backtest.", "validate.", "comply.")
FACT_SCOPE: dict[str, tuple[str, ...]] = {
    "data_analyst": ("explore.",),
    "design_lead": ("explore.", "ideate."),
    "modeler": ("explore.", "ideate."),
    "experiment": ("explore.", "ideate."),
    "outcomes_analyst": ("explore.", "ideate.", "model.", "backtest."),
    "validator": ("explore.", "ideate.", "model.", "backtest.", "validate."),
    "risk_analyst": UPSTREAM,
    "writer": UPSTREAM,
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
    }


def build(ctx, agent: str, data: dict[str, Any], fresh: dict | None = None) -> dict[str, Any]:
    stage = AGENT_STAGE[agent]
    challenges = [] if agent == "experiment" else [
        {"id": c.id, "source": c.source, "severity": c.severity, "message": c.message,
         "evidence": c.evidence, "remedy": c.remedy}
        for c in ctx.challenges_for(stage)
    ]
    return {
        "agent": agent,
        "stage": stage,
        "project": project_brief(ctx),
        **data,
        "facts": facts_mod.collect(ctx, fresh=fresh, prefixes=FACT_SCOPE[agent]),
        "sponsor_answers": ctx.sponsor_answers(),
        "challenges": challenges,
    }
