"""Stage 2 — Idea generation.

Proposes a ranked set of candidate model specifications (family + feature strategy) for the modeling
stage to search. Deterministically enumerates task-appropriate model families and feature strategies,
ranking them by an interpretability/expected-fit heuristic informed by the data profile. When an LLM
brain is available it additionally drafts a natural-language rationale and may suggest extra ideas;
the deterministic plan always stands on its own so the stage runs offline.
"""

from __future__ import annotations

import json
import re

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..modeling.fit import DEFAULT_FAMILIES
from .base import Stage, register_stage


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# Lower interpretability rank = more interpretable / preferred for regulated use.
_INTERPRETABILITY = {
    "ols": 0.95, "logit": 0.95, "ridge": 0.85, "lasso": 0.88, "elasticnet": 0.83,
    "ridge_logit": 0.85, "lasso_logit": 0.88, "random_forest": 0.45, "gradient_boosting": 0.4,
}


@register_stage
class IdeateStage(Stage):
    name = "ideate"
    requires = ("explore",)
    description = "Propose and rank candidate model specifications."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        profile = ctx.require("explore").payload
        families = cfg.search.model_families or DEFAULT_FAMILIES.get(
            cfg.task.value, DEFAULT_FAMILIES["regression"]
        )
        top_feats = [c["feature"] for c in profile.get("top_correlations", [])[:5]]

        hypotheses: list[dict] = []
        hid = 0
        for family in families:
            interp = _INTERPRETABILITY.get(family, 0.5)
            for strategy in ("top", "all"):
                hid += 1
                # Priority blends interpretability with a mild preference for parsimonious feature sets.
                parsimony_bonus = 0.05 if strategy == "top" else 0.0
                priority = round(min(1.0, interp + parsimony_bonus), 3)
                rationale = self._rationale(family, strategy, top_feats, cfg.task.value)
                hypotheses.append({
                    "id": f"h{hid}",
                    "family": family,
                    "feature_strategy": strategy,
                    "interpretable": interp >= 0.7,
                    "priority": priority,
                    "rationale": rationale,
                })
        hypotheses.sort(key=lambda h: h["priority"], reverse=True)

        notes = (
            f"{len(families)} model families x feature strategies = {len(hypotheses)} hypotheses. "
            f"Strongest signals: {', '.join(top_feats) or 'n/a'}."
        )
        proposed_transforms: list[dict] = []
        if ctx.brain.available:
            extra_note, proposed_transforms = self._llm_propose(ctx, profile, families)
            notes += " " + extra_note

        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        payload = {
            "task": cfg.task.value,
            "families": families,
            "search_budget": cfg.search.max_candidates,
            "hypotheses": hypotheses,
            "proposed_transforms": proposed_transforms,  # LLM-authored, engine-validated (target-hidden)
            "notes": notes,
        }
        res.add_artifact(ctx.save_json("stages/ideate/hypotheses.json", payload))
        res.payload = payload
        res.metrics = {"n_hypotheses": len(hypotheses), "n_families": len(families)}
        res.summary = f"Generated {len(hypotheses)} ranked hypotheses across {len(families)} families."
        if not hypotheses:
            res.verdict = Verdict.FAIL
            res.add_finding(Finding(id="no-ideas", severity=Severity.HIGH, category="idea-gen",
                                    message="No candidate model families resolved for this task."))
        return res

    @staticmethod
    def _rationale(family: str, strategy: str, top_feats: list[str], task: str) -> str:
        feat_txt = f"using {'the strongest predictors' if strategy == 'top' else 'all features'}"
        if family in ("ols", "logit"):
            return f"Baseline interpretable {family.upper()} {feat_txt}; defensible and easy to validate."
        if family in ("ridge", "lasso", "elasticnet", "ridge_logit", "lasso_logit"):
            return f"Regularized linear ({family}) {feat_txt} to control variance/collinearity."
        return f"Flexible {family} {feat_txt} to capture nonlinearity; weigh against interpretability."

    @staticmethod
    def _llm_propose(ctx: RunContext, profile: dict, families: list[str]) -> tuple[str, list[dict]]:
        """LLM proposes feature-engineering transforms; the engine validates them (target-hidden).

        Reasoning *proposes*; the engine *disposes* — a proposed transform is only retained if it
        evaluates safely on a features-only view of the data. The exchange is logged for audit.
        """
        from ..modeling.transforms import SAFE_NP_FUNCS, TransformSpec, apply_transforms

        prompt = (
            "You are COGNOS's idea-generation agent. Propose up to 3 feature-engineering transforms as "
            "expressions over the EXISTING feature columns, using only np.<fn> "
            f"(fn in {sorted(SAFE_NP_FUNCS)}) and arithmetic. Do NOT reference the target.\n"
            f"Task: {profile.get('task')}\nFeature columns: {profile.get('features')}\n"
            f"Top correlations: {profile.get('top_correlations')}\n"
            'Respond with ONLY JSON: {"transforms": [{"name": <str>, "expr": <str>}], "note": <str>}.'
        )
        try:
            raw = ctx.brain.generate(prompt, max_tokens=500)
        except Exception:
            return ("", [])
        ctx.log_reasoning("ideate", "propose-transforms", prompt, raw)

        obj = _parse_json(raw)
        specs = [TransformSpec(name=str(t["name"]), expr=str(t["expr"]))
                 for t in (obj.get("transforms") or [])
                 if isinstance(t, dict) and t.get("name") and t.get("expr")]
        if not specs:
            return (f"LLM note: {obj.get('note', '')}".strip(), [])
        try:
            df = ctx.load_dataset()
            features = profile.get("features", [])
            _, applied, rejected = apply_transforms(df[features], specs)  # target-hidden: df[features]
        except Exception:
            return (f"LLM note: {obj.get('note', '')}".strip(), [])
        note = (f"LLM proposed {len(specs)} transform(s); {len(applied)} validated, "
                f"{len(rejected)} rejected. {obj.get('note', '')}").strip()
        return (note, [s.to_dict() for s in applied])
