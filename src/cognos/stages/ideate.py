"""Stage 2 — Idea generation.

Proposes a ranked set of candidate model specifications (family + feature strategy) for the modeling
stage to search. Deterministically enumerates task-appropriate model families and feature strategies,
ranking them by an interpretability/expected-fit heuristic informed by the data profile. When an LLM
brain is available it additionally drafts a natural-language rationale and may suggest extra ideas;
the deterministic plan always stands on its own so the stage runs offline.
"""

from __future__ import annotations

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..modeling.fit import DEFAULT_FAMILIES
from .base import Stage, register_stage

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
        if ctx.brain.available:
            notes += " " + self._llm_notes(ctx, profile, families)

        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        payload = {
            "task": cfg.task.value,
            "families": families,
            "search_budget": cfg.search.max_candidates,
            "hypotheses": hypotheses,
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
    def _llm_notes(ctx: RunContext, profile: dict, families: list[str]) -> str:
        prompt = (
            "You are COGNOS's idea-generation agent. Given this dataset profile, suggest 1-2 concise "
            "feature-engineering or modeling ideas worth trying. Be specific and brief.\n\n"
            f"Task: {profile.get('task')}\nFeatures: {profile.get('features')}\n"
            f"Top correlations: {profile.get('top_correlations')}\nFamilies: {families}"
        )
        try:
            return "LLM ideas: " + ctx.brain.generate(prompt, max_tokens=300).strip()
        except Exception:
            return ""
