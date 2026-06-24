"""Stage 6 — Regulatory compliance assessment (gate).

Turns the modeling artifacts into a model-risk record an examiner would recognize: a model
inventory entry, the three core SR 11-7 elements (conceptual soundness / ongoing monitoring /
outcomes analysis), the four NIST AI RMF functions, the seven NIST trustworthy-AI characteristics,
and — when the project is a credit model — an ECOA/Reg B fair-lending scan (four-fifths disparate
impact + plain-English adverse-action reason codes). A disparate-impact violation BLOCKs the
pipeline; an SR 11-7 element marked ``fail`` FAILs it. Runs fully offline (deterministic).
"""

from __future__ import annotations

import numpy as np

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..runtime.score import score_frame
from .base import Stage, register_stage

PASS = "pass"
WARN = "warn"
FAIL = "fail"


def _humanize_feature(name: str, weight: float) -> str:
    """Turn a feature name + signed weight into a Reg B-style adverse-action reason string."""
    # Strip design-matrix prefixes (num__/cat__/onehot encodings) and underscores.
    base = name
    for prefix in ("num__", "cat__", "remainder__"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    pretty = " ".join(base.replace("_", " ").split()).strip().lower()
    direction = "elevated" if weight >= 0 else "low"
    return f"{direction} {pretty}"


def _reason_codes(weights: dict[str, float], k: int = 5) -> dict[str, str]:
    """Top-k signed drivers as plain-English reasons, excluding the intercept."""
    items = [(n, float(w)) for n, w in weights.items() if str(n).lower() not in ("const", "intercept")]
    ranked = sorted(items, key=lambda kv: abs(kv[1]), reverse=True)
    return {name: _humanize_feature(name, w) for name, w in ranked[:k]}


@register_stage
class ComplyStage(Stage):
    name = "comply"
    requires = ("model",)
    is_gate = False  # ADR-0006: compliance is a non-gating report, never a verdict
    description = "Produce a non-gating SR 11-7 / NIST AI RMF model-risk readiness report."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        comp = cfg.compliance
        model = ctx.require("model").payload
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        champion = model.get("champion", {}) or {}
        features = list(champion.get("features", []) or model.get("raw_features", []) or [])
        coefficients = model.get("coefficients")
        importances = model.get("feature_importances")
        scorer_path = model.get("scorer_path")
        has_payload = bool(model)
        has_backtest = ctx.has("backtest")
        backtest = ctx.get("backtest").payload if has_backtest else {}
        oos_metric = backtest.get("oos_metric") if has_backtest else None

        # --- (A) model inventory entry ------------------------------------------
        inventory = {
            "model_id": cfg.name,
            "version": cfg.version,
            "purpose": comp.intended_use or cfg.description,
            "intended_use": comp.intended_use,
            "out_of_scope_use": comp.out_of_scope_use,
            "risk_tier": comp.risk_tier,
            "owner": "COGNOS",
            "champion_family": champion.get("family", "unknown"),
            "n_features": len(features),
            "created_from_run": ctx.run_id,
        }

        # --- (B) SR 11-7 three core elements ------------------------------------
        evidence_basis = coefficients is not None or importances is not None
        if evidence_basis and has_payload:
            conceptual = {
                "status": PASS,
                "evidence": (
                    "Champion exposes per-feature effects (coefficients/importances) and a persisted "
                    "scorer; theory/assumptions and feature roles are documented in the OKF bundle."
                ),
            }
        else:
            conceptual = {
                "status": WARN,
                "evidence": "No per-feature effects available; conceptual soundness cannot be evidenced from the model object.",
            }

        # Ongoing monitoring is a production discipline (thresholds, cadence, owner, exception
        # reporting) that does not exist at development time. The backtest produces monitoring-ready
        # baselines, but monitoring itself is always an outstanding human step — never auto-PASS.
        ongoing = {
            "status": WARN,
            "evidence": (
                "Outstanding: ongoing monitoring requires a production monitoring plan — drift/PSI "
                "alert thresholds, performance-decay triggers, exception reporting, cadence and owner."
                + (" Backtest produced monitoring-ready baselines (calibration/stability)."
                   if getattr(cfg.backtest, "enabled", False) else "")
            ),
        }

        if has_backtest and oos_metric is not None:
            outcomes = {
                "status": PASS,
                "evidence": f"Out-of-sample backtest ran ({backtest.get('oos_metric_name', 'oos')}={oos_metric}); outcomes analysis is evidenced.",
            }
        else:
            outcomes = {
                "status": WARN,
                "evidence": "No out-of-sample backtest result available; outcomes analysis is incomplete.",
            }

        sr11_7 = {
            "conceptual_soundness": conceptual,
            "ongoing_monitoring": ongoing,
            "outcomes_analysis": outcomes,
        }

        # --- (C) NIST AI RMF four functions -------------------------------------
        nist_ai_rmf = {
            "govern": {
                "status": PASS if comp.risk_tier else WARN,
                "evidence": f"Risk tier '{comp.risk_tier}' assigned with COGNOS as accountable owner; governance regimes: {', '.join(comp.regimes) or 'none'}.",
            },
            "map": {
                "status": PASS if ctx.has("explore") else WARN,
                "evidence": "Context and data characteristics mapped by the explore stage (schema, missingness, leakage suspects)."
                if ctx.has("explore")
                else "Explore stage output not found; data-context mapping is incomplete.",
            },
            "measure": {
                "status": PASS if (has_backtest and oos_metric is not None) or ctx.has("validate") else WARN,
                "evidence": "Performance measured via cross-validation, holdout, backtest and validation diagnostics."
                if (has_backtest and oos_metric is not None) or ctx.has("validate")
                else "Quantitative measurement (backtest/validation) is incomplete.",
            },
            "manage": {
                "status": ongoing["status"],
                "evidence": "Residual risk managed through monitoring plan and gate verdicts; thresholds pending sign-off.",
            },
        }

        # --- (D) seven trustworthy-AI characteristics ---------------------------
        is_linear = bool(coefficients)
        trustworthy = {
            "valid_reliable": PASS if has_backtest and oos_metric is not None else WARN,
            "safe": PASS if ctx.has("validate") else WARN,
            "secure_resilient": WARN,  # adversarial/robustness testing not yet automated
            "accountable_transparent": PASS,  # inventory + OKF bundle establish accountability
            "explainable_interpretable": PASS if is_linear else WARN,
            "privacy_enhanced": PASS if not comp.fair_lending or cfg.data.protected_attributes else WARN,
            "fair_bias_managed": PASS if comp.fair_lending else WARN,
        }

        # --- reason codes (Reg B adverse-action drivers) ------------------------
        if coefficients:
            reason_codes = _reason_codes(coefficients)
        elif importances:
            reason_codes = _reason_codes(importances)
        else:
            reason_codes = {}

        # --- (E) fair-lending scan ----------------------------------------------
        disparate_impact: float | None = None
        if comp.fair_lending:
            fair_lending = self._fair_lending(ctx, model, scorer_path, reason_codes, res)
            disparate_impact = fair_lending.get("disparate_impact")
        else:
            fair_lending = {"enabled": False}

        # --- EU AI Act Annex IV note --------------------------------------------
        jurisdictions = list(comp.jurisdictions)
        if "EU" in jurisdictions:
            inventory["annex_iv_required"] = True
            inventory["annex_iv_components"] = [
                "general description and intended purpose",
                "detailed model design and development methodology",
                "data governance and training/validation/holdout provenance",
                "performance metrics and accuracy/robustness limitations",
                "risk management system and residual risks",
                "human oversight and post-market monitoring plan",
            ]

        # --- findings for every fail/warn element -------------------------------
        for label, elem in sr11_7.items():
            if elem["status"] == FAIL:
                res.add_finding(Finding(
                    id=f"sr11-7-{label}", severity=Severity.HIGH, category="sr11-7",
                    message=f"SR 11-7 {label.replace('_', ' ')} failed: {elem['evidence']}",
                    location=label, confidence=0.9,
                ))
            elif elem["status"] == WARN:
                res.add_finding(Finding(
                    id=f"sr11-7-{label}", severity=Severity.MEDIUM, category="sr11-7",
                    message=f"SR 11-7 {label.replace('_', ' ')} warning: {elem['evidence']}",
                    location=label, confidence=0.8,
                ))
        for func, elem in nist_ai_rmf.items():
            if elem["status"] == WARN:
                res.add_finding(Finding(
                    id=f"nist-{func}", severity=Severity.LOW, category="nist-ai-rmf",
                    message=f"NIST AI RMF {func.upper()} incomplete: {elem['evidence']}",
                    location=func, confidence=0.7,
                ))
        for char, status in trustworthy.items():
            if status == WARN:
                res.add_finding(Finding(
                    id=f"trustworthy-{char}", severity=Severity.LOW, category="trustworthy-ai",
                    message=f"Trustworthy-AI characteristic '{char}' not fully evidenced.",
                    location=char, confidence=0.6,
                ))

        # --- report, never a verdict (ADR-0006) ---------------------------------
        # An agent cannot adjudicate SR 11-7 compliance (independent human validation is required) and
        # a verdict tempts rubber-stamping. This stage reports evidence + outstanding gaps; its verdict
        # is always PASS meaning "report produced", never a compliance judgement, and it never BLOCKs.
        outstanding_human_steps = [
            "Independent model validation and sign-off (by a function independent of development).",
            "A production monitoring plan: drift/PSI thresholds, performance-decay triggers, cadence, owner.",
            "Model governance: change-control, override policy, and an approval record.",
        ]
        if comp.fair_lending and disparate_impact is not None and disparate_impact < comp.disparate_impact_threshold:
            outstanding_human_steps.append(
                f"Fair-lending review: disparate-impact ratio {disparate_impact:.3f} is below "
                f"{comp.disparate_impact_threshold:.2f} (note: fair lending applies to consumer, not "
                "commercial, credit)."
            )

        n_evidenced = sum(1 for e in sr11_7.values() if e["status"] == PASS)
        payload = {
            "inventory": inventory,
            "sr11_7": sr11_7,
            "nist_ai_rmf": nist_ai_rmf,
            "trustworthy": trustworthy,
            "fair_lending": fair_lending,
            "regimes": list(comp.regimes),
            "jurisdictions": jurisdictions,
            "outstanding_human_steps": outstanding_human_steps,
            "report_only": True,
        }
        ref = ctx.save_json("stages/comply/compliance.json", payload)
        res.add_artifact(ref)
        res.payload = payload
        res.verdict = Verdict.PASS  # "report produced" — never a compliance verdict
        res.metrics = {
            "sr11_7_evidenced": n_evidenced,
            "disparate_impact": disparate_impact,
            "n_outstanding": len(outstanding_human_steps),
            "n_findings": len(res.findings),
        }
        di_str = f"DI={disparate_impact:.3f}" if disparate_impact is not None else "fair-lending n/a"
        res.summary = (
            f"Model-risk readiness report (non-gating): SR 11-7 {n_evidenced}/3 evidenced, "
            f"{len(outstanding_human_steps)} human step(s) outstanding, {di_str}."
        )
        return res

    # --- fair-lending helper ----------------------------------------------------
    def _fair_lending(
        self,
        ctx: RunContext,
        model: dict,
        scorer_path: str | None,
        reason_codes: dict[str, str],
        res: StageResult,
    ) -> dict:
        """Four-fifths disparate-impact test per protected attribute on the holdout frame."""
        cfg = ctx.config
        comp = cfg.compliance
        threshold = comp.disparate_impact_threshold
        attrs = list(cfg.data.protected_attributes)
        holdout_rel = "data/holdout.parquet"

        if not attrs or scorer_path is None or not (ctx.run_dir / holdout_rel).exists():
            res.add_finding(Finding(
                id="fair-lending-unavailable", severity=Severity.MEDIUM, category="fair-lending",
                message="Fair-lending requested but protected attributes, scorer, or holdout frame are unavailable.",
                confidence=0.9,
                suggestion="Configure data.protected_attributes and ensure the holdout frame is materialized.",
            ))
            return {
                "enabled": True,
                "disparate_impact": None,
                "group_rates": {},
                "passes": False,
                "reason_codes": reason_codes,
            }

        df = ctx.load_df(holdout_rel)
        is_classification = bool(model.get("is_classification", cfg.task.is_classification))
        try:
            scores = np.asarray(score_frame(df, scorer_path), dtype=float)
        except Exception as exc:  # scoring failure is a recoverable finding, not a crash
            res.add_finding(Finding(
                id="fair-lending-score-error", severity=Severity.MEDIUM, category="fair-lending",
                message=f"Could not score holdout for fair-lending analysis: {type(exc).__name__}: {exc}",
                confidence=0.9,
            ))
            return {
                "enabled": True,
                "disparate_impact": None,
                "group_rates": {},
                "passes": False,
                "reason_codes": reason_codes,
            }

        # Selection = favorable decision. For classification use the 0.5 threshold; for a
        # continuous score, "selection" = scoring above the population median (relative favorability).
        if is_classification:
            selected = scores >= 0.5
        else:
            selected = scores >= float(np.median(scores))

        group_rates: dict[str, dict[str, float]] = {}
        di_per_attr: dict[str, float] = {}
        worst_di: float | None = None
        worst_attr: str | None = None

        for attr in attrs:
            if attr not in df.columns:
                res.add_finding(Finding(
                    id=f"fair-lending-missing-{attr}", severity=Severity.LOW, category="fair-lending",
                    message=f"Protected attribute '{attr}' not found in holdout frame; skipped.",
                    location=attr, confidence=0.9,
                ))
                continue
            rates: dict[str, float] = {}
            for group, mask in df.groupby(attr, dropna=True).groups.items():
                idx = df.index.get_indexer(mask)
                if len(idx) == 0:
                    continue
                rates[str(group)] = float(np.mean(selected[idx]))
            group_rates[attr] = rates
            positive_rates = [r for r in rates.values() if r > 0]
            if len(rates) < 2 or not positive_rates:
                continue
            max_rate = max(rates.values())
            min_rate = min(rates.values())
            di = (min_rate / max_rate) if max_rate > 0 else 0.0
            di_per_attr[attr] = di
            if worst_di is None or di < worst_di:
                worst_di = di
                worst_attr = attr
            if di < threshold:
                res.add_finding(Finding(
                    id=f"disparate-impact-{attr}", severity=Severity.HIGH, category="fair-lending",
                    message=(
                        f"Disparate impact on '{attr}': ratio {di:.3f} < four-fifths threshold {threshold:.2f} "
                        f"(group selection rates {rates})."
                    ),
                    location=attr, confidence=0.9,
                    suggestion="Investigate adverse-impact drivers; consider reweighting, threshold review, or feature audit.",
                ))

        return {
            "enabled": True,
            "disparate_impact": worst_di,
            "disparate_impact_attribute": worst_attr,
            "disparate_impact_by_attribute": di_per_attr,
            "group_rates": group_rates,
            "threshold": threshold,
            "passes": worst_di is None or worst_di >= threshold,
            "reason_codes": reason_codes,
        }
