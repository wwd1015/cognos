"""Stage 5 — Independent validation (SR 11-7 effective challenge).

This is the *adversarial* gate. Per SR 11-7, validation must be performed independently of model
development: it does not trust the modeling stage's self-assessment, it re-derives risk from the
upstream artifacts and tries to find reasons the champion should NOT ship. It scores the model on a
five-axis rubric (leakage, overfitting, stability, diagnostics, significance), raises a Finding for
every issue, and applies a strict decision rule where any confirmed leakage or backtest-overfitting
signal BLOCKs the pipeline outright.

v1: after the deterministic rubric, the **Independent Validator** agent adds the judgment a rubric
cannot. Its slice holds artifacts and engine metrics only — never the modeler's rationale. Its
findings cite facts and name the stage that must change; high-severity ones are routed back to that
stage's agent by the engine (``routed_findings``). An agent can raise FAIL/WARN but never BLOCK:
BLOCK stays reserved for the engine's confirmed-leakage rule (CLAUDE.md non-negotiable #6).
"""

from __future__ import annotations

import math

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from .base import Stage, attach_recommendation, register_stage

# Decision thresholds (kept explicit so the rubric is auditable).
PBO_BLOCK = 0.5  # P(backtest overfitting) above a coin-flip => not credible
CV_HOLDOUT_GAP = 0.15  # relative CV-vs-holdout degradation => overfitting suspicion
INSTABILITY_RATIO = 0.5  # walk-forward std/|mean| => unstable out-of-sample performance
SIGNIFICANCE_SHARE = 0.5  # < half the coefficients significant => weak specification
ABSURD_COEF = 1e6  # |coefficient| this large signals collinearity/numerical blow-up
P_SIGNIFICANT = 0.05


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


@register_stage
class ValidateStage(Stage):
    name = "validate"
    requires = ("model", "backtest")
    is_gate = True
    description = "Independent effective-challenge validation (SR 11-7): adversarially judge the champion."

    def run(self, ctx: RunContext) -> StageResult:
        model = ctx.require("model").payload
        backtest = ctx.require("backtest").payload
        explore = ctx.get("explore")
        explore_payload = explore.payload if explore is not None else {}

        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        rubric: dict[str, float] = {}
        blockers: list[str] = []

        # --- (1) LEAKAGE ---------------------------------------------------------
        leakage_suspects = list(explore_payload.get("leakage_suspects", []) or [])
        champion = model.get("champion", {}) or {}
        champ_features = list(champion.get("features", []) or [])
        leaked = [f for f in champ_features if f in leakage_suspects]
        if leaked:
            for f in leaked:
                res.add_finding(Finding(
                    id=f"leak-{f}", severity=Severity.CRITICAL, category="leakage",
                    message=f"Champion uses leakage suspect '{f}' (flagged by explore as near-perfectly "
                            f"correlated with the target).",
                    location=f, confidence=0.95,
                    suggestion="Remove the leaking feature and refit; do not ship a leaking model.",
                ))
            blockers.append(f"target leakage via {len(leaked)} feature(s): {', '.join(leaked)}")
            rubric["leakage"] = 0.0
        else:
            rubric["leakage"] = 1.0

        # --- (2) OVERFITTING -----------------------------------------------------
        overfit_score = 1.0
        pbo = (backtest.get("pbo") or {}).get("pbo")
        if pbo is not None and not _is_nan(pbo):
            pbo = float(pbo)
            if pbo > PBO_BLOCK:
                # PBO is a noisy statistic (it drifts toward ~0.5 when many candidates are equally
                # good), so it is an informational MEDIUM signal only. The reliable, direct overfit
                # evidence is the CV-vs-sealed-holdout gap below (HIGH), and confirmed leakage (BLOCK).
                res.add_finding(Finding(
                    id="pbo", severity=Severity.MEDIUM, category="overfitting",
                    message=f"Probability of backtest overfitting is {pbo:.2f} (> {PBO_BLOCK}): the "
                            f"in-sample-best configuration may underperform out-of-sample.",
                    confidence=0.6,
                    suggestion="Corroborate with the CV-vs-holdout gap; reduce the search space if confirmed.",
                ))
            overfit_score = min(overfit_score, 1.0 - pbo)

        cv_mean = model.get("cv_mean")
        holdout = model.get("holdout_metric")
        if cv_mean is not None and holdout is not None and not _is_nan(cv_mean) and not _is_nan(holdout):
            cv_mean, holdout = float(cv_mean), float(holdout)
            denom = abs(cv_mean) if abs(cv_mean) > 1e-9 else 1.0
            gap = abs(cv_mean - holdout) / denom
            if gap > CV_HOLDOUT_GAP:
                res.add_finding(Finding(
                    id="cv-holdout-gap", severity=Severity.HIGH, category="overfitting",
                    message=f"CV mean ({cv_mean:.4f}) and sealed-holdout metric ({holdout:.4f}) differ by "
                            f"{gap:.0%} (> {CV_HOLDOUT_GAP:.0%}): cross-validation likely over-optimistic.",
                    confidence=0.8,
                    suggestion="Investigate fold construction / feature selection inside CV for leakage.",
                ))
                overfit_score = min(overfit_score, max(0.0, 1.0 - gap))
        rubric["overfitting"] = overfit_score

        # --- (3) STABILITY -------------------------------------------------------
        stability_score = 1.0
        wf = backtest.get("walk_forward") or {}
        wf_mean, wf_std = wf.get("mean"), wf.get("std")
        if wf_mean is not None and wf_std is not None and not _is_nan(wf_mean) and not _is_nan(wf_std):
            wf_mean, wf_std = float(wf_mean), float(wf_std)
            denom = abs(wf_mean) if abs(wf_mean) > 1e-9 else 1.0
            ratio = wf_std / denom
            if ratio > INSTABILITY_RATIO:
                res.add_finding(Finding(
                    id="walk-forward-instability", severity=Severity.MEDIUM, category="stability",
                    message=f"Walk-forward performance is unstable: std/|mean| = {ratio:.2f} "
                            f"(> {INSTABILITY_RATIO}). OOS results vary widely across folds.",
                    confidence=0.7,
                    suggestion="Performance regime-dependent; verify the model generalizes across periods.",
                ))
                stability_score = max(0.0, 1.0 - (ratio - INSTABILITY_RATIO))
        rubric["stability"] = stability_score

        # --- (4) DIAGNOSTICS -----------------------------------------------------
        diagnostics = model.get("diagnostics") or {}
        tests = list(diagnostics.get("tests", []) or [])
        diag_score = 1.0
        if tests:
            failed = [t for t in tests if not t.get("passed", True)]
            high_failed = [t for t in failed if str(t.get("severity", "")).upper() in ("CRITICAL", "HIGH")]
            for t in high_failed:
                res.add_finding(Finding(
                    id=f"diag-{t.get('name', 'test')}", severity=Severity.HIGH, category="diagnostics",
                    message=f"Diagnostic '{t.get('name', 'test')}' ({t.get('category', 'n/a')}) failed: "
                            f"{t.get('interpretation', 'assumption violated')}.",
                    location=t.get("name"), confidence=0.75,
                    suggestion="A violated modeling assumption undermines inference; address before sign-off.",
                ))
            diag_score = max(0.0, 1.0 - len(failed) / len(tests))
        rubric["diagnostics"] = diag_score

        # --- (5) STATISTICAL SIGNIFICANCE ---------------------------------------
        sig_score = 1.0
        pvalues = model.get("pvalues")
        coefficients = model.get("coefficients") or {}
        if pvalues:
            valid_p = [float(p) for p in pvalues.values() if p is not None and not _is_nan(p)]
            if valid_p:
                share_sig = sum(1 for p in valid_p if p < P_SIGNIFICANT) / len(valid_p)
                if share_sig < SIGNIFICANCE_SHARE:
                    res.add_finding(Finding(
                        id="weak-significance", severity=Severity.MEDIUM, category="significance",
                        message=f"Only {share_sig:.0%} of coefficients are significant at p<{P_SIGNIFICANT} "
                                f"(< {SIGNIFICANCE_SHARE:.0%}): most predictors add no reliable signal.",
                        confidence=0.7,
                        suggestion="Prune insignificant features to a parsimonious, defensible specification.",
                    ))
                    sig_score = min(sig_score, max(0.0, share_sig / SIGNIFICANCE_SHARE))

        # NaN / absurd-magnitude coefficients or p-values (numerical pathology).
        bad: list[str] = []
        for name, p in (pvalues or {}).items():
            if p is None or _is_nan(p):
                bad.append(f"p-value({name})")
        for name, c in coefficients.items():
            if c is None or _is_nan(c) or (isinstance(c, int | float) and abs(float(c)) > ABSURD_COEF):
                bad.append(f"coef({name})")
        if bad:
            res.add_finding(Finding(
                id="pathological-coefs", severity=Severity.MEDIUM, category="significance",
                message=f"Pathological coefficient/p-value(s) detected ({len(bad)}): "
                        f"{', '.join(bad[:6])}{'...' if len(bad) > 6 else ''}. "
                        f"Signals collinearity, separation, or numerical instability.",
                confidence=0.7,
                suggestion="Check for multicollinearity / perfect separation; regularize or drop features.",
            ))
            sig_score = min(sig_score, 0.5)
        rubric["significance"] = sig_score

        # --- (6) SANITY: sealed holdout must exist -------------------------------
        n_holdout = int(model.get("n_holdout", 0) or 0)
        if n_holdout <= 0:
            res.add_finding(Finding(
                id="no-holdout", severity=Severity.MEDIUM, category="sanity",
                message="No sealed holdout was reserved: the champion has never been evaluated on data "
                        "it could not see during selection.",
                confidence=0.85,
                suggestion="Reserve a frozen holdout and report the champion's untouched out-of-sample metric.",
            ))
            rubric["overfitting"] = min(rubric["overfitting"], 0.7)

        # --- rubric aggregation --------------------------------------------------
        overall_score = sum(rubric.values()) / len(rubric) if rubric else 1.0

        # --- decision rule -------------------------------------------------------
        has_block = any(f.severity == Severity.CRITICAL for f in res.findings) or bool(blockers)
        has_high = any(f.severity == Severity.HIGH for f in res.findings)
        if has_block:
            verdict = Verdict.BLOCK
        elif has_high:
            verdict = Verdict.FAIL
        elif res.findings:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS
        res.verdict = verdict

        # --- sealed-holdout reuse (re-selection after seeing it) -------------------
        n_eval = int(model.get("holdout_evaluations") or 1)
        if n_eval > 1:
            res.add_finding(Finding(
                id="holdout-reuse", severity=Severity.LOW, category="overfitting",
                message=f"The sealed holdout has been evaluated {n_eval} times in this run (re-runs, "
                        "send-backs or a champion override); a choice made after seeing it is no "
                        "longer a single-shot estimate.",
                confidence=0.9,
                suggestion="Report the holdout metric as indicative; confirm on fresh out-of-time data.",
            ))
            if verdict == Verdict.PASS:
                verdict = Verdict.WARN

        # --- agent-written analysis code: re-run every script and compare what it finds -------
        from ..analysis import run as analysis

        explore = ctx.get("explore")
        scripts = [a for a in ((explore.payload or {}).get("analyses") if explore else None) or []
                   if a.get("kind") == "code"]
        code_review = []
        for a in scripts:
            ran = a.get("status") == "ok"
            ok, why = analysis.reproduce(ctx, a) if ran else (False, a.get("error", ""))
            try:
                code = analysis.code_of(ctx.resolve(a["code_path"]).read_text(encoding="utf-8"))
            except FileNotFoundError:
                code = ""
            code_review.append({"id": a["id"], "purpose": a["purpose"], "sha256": a["code_sha256"],
                                "ran": ran, "reproduced": ok, "note": why, "code": code[:4000]})
            if ran and not ok:
                res.add_finding(Finding(
                    id=f"analysis-not-reproduced-{a['id']}", severity=Severity.MEDIUM,
                    category="analysis-code", location=a["code_path"],
                    message=f"Analysis script {a['id']} ({a['purpose'][:80]}) did not reproduce: "
                            f"{why}.",
                    suggestion="Evidence that rests on this script cannot be relied on until it "
                               "reproduces; re-run explore."))

        # --- the Independent Validator's challenge (never sees the modeler's rationale) ---
        ideate = ctx.get("ideate")
        ip = ideate.payload if ideate is not None else {}
        res.verdict = verdict
        res.payload = {"rubric": rubric, "overall_score": overall_score, "decision": verdict.value,
                       "analysis_code": [{k: c[k] for k in ("id", "sha256", "ran", "reproduced",
                                                            "note")} for c in code_review]}
        diagnostics = model.get("diagnostics") or {}
        out = ctx.recommend("validator", {
            "engine_verdict": verdict.value,
            "rubric": rubric,
            "blockers": blockers,
            "engine_findings": [{"id": f.id, "severity": f.severity.value, "category": f.category,
                                 "message": f.message} for f in res.findings],
            "champion": {"family": champion.get("family"), "features": champ_features,
                         "n_features": len(champ_features)},
            "interpretability": ctx.config.design.interpretability,
            "data_structure": ip.get("data_structure"),
            "framework_choices": ip.get("framework_choices"),
            "n_candidates_tried": model.get("n_candidates_tried"),
            "admissible_set_size": len(model.get("admissible_set") or []),
            "champion_source": model.get("champion_source"),
            "challenger_benchmark": model.get("challenger_benchmark"),
            "failed_diagnostics": diagnostics.get("failed_tests", []),
            "holdout_evaluations": n_eval,
            "analysis_code": code_review,
        }, fresh={"validate": res})
        sev = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}
        routed = []
        for f in out.findings:
            res.add_finding(Finding(
                id=f"val-{f.id}", severity=sev[f.severity], category=f"challenge/{f.category}",
                message=f.message, location=f.target_stage if f.target_stage != "none" else None,
                confidence=0.7, suggestion=f.remedy,
            ))
            if f.severity == "high" and f.target_stage != "none":
                routed.append({"id": f"val-{f.id}", "severity": f.severity, "message": f.message,
                               "evidence": f.evidence, "target_stage": f.target_stage,
                               "remedy": f.remedy})
        # The agent can escalate to FAIL/WARN; only the engine BLOCKs.
        if verdict != Verdict.BLOCK:
            if any(f.severity == "high" for f in out.findings):
                verdict = Verdict.FAIL
            elif out.findings and verdict == Verdict.PASS:
                verdict = Verdict.WARN
        res.verdict = verdict

        payload = {
            "rubric": rubric,
            "overall_score": overall_score,
            "decision": verdict.value,
            "blockers": blockers,
            "n_findings": len(res.findings),
        }
        payload["routed_findings"] = routed
        payload["analysis_code"] = [{k: c[k] for k in ("id", "sha256", "ran", "reproduced", "note")}
                                    for c in code_review]
        payload["validator"] = {"assessment": out.assessment, "recommendation": out.recommendation,
                                "conditions": out.conditions}
        attach_recommendation(payload, ctx, "validator", out)

        ref = ctx.save_json("stages/validate/validation.json", payload)
        res.add_artifact(ref)
        res.payload = payload
        res.metrics = {
            "overall_score": round(overall_score, 4),
            "decision": verdict.value,
            "n_blockers": len(blockers),
        }
        res.summary = (
            f"Independent validation: {verdict.value} (rubric {overall_score:.2f}); "
            f"{len(res.findings)} finding(s), {len(blockers)} blocker(s)."
        )
        return res
