"""Stage 7 — Documentation.

Emits the model development run as an OKF bundle (one markdown *concept* per artifact) so that a
downstream agent — human or machine — gets curated, cross-linked context with docs<->code
traceability anchors. On top of the white paper this stage also produces a Google Model Card (the
9-section regulator-facing summary) and, for EU deployments, an EU AI Act Annex IV technical
documentation pack. Every methodology/scoring claim is anchored to the code that implements it
(``{@code:...#symbol}``) so the consistency-review stage can verify the docs match the deployment.

The stage is deterministic and fully offline: an optional LLM brain only *polishes* prose, never
gates the output.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..okf import OKFBundle, OKFConcept
from .base import Stage, register_stage


def _fmt(value: Any, nd: int = 4) -> str:
    """Render a metric for prose, tolerating None/NaN/non-numeric inputs."""
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{nd}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table; returns an em-dash placeholder when there are no rows."""
    if not rows:
        return "_No data available._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


@register_stage
class DocumentStage(Stage):
    name = "document"
    requires = ("model",)
    is_gate = False
    description = "Write the white paper as an OKF bundle plus a Google Model Card and EU Annex IV pack."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        model_res = ctx.require("model")
        mp = model_res.payload or {}
        explore = ctx.get("explore")
        ideate = ctx.get("ideate")
        backtest = ctx.get("backtest")
        validate = ctx.get("validate")
        comply = ctx.get("comply")
        ep = explore.payload if explore else {}
        ip = ideate.payload if ideate else {}
        bp = backtest.payload if backtest else {}
        vp = validate.payload if validate else {}
        cp = comply.payload if comply else {}

        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        if not mp:
            res.verdict = Verdict.FAIL
            res.summary = "Model payload missing — cannot author documentation."
            return res

        bundle = OKFBundle(ctx.docs_dir)
        champion = mp.get("champion", {}) or {}
        family = champion.get("family", "unknown")
        metric_name = mp.get("metric", cfg.metric.name)
        code_links: list[str] = []  # "path#symbol" anchors emitted across the bundle

        def emit(concept: OKFConcept) -> None:
            bundle.add(concept)
            for path, symbol in concept.code_anchors():
                code_links.append(f"{path}#{symbol}" if symbol else path)

        bundle.log_event("Creation", f"Initialized OKF bundle for project '{cfg.name}'.")

        # --- 1. overview ---------------------------------------------------------
        emit(OKFConcept(
            name="overview", type="model_overview",
            title="Model Overview", description="Purpose, task, and champion summary.",
            tags=["overview"],
            body=(
                f"# {cfg.name} — Model Overview\n\n"
                f"{cfg.description or 'A COGNOS-developed model.'}\n\n"
                f"- **Task:** {cfg.task.value}\n"
                f"- **Target:** {cfg.data.target}\n"
                f"- **Primary metric:** {metric_name} ({cfg.metric.direction.value})\n"
                f"- **Champion family:** {family}\n"
                f"- **CV {metric_name}:** {_fmt(mp.get('cv_mean'))} ± {_fmt(mp.get('cv_std'))}\n"
                f"- **Frozen-holdout {metric_name}:** {_fmt(mp.get('holdout_metric'))}\n\n"
                "## Intended use\n\n"
                f"{cfg.compliance.intended_use or 'See the [model card](./model_card.md).'}\n\n"
                "Read on: [dataset](./dataset.md) · [methodology](./methodology.md) · "
                "[model](./model.md) · [model card](./model_card.md) · [limitations](./limitations.md)."
            ),
        ))

        # --- 2. dataset ----------------------------------------------------------
        leakage = ep.get("leakage_suspects", []) or []
        feats = ep.get("features", []) or champion.get("features", [])
        emit(OKFConcept(
            name="dataset", type="dataset",
            title="Dataset Profile", description="Shape, features, and leakage notes from exploration.",
            resource=ctx.rel(ctx.resolve("stages/explore/profile.json")),
            tags=["data"],
            body=(
                "# Dataset Profile\n\n"
                f"- **Rows:** {ep.get('n_rows', 'n/a')}\n"
                f"- **Columns:** {ep.get('n_cols', 'n/a')}\n"
                f"- **Model features ({len(feats)}):** {', '.join(map(str, feats[:40])) or 'n/a'}\n"
                f"- **Numeric features:** {len(ep.get('numeric_features', []) or [])}\n"
                f"- **Categorical features:** {len(ep.get('categorical_features', []) or [])}\n\n"
                "## Leakage notes\n\n"
                + (
                    "Target-leakage suspects flagged by exploration: "
                    + ", ".join(map(str, leakage)) + "."
                    if leakage else
                    "No high-correlation target-leakage suspects were flagged during exploration."
                )
                + "\n\nProtected attributes are excluded from the model feature set "
                "(disparate-treatment avoidance). See [methodology](./methodology.md)."
            ),
        ))

        # --- 3. methodology (carries the core code anchors) ----------------------
        challenger = mp.get("challenger_benchmark") or {}
        diagnostics = mp.get("diagnostics", {}) or {}
        challenger_line = (
            f"- **Challenger benchmark (not deployed):** ensemble of "
            f"{challenger.get('n_members', 0)} member(s); headroom vs champion="
            f"{challenger.get('headroom_vs_champion', False)}\n\n"
            if challenger else "\n"
        )
        emit(OKFConcept(
            name="methodology", type="methodology",
            title="Methodology", description="Search procedure and statistical battery.",
            tags=["method"],
            body=(
                "# Methodology\n\n"
                "## Champion search\n\n"
                "Candidates are explored with a *ratchet* search: each accepted experiment must beat "
                "the incumbent on leakage-safe cross-validation (all preprocessing fit inside each "
                "training fold) before it becomes the new incumbent. The final score is read once on a "
                "sealed *frozen holdout* never touched during search. The deployed model is the single "
                "interpretable champion. The ratchet loop is implemented in "
                "{@code:src/cognos/modeling/search.py#ratchet_search}, orchestrated by the modeling "
                "stage at {@code:src/cognos/stages/model.py}.\n\n"
                f"- **Candidates tried:** {mp.get('n_candidates_tried', 'n/a')}\n"
                f"- **Hypothesis families considered:** {', '.join(ip.get('families', []) or []) or 'n/a'}\n"
                f"{challenger_line}"
                "## Statistical battery\n\n"
                f"A battery of {diagnostics.get('n_run', 0)} statistical tests is run on the champion "
                f"({diagnostics.get('n_passed', 0)} passed, {diagnostics.get('n_failed', 0)} failed). "
                "See [diagnostics](./diagnostics.md) for the full table.\n\n"
                "## Deployment scoring\n\n"
                "The trained champion is persisted as a picklable scorer bundle. The deployment "
                "scoring entry point is {@code:src/cognos/runtime/score.py#score_row}, which evaluates "
                "a single row as the IMPACT derived-field contract. See [model](./model.md)."
            ),
        ))

        # --- 4. model ------------------------------------------------------------
        hp = champion.get("hyperparams", {}) or {}
        hp_rows = [[str(k), str(v)] for k, v in hp.items()]
        emit(OKFConcept(
            name="model", type="model",
            title="Champion Model", description="Family, hyperparameters, and headline metrics.",
            resource="models/champion_scorer.joblib",
            tags=["model"],
            body=(
                "# Champion Model\n\n"
                f"- **Family:** {family}\n"
                f"- **Description:** {champion.get('description', 'n/a')}\n"
                f"- **CV {metric_name}:** {_fmt(mp.get('cv_mean'))} ± {_fmt(mp.get('cv_std'))}\n"
                f"- **Frozen-holdout {metric_name}:** {_fmt(mp.get('holdout_metric'))}\n"
                f"- **Train / holdout rows:** {mp.get('n_train', 'n/a')} / {mp.get('n_holdout', 'n/a')}\n\n"
                "## Hyperparameters\n\n"
                + _table(["parameter", "value"], hp_rows) + "\n\n"
                "## Scoring\n\n"
                "Deployment-time scoring is served by "
                "{@code:src/cognos/runtime/score.py#score_row}. See [coefficients](./coefficients.md), "
                "[diagnostics](./diagnostics.md), and [methodology](./methodology.md)."
            ),
        ))

        # --- 5. coefficients / feature importances -------------------------------
        coefs = mp.get("coefficients") or {}
        pvals = mp.get("pvalues") or {}
        importances = mp.get("feature_importances") or {}
        if coefs:
            coef_rows = [
                [str(name), _fmt(coefs.get(name)), _fmt(pvals.get(name))]
                for name in coefs
            ]
            coef_body = "## Coefficients\n\n" + _table(["feature", "coefficient", "p-value"], coef_rows)
        elif importances:
            imp_rows = [[str(name), _fmt(val)] for name, val in importances.items()]
            coef_body = "## Feature importances\n\n" + _table(["feature", "importance"], imp_rows)
        else:
            coef_body = "_No coefficients or feature importances are available for this model family._"
        emit(OKFConcept(
            name="coefficients", type="coefficients",
            title="Coefficients & Importances",
            description="Per-feature effect sizes (or importances).",
            tags=["model", "interpretability"],
            body="# Coefficients & Importances\n\n" + coef_body
            + "\n\nBack to [model](./model.md).",
        ))

        # --- 6. diagnostics ------------------------------------------------------
        tests = diagnostics.get("tests", []) or []
        diag_rows = [
            [
                str(t.get("name", "")), str(t.get("category", "")),
                _fmt(t.get("statistic")), _fmt(t.get("pvalue")),
                "pass" if t.get("passed") else "fail", str(t.get("interpretation", "")),
            ]
            for t in tests
        ]
        emit(OKFConcept(
            name="diagnostics", type="diagnostics",
            title="Statistical Diagnostics", description="The full statistical test battery.",
            tags=["diagnostics"],
            body=(
                "# Statistical Diagnostics\n\n"
                f"{diagnostics.get('n_passed', 0)} of {diagnostics.get('n_run', 0)} tests passed; "
                f"max failed severity: {diagnostics.get('max_failed_severity', 'none')}.\n\n"
                + _table(["test", "category", "statistic", "p-value", "result", "interpretation"], diag_rows)
                + "\n\nThe battery is implemented in {@code:src/cognos/stages/stat_tests.py#run_battery}. "
                "See [model](./model.md)."
            ),
        ))

        # --- 7. backtest ---------------------------------------------------------
        if bp:
            pbo = bp.get("pbo") or {}
            dsr = bp.get("deflated_sharpe") or {}
            backtest_body = (
                "# Out-of-Sample Backtest\n\n"
                f"- **Scheme:** {bp.get('scheme', 'n/a')}\n"
                f"- **OOS {bp.get('oos_metric_name', metric_name)}:** {_fmt(bp.get('oos_metric'))}\n"
                f"- **Scored rows:** {bp.get('scored_rows', 'n/a')}\n"
                f"- **PBO (prob. of backtest overfitting):** {_fmt(pbo.get('pbo'))}\n"
                f"- **Deflated Sharpe ratio:** {_fmt(dsr.get('deflated_sharpe'))}\n"
                f"- **IMPACT used:** {bp.get('used_impact', False)} — "
                f"{bp.get('impact_note', 'n/a')}\n\n"
                f"{bp.get('strategy_note', '')}\n\n"
                "The backtest is implemented at {@code:src/cognos/stages/backtest.py}. "
                "See [model](./model.md)."
            )
        else:
            backtest_body = (
                "# Out-of-Sample Backtest\n\n"
                "_No backtest stage results are available for this run._\n\n"
                "The backtest is implemented at {@code:src/cognos/stages/backtest.py}."
            )
        emit(OKFConcept(
            name="backtest", type="backtest",
            title="Backtest", description="Out-of-sample performance, PBO, and DSR.",
            resource=ctx.rel(ctx.resolve("stages/backtest/scored.parquet")),
            tags=["backtest"], body=backtest_body,
        ))

        # --- 8. validation -------------------------------------------------------
        if vp:
            val_summary = validate.summary if validate else ""
            emit(OKFConcept(
                name="validation", type="validation",
                title="Independent Validation", description="Summary of the validation gate.",
                tags=["validation"],
                body=(
                    "# Independent Validation\n\n"
                    f"**Verdict:** {validate.verdict.value if validate else 'n/a'}\n\n"
                    f"{val_summary or 'See the validation stage result.'}\n\n"
                    f"Findings raised: {len(validate.findings) if validate else 0}.\n\n"
                    "See [model](./model.md) and [diagnostics](./diagnostics.md)."
                ),
            ))

        # --- 9. compliance -------------------------------------------------------
        if cp:
            emit(OKFConcept(
                name="compliance", type="compliance",
                title="Compliance Summary",
                description="SR 11-7 / NIST AI RMF / fair-lending review.",
                tags=["compliance"],
                body=(
                    "# Compliance Summary\n\n"
                    f"- **Regimes:** {', '.join(cfg.compliance.regimes) or 'n/a'}\n"
                    f"- **Risk tier:** {cfg.compliance.risk_tier}\n"
                    f"- **Jurisdictions:** {', '.join(cfg.compliance.jurisdictions) or 'n/a'}\n"
                    f"- **Fair lending checks:** {cfg.compliance.fair_lending}\n"
                    f"- **Verdict:** {comply.verdict.value if comply else 'n/a'}\n\n"
                    f"{comply.summary if comply else 'See the compliance stage result.'}\n\n"
                    "See the [model card](./model_card.md) and [limitations](./limitations.md)."
                ),
            ))

        # --- 10. model card (Google's 9 sections) --------------------------------
        emit(OKFConcept(
            name="model_card", type="model_card",
            title="Model Card", description="Google Model Card — 9 standard sections.",
            tags=["model_card", "governance"],
            body=self._model_card_body(cfg, mp, ep, bp, cp, family, metric_name),
        ))

        # --- 11. limitations / caveats -------------------------------------------
        emit(OKFConcept(
            name="limitations", type="caveats",
            title="Limitations & Assumptions",
            description="Known limitations, assumptions, and recommendations.",
            tags=["caveats"],
            body=(
                "# Limitations & Assumptions\n\n"
                "- The model is valid only within the distribution of its training data; "
                "monitor for drift before relying on out-of-distribution scores.\n"
                f"- Out-of-scope use: {cfg.compliance.out_of_scope_use or 'see the model card.'}\n"
                + (
                    "- Exploration flagged potential target-leakage suspects "
                    f"({', '.join(map(str, leakage))}); confirm prediction-time availability.\n"
                    if leakage else
                    "- No target-leakage suspects were flagged, but feature availability at "
                    "prediction time should still be confirmed.\n"
                )
                + "- The frozen-holdout metric is a single-shot estimate; real-world performance "
                "may differ.\n\n"
                "See the [model card](./model_card.md) and [methodology](./methodology.md)."
            ),
        ))

        concept_names = [
            "overview", "dataset", "methodology", "model", "coefficients",
            "diagnostics", "backtest", "limitations", "model_card",
        ]
        if vp:
            concept_names.insert(7, "validation")
        if cp:
            concept_names.insert(8 if vp else 7, "compliance")

        # --- 12. EU AI Act Annex IV (only for EU deployments) --------------------
        if "EU" in cfg.compliance.jurisdictions:
            emit(OKFConcept(
                name="annex_iv", type="technical_documentation",
                title="EU AI Act — Annex IV Technical Documentation",
                description="The 9 Annex IV components for EU deployment.",
                tags=["eu_ai_act", "annex_iv", "governance"],
                body=self._annex_iv_body(cfg, mp, ep, bp, family, metric_name),
            ))
            concept_names.append("annex_iv")

        bundle.log_event(
            "Creation",
            f"Authored {len(concept_names)} concepts including the Google Model Card"
            + (" and EU Annex IV pack" if "EU" in cfg.compliance.jurisdictions else "") + ".",
        )
        bundle.finalize(
            title=f"{cfg.name} — Model White Paper",
            description=cfg.description or "COGNOS model-development knowledge bundle (OKF v0.1).",
        )

        # --- single-file human-readable white paper ------------------------------
        whitepaper = self._whitepaper(cfg, bundle, concept_names)
        wp_ref = ctx.save_text("stages/document/whitepaper.md", whitepaper, kind="text")
        res.add_artifact(wp_ref)

        # --- code-link integrity findings ----------------------------------------
        has_code_links = bool(code_links)
        if not has_code_links:
            res.add_finding(Finding(
                id="no-code-links", severity=Severity.MEDIUM, category="traceability",
                message="No docs<->code traceability anchors were emitted.",
                location="document",
            ))

        res.add_artifact(ArtifactRef(
            name="okf_bundle", kind="okf", path=ctx.rel(ctx.docs_dir),
            description="OKF white-paper bundle (index.md + concepts).",
        ))
        ctx.save_json("stages/document/result.payload.json", {"concepts": concept_names})

        res.payload = {
            "bundle_dir": ctx.rel(ctx.docs_dir),
            "n_concepts": len(concept_names),
            "concepts": concept_names,
            "model_card": "docs/model_card.md",
            "code_links": sorted(set(code_links)),
            "has_code_links": has_code_links,
            "whitepaper": "stages/document/whitepaper.md",
        }
        res.metrics = {"n_concepts": len(concept_names), "n_code_links": len(set(code_links))}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS
        res.summary = (
            f"Wrote OKF bundle with {len(concept_names)} concepts, a Google Model Card, "
            + ("an EU Annex IV pack, " if "EU" in cfg.compliance.jurisdictions else "")
            + f"and {len(set(code_links))} docs<->code anchor(s)."
        )
        return res

    # --- builders ----------------------------------------------------------------
    def _model_card_body(
        self, cfg, mp: dict, ep: dict, bp: dict, cp: dict, family: str, metric_name: str,
    ) -> str:
        """Render the 9-section Google Model Card from available payloads + config."""
        holdout = _fmt(mp.get("holdout_metric"))
        diagnostics = mp.get("diagnostics", {}) or {}
        return (
            "# Model Card\n\n"
            "## 1 Model Details\n\n"
            f"- **Name:** {cfg.name} (v{cfg.version})\n"
            f"- **Type:** {family} ({cfg.task.value})\n"
            f"- **Developed with:** COGNOS automated model-development pipeline\n"
            f"- **Description:** {cfg.description or 'n/a'}\n\n"
            "## 2 Intended Use\n\n"
            f"{cfg.compliance.intended_use or 'Intended use not specified.'}\n\n"
            f"**Out-of-scope use:** {cfg.compliance.out_of_scope_use or 'Not specified.'}\n\n"
            "## 3 Factors\n\n"
            "Relevant factors include the input feature distribution and, for fairness, the "
            f"protected attributes: {', '.join(cfg.data.protected_attributes) or 'none declared'}.\n\n"
            "## 4 Metrics\n\n"
            f"- **Primary metric:** {metric_name} ({cfg.metric.direction.value})\n"
            f"- **CV {metric_name}:** {_fmt(mp.get('cv_mean'))} ± {_fmt(mp.get('cv_std'))}\n"
            f"- **Frozen-holdout {metric_name}:** {holdout}\n\n"
            "## 5 Evaluation Data\n\n"
            f"A sealed frozen holdout of {mp.get('n_holdout', 'n/a')} rows, never touched during "
            "model search. See [dataset](./dataset.md).\n\n"
            "## 6 Training Data\n\n"
            f"{mp.get('n_train', 'n/a')} training rows over {len(ep.get('features', []) or [])} "
            f"features drawn from the source dataset ({ep.get('n_rows', 'n/a')} total rows).\n\n"
            "## 7 Quantitative Analyses\n\n"
            f"{diagnostics.get('n_passed', 0)} of {diagnostics.get('n_run', 0)} statistical tests "
            "passed; see [diagnostics](./diagnostics.md). "
            + (
                f"Out-of-sample backtest {bp.get('oos_metric_name', metric_name)}="
                f"{_fmt(bp.get('oos_metric'))}; see [backtest](./backtest.md)."
                if bp else "No backtest was run."
            )
            + "\n\n"
            "## 8 Ethical Considerations\n\n"
            + (
                f"Fair-lending checks ({', '.join(cfg.compliance.regimes)}) were performed; "
                f"see [compliance](./compliance.md). Verdict: "
                f"{cp.get('verdict', 'see compliance stage')}.\n\n"
                if cp else
                "Protected attributes are excluded from model features (disparate-treatment "
                "avoidance). Run the compliance stage for disparate-impact testing.\n\n"
            )
            + "## 9 Caveats & Recommendations\n\n"
            "Validity is bounded by the training distribution; monitor for drift and re-validate "
            "before high-stakes use. See [limitations](./limitations.md)."
        )

    def _annex_iv_body(
        self, cfg, mp: dict, ep: dict, bp: dict, family: str, metric_name: str,
    ) -> str:
        """Render the 9 EU AI Act Annex IV technical-documentation components."""
        diagnostics = mp.get("diagnostics", {}) or {}
        return (
            "# EU AI Act — Annex IV Technical Documentation\n\n"
            "## 1 General description of the AI system\n\n"
            f"{cfg.name} (v{cfg.version}): a {family} {cfg.task.value} model. "
            f"Intended purpose: {cfg.compliance.intended_use or 'see the model card.'}\n\n"
            "## 2 Detailed description of elements and development process\n\n"
            "Developed via the COGNOS ratchet search with nested cross-validation and a sealed "
            "frozen holdout. See [methodology](./methodology.md).\n\n"
            "## 3 Monitoring, functioning and control\n\n"
            "Deployment scoring is served by {@code:src/cognos/runtime/score.py#score_row}; the "
            "model card section 9 lists operating caveats.\n\n"
            "## 4 Risk management system\n\n"
            f"Risk tier: {cfg.compliance.risk_tier}. Regimes applied: "
            f"{', '.join(cfg.compliance.regimes) or 'n/a'}. See [compliance](./compliance.md).\n\n"
            "## 5 Changes through the lifecycle\n\n"
            "Tracked in the bundle change log (log.md) and versioned config "
            f"(v{cfg.version}).\n\n"
            "## 6 Standards and specifications applied\n\n"
            "OKF v0.1 documentation; Google Model Card schema; SR 11-7 / NIST AI RMF where "
            "applicable.\n\n"
            "## 7 EU declaration of conformity\n\n"
            "To be issued by the deployer upon completion of the conformity assessment.\n\n"
            "## 8 Post-market monitoring plan\n\n"
            "Monitor input drift and performance against the frozen-holdout "
            f"{metric_name}={_fmt(mp.get('holdout_metric'))}; re-validate on degradation.\n\n"
            "## 9 Records, performance metrics and test results\n\n"
            f"{diagnostics.get('n_passed', 0)}/{diagnostics.get('n_run', 0)} statistical tests "
            "passed; see [diagnostics](./diagnostics.md) and [backtest](./backtest.md)."
        )

    def _whitepaper(self, cfg, bundle: OKFBundle, names: list[str]) -> str:
        """Concatenate the key concept bodies into a single human-readable white paper."""
        parts = [
            f"# {cfg.name} — Model White Paper",
            "",
            cfg.description or "A COGNOS-developed model.",
            "",
            "_Generated by the COGNOS document stage (OKF v0.1)._",
            "",
        ]
        concepts = {c.name: c for c in bundle.concepts()}
        for name in names:
            concept = concepts.get(name)
            if concept is None:
                continue
            parts.append("\n---\n")
            parts.append(concept.body.strip())
        return "\n".join(parts) + "\n"
