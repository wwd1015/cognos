"""Stage 7 — Documentation.

Emits the model development run as an OKF bundle (one markdown *concept* per artifact) so that a
downstream agent — human or machine — gets curated, cross-linked context with docs<->code
traceability anchors. On top of the white paper this stage also produces a Google Model Card (the
9-section regulator-facing summary) and, for EU deployments, an EU AI Act Annex IV technical
documentation pack. Every methodology/scoring claim is anchored to the code that implements it
(``{@code:...#symbol}``) so the consistency-review stage can verify the docs match the deployment.

v1: the **Technical Writer** agent drafts the narrative sections with ``{{fact:<id>}}``
placeholders that the engine renders (an agent never types a metric), and the bundle gains a
**decision log** — every agent recommendation, every human gate decision, and every challenge with
its response — so the white paper records who recommended what and who decided.
"""

from __future__ import annotations

from typing import Any

from ..analysis import consult
from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..okf import OKFBundle, OKFConcept
from .base import Stage, attach_recommendation, register_stage


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

        # --- 1b. engagement: the business intent and, for an update, the change record ----
        intake = ctx.get("intake")
        tp = (intake.payload if intake is not None else None) or {}
        if tp:
            emit(OKFConcept(
                name="engagement", type="engagement",
                title="Business Intent" + (" & Model Change Record" if tp.get("update") else ""),
                description="What the sponsor asked for, what stayed open, and what an update "
                            "changed.",
                resource=ctx.rel(ctx.resolve("stages/intake/brief.json")),
                tags=["intent", "governance"],
                body=self._engagement_body(cfg, tp, mp, metric_name),
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

        # --- 2b. exploratory analysis: what was run, and every script the analyst wrote --------
        analyses = ep.get("analyses") or []
        if analyses or ep.get("source"):
            emit(OKFConcept(
                name="analysis", type="analysis",
                title="Data Source & Exploratory Analysis",
                description="Where the data came from, the dependent variable, the analyses "
                            "run, and the code of every agent-written script.",
                resource=ctx.rel(ctx.resolve("stages/explore/analyses")),
                tags=["data", "analysis", "code"],
                body=self._analysis_body(ctx, ep, vp),
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

        # --- narrative (Technical Writer) + decision log ------------------------------
        narrative_payload = self._narrative(ctx, cfg, ip, emit)
        for f in consult.findings(narrative_payload.get("tool_runs", [])):
            res.add_finding(f)

        concept_names = [
            "overview", "narrative", "decisions", "dataset", "methodology", "model",
            "coefficients", "diagnostics", "backtest", "limitations", "model_card",
        ]
        if vp:
            concept_names.insert(9, "validation")
        if cp:
            concept_names.insert(10 if vp else 9, "compliance")
        if tp:
            concept_names.insert(1, "engagement")
        if analyses or ep.get("source"):
            concept_names.insert(concept_names.index("dataset") + 1, "analysis")

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
            **narrative_payload,
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
            "Developed via the COGNOS ratchet search with leakage-safe cross-validation and a sealed "
            "frozen (out-of-time when dated) holdout. See [methodology](./methodology.md).\n\n"
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

    @staticmethod
    def _join_body(linked: dict | None) -> list[str]:
        """How several sources became the modelling table: each source, the join as run, and
        who decided it."""
        if not linked:
            return []
        plan, report = linked["plan"], linked["report"]
        who = {"profile": "stated in the project profile", "decision": "set at the data gate",
               "agent": "proposed by the Data Analyst, confirmed at the data gate"}
        reasons = (linked.get("rationale") or {}).get("joins") or {}
        parts = [
            "## Sources and join\n",
            _table(["Source", "Connector", "Location", "Rows", "Columns", "SHA-256"],
                   [[t["name"], t.get("connector") or "n/a",
                     " ".join(str(t.get("query") or t.get("location") or "n/a").split())[:120],
                     str(t["n_rows"]), str(t["n_cols"]),
                     f"`{str(t.get('snapshot_sha256') or '')[:16]}`"]
                    for t in linked["tables"]]), "",
            f"- **Base table:** `{plan['base']}` ({report.get('base_rows', report['n_rows'])} "
            f"rows; {report['n_rows']} in the modelling table; a join never adds a row)",
            f"- **Join:** {who.get(plan['source'], 'n/a')}\n",
        ]
        if report["steps"]:
            parts += [_table(["Added", "On", "Rows matched", "Unmatched rows", "Rows per key", "Why"],
                             [[s["right"], "; ".join(s["on"]), f"{s['match_rate']:.1%}",
                               (f"{s['dropped_rows']} dropped" if s.get("dropped_rows") else
                                "none" if s["match_rate"] >= 1 else "kept with missing values"),
                               {"none": "one", "aggregate": "several, aggregated",
                                "first": "several, first kept"}.get(s["reduced"], s["reduced"]),
                               str(reasons.get(s["link"], "")).replace("|", "/")]
                              for s in report["steps"]]), ""]
        if linked.get("left_out"):
            parts.append(f"- **Not joined:** {', '.join(linked['left_out'])}\n")
        return parts

    @staticmethod
    def _analysis_body(ctx: RunContext, ep: dict, vp: dict) -> str:
        """Provenance, the target decision and the analysis log, then each script in full. The
        code is printed as recorded: it is a development artifact a validator has to read."""
        from ..analysis import run as analysis

        src = ep.get("source") or {}
        how = {"profile": "fixed in the project profile", "decision": "set at the data gate",
               "agent": "proposed by the Data Analyst, confirmed at the data gate"}
        parts = [
            "# Data Source & Exploratory Analysis\n",
            "## Data source\n",
            f"- **Connector:** {src.get('connector', 'n/a')} (`{src.get('kind', 'n/a')}`)",
            f"- **Location:** {src.get('location') or 'n/a'}"
            + (f" — query: `{' '.join(str(src['query']).split())[:300]}`" if src.get("query") else ""),
            f"- **Snapshot:** {src.get('n_rows', 'n/a')} rows × {src.get('n_cols', 'n/a')} columns, "
            f"fetched {src.get('fetched_at', 'n/a')}, SHA-256 `{str(src.get('snapshot_sha256', ''))[:16]}`\n",
            *DocumentStage._join_body(ep.get("linking")),
            "## Dependent variable\n",
            f"- **Target:** `{ep.get('target', 'n/a')}` ({ep.get('task', 'n/a')}), "
            f"{how.get(ep.get('target_source', ''), 'n/a')}",
            f"- **Basis:** {ep.get('target_rationale') or 'n/a'}\n",
            "## Features considered\n",
            _table(["Feature", "Expected", "Why"],
                   [[c["column"], c["relationship"], c["rationale"].replace("|", "/")]
                    for c in ep.get("feature_candidates", [])]), "",
            "## Analyses run\n",
            _table(["Id", "By", "What", "Question", "Status"],
                   [[a["id"], "agent-written code" if a["kind"] == "code" else
                     ("tool" if a.get("origin") == "cognos" else f"plugin tool ({a.get('origin')})"),
                     a.get("tool") or "script", a["purpose"].replace("|", "/"),
                     a["status"] + (f": {a['error'][:80]}" if a.get("error") else "")]
                    for a in ep.get("analyses", [])]),
        ]
        scripts = [a for a in ep.get("analyses", []) if a["kind"] == "code"]
        if scripts:
            checked = {c["id"]: c for c in vp.get("analysis_code", [])}
            parts += ["", "## Analysis code\n",
                      "Written by the Data Analyst agent and executed by the engine in a "
                      "restricted process. Each script was shown to the model developer at the "
                      "data gate and re-run by validation.\n"]
            for a in scripts:
                c = checked.get(a["id"])
                status = ("not re-run" if c is None else "reproduced at validation" if c["reproduced"]
                          else f"NOT reproduced at validation ({c['note']})")
                try:
                    code = analysis.code_of(ctx.resolve(a["code_path"]).read_text(encoding="utf-8"))
                except FileNotFoundError:
                    code = "# (script file missing)"
                parts += [f"### {a['id']} — {a['purpose']}\n",
                          f"- **File:** `{a['code_path']}`  **SHA-256:** `{a['code_sha256'][:16]}`",
                          f"- **Result:** {a['status']}; {status}\n",
                          "```python", code.strip(), "```", ""]
        # Tools the other stages' agents requested (built in or from a plugin), and the ones
        # registered for a stage that could not be run: a reviewer must see both.
        runs, not_run = [], []
        for stage in ("intake", "ideate", "model", "backtest", "validate", "comply"):
            res = ctx.get(stage)
            for r in consult.recorded(ctx, stage):
                checks = r.get("checks") or []
                failed = [c["name"] for c in checks if not c["passed"]]
                runs.append([stage, r["id"], r["tool"],
                             "cognos" if r.get("origin") == "cognos" else
                             f"plugin ({r.get('origin')}" + (f" {r['version']})" if r.get("version")
                                                            else ")"),
                             r["purpose"].replace("|", "/"),
                             r["status"] + (f": {r['error'][:80]}" if r.get("error") else ""),
                             "—" if not checks else (f"FAILED: {', '.join(failed)}" if failed
                                                     else f"{len(checks)} passed")])
            for t in ((res.payload if res is not None else None) or {}).get(
                    "tools_unavailable") or []:
                not_run.append([stage, t["name"], t.get("origin", ""), t.get("note", "")])
        if runs:
            parts += ["", "## Tools run by stage\n",
                      "Requested by each stage's agent and executed by the engine. A tool is "
                      "reviewed code; its numbers are recorded as engine facts.\n",
                      _table(["Stage", "Id", "Tool", "From", "Question", "Status", "Checks"], runs)]
        if not_run:
            parts += ["", "## Tools registered but not run\n",
                      _table(["Stage", "Tool", "From", "Why not"], not_run)]
        return "\n".join(parts)

    @staticmethod
    def _engagement_body(cfg, tp: dict, mp: dict, metric_name: str) -> str:
        """The intake brief as recorded, with design fields read from the effective config (an
        answer given after intake wins). Numbers are the engine's, from this run and, for an
        update of an earlier COGNOS run, from that run's record."""
        basis = {"stated": "intent document", "answered": "sponsor answer",
                 "profile": "project profile", "inferred": "inferred, unconfirmed",
                 "missing": "open"}
        rows = []
        for b in tp.get("brief", []):
            value, src = b.get("value") or "", basis.get(b.get("basis", ""), "")
            live = str(getattr(cfg.design, b["field"], "") or "") if b["field"] in (
                "use_case", "horizon", "default_definition", "segment") else ""
            if live and live != value:
                value, src = live, "sponsor answer"
            if not value and not b.get("required"):
                value, src = "not stated", ""
            rows.append([b["label"], value.replace("|", "/") or "_open_", src])
        parts = [
            "# Business Intent\n",
            f"- **Development mode:** {tp.get('kind_label', tp.get('kind', 'n/a'))}",
            f"- **Objective:** {tp.get('objective') or 'n/a'}",
            f"- **Intent at intake:** {str(tp.get('clarity', 'n/a')).replace('_', ' ')}\n",
            _table(["Field", "Sponsor position", "Source"], rows), "",
            "## Open at intake\n",
            "\n".join(f"- {q['question']}" for q in tp.get("questions", [])) or "Nothing.", "",
            "## Documents received\n",
            _table(["Document", "Role", "SHA-256"],
                   [[d["name"], d["role"], (d.get("sha256") or "")[:16] or "unreadable"]
                    for d in tp.get("documents", [])]),
        ]
        up = tp.get("update")
        if up:
            parts += [
                "", "## Model change record\n",
                f"- **Scope:** {str(up.get('scope')).replace('_', ' ')} — {up.get('rationale', '')}",
                f"- **Existing model family:** {up.get('incumbent_family') or 'not identified'}\n",
                _table(["Requested change", "Type", "Enters at"],
                       [[c["change"].replace("|", "/"), c["type"].replace("_", " "), c["affects"]]
                        for c in up.get("change_items", [])]),
            ]
            prior = (tp.get("prior") or {}).get("run")
            if prior:
                champion = mp.get("champion", {}) or {}
                same = prior.get("metric") == metric_name
                parts += [
                    "", "## Existing model against this update\n",
                    f"The existing model is COGNOS run `{prior.get('run_id')}`. Both columns are "
                    "what each run recorded; the two runs keep separate sealed holdouts, so the "
                    "holdout figures are not measured on the same sample.\n",
                    _table(["", "Existing model", "This update"], [
                        ["Champion family", str(prior.get("champion_family") or "n/a"),
                         str(champion.get("family", "n/a"))],
                        ["Features", str(len(prior.get("features") or [])),
                         str(len(champion.get("features", []) or []))],
                        ["Metric", str(prior.get("metric") or "n/a"), metric_name],
                        [f"CV {metric_name}" if same else "CV metric",
                         _fmt(prior.get("cv_mean")), _fmt(mp.get("cv_mean"))],
                        [f"Holdout {metric_name}" if same else "Holdout metric",
                         _fmt(prior.get("holdout_metric")), _fmt(mp.get("holdout_metric"))],
                        ["Validation verdict", str(prior.get("validation_verdict") or "n/a"),
                         "see the validation section"],
                    ]),
                ]
        return "\n".join(parts)

    def _narrative(self, ctx: RunContext, cfg, ip: dict, emit) -> dict:
        """Ask the Technical Writer for the narrative; render placeholders; emit the decision log."""
        from ..agents import facts as facts_mod
        from ..agents.contracts import FRIENDLY, SECTIONS
        from ..agents.slices import fact_scope
        from ..engine.state import RunState

        state = RunState.load(ctx.run_dir) if RunState.exists(ctx.run_dir) else None
        decisions = [{"gate": d.gate, "action": d.action, "actor": d.actor,
                      "seat": d.seat or d.actor, "reason": d.reason, "at": d.at}
                     for d in (state.decisions if state else [])]
        challenge_log = [{"id": c.id, "source": c.source, "stage": c.target_stage,
                          "severity": c.severity, "message": c.message, "status": c.status,
                          "response": c.response} for c in (state.challenges if state else [])]
        tool_runs = ctx.consult_tools("writer", {"sections_required": SECTIONS})
        out = ctx.recommend("writer", {
            "tool_runs": tool_runs,
            "sections_required": SECTIONS,
            "framework_choices": ip.get("framework_choices") or [],
            "design": cfg.design.model_dump(),
            "excluded_columns": list(ctx.overrides.exclude_columns),
            "open_questions": [q["question"] for q in ip.get("open_questions", [])],
            "decisions": decisions,
            "challenge_log": challenge_log,
        })
        facts = facts_mod.collect(ctx, prefixes=fact_scope("writer"))
        titles = {"executive_summary": "Executive summary",
                  "methodology_rationale": "Methodology rationale",
                  "alternatives_considered": "Alternatives considered",
                  "limitations": "Limitations and assumptions",
                  "use_and_monitoring": "Use and monitoring"}
        rendered = {s.section: facts_mod.render(s.markdown, facts) for s in out.sections}
        body = ["# Narrative", "",
                "_Drafted by the Technical Writer agent; every number is rendered from a recorded "
                "fact._", ""]
        for key in SECTIONS:
            body += [f"## {titles[key]}", "", rendered.get(key, "").strip(), ""]
        emit(OKFConcept(name="narrative", type="narrative", title="Narrative",
                        description="Agent-drafted narrative with engine-rendered numbers.",
                        tags=["narrative"], body="\n".join(body)))

        # Decision log: who recommended what, who decided.
        recs = []
        for stage in ("intake", "explore", "ideate", "model", "backtest", "validate", "comply"):
            r = ctx.get(stage)
            rec = (r.payload.get("recommendation") if r is not None else None) or {}
            if rec:
                recs.append([stage, FRIENDLY.get(rec.get("agent", ""), rec.get("agent", "")),
                             rec.get("provider", "") if rec.get("model") in (None, rec.get("provider"), "heuristic", "replay")
                             else f"{rec.get('provider', '')}:{rec.get('model', '')}",
                             str(rec.get("output", {}).get("summary", "")).replace("|", "/")])
        log = ["# Decision log", "",
               "Agents recommend. A person decides in a seat: the model developer, the "
               "independent reviewer, or the approver. Express preparation accepts gates so a "
               "run can finish; that is not a signature. A sealed package is the signature.", "",
               "## Agent recommendations", "",
               _table(["Stage", "Agent", "Backend", "Recommendation"], recs), "",
               "## Gate decisions", "",
               _table(["Gate", "Action", "Seat", "Reason", "At"],
                      [[d["gate"], d["action"], d["seat"], d["reason"].replace("|", "/") or "-",
                        d["at"]] for d in decisions]), "",
               "## Challenges and responses", "",
               _table(["Id", "Source", "Stage", "Severity", "Challenge", "Status", "Response"],
                      [[c["id"], c["source"], c["stage"], c["severity"],
                        c["message"].replace("|", "/"), c["status"],
                        (c["response"] or "-").replace("|", "/")] for c in challenge_log])]
        emit(OKFConcept(name="decisions", type="decision_log", title="Decision Log",
                        description="Agent recommendations, human gate decisions, and challenges.",
                        tags=["governance", "audit"], body="\n".join(log)))
        payload: dict = {"narrative_sections": rendered}
        attach_recommendation(payload, ctx, "writer", out)
        return payload

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
