"""Stage 2 — Idea generation (the design stage).

Works the way a senior modeler opens a commercial-risk engagement: first understand the *data
structure* (cross-sectional vs. vintage/panel, event counts), then pick the *econometric framework*
(structural vs. reduced-form vs. hazard vs. ML challenger) before any algorithm, and triangulate the
*design brief* with the model sponsor (the MD) — every unanswered design point becomes an explicit
open question, never a silent assumption. The output is a ranked slate of candidate model
specifications (family + feature strategy) for the modeling stage to search, plus a human-readable
``design_brief.md``.

Everything above is deterministic (rules over the explore profile + config) so the stage runs
offline. v1: the **Design Lead** agent then decides every framework's role, ranks the slate (only
engine-fittable families), may propose feature transforms (engine-validated, target-hidden) and
adds sponsor questions — reasoning proposes, the engine disposes; the human reviews at gate_design.
"""

from __future__ import annotations

import hashlib
import re

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..config import CognosConfig
from ..context import RunContext
from ..modeling.fit import DEFAULT_FAMILIES
from ..modeling.transforms import TransformSpec, apply_transforms
from .base import Stage, attach_recommendation, register_stage


def _qhash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:6]


_DESIGN_FIELDS = ("use_case", "horizon", "default_definition", "segment")


def _gap_spec(q: dict) -> dict:
    """An open question as the engine's gap record: what an answer fills and where it re-enters."""
    field = q.get("design_field")
    if field is None and q["id"].startswith("design-") and q["id"][7:] in _DESIGN_FIELDS:
        field = q["id"][7:]
    category = q.get("category") or ("design" if q.get("source") == "design-brief" else "data")
    reentry = "explore" if q["id"] == "data-leakage" else "ideate"
    return {"id": q["id"], "question": q["question"], "category": category,
            "source": q.get("source", "engine"), "design_field": field, "reentry": reentry}


# Lower interpretability rank = more interpretable / preferred for regulated use.
_INTERPRETABILITY = {
    "ols": 0.95, "logit": 0.95, "probit": 0.93, "cloglog": 0.92,
    "hazard_logit": 0.9, "hazard_cloglog": 0.9,
    "ridge": 0.85, "lasso": 0.88, "elasticnet": 0.83,
    "ridge_logit": 0.85, "lasso_logit": 0.88, "poisson": 0.9, "gamma": 0.9, "tweedie": 0.87,
    "random_forest": 0.45, "gradient_boosting": 0.4,
}
_TREE_FAMILIES = {"random_forest", "gradient_boosting"}

# Column-name hints used to judge framework applicability from the data alone. Matched on
# underscore-separated name tokens (not substrings), so e.g. "operating_margin" ≠ "rating".
_MARKET_HINTS = ("equity_vol", "asset_vol", "market_cap", "equity_value", "stock_price",
                 "share_price", "mkt")
_RATING_HINTS = ("rating", "grade")
_DATE_HINTS = ("date", "vintage", "asof", "as_of", "cohort", "snapshot", "period")

# Events-per-variable below this suggests the sample cannot support a wide feature set
# (Peduzzi et al. 1996 rule of thumb for logistic regression).
_EPV_FLOOR = 10.0


def _engine_families(cfg: CognosConfig) -> list[str]:
    """Every family the engine can fit for this task — the LLM's proposal space, which may be
    wider than the deterministic slate (e.g. trees when the regulated default excludes them)."""
    if cfg.task.is_classification:
        fams = ["logit", "probit", "cloglog", "ridge_logit", "lasso_logit",
                "random_forest", "gradient_boosting"]
        if cfg.data.event_time_col:  # hazard families need event timing to be fittable
            fams += ["hazard_logit", "hazard_cloglog"]
        return fams
    if cfg.task.value == "timeseries":
        return ["ols", "ridge", "lasso"]
    return ["ols", "ridge", "lasso", "elasticnet", "poisson", "gamma", "tweedie",
            "random_forest", "gradient_boosting"]


def _incumbent(ctx: RunContext, cfg: CognosConfig, profile: dict) -> dict | None:
    """Model update: what the run knows about the existing model's specification. The family comes
    from the confirmed intake brief; the features are the dataset columns the prior artifacts name
    (or the champion's features, when the prior model is an earlier COGNOS run). No scores."""
    intake = ctx.get("intake")
    p = (intake.payload if intake is not None else None) or {}
    update = p.get("update")
    if not update:
        return None
    from .. import engagement as eg

    prior_run = (p.get("prior") or {}).get("run") or {}
    features = list(profile.get("features", []))
    try:
        corpus = ctx.load_json("stages/intake/corpus.json").get("documents", [])
    except (FileNotFoundError, ValueError):
        corpus = []
    texts = [d.get("text") or "" for d in corpus if d.get("role") not in ("intent", "supporting")]
    named = ([f for f in features if f in set(prior_run.get("features") or [])]
             or eg.columns_mentioned(features, texts))
    family = update.get("incumbent_family")
    return {"family": family, "fittable": bool(family) and family in _engine_families(cfg),
            "features_in_data": named, "scope": update.get("scope"),
            "change_items": update.get("change_items", [])}


def _match_hints(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    """Columns whose underscore-separated tokens (or token runs) equal a hint."""
    out = []
    for c in columns:
        tokens = [t for t in re.split(r"[^a-z0-9]+", c.lower()) if t]
        runs = tokens + ["_".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        if any(h in runs for h in hints):
            out.append(c)
    return out


def _data_structure(cfg: CognosConfig, profile: dict) -> dict:
    """What kind of sample is this? Shape, size, and (for PD-style tasks) event support."""
    features = profile.get("features", [])
    if cfg.task.value == "timeseries":
        shape = "timeseries"
    elif cfg.data.datetime_col:
        shape = "panel"  # time-indexed observations (origination vintages / snapshot cohorts)
    else:
        shape = "cross_sectional"
    structure: dict = {
        "shape": shape,
        "datetime_col": cfg.data.datetime_col,
        "n_rows": profile.get("n_rows"),
        "n_features": len(features),
        "n_numeric": len(profile.get("numeric_features", [])),
        "n_categorical": len(profile.get("categorical_features", [])),
        "leakage_suspects": list(profile.get("leakage_suspects", [])),
    }
    if cfg.task.is_classification:
        classes = (profile.get("target_summary") or {}).get("classes") or {}
        counts = [int(v) for v in classes.values()]
        events = min(counts) if counts else None  # the rare class is the event in PD-style data
        structure["n_events"] = events
        structure["event_rate"] = (events / profile["n_rows"]) if events and profile.get("n_rows") else None
        if events is not None and features:
            # Approximate: one parameter per feature (categorical dummies make this optimistic).
            structure["events_per_variable"] = round(events / len(features), 1)
    return structure


def _framework_assessment(cfg: CognosConfig, profile: dict, structure: dict) -> list[dict]:
    """Which econometric frameworks fit this problem — and, for SR 11-7 'alternatives considered'
    documentation, which were rejected and why. Deterministic rules over the data structure."""
    all_cols = list(profile.get("dtypes", {}).keys())
    feats = profile.get("features", [])
    frameworks: list[dict] = []

    if cfg.task.is_classification:
        frameworks.append({
            "framework": "reduced_form_pd",
            "label": "Reduced-form PD (obligor scorecard / GLM)",
            "applicable": True,
            "role": "primary",
            "reason": "Binary outcome with obligor-level financial and facility covariates — the "
                      "standard framework for commercial PD at origination or surveillance.",
            "engine_families": ["logit", "ridge_logit", "lasso_logit"],
            "reference": "Altman (1968); Ohlson (1980); Basel IRB PD; SR 11-7",
        })
        panel = structure["shape"] == "panel"
        has_event_time = bool(cfg.data.event_time_col)
        frameworks.append({
            "framework": "discrete_time_hazard",
            "label": "Discrete-time hazard (survival)",
            "applicable": True if has_event_time else ("partial" if panel else False),
            "role": "candidate" if (has_event_time or panel) else "rejected",
            "reason": ("Event timing present (data.event_time_col) — the engine panel-expands "
                       "obligor-periods and fits a full discrete-time hazard (logit or cloglog "
                       "grouped-time PH link) with a PD term structure." if has_event_time else
                       ("Time-indexed cohorts present — a period-indexed logit reads as a "
                        "discrete-time hazard; set data.event_time_col (default period per "
                        "obligor) to unlock the full hazard families and PD term structure."
                        if panel else
                        "No datetime column — a hazard framework needs time-indexed observations "
                        "(set data.datetime_col if this sample has vintages/snapshots).")),
            "engine_families": (["hazard_logit", "hazard_cloglog"] if has_event_time
                                else (["logit"] if panel else [])),
            "reference": "Shumway (2001) hazard bankruptcy model",
        })
        market_cols = _match_hints(feats, _MARKET_HINTS)
        structural_on = bool(cfg.structural.enabled) and bool(market_cols)
        frameworks.append({
            "framework": "structural_merton",
            "label": "Structural (Merton distance-to-default)",
            "applicable": bool(market_cols),
            "role": "candidate" if structural_on else ("available" if market_cols else "rejected"),
            "reason": ((f"Market observables detected ({', '.join(market_cols)}); the structural: "
                        "config block is enabled — the engine solves for distance-to-default, "
                        "feeds it to the champion (hybrid), and scores the pure structural PD as "
                        "a challenger benchmark.") if structural_on else
                       (f"Market observables detected ({', '.join(market_cols)}) — enable the "
                        "structural: config block to compute distance-to-default (hybrid feature "
                        "+ benchmark)." if market_cols else
                        "Requires traded-market observables (equity value/volatility, liability "
                        "structure) to compute distance-to-default; none present — typical for "
                        "private middle-market obligors.")),
            "engine_families": [],
            "reference": "Merton (1974); KMV/Moody's EDF",
        })
        rating_cols = _match_hints(all_cols, _RATING_HINTS)
        migration_on = bool(cfg.migration.enabled) and bool(rating_cols)
        frameworks.append({
            "framework": "transition_matrix",
            "label": "Rating-transition matrix (migration)",
            "applicable": bool(rating_cols),
            "role": "candidate" if migration_on else ("available" if rating_cols else "rejected"),
            "reason": ((f"Rating columns detected ({', '.join(rating_cols)}); the migration: "
                        "config block is enabled — the engine estimates a cohort transition "
                        "matrix on the training partition (NR-adjusted), feeds the rating-implied "
                        "horizon PD to the champion (hybrid), scores the pure-migration PD as a "
                        "challenger benchmark, and reports a by-rating expected-loss forecast.")
                       if migration_on else
                       (f"Rating columns detected ({', '.join(rating_cols)}) — enable the "
                        "migration: config block (rating_col + next_rating_col) to estimate a "
                        "transition matrix (hybrid feature + benchmark + loss forecast)."
                        if rating_cols else
                        "Requires a rating history (grade at successive snapshots, e.g. an "
                        "internal grade or agency rating panel); none present.")),
            "engine_families": [],
            "reference": "CreditMetrics (1997); S&P CreditPro rating-migration practice",
        })
    elif cfg.task.value == "timeseries":
        frameworks.append({
            "framework": "reduced_form_forecast",
            "label": "Reduced-form forecasting (regression on time features)",
            "applicable": True,
            "role": "primary",
            "reason": "Time-ordered target; evaluation must be walk-forward (no shuffled CV).",
            "engine_families": ["ols", "ridge", "lasso"],
            "reference": "out-of-time backtesting practice; SR 11-7 outcomes analysis",
        })
    else:
        frameworks.append({
            "framework": "reduced_form_glm",
            "label": "Reduced-form GLM / linear regression",
            "applicable": True,
            "role": "primary",
            "reason": "Continuous outcome with tabular covariates — interpretable coefficients "
                      "with valid inference for documentation.",
            "engine_families": ["ols", "ridge", "lasso", "elasticnet", "poisson", "gamma", "tweedie"],
            "reference": "GLM practice (McCullagh & Nelder); SR 11-7",
        })

    interp = cfg.design.interpretability
    frameworks.append({
        "framework": "ml_challenger",
        "label": "Machine-learning challenger (trees/boosting)",
        "applicable": True,
        "role": "challenger" if interp == "required" else "candidate",
        "reason": ("Interpretability is required for the deployed model, so nonlinear learners "
                   "serve as challenger benchmarks quantifying the predictive ceiling."
                   if interp == "required" else
                   "Interpretability is negotiable — nonlinear learners compete for champion, "
                   "subject to validation and documentation burden."),
        "engine_families": ["gradient_boosting", "random_forest"],
        "reference": "champion–challenger practice; SR 11-7 benchmarking",
    })
    return frameworks


def _open_questions(cfg: CognosConfig, profile: dict, structure: dict) -> list[dict]:
    """MD triangulation: the design points ideate refuses to assume silently."""
    q: list[dict] = []
    templates = {
        "use_case": "What decision will the model support (origination underwriting, portfolio "
                    "surveillance, CECL/IFRS 9, IRB, stress testing)? The use case fixes the "
                    "target definition, horizon, and documentation depth.",
        "horizon": "What outcome window defines the target (e.g. 12-month default)? The window "
                   "must match how the target column was labelled.",
        "default_definition": f"What event definition labels '{cfg.data.target}' (e.g. 90+ DPD, "
                              "nonaccrual, bankruptcy)? Validation re-derives risk from this.",
        "segment": "What portfolio segment does this sample represent (C&I, CRE, small business…)? "
                   "Pooling heterogeneous segments biases coefficients.",
    }
    for field in cfg.design.unanswered():
        q.append({"id": f"design-{field}", "question": templates[field], "source": "design-brief"})

    leaks = structure.get("leakage_suspects") or []
    if leaks:
        q.append({"id": "data-leakage", "source": "data",
                  "question": f"Explore flagged {', '.join(leaks)} as possible target leakage — "
                              "confirm whether each is in the information set at prediction time; "
                              "if not, add it to data.drop_columns and re-run."})
    if structure["shape"] == "cross_sectional" and cfg.task.value != "timeseries":
        datey = _match_hints(list(profile.get("dtypes", {}).keys()), _DATE_HINTS)
        datey = [c for c in datey if c != cfg.data.target]
        if datey:
            q.append({"id": "data-panel", "source": "data",
                      "question": f"Column(s) {', '.join(datey)} look time-related but "
                                  "data.datetime_col is not set — if this is a vintage/panel "
                                  "sample, set datetime_col to unlock out-of-time evaluation."})
    epv = structure.get("events_per_variable")
    if epv is not None and epv < _EPV_FLOOR:
        q.append({"id": "data-epv", "source": "data",
                  "question": f"Only {structure['n_events']} events for {structure['n_features']} "
                              f"candidate features (≈{epv} events per variable, below the ~10 rule "
                              "of thumb) — confirm appetite for a short feature list, a coarser "
                              "segmentation, or a longer sampling window."})

    # Framework-unlock questions: capabilities the data supports but the config has not switched on.
    if cfg.task.is_classification and not cfg.data.event_time_col:
        timing = _match_hints(list(profile.get("dtypes", {}).keys()),
                              ("default_quarter", "default_month", "event_time",
                               "time_to_default", "months_to_default"))
        if timing:
            q.append({"id": "data-event-time", "source": "data",
                      "question": f"Column(s) {', '.join(timing)} look like event timing — set "
                                  "data.event_time_col to unlock the discrete-time hazard "
                                  "families and a PD term structure."})
    if cfg.task.is_classification and not cfg.structural.enabled:
        market = _match_hints(profile.get("features", []), _MARKET_HINTS)
        if market:
            q.append({"id": "data-structural", "source": "data",
                      "question": f"Market observables detected ({', '.join(market)}) — enable "
                                  "the structural: config block to compute Merton "
                                  "distance-to-default (hybrid feature + structural benchmark)."})
    if cfg.task.is_classification and not cfg.migration.enabled:
        ratingish = _match_hints(list(profile.get("dtypes", {}).keys()), _RATING_HINTS)
        ratingish = [c for c in ratingish if c != cfg.data.target]
        if ratingish:
            q.append({"id": "data-migration", "source": "data",
                      "question": f"Rating column(s) detected ({', '.join(ratingish)}) — enable "
                                  "the migration: config block (rating_col + next_rating_col) to "
                                  "estimate a rating-transition matrix: rating-implied PD as a "
                                  "hybrid feature, a pure-migration challenger benchmark, and a "
                                  "by-rating expected-loss forecast."})
    if cfg.portfolio.enabled and cfg.portfolio.asset_correlation is None:
        q.append({"id": "design-asset-correlation", "source": "design-brief",
                  "question": f"Portfolio simulation uses the Basel IRB asset-correlation formula "
                              f"and LGD={cfg.portfolio.lgd} — confirm both against portfolio "
                              "evidence or supply calibrated values."})
    if cfg.stress.enabled and not cfg.stress.scenarios:
        q.append({"id": "design-scenarios", "source": "design-brief",
                  "question": "Stress testing is enabled but no scenarios are defined — supply "
                              "the scenario set (e.g. baseline/adverse/severely adverse macro "
                              "shocks) in stress.scenarios."})
    return q[:8]


@register_stage
class IdeateStage(Stage):
    name = "ideate"
    requires = ("explore",)
    description = "Assess data structure and frameworks; propose and rank candidate model specs."

    def run(self, ctx: RunContext) -> StageResult:
        from ..modeling.transforms import SAFE_NP_FUNCS

        cfg = ctx.config
        profile = ctx.profile()  # human-excluded columns are already gone
        structure = _data_structure(cfg, profile)
        frameworks = _framework_assessment(cfg, profile, structure)
        questions = _open_questions(cfg, profile, structure)

        families = list(cfg.search.model_families or DEFAULT_FAMILIES.get(
            cfg.task.value, DEFAULT_FAMILIES["regression"]
        ))
        # Event timing unlocks the survival framework: add the hazard families to the default
        # slate (an explicit search.model_families list is respected as-is).
        if (not cfg.search.model_families and cfg.task.is_classification
                and cfg.data.event_time_col):
            families += [f for f in ("hazard_logit", "hazard_cloglog") if f not in families]
        # Under a hard/soft interpretability constraint the ratchet should spend its budget on the
        # defensible families first; the search honors this ordering when the budget binds.
        if cfg.design.interpretability in ("required", "preferred"):
            families.sort(key=lambda f: _INTERPRETABILITY.get(f, 0.5), reverse=True)
        # Model update: the existing specification is the benchmark the update has to beat, so
        # its family is always searched (an explicit search.model_families list is respected).
        incumbent = _incumbent(ctx, cfg, profile)
        if (incumbent and incumbent["fittable"] and not cfg.search.model_families
                and incumbent["family"] not in families):
            families.insert(0, incumbent["family"])

        # Leakage suspects never seed the parsimonious feature strategy.
        leaks = set(structure["leakage_suspects"])
        top_feats = [c["feature"] for c in profile.get("top_correlations", [])
                     if c["feature"] not in leaks][:5]

        epv = structure.get("events_per_variable")
        epv_low = epv is not None and epv < _EPV_FLOOR
        default_slate = self._default_slate(families, top_feats, cfg, structure, epv_low)
        if incumbent and incumbent["fittable"]:
            default_slate = self._incumbent_first(default_slate, incumbent)

        # --- the Design Lead's recommendation (engine-checked) ---------------------
        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        res.payload = {"data_structure": structure, "open_questions": questions,
                       "families": families}
        features = list(profile.get("features", []))

        def check_transforms(out, sl) -> list[str]:
            specs = [TransformSpec(name=t.name, expr=t.expr) for t in out.transforms]
            if not specs:
                return []
            _, _, rejected = apply_transforms(ctx.load_dataset()[features], specs)
            return [f"transform {spec.name!r} rejected by the engine ({why}); fix or drop it"
                    for spec, why in rejected]

        tool_runs = ctx.consult_tools("design_lead", {"data_structure": structure}, res=res)
        out = ctx.recommend("design_lead", {
            "tool_runs": tool_runs,
            "data_structure": structure,
            "framework_assessment": [{k: f[k] for k in ("framework", "label", "applicable", "role",
                                                          "reason")} for f in frameworks],
            "fittable_families": _engine_families(cfg),
            "default_slate": default_slate,
            "features": features,
            "clean_top_features": top_feats,
            "open_questions": questions,
            "safe_np_funcs": sorted(SAFE_NP_FUNCS),
            "search_budget": cfg.search.max_candidates,
            "interpretability": cfg.design.interpretability,
            **({"incumbent": incumbent} if incumbent else {}),
        }, fresh={"ideate": res}, check=check_transforms)

        # --- engine disposes: the slate (a human edit at gate_design wins) --------------
        if ctx.overrides.slate:
            slate, slate_source = ctx.overrides.slate, "human"
        else:
            slate = [i.model_dump() for i in out.slate]
            slate_source = "agent" if ctx.runner.kind != "heuristic" else "engine"
        hypotheses = []
        for i, item in enumerate(sorted(slate, key=lambda h: -float(h.get("priority", 0.5))), 1):
            fam = item["family"]
            hypotheses.append({
                "id": f"h{i}", "family": fam, "feature_strategy": item.get("feature_strategy", "all"),
                "framework": self._family_framework(fam, cfg, structure),
                "role": item.get("role", "candidate"),
                "interpretable": _INTERPRETABILITY.get(fam, 0.5) >= 0.7,
                "priority": round(float(item.get("priority", 0.5)), 3),
                "source": slate_source,
                "rationale": item.get("rationale", ""),
            })
        slate_families = list(dict.fromkeys(h["family"] for h in hypotheses))
        families = ([f for f in families if f in slate_families]
                    + [f for f in slate_families if f not in families])
        proposed_transforms = [{"name": t.name, "expr": t.expr, "rationale": t.rationale}
                               for t in out.transforms]
        framework_choices = [f.model_dump() for f in out.frameworks]
        for q in out.sponsor_questions:
            questions.append({"id": f"agent-{_qhash(q.question)}", "question": q.question,
                              "source": "agent", "category": q.category,
                              "design_field": None if q.design_field == "none" else q.design_field})
        questions = questions[:10]

        notes = (
            f"{len(families)} model families x feature strategies = {len(hypotheses)} hypotheses. "
            f"Strongest clean signals: {', '.join(top_feats) or 'n/a'}."
        )
        if epv_low:
            notes += (f" Low event support (EPV≈{epv}) — parsimonious feature sets up-weighted.")
        if incumbent:
            notes += (f" Model update ({str(incumbent['scope']).replace('_', ' ')}): existing "
                      f"family {incumbent['family'] or 'not identified'}; "
                      f"{len(incumbent['features_in_data'])} of its inputs found in this data.")
        notes += f" Design Lead: {out.summary}"

        payload = {
            "task": cfg.task.value,
            "families": families,
            "search_budget": cfg.search.max_candidates,
            "data_structure": structure,
            "clean_top_features": top_feats,  # strongest predictors net of leakage suspects
            "framework_assessment": frameworks,
            "framework_choices": framework_choices,
            "fittable_families": _engine_families(cfg),
            "open_questions": questions,
            "questions": [_gap_spec(q) for q in questions],
            "design": cfg.design.model_dump(),
            "hypotheses": hypotheses,
            "proposed_transforms": proposed_transforms,  # agent-authored, engine-validated
            "notes": notes,
        }
        if incumbent:
            payload["incumbent"] = incumbent
        attach_recommendation(payload, ctx, "design_lead", out)
        res.add_artifact(ctx.save_json("stages/ideate/hypotheses.json", payload))
        brief = self._design_brief_md(cfg, structure, frameworks, questions, hypotheses, notes,
                                      framework_choices)
        res.add_artifact(ctx.save_text("stages/ideate/design_brief.md", brief, kind="markdown"))

        if questions:
            res.add_finding(Finding(
                id="design-open-questions", severity=Severity.LOW, category="design",
                message=f"{len(questions)} open design question(s) for the model sponsor — see "
                        "stages/ideate/design_brief.md.",
                suggestion="Answer them at the design gate (or in the config's design: block); "
                           "ideate re-runs with the answers.",
            ))
        if epv_low:
            res.add_finding(Finding(
                id="design-epv", severity=Severity.MEDIUM, category="design/sample-size",
                message=f"Events-per-variable ≈{epv} (events={structure['n_events']}, "
                        f"features={structure['n_features']}) is below the ~{_EPV_FLOOR:.0f} rule "
                        "of thumb — coefficient estimates may be unstable.",
                suggestion="Prefer parsimonious candidates; consider a longer window or coarser "
                           "segmentation.",
            ))

        res.payload = payload
        res.metrics = {
            "n_hypotheses": len(hypotheses),
            "n_families": len(families),
            "n_frameworks_applicable": sum(1 for f in frameworks if f["applicable"]),
            "n_open_questions": len(questions),
        }
        res.summary = (
            f"Generated {len(hypotheses)} ranked hypotheses across {len(families)} families; "
            f"{res.metrics['n_frameworks_applicable']}/{len(frameworks)} frameworks applicable; "
            f"{len(questions)} open design question(s)."
        )
        if not hypotheses:
            res.verdict = Verdict.FAIL
            res.add_finding(Finding(id="no-ideas", severity=Severity.HIGH, category="idea-gen",
                                    message="No candidate model families resolved for this task."))
        return res

    def _default_slate(self, families: list[str], top_feats: list[str], cfg: CognosConfig,
                       structure: dict, epv_low: bool) -> list[dict]:
        """The engine's deterministic slate — the heuristic Design Lead adopts it as-is."""
        parsimony_bonus = 0.12 if epv_low else 0.05  # few events => lean on parsimonious sets
        slate: list[dict] = []
        for family in families:
            interp = _INTERPRETABILITY.get(family, 0.5)
            tree = family in _TREE_FAMILIES
            base = interp
            if tree and cfg.design.interpretability == "flexible":
                base = min(0.75, interp + 0.3)  # trees compete when the sponsor allows it
            role = "challenger" if (tree and cfg.design.interpretability == "required") else "candidate"
            for strategy in ("top", "all"):
                priority = round(min(1.0, base + (parsimony_bonus if strategy == "top" else 0.0)), 3)
                slate.append({
                    "family": family, "feature_strategy": strategy, "role": role,
                    "priority": priority,
                    "rationale": self._rationale(family, strategy, top_feats, cfg, structure),
                })
        return sorted(slate, key=lambda h: h["priority"], reverse=True)

    @staticmethod
    def _incumbent_first(slate: list[dict], incumbent: dict) -> list[dict]:
        """Model update: rank the existing model's family first and say why. A recalibration or
        re-estimation stays close to it; a redevelopment keeps it as the benchmark to beat."""
        close = incumbent.get("scope") in ("recalibrate", "re_estimate")
        note = ("Existing model's family: the update request keeps this specification."
                if close else "Existing model's family: the benchmark a redevelopment must beat.")
        out = []
        for item in slate:
            if item["family"] == incumbent["family"]:
                item = {**item, "priority": 1.0 if item["feature_strategy"] == "all" else 0.99,
                        "rationale": f"{note} {item['rationale']}"}
            out.append(item)
        return sorted(out, key=lambda h: (h["priority"], h["family"] == incumbent["family"]),
                      reverse=True)

    @staticmethod
    def _family_framework(family: str, cfg: CognosConfig, structure: dict) -> str:
        if family in _TREE_FAMILIES:
            return "ml_challenger"
        if cfg.task.is_classification:
            return "discrete_time_hazard" if family.startswith("hazard_") else "reduced_form_pd"
        if cfg.task.value == "timeseries":
            return "reduced_form_forecast"
        return "reduced_form_glm"

    @staticmethod
    def _rationale(family: str, strategy: str, top_feats: list[str], cfg: CognosConfig,
                   structure: dict) -> str:
        feat_txt = f"using {'the strongest clean predictors' if strategy == 'top' else 'all features'}"
        if family in ("ols", "logit"):
            base = f"Baseline interpretable {family.upper()} {feat_txt}; defensible and easy to validate."
            if family == "logit" and structure["shape"] == "panel":
                base += " On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001)."
            return base
        if family in ("probit", "cloglog"):
            note = (" cloglog is the grouped-time proportional-hazards link." if family == "cloglog"
                    else "")
            return (f"Binary GLM with the {family} link {feat_txt}; statsmodels inference with "
                    f"valid p-values.{note}")
        if family.startswith("hazard_"):
            link = family.split("_", 1)[1]
            return (f"Discrete-time hazard ({link} link) on the obligor-period panel {feat_txt} — "
                    "PD term structure over the outcome window (Shumway 2001).")
        if family in ("ridge", "lasso", "elasticnet", "ridge_logit", "lasso_logit"):
            return f"Regularized linear ({family}) {feat_txt} to control variance/collinearity."
        if family in ("poisson", "gamma", "tweedie"):
            return f"GLM ({family}) {feat_txt} when the outcome's error structure is non-Gaussian."
        if cfg.design.interpretability == "required":
            return (f"Flexible {family} {feat_txt} as a challenger benchmark only — quantifies the "
                    "predictive ceiling; the deployed model must stay interpretable.")
        return f"Flexible {family} {feat_txt} to capture nonlinearity; weigh against interpretability."

    @staticmethod
    def _design_brief_md(cfg: CognosConfig, structure: dict, frameworks: list[dict],
                         questions: list[dict], hypotheses: list[dict], notes: str,
                         choices: list[dict] | None = None) -> str:
        d = cfg.design
        answered = {
            "Use case": d.use_case, "Horizon": d.horizon,
            "Default definition": d.default_definition, "Segment": d.segment,
            "Interpretability": d.interpretability, "Sponsor notes": d.notes,
        }
        lines = [f"# Design brief — {cfg.name}", "",
                 "## Sponsor design brief (MD triangulation)", ""]
        for k, v in answered.items():
            lines.append(f"- **{k}:** {v.strip() or '_unanswered — see open questions_'}")
        lines += ["", "## Data structure", ""]
        for k, v in structure.items():
            lines.append(f"- {k}: {v}")
        lines += ["", "## Framework assessment (alternatives considered)", "",
                  "| Framework | Applicable | Role | Reason | Reference |",
                  "|---|---|---|---|---|"]
        for f in frameworks:
            lines.append(f"| {f['label']} | {f['applicable']} | {f['role']} | {f['reason']} "
                         f"| {f['reference']} |")
        if choices:
            lines += ["", "## Design Lead decisions", "", "| Framework | Decision | Reason |",
                      "|---|---|---|"]
            for c in choices:
                lines.append(f"| {c['framework']} | {c['decision']} | {c['reason']} |")
        lines += ["", "## Open questions for the sponsor", ""]
        if questions:
            for q in questions:
                lines.append(f"- **[{q['id']}]** {q['question']}")
        else:
            lines.append("- none — the design brief is fully specified.")
        lines += ["", "## Ranked hypothesis slate (top 10)", "",
                  "| # | Family | Features | Framework | Role | Priority | Rationale |",
                  "|---|---|---|---|---|---|---|"]
        for h in hypotheses[:10]:
            lines.append(f"| {h['id']} | {h['family']} | {h['feature_strategy']} | {h['framework']} "
                         f"| {h['role']} | {h['priority']} | {h['rationale']} |")
        lines += ["", f"_Notes: {notes}_", ""]
        return "\n".join(lines)
