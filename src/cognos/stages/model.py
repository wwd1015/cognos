"""Stage 3 — Modeling & statistical testing.

The heart of COGNOS. Seals a holdout (frozen substrate), runs the budget-aware ratchet search over
the CASH space with leakage-safe nested CV, refits the champion on the full training set (adding
statsmodels inference for linear families), runs the statistical diagnostic battery, evaluates the
champion on the sealed holdout, and (optionally) reports an ensemble as a labelled challenger
benchmark — the deployed model is always the single interpretable champion. Persists a deployable
scorer the backtest/IMPACT stage embeds as a derived field.

v1: the engine computes an **admissible set** — the evaluated candidates within one CV standard
error of the best (plus the best interpretable one when interpretability is required) — and the
**Modeler** agent chooses the champion from it, with economic sign checks, *before* the sealed
holdout is scored (it never sees holdout evidence). A human override at gate_champion wins; every
evaluation of the sealed holdout is counted (``holdout_evaluations``) so re-selection is visible to
the validator. The search itself is cached by an input fingerprint, so a send-back or an override
re-finalizes without re-searching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..modeling import fit_full, greedy_ensemble, holdout_split, ratchet_search, score
from ..modeling.fit import HAZARD_FAMILIES
from ..modeling.metrics import metric_direction
from ..runtime.score import save_scorer
from . import stat_tests
from .base import Stage, attach_recommendation, register_stage

_TREES = {"random_forest", "gradient_boosting"}
MAX_ADMISSIBLE = 6

OVERFIT_GAP_FRAC = 0.15  # relative degradation cv->holdout that triggers an overfitting finding


@register_stage
class ModelStage(Stage):
    name = "model"
    requires = ("explore",)
    description = "Search, fit, statistically test, and select the champion model."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        df = ctx.load_dataset()
        profile = ctx.profile()  # human-excluded columns are already gone
        features = list(profile["features"])
        metric = cfg.metric.name
        is_clf = cfg.task.is_classification
        is_ts = cfg.task.value == "timeseries"

        # --- structural (Merton) augmentation: deterministic solver, target-hidden ---
        structural_info, structural_spec = None, None
        sc = cfg.structural
        if sc.enabled and all(c and c in df.columns
                              for c in (sc.equity_value_col, sc.equity_vol_col, sc.debt_col)):
            from ..modeling.structural import StructuralSpec, augment_frame

            structural_spec = StructuralSpec(
                equity_value_col=sc.equity_value_col, equity_vol_col=sc.equity_vol_col,
                debt_col=sc.debt_col, risk_free_rate=sc.risk_free_rate,
                horizon_years=sc.horizon_years,
            )
            df, s_info = augment_frame(df, structural_spec)
            if structural_spec.dd_feature not in features:
                features.append(structural_spec.dd_feature)  # hybrid mode: DD feeds the champion
            structural_info = {"spec": structural_spec.to_dict(), **s_info}

        # --- frozen substrate: seal the holdout BEFORE any search --------------------
        train_df, holdout_df = holdout_split(
            df, holdout_fraction=cfg.search.holdout_fraction,
            datetime_col=cfg.data.datetime_col, random_state=cfg.search.random_state,
        )

        # --- rating-migration augmentation: matrix FITTED on the training partition --
        # Unlike the deterministic Merton solver above, the transition matrix is estimated, so it
        # must never see the sealed holdout. The horizon PD it implies replaces the raw rating as
        # the champion's feature (they are exact functions of each other — keeping both would make
        # the K-1 inference design perfectly collinear).
        migration_info, migration_spec, migration_model = None, None, None
        migration_findings: list[Finding] = []
        mc = cfg.migration
        if (mc.enabled and mc.rating_col and mc.rating_col in df.columns
                and mc.next_rating_col and mc.next_rating_col in df.columns):
            from ..modeling.migration import (
                MigrationSpec,
                fit_migration,
                infer_scale,
            )
            from ..modeling.migration import (
                augment_frame as migration_augment,
            )

            scale = list(mc.rating_scale) or infer_scale(
                train_df[mc.rating_col], train_df[mc.next_rating_col],
                default_state=mc.default_state, withdrawn_states=tuple(mc.withdrawn_states))
            migration_spec = MigrationSpec(
                rating_col=mc.rating_col, next_rating_col=mc.next_rating_col, scale=scale,
                default_state=mc.default_state, withdrawn_states=list(mc.withdrawn_states),
                horizon_periods=mc.horizon_periods, smoothing=mc.smoothing,
                monotone_pd=mc.monotone_pd,
            )
            migration_model = fit_migration(train_df, migration_spec,
                                            condition_col=mc.condition_col)
            train_df, m_info = migration_augment(train_df, migration_model)
            if len(holdout_df):
                holdout_df, _ = migration_augment(holdout_df, migration_model)
            features = [f for f in features if f != mc.rating_col]
            if migration_spec.pd_feature not in features:
                features.append(migration_spec.pd_feature)  # hybrid mode: migration PD feeds the champion
            migration_info = {
                "spec": migration_spec.to_dict(),
                "scale_source": "configured" if mc.rating_scale else "inferred",
                **m_info,
                "estimate": migration_model["estimate"],
                "term_structure": migration_model["term_structure"],
                "pd_map": migration_model["pd_map"],
                "condition_col": migration_model["condition_col"],
                "conditional": {k: {kk: vv for kk, vv in v.items() if kk != "matrix"}
                                for k, v in migration_model["conditional"].items()},
            }
            migration_findings = self._check_migration(migration_model)

        ctx.save_df("data/train.parquet", train_df)
        if len(holdout_df):
            ctx.save_df("data/holdout.parquet", holdout_df)

        X_train = train_df[features]
        y_train = coerce_target(train_df, cfg)

        # --- discrete-time hazard metadata (survival framework) ---------------------
        # The event-time column rides along in the search frame as metadata (never a feature);
        # hazard candidates panel-expand inside their estimator, so CV stays obligor-level.
        etc = cfg.data.event_time_col
        hazard_meta = None
        X_search = X_train
        if etc and etc in train_df.columns:
            observed = pd.to_numeric(train_df[etc], errors="coerce")
            horizon = int(cfg.data.horizon_periods or max(1.0, float(np.nanmax(observed.to_numpy()))
                                                          if observed.notna().any() else 1.0))
            hazard_meta = {"event_time_col": etc, "horizon": horizon}
            X_search = train_df[[*features, etc]]

        # --- ratchet search ---------------------------------------------------------
        families = ctx.get("ideate").payload.get("families") if ctx.has("ideate") else None
        if ctx.overrides.slate:  # the human's slate edit at gate_design wins
            families = list(dict.fromkeys(h["family"] for h in ctx.overrides.slate))
        families = families or (cfg.search.model_families or None)
        fingerprint = self._fingerprint(cfg, X_search, y_train, features, families, hazard_meta)
        sr = self._cached_search(ctx, fingerprint, lambda: ratchet_search(
            X_search, y_train, task=cfg.task.value, metric=metric, is_classification=is_clf,
            is_timeseries=is_ts, families=families, max_candidates=cfg.search.max_candidates,
            folds=cfg.search.cv_folds, random_state=cfg.search.random_state,
            complexity_penalty=cfg.search.complexity_penalty, time_budget_s=cfg.search.time_budget_s,
            max_features=cfg.search.max_features_per_candidate,
            feature_columns=features, hazard_meta=hazard_meta,
        ))
        ctx.save_text("stages/model/ledger.tsv", sr.ledger_tsv(), kind="tsv")
        ctx.save_json("stages/model/ledger.json", sr.ledger_records())

        # --- ensemble of survivors (Caruana) ---------------------------------------
        # The deployed model is always the single interpretable champion (ADR-0007). An ensemble, when
        # explicitly enabled, is computed only as a labelled predictive-ceiling *challenger benchmark*
        # ("how much accuracy would a blend buy?") — it is reported, never silently shipped.
        challenger_benchmark = None
        if cfg.search.ensemble and len(sr.evaluated) >= 2:
            top = sorted(sr.evaluated, key=lambda cv: cv[1].mean,
                         reverse=metric_direction(metric) == "maximize")[:8]
            ens = greedy_ensemble([cv.oof_pred for _, cv in top], y_train,
                                  metric=metric, is_classification=is_clf)
            if ens is not None:
                challenger_benchmark = {
                    "kind": "ensemble (Caruana)",
                    "deployed": False,
                    "note": "Exploratory predictive-ceiling benchmark only; the deployed model is the single champion.",
                    "n_members": len(set(ens.member_indices)),
                    "weights": {top[i][0].label(): w for i, w in ens.weights.items()},
                    "benchmark_score": ens.ensemble_score,
                    "champion_single_score": ens.best_single_score,
                    "headroom_vs_champion": ens.improved,
                }

        # --- persist per-candidate OOF performance for an honest PBO ----------------
        # PBO must be computed over the full set of configurations tried (Bailey/López de Prado),
        # not a hand-picked few. We save the per-sample performance matrix of every evaluated
        # candidate so the backtest stage can compute PBO over the real search library.
        oof_perf_path = self._save_oof_perf(ctx, sr, y_train, is_clf)

        # --- agent-guided refinement (ADR-0001 stage B; opt-in; the agent proposes) ---
        guided_info, guided_entry = None, None
        # Guided refinement refits via the generic path; a hazard champion has its own fit contract.
        if (cfg.search.guided and ctx.runner.kind != "heuristic"
                and sr.champion.family not in HAZARD_FAMILIES):
            from ..modeling.guided import guided_search
            from .ideate import _engine_families

            gr = guided_search(
                lambda c: ctx.recommend("experiment", c).model_dump(),
                X_train, y_train, champion=sr.champion, champion_cv=sr.champion_cv,
                metric=metric, direction=metric_direction(metric), is_classification=is_clf,
                allowed_families=[f for f in _engine_families(cfg) if f not in HAZARD_FAMILIES],
                is_timeseries=is_ts, folds=cfg.search.cv_folds,
                random_state=cfg.search.random_state, rounds=cfg.search.guided_rounds,
            )
            guided_info = {"rounds": len(gr.rounds), "accepted": sum(1 for x in gr.rounds if x.accepted),
                           "improved": gr.improved, "transforms": [t.to_dict() for t in gr.transforms],
                           "history": [{"round": r.idx, "accepted": r.accepted, "score": r.score,
                                        "note": r.note} for r in gr.rounds]}
            if gr.improved:  # engine verified the agent-proposed candidate beats the incumbent
                guided_entry = (gr.champion, gr.champion_cv, gr.transforms, list(sr.champion.features))

        # --- admissible set + the Modeler's choice (before the holdout is touched) ------
        admissible, lookup = self._admissible(sr, cfg, metric, X_train, y_train, is_clf,
                                              guided_entry)
        rec = self._modeler_choice(ctx, admissible, sr, metric, fingerprint)
        champion_id, champion_source = rec.champion, "agent"
        if ctx.overrides.champion and ctx.overrides.champion in lookup:
            champion_id, champion_source = ctx.overrides.champion, "human"
        champion_cand, champion_cv, champion_transforms, base_features = lookup[champion_id]
        prev = ctx.get("model")
        holdout_evaluations = int((prev.payload.get("holdout_evaluations") or 0) if prev else 0) + 1

        # --- refit champion (with any kept transforms) + statsmodels inference ------
        hazard_info = None
        if champion_cand.family in HAZARD_FAMILIES:
            from ..modeling.hazard import fit_full_hazard, term_structure

            X_fit = X_train
            fitted = fit_full_hazard(champion_cand, X_train, y_train,
                                     event_time=train_df[etc], horizon=hazard_meta["horizon"],
                                     task=cfg.task.value)
            hazard_info = {
                "horizon_periods": hazard_meta["horizon"],
                "event_time_col": etc,
                "term_structure": term_structure(fitted.pipeline,
                                                 X_train[champion_cand.features]),
            }
        else:
            if champion_transforms:
                from ..modeling.transforms import apply_transforms

                X_fit, _, _ = apply_transforms(X_train[base_features], champion_transforms)
            else:
                X_fit = X_train
            fitted = fit_full(champion_cand, X_fit, y_train, task=cfg.task.value,
                              is_classification=is_clf)
        fitted.transforms = champion_transforms
        fitted.base_features = base_features
        if structural_spec is not None:
            # Serving recomputes DD from the market observables, so the scorer's base features are
            # the raw inputs (market columns in, engineered DD out) — mirrors the transform contract.
            market = [structural_spec.equity_value_col, structural_spec.equity_vol_col,
                      structural_spec.debt_col]
            fitted.base_features = list(dict.fromkeys(
                [f for f in base_features if f != structural_spec.dd_feature] + market))
            fitted.structural = structural_spec.to_dict()
        if migration_model is not None and migration_spec is not None:
            # Serving recomputes migration_pd from the rating column via the fitted matrix, so the
            # scorer's base features carry the raw rating in and the engineered PD out.
            current = list(fitted.base_features or base_features)
            fitted.base_features = list(dict.fromkeys(
                [f for f in current if f != migration_spec.pd_feature] + [migration_spec.rating_col]))
            fitted.migration = {k: migration_model[k] for k in ("spec", "pd_map", "fill")}
        diagnostics = stat_tests.run_battery(fitted, X_fit, y_train,
                                             is_classification=is_clf, is_timeseries=is_ts)

        # --- sealed-holdout evaluation ---------------------------------------------
        holdout_metric = None
        if len(holdout_df):
            Xh = holdout_df[features]
            yh = coerce_target(holdout_df, cfg)
            if is_clf:
                proba = fitted.predict_proba(Xh)
                holdout_metric = score(metric, yh, (proba >= 0.5).astype(int), y_proba=proba)
            else:
                holdout_metric = score(metric, yh, fitted.predict(Xh))
            # Pure-structural challenger benchmark: PD = N(-DD) scored directly on the holdout —
            # a labelled reference point, never the deployed model.
            if structural_info is not None and is_clf and structural_spec is not None:
                pd_struct = holdout_df[structural_spec.pd_feature].to_numpy(dtype=float)
                auc = score("roc_auc", yh, (pd_struct >= 0.5).astype(int), y_proba=pd_struct)
                structural_info["benchmark"] = {
                    "kind": "pure structural Merton PD (challenger benchmark)",
                    "deployed": False,
                    "holdout_roc_auc": auc,
                    "holdout_gini": 2 * auc - 1,
                }
            # Pure-migration challenger benchmark: the rating-implied PD scored directly on the
            # sealed holdout — how far the agency matrix alone gets you, never the deployed model.
            if migration_info is not None and is_clf and migration_spec is not None:
                pd_mig = holdout_df[migration_spec.pd_feature].to_numpy(dtype=float)
                auc = score("roc_auc", yh, (pd_mig >= 0.5).astype(int), y_proba=pd_mig)
                migration_info["benchmark"] = {
                    "kind": "pure rating-migration PD (challenger benchmark)",
                    "deployed": False,
                    "holdout_roc_auc": auc,
                    "holdout_gini": 2 * auc - 1,
                }

        # --- migration expected-loss forecast (reported; never feeds selection) -----
        if migration_model is not None and migration_spec is not None:
            from ..modeling.migration import expected_loss_forecast

            # Forecast on the out-of-time book when one exists (the closest thing to the current
            # portfolio); otherwise the training book.
            book = holdout_df if len(holdout_df) else train_df
            ead = (book[mc.ead_column].to_numpy(dtype=float)
                   if mc.ead_column and mc.ead_column in book.columns else None)
            migration_info["loss_forecast"] = expected_loss_forecast(
                book[migration_spec.rating_col], migration_model, lgd=mc.lgd, ead=ead)
            migration_info["loss_forecast"]["book"] = ("out_of_time_holdout" if len(holdout_df)
                                                       else "training")

        # --- persist deployable scorer ---------------------------------------------
        scorer_path = str((ctx.models_dir / "champion_scorer.joblib").resolve())
        save_scorer(scorer_path, fitted)
        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        res.add_artifact(ArtifactRef(name="champion_scorer", kind="model",
                                     path=ctx.rel(ctx.models_dir / "champion_scorer.joblib"),
                                     description="Picklable scorer embedded by IMPACT / serving."))
        coefficients = fitted.coefficients()
        pvalues = fitted.pvalues()
        importances = fitted.feature_importances()
        res.add_artifact(ctx.save_json("stages/model/coefficients.json",
                                       {"coefficients": coefficients, "pvalues": pvalues,
                                        "feature_importances": importances}))
        res.add_artifact(ctx.save_json("stages/model/diagnostics.json", diagnostics))

        # --- findings ---------------------------------------------------------------
        for f in migration_findings:
            res.add_finding(f)
        for t in diagnostics["failed_tests"]:
            test = next(x for x in diagnostics["tests"] if x["name"] == t)
            res.add_finding(Finding(id=f"diag-{t}", severity=Severity(test["severity"]),
                                    category=f"diagnostic/{test['category']}",
                                    message=test["interpretation"], location=t))
        if holdout_metric is not None:
            denom = abs(champion_cv.mean) or 1.0
            if metric_direction(metric) == "maximize":
                gap = (champion_cv.mean - holdout_metric) / denom
            else:
                gap = (holdout_metric - champion_cv.mean) / denom
            if gap > OVERFIT_GAP_FRAC:
                res.add_finding(Finding(id="overfit-gap", severity=Severity.HIGH, category="overfitting",
                                        message=f"Holdout {metric} degrades {gap:.0%} vs CV — possible overfitting.",
                                        confidence=0.8))

        payload = {
            "champion": champion_cand.to_dict(),
            "champion_id": champion_id,
            "champion_label": champion_cand.label(),
            "champion_source": champion_source,
            "admissible_set": admissible,
            "search_fingerprint": fingerprint,
            "holdout_evaluations": holdout_evaluations,
            "metric": metric,
            "direction": metric_direction(metric),
            "cv_mean": champion_cv.mean,
            "cv_std": champion_cv.std,
            "holdout_metric": holdout_metric,
            "n_candidates_tried": sr.n_tried,
            "n_configs_for_deflation": sr.n_tried,
            "cv_fold_scores": [float(s) for s in champion_cv.fold_scores],
            "oof_perf_path": oof_perf_path,
            "n_search_strategies": len(sr.evaluated),
            "coefficients": coefficients,
            "pvalues": pvalues,
            "feature_importances": importances,
            "diagnostics": diagnostics,
            "challenger_benchmark": challenger_benchmark,
            "hazard": hazard_info,
            "structural": structural_info,
            "migration": migration_info,
            "guided": guided_info,
            "transforms": [t.to_dict() for t in champion_transforms],
            "base_features": base_features,
            "scorer_path": scorer_path,
            "raw_features": features,
            "is_classification": is_clf,
            "n_train": int(len(train_df)),
            "n_holdout": int(len(holdout_df)),
        }
        attach_recommendation(payload, ctx, "modeler", rec)
        if migration_info is not None:
            res.add_artifact(ctx.save_json("stages/model/migration.json", migration_info))
        res.add_artifact(ctx.save_json("stages/model/summary.json", payload))
        res.payload = payload
        res.metrics = {"cv_mean": champion_cv.mean, "cv_std": champion_cv.std,
                       "holdout_metric": holdout_metric, "n_candidates_tried": sr.n_tried,
                       "champion": champion_cand.family, "n_transforms": len(champion_transforms)}
        worst = diagnostics["max_failed_severity"]
        res.add_artifact(ArtifactRef(name="oof_perf", kind="json", path=oof_perf_path,
                                     description="Per-candidate OOF performance matrix for PBO."))
        res.verdict = Verdict.WARN if (res.findings and (worst in ("MEDIUM", "HIGH"))) else Verdict.PASS
        guided_note = (f" | guided +{len(champion_transforms)} transform(s)"
                       if champion_transforms else "")
        res.summary = (
            f"Champion {champion_id} {champion_cand.label()} ({champion_source}) | "
            f"CV {metric}={champion_cv.mean:.4f}"
            f"±{champion_cv.std:.4f}"
            + (f" | holdout={holdout_metric:.4f}" if holdout_metric is not None else "")
            + f" | tried {sr.n_tried} candidates{guided_note} | "
            f"diagnostics {diagnostics['n_passed']}/{diagnostics['n_run']} passed."
        )
        return res

    # --- v1 helpers: search cache, admissible set, the Modeler's choice -----------------
    @staticmethod
    def _fingerprint(cfg, X_search, y_train, features, families, hazard_meta) -> str:
        import hashlib
        import json

        data_hash = int(pd.util.hash_pandas_object(X_search, index=False).sum()) & 0xFFFFFFFF
        key = {"features": features, "families": families, "hazard": hazard_meta,
               "search": cfg.search.model_dump(exclude={"guided", "guided_rounds"}),
               "task": cfg.task.value, "metric": cfg.metric.name, "rows": len(X_search),
               "data": data_hash, "y": float(np.nansum(np.asarray(y_train, dtype=float)))}
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @staticmethod
    def _cached_search(ctx: RunContext, fingerprint: str, search):
        import joblib

        path = ctx.resolve("stages/model/search_cache.joblib")
        if path.exists():
            try:
                cached = joblib.load(path)
                if cached.get("fingerprint") == fingerprint:
                    return cached["sr"]
            except Exception:  # a stale/corrupt cache just means searching again
                pass
        sr = search()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"fingerprint": fingerprint, "sr": sr}, path)
        return sr

    @staticmethod
    def _signs(coefs: dict | None, features: list[str]) -> list[dict]:
        if not coefs:
            return []
        out = []
        for f in features:
            key = next((k for k in coefs if k == f or k.endswith(f"__{f}")), None)
            if key is None:
                continue
            v = coefs[key]
            out.append({"feature": f, "sign": "0" if abs(v) < 1e-12 else ("+" if v > 0 else "-")})
        return out

    def _admissible(self, sr, cfg, metric: str, X_train, y_train, is_clf: bool,
                    guided_entry) -> tuple[list[dict], dict]:
        """Candidates statistically indistinguishable from the best (one-standard-error rule)."""
        maximize = metric_direction(metric) == "maximize"
        ok_idx = [r.idx for r in sr.ledger if r.status != "crash"]
        entries = []  # (id, cand, cv, transforms, base_features)
        for idx, (cand, cv) in zip(ok_idx, sr.evaluated, strict=False):
            entries.append((f"c{idx}", cand, cv, [], list(cand.features)))
        best_mean, best_std = sr.champion_cv.mean, sr.champion_cv.std
        champ_id = next(e[0] for e in entries if e[1] is sr.champion)

        def within(cv) -> bool:
            return cv.mean >= best_mean - best_std if maximize else cv.mean <= best_mean + best_std

        chosen = [e for e in entries if within(e[2])]
        chosen.sort(key=lambda e: (e[0] != champ_id, -e[2].mean if maximize else e[2].mean))
        required = cfg.design.interpretability == "required"
        if required and all(e[1].family in _TREES for e in chosen):
            interp = [e for e in entries if e[1].family not in _TREES]
            if interp:
                chosen.append(sorted(interp, key=lambda e: -e[2].mean if maximize else e[2].mean)[0])
        chosen = chosen[:MAX_ADMISSIBLE]
        if guided_entry is not None:
            g_cand, g_cv, g_tr, g_base = guided_entry
            chosen.insert(0, ("g1", g_cand, g_cv, g_tr, g_base))

        admissible, lookup = [], {}
        for cid, cand, cv, transforms, base in chosen:
            lookup[cid] = (cand, cv, transforms, base)
            signs: list[dict] = []
            if cand.family not in _TREES and cand.family not in HAZARD_FAMILIES:
                try:
                    X_fit = X_train
                    if transforms:
                        from ..modeling.transforms import apply_transforms

                        X_fit, _, _ = apply_transforms(X_train[base], transforms)
                    fitted = fit_full(cand, X_fit, y_train, task=cfg.task.value,
                                      is_classification=is_clf)
                    signs = self._signs(fitted.coefficients(), list(cand.features))
                except Exception:  # signs are advisory; a failed refit just omits them
                    signs = []
            admissible.append({
                "id": cid, "label": cand.label(), "family": cand.family,
                "n_features": len(cand.features), "features": list(cand.features),
                "cv_mean": round(float(cv.mean), 6), "cv_std": round(float(cv.std), 6),
                "role": "challenger" if (cand.family in _TREES and required) else "candidate",
                "interpretable": cand.family not in _TREES,
                "transforms": [t.to_dict() for t in transforms],
                "coefficient_signs": signs,
                "ratchet_champion": cid == champ_id,
            })
        # The ratchet champion leads unless a guided candidate beat it.
        return admissible, lookup

    @staticmethod
    def _modeler_choice(ctx: RunContext, admissible: list[dict], sr, metric: str,
                        fingerprint: str):
        """Ask the Modeler — or reuse its last answer when only a human override changed."""
        from ..agents.contracts import ModelerOutput

        prev = ctx.get("model")
        prev_rec = (prev.payload.get("recommendation") or {}) if prev else {}
        if (ctx.overrides.champion and prev is not None
                and prev.payload.get("search_fingerprint") == fingerprint
                and prev_rec.get("output") and not ctx.challenges_for("model")):
            ctx.runner.last["modeler"] = {**{k: v for k, v in prev_rec.items() if k != "output"}, "reused": True}
            return ModelerOutput.model_validate(prev_rec["output"])
        return ctx.recommend("modeler", {
            "metric": metric,
            "metric_direction": metric_direction(metric),
            "interpretability": ctx.config.design.interpretability,
            "admissible_set": admissible,
            "n_candidates_tried": sr.n_tried,
            "ledger_top": [{"label": r.label, "family": r.family, "n_features": r.n_features,
                            "cv_mean": round(r.metric_value, 6), "status": r.status}
                           for r in sorted(sr.ledger, key=lambda r: r.metric_value,
                                           reverse=metric_direction(metric) == "maximize")[:10]],
        })

    @staticmethod
    def _check_migration(model: dict) -> list[Finding]:
        """Rank-order and support diagnostics on the fitted transition matrix (never a crash)."""
        est = model["estimate"]
        diag = est["diagnostics"]
        findings: list[Finding] = []
        if not diag["default_col_monotone"]:
            findings.append(Finding(
                id="migration-rank-order", severity=Severity.MEDIUM, category="migration",
                message="Estimated one-period PDs are not monotone across the rating scale — the "
                        "matrix does not rank-order (thin rows, a wrong scale order, or a mixed "
                        "sample).",
                suggestion="Check migration.rating_scale ordering and per-row support; consider "
                           "pooling adjacent grades, a longer history, or migration.monotone_pd.",
            ))
        elif diag.get("monotonized_rows"):
            findings.append(Finding(
                id="migration-monotonized", severity=Severity.LOW, category="migration",
                message=f"Raw default-column estimates violated rank order at grade(s) "
                        f"{', '.join(diag['monotonized_rows'])} (small-sample artifact); PDs were "
                        "monotonized via weighted PAVA — raw and adjusted values are in the run "
                        "record.",
                suggestion="Expected with sparse high-grade defaults; document the adjustment in "
                           "the matrix-estimation section.",
            ))
        if diag["thin_rows"]:
            findings.append(Finding(
                id="migration-thin-rows", severity=Severity.LOW, category="migration",
                message=f"Rating grade(s) {', '.join(diag['thin_rows'])} have fewer than 30 "
                        "observed transitions — their row of the matrix leans on smoothing.",
                suggestion="Pool thin grades or extend the history window.",
            ))
        if est["withdrawn_share"] > 0.25:
            findings.append(Finding(
                id="migration-withdrawn", severity=Severity.LOW, category="migration",
                message=f"{est['withdrawn_share']:.0%} of transitions end in a withdrawn rating "
                        "(NR-adjusted out of the denominator) — a high share can bias the matrix "
                        "if withdrawal correlates with credit state.",
                suggestion="Document the NR treatment; consider a withdrawal-adjusted robustness "
                           "check.",
            ))
        return findings

    @staticmethod
    def _save_oof_perf(ctx: RunContext, sr, y_train, is_clf: bool) -> str:
        """Save an (n_train x n_candidates) per-sample performance matrix (higher = better)."""
        y = np.asarray(y_train, dtype=float)
        cols, labels = [], []
        for cand, cv in sr.evaluated:
            pred = np.asarray(cv.oof_pred, dtype=float)
            if is_clf:
                p = np.clip(pred, 1e-6, 1 - 1e-6)
                perf = np.where(np.isnan(pred), np.nan, y * np.log(p) + (1 - y) * np.log(1 - p))
            else:
                perf = -((y - pred) ** 2)
            cols.append(perf)
            labels.append(cand.label())
        mat = np.column_stack(cols) if cols else np.zeros((len(y), 0))
        relpath = "stages/model/oof_perf.npz"
        np.savez(ctx.resolve(relpath), perf=mat, y=y, labels=np.array(labels, dtype=object))
        return relpath
