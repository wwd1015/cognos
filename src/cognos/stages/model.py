"""Stage 3 — Modeling & statistical testing.

The heart of COGNOS. Seals a holdout (frozen substrate), runs the budget-aware ratchet search over
the CASH space with leakage-safe nested CV, refits the champion on the full training set (adding
statsmodels inference for linear families), runs the statistical diagnostic battery, evaluates the
champion on the sealed holdout, and (optionally) reports an ensemble as a labelled challenger
benchmark — the deployed model is always the single interpretable champion. Persists a deployable
scorer the backtest/IMPACT stage embeds as a derived field.
"""

from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..modeling import fit_full, greedy_ensemble, holdout_split, ratchet_search, score
from ..modeling.metrics import metric_direction
from ..runtime.score import save_scorer
from . import stat_tests
from .base import Stage, register_stage

OVERFIT_GAP_FRAC = 0.15  # relative degradation cv->holdout that triggers an overfitting finding


@register_stage
class ModelStage(Stage):
    name = "model"
    requires = ("explore",)
    description = "Search, fit, statistically test, and select the champion model."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        df = ctx.load_dataset()
        profile = ctx.require("explore").payload
        features = profile["features"]
        metric = cfg.metric.name
        is_clf = cfg.task.is_classification
        is_ts = cfg.task.value == "timeseries"

        # --- frozen substrate: seal the holdout BEFORE any search --------------------
        train_df, holdout_df = holdout_split(
            df, holdout_fraction=cfg.search.holdout_fraction,
            datetime_col=cfg.data.datetime_col, random_state=cfg.search.random_state,
        )
        ctx.save_df("data/train.parquet", train_df)
        if len(holdout_df):
            ctx.save_df("data/holdout.parquet", holdout_df)

        X_train = train_df[features]
        y_train = coerce_target(train_df, cfg)

        # --- ratchet search ---------------------------------------------------------
        families = ctx.get("ideate").payload.get("families") if ctx.has("ideate") else None
        families = families or (cfg.search.model_families or None)
        sr = ratchet_search(
            X_train, y_train, task=cfg.task.value, metric=metric, is_classification=is_clf,
            is_timeseries=is_ts, families=families, max_candidates=cfg.search.max_candidates,
            folds=cfg.search.cv_folds, random_state=cfg.search.random_state,
            complexity_penalty=cfg.search.complexity_penalty, time_budget_s=cfg.search.time_budget_s,
            max_features=cfg.search.max_features_per_candidate,
        )
        ctx.save_text("stages/model/ledger.tsv", sr.ledger_tsv(), kind="tsv")

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

        # --- LLM-guided refinement (ADR-0001 stage B; opt-in; reasoning proposes) ---
        champion_cand, champion_cv = sr.champion, sr.champion_cv
        champion_transforms: list = []
        guided_info = None
        if cfg.search.guided and ctx.brain.available:
            from ..modeling.guided import guided_search

            gr = guided_search(
                ctx.brain, X_train, y_train, profile=profile, champion=sr.champion,
                champion_cv=sr.champion_cv, metric=metric, direction=metric_direction(metric),
                is_classification=is_clf, is_timeseries=is_ts, folds=cfg.search.cv_folds,
                random_state=cfg.search.random_state, rounds=cfg.search.guided_rounds,
                log_fn=lambda p, r: ctx.log_reasoning("model", "guided-search", p, r),
            )
            guided_info = {"rounds": len(gr.rounds), "accepted": sum(1 for x in gr.rounds if x.accepted),
                           "improved": gr.improved, "transforms": [t.to_dict() for t in gr.transforms]}
            if gr.improved:  # engine verified the LLM-proposed champion beats the incumbent
                champion_cand, champion_cv, champion_transforms = gr.champion, gr.champion_cv, gr.transforms

        # --- refit champion (with any kept transforms) + statsmodels inference ------
        base_features = list(sr.champion.features)
        if champion_transforms:
            from ..modeling.transforms import apply_transforms

            X_fit, _, _ = apply_transforms(X_train[base_features], champion_transforms)
        else:
            X_fit = X_train
        fitted = fit_full(champion_cand, X_fit, y_train, task=cfg.task.value, is_classification=is_clf)
        fitted.transforms = champion_transforms
        fitted.base_features = base_features
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
            "guided": guided_info,
            "transforms": [t.to_dict() for t in champion_transforms],
            "base_features": base_features,
            "scorer_path": scorer_path,
            "raw_features": features,
            "is_classification": is_clf,
            "n_train": int(len(train_df)),
            "n_holdout": int(len(holdout_df)),
        }
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
            f"Champion {champion_cand.label()} | CV {metric}={champion_cv.mean:.4f}"
            f"±{champion_cv.std:.4f}"
            + (f" | holdout={holdout_metric:.4f}" if holdout_metric is not None else "")
            + f" | tried {sr.n_tried} candidates{guided_note} | "
            f"diagnostics {diagnostics['n_passed']}/{diagnostics['n_run']} passed."
        )
        return res

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
