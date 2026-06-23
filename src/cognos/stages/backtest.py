"""Stage 4 — Standardized testing / backtesting (IMPACT integration).

Two jobs: (1) embed the champion model as a derived field in an IMPACT EntityConfig and run
``EntityPipeline`` to produce a standardized scored feature table (with a transparent built-in
fallback when IMPACT is unavailable); (2) compute backtest-overfitting analytics on top — out-of-
sample performance, walk-forward stability, Probability of Backtest Overfitting (PBO via CSCV), and
the Deflated Sharpe Ratio when a returns column is present.
"""

from __future__ import annotations

import numpy as np

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..integrations.impact_adapter import score_with_impact
from ..modeling.backtest_stats import deflated_sharpe_ratio, pbo_cscv
from ..modeling.metrics import score
from .base import Stage, register_stage


@register_stage
class BacktestStage(Stage):
    name = "backtest"
    requires = ("model",)
    description = "Score via IMPACT and compute backtest-overfitting analytics (PBO, DSR)."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        model = ctx.require("model").payload
        features = model["raw_features"]
        metric = model["metric"]
        is_clf = model["is_classification"]
        scorer_path = model["scorer_path"]
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        # --- IMPACT scoring of the sealed holdout (or full data) --------------------
        eval_df = ctx.load_df("data/holdout.parquet") if (ctx.run_dir / "data/holdout.parquet").exists() \
            else ctx.load_dataset()
        impact_res = score_with_impact(
            eval_df, model_path=scorer_path, raw_features=features,
            work_dir=ctx.stage_dir("backtest"), entity_name=cfg.impact.entity_name,
            primary_key=cfg.impact.primary_key, prefer_impact=cfg.impact.enabled,
        )
        ctx.save_df("stages/backtest/scored.parquet", impact_res.scored_df)
        if impact_res.config_path:
            res.add_artifact(ctx.save_text(
                "stages/backtest/impact_entity.yaml",
                open(impact_res.config_path).read(), kind="text"))
        if impact_res.validation.get("error_count", 0):
            res.add_finding(Finding(id="impact-validation", severity=Severity.MEDIUM,
                                    category="data-quality",
                                    message=f"IMPACT reported {impact_res.validation['error_count']} validation error(s)."))

        # --- out-of-sample metric on the scored holdout ----------------------------
        oos_metric = None
        if cfg.data.target in eval_df.columns:
            y_oos = coerce_target(eval_df, cfg)
            s = impact_res.scored_df["score"].to_numpy()
            if is_clf:
                oos_metric = score(metric, y_oos, (s >= 0.5).astype(int), y_proba=s)
            else:
                oos_metric = score(metric, y_oos, s)

        # --- walk-forward stability + PBO over the full search library -------------
        walk_forward, pbo, strategy_note = self._overfitting_analytics(ctx)

        # --- Deflated Sharpe (only when a returns column is configured) -------------
        dsr = None
        rc = cfg.backtest.returns_column
        if cfg.backtest.deflate_sharpe and rc and rc in eval_df.columns:
            s = impact_res.scored_df["score"].to_numpy()
            thr = float(np.median(s))
            position = np.sign(s - thr) if is_clf else (s - np.mean(s))
            strat_returns = position * eval_df[rc].to_numpy()
            dsr = deflated_sharpe_ratio(strat_returns, n_trials=model["n_candidates_tried"])

        # --- findings + verdict -----------------------------------------------------
        if pbo and not np.isnan(pbo.get("pbo", float("nan"))) and pbo["pbo"] > 0.5:
            res.add_finding(Finding(id="pbo-high", severity=Severity.HIGH, category="overfitting",
                                    message=f"PBO={pbo['pbo']:.0%} (>50%): the selected model is likely backtest-overfit.",
                                    confidence=0.75))
        if dsr and not np.isnan(dsr.get("deflated_sharpe", float("nan"))) and dsr["deflated_sharpe"] < 0.5:
            res.add_finding(Finding(id="dsr-low", severity=Severity.MEDIUM, category="overfitting",
                                    message=f"Deflated Sharpe prob={dsr['deflated_sharpe']:.2f} (<0.5): edge not significant after multiple-testing correction."))

        payload = {
            "used_impact": impact_res.used_impact,
            "impact_note": impact_res.note,
            "impact_config_path": ctx.rel(ctx.resolve("stages/backtest/impact_entity.yaml"))
            if impact_res.config_path else None,
            "scored_rows": int(len(impact_res.scored_df)),
            "scheme": cfg.backtest.scheme,
            "oos_metric": oos_metric,
            "oos_metric_name": metric,
            "walk_forward": walk_forward,
            "pbo": pbo,
            "deflated_sharpe": dsr,
            "n_trials": model["n_candidates_tried"],
            "strategy_note": strategy_note,
        }
        res.add_artifact(ctx.save_json("stages/backtest/backtest.json", payload))
        res.payload = payload
        res.metrics = {"oos_metric": oos_metric, "pbo": (pbo or {}).get("pbo"),
                       "used_impact": impact_res.used_impact,
                       "wf_mean": (walk_forward or {}).get("mean")}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS
        impact_word = "IMPACT" if impact_res.used_impact else "built-in scorer"
        res.summary = (
            f"Scored {len(impact_res.scored_df)} rows via {impact_word}"
            + (f"; OOS {metric}={oos_metric:.4f}" if oos_metric is not None else "")
            + (f"; PBO={pbo['pbo']:.0%}" if pbo and not np.isnan(pbo.get('pbo', float('nan'))) else "")
            + "."
        )
        return res

    def _overfitting_analytics(self, ctx):
        """PBO over the model stage's full search library + champion walk-forward stability.

        Reuses the per-candidate OOF performance matrix persisted by the model stage (Bailey/López
        de Prado CSCV is meaningful only over the real set of configurations tried), and the
        champion's CV fold scores for walk-forward stability — no extra model fitting here.
        """
        cfg = ctx.config
        model = ctx.require("model").payload
        wf = model.get("cv_fold_scores") or []
        walk_forward = {
            "scheme": "time_series_split" if cfg.task.value == "timeseries" else "kfold",
            "fold_scores": [float(s) for s in wf],
            "mean": float(np.mean(wf)) if wf else None,
            "std": float(np.std(wf)) if wf else None,
        }
        pbo = {"pbo": float("nan"), "n_strategies": 0, "note": "no search library"}
        note = "PBO unavailable."
        try:
            path = ctx.resolve(model.get("oof_perf_path", "") or "stages/model/oof_perf.npz")
            if path.exists():
                data = np.load(path, allow_pickle=True)
                mat = np.asarray(data["perf"], dtype=float)
                if mat.ndim == 2 and mat.shape[1] >= 2:
                    valid = ~np.isnan(mat).any(axis=1)
                    if int(valid.sum()) >= 8:
                        pbo = pbo_cscv(mat[valid], n_splits=min(10, max(4, cfg.backtest.n_splits * 2)))
                        note = f"PBO over {mat.shape[1]} search configurations (CSCV)."
        except Exception as exc:  # analytics must never break the stage
            pbo = {"pbo": float("nan"), "n_strategies": 0, "note": f"{type(exc).__name__}: {exc}"}
        return walk_forward, pbo, note
