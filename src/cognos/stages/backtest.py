"""Stage 4 — Standardized testing / backtesting (IMPACT integration + outcomes analysis).

Two jobs: (1) embed the champion model as an IMPACT derived field and run ``EntityPipeline`` to
produce a standardized scored feature table (transparent built-in fallback when IMPACT is absent);
(2) compute SR 11-7 **outcomes analysis** on an out-of-time sample by default — discrimination
(Gini/KS), calibration (expected vs observed), and population stability (PSI). Trading-strategy
metrics (Probability of Backtest Overfitting, Deflated Sharpe) are computed only in an explicit
returns/trading mode, since they presume a P&L series. See ADR-0005.
"""

from __future__ import annotations

import numpy as np

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..integrations.impact_adapter import score_with_impact
from ..modeling.backtest_stats import deflated_sharpe_ratio, pbo_cscv
from ..modeling.credit_metrics import credit_outcomes
from ..modeling.metrics import score
from ..runtime.score import score_frame
from .base import Stage, register_stage


@register_stage
class BacktestStage(Stage):
    name = "backtest"
    requires = ("model",)
    description = "Score via IMPACT and run outcomes analysis (Gini/KS, calibration, PSI) on an OOT sample."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        model = ctx.require("model").payload
        features = model["raw_features"]
        metric = model["metric"]
        is_clf = model["is_classification"]
        scorer_path = model["scorer_path"]
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        # The sealed holdout is an out-of-time sample when a datetime column was configured (the model
        # stage splits time-ordered in that case) — exactly what credit backtesting requires.
        has_holdout = (ctx.run_dir / "data/holdout.parquet").exists()
        eval_df = ctx.load_df("data/holdout.parquet") if has_holdout else ctx.load_dataset()
        is_oot = bool(cfg.data.datetime_col) and has_holdout

        # --- IMPACT scoring of the evaluation sample -------------------------------
        impact_res = score_with_impact(
            eval_df, model_path=scorer_path, raw_features=features,
            work_dir=ctx.stage_dir("backtest"), entity_name=cfg.impact.entity_name,
            primary_key=cfg.impact.primary_key, prefer_impact=cfg.impact.enabled,
        )
        ctx.save_df("stages/backtest/scored.parquet", impact_res.scored_df)
        if impact_res.config_path:
            res.add_artifact(ctx.save_text(
                "stages/backtest/impact_entity.yaml", open(impact_res.config_path).read(), kind="text"))
        if impact_res.validation.get("error_count", 0):
            res.add_finding(Finding(id="impact-validation", severity=Severity.MEDIUM, category="data-quality",
                                    message=f"IMPACT reported {impact_res.validation['error_count']} validation error(s)."))

        scores = impact_res.scored_df["score"].to_numpy()
        oos_metric = None
        if cfg.data.target in eval_df.columns:
            y_oos = coerce_target(eval_df, cfg)
            oos_metric = (score(metric, y_oos, (scores >= 0.5).astype(int), y_proba=scores)
                          if is_clf else score(metric, y_oos, scores))

        # --- outcomes analysis (default for a PD/scoring model) --------------------
        outcomes = None
        if is_clf and cfg.data.target in eval_df.columns:
            dev_scores = self._dev_scores(ctx, scorer_path)
            outcomes = credit_outcomes(coerce_target(eval_df, cfg), scores, dev_scores=dev_scores)
            self._outcome_findings(res, outcomes)

        # --- walk-forward stability (generic) -------------------------------------
        wf = model.get("cv_fold_scores") or []
        walk_forward = {
            "scheme": "out_of_time" if is_oot else ("time_series_split" if cfg.data.datetime_col else "kfold"),
            "fold_scores": [float(s) for s in wf],
            "mean": float(np.mean(wf)) if wf else None,
            "std": float(np.std(wf)) if wf else None,
        }

        # --- trading mode only: PBO + Deflated Sharpe -----------------------------
        trading = None
        rc = cfg.backtest.returns_column
        if rc and rc in eval_df.columns:
            trading = self._trading_metrics(ctx, cfg, model, eval_df, scores, is_clf)
            pbo = (trading.get("pbo") or {}).get("pbo")
            if pbo is not None and not np.isnan(pbo) and pbo > 0.5:
                res.add_finding(Finding(id="pbo-high", severity=Severity.HIGH, category="overfitting",
                                        message=f"PBO={pbo:.0%} (>50%): strategy likely backtest-overfit.",
                                        confidence=0.75))

        payload = {
            "used_impact": impact_res.used_impact,
            "impact_note": impact_res.note,
            "scored_rows": int(len(impact_res.scored_df)),
            "evaluation_sample": "out_of_time" if is_oot else ("holdout" if has_holdout else "full"),
            "oos_metric": oos_metric,
            "oos_metric_name": metric,
            "outcomes_analysis": outcomes,
            "walk_forward": walk_forward,
            "trading": trading,
            "n_trials": model.get("n_candidates_tried"),
        }
        res.add_artifact(ctx.save_json("stages/backtest/backtest.json", payload))
        res.payload = payload
        res.metrics = {"oos_metric": oos_metric, "used_impact": impact_res.used_impact,
                       "gini": (outcomes or {}).get("gini"), "psi": (outcomes or {}).get("psi")}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS

        impact_word = "IMPACT" if impact_res.used_impact else "built-in scorer"
        bits = [f"Scored {len(impact_res.scored_df)} rows via {impact_word}"]
        if oos_metric is not None:
            bits.append(f"OOS {metric}={oos_metric:.4f}")
        if outcomes:
            bits.append(f"Gini={outcomes['gini']:.3f}, KS={outcomes['ks']:.3f}, "
                        f"PSI={outcomes['psi']:.3f} ({outcomes['psi_label']})")
        res.summary = "; ".join(bits) + "."
        return res

    # --- helpers ----------------------------------------------------------------
    def _dev_scores(self, ctx: RunContext, scorer_path: str):
        """Development-set scores (for PSI dev->OOT). None if the train frame is unavailable."""
        try:
            if (ctx.run_dir / "data/train.parquet").exists():
                return score_frame(ctx.load_df("data/train.parquet"), scorer_path)
        except Exception:
            pass
        return None

    @staticmethod
    def _outcome_findings(res: StageResult, outcomes: dict) -> None:
        if outcomes["psi_label"] == "significant shift":
            res.add_finding(Finding(id="psi-shift", severity=Severity.MEDIUM, category="stability",
                                    message=f"Population shift dev->OOT: PSI={outcomes['psi']:.3f} (>0.25).",
                                    confidence=0.8))
        ece = outcomes["expected_calibration_error"]
        if ece is not None and not np.isnan(ece) and ece > 0.10:
            res.add_finding(Finding(id="calibration", severity=Severity.MEDIUM, category="calibration",
                                    message=f"Poor calibration: mean |observed-predicted|={ece:.3f} (>0.10).",
                                    confidence=0.7))
        g = outcomes["gini"]
        if g is not None and not np.isnan(g) and g < 0.2:
            res.add_finding(Finding(id="weak-discrimination", severity=Severity.LOW, category="discrimination",
                                    message=f"Weak discrimination: Gini={g:.3f} (<0.2).", confidence=0.7))

    def _trading_metrics(self, ctx: RunContext, cfg, model: dict, eval_df, scores, is_clf: bool) -> dict:
        rc = cfg.backtest.returns_column
        thr = float(np.median(scores))
        position = np.sign(scores - thr) if is_clf else (scores - np.mean(scores))
        strat_returns = position * eval_df[rc].to_numpy()
        dsr = (deflated_sharpe_ratio(strat_returns, n_trials=model.get("n_candidates_tried", 1))
               if cfg.backtest.deflate_sharpe else None)
        return {"deflated_sharpe": dsr, "pbo": self._pbo(ctx, cfg, model)}

    def _pbo(self, ctx: RunContext, cfg, model: dict) -> dict:
        try:
            path = ctx.resolve(model.get("oof_perf_path", "") or "stages/model/oof_perf.npz")
            if path.exists():
                data = np.load(path, allow_pickle=True)
                mat = np.asarray(data["perf"], dtype=float)
                if mat.ndim == 2 and mat.shape[1] >= 2:
                    valid = ~np.isnan(mat).any(axis=1)
                    if int(valid.sum()) >= 8:
                        return pbo_cscv(mat[valid], n_splits=min(10, max(4, cfg.backtest.n_splits * 2)))
        except Exception as exc:
            return {"pbo": float("nan"), "note": f"{type(exc).__name__}: {exc}"}
        return {"pbo": float("nan"), "note": "no search library"}
