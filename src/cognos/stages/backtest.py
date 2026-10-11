"""Stage 4 — Outcomes analysis of the candidate pool, and the champion it points to (ADR-0016).

Models that rank together on the development sample can forecast very differently. This stage
is where that shows: every pool member is scored on the comparison sample (the test sample,
out of time when the plan says so) and measured the same way — forecast error and bias, or
discrimination, calibration and population stability — and, when scenarios are defined, under
each scenario. The **Outcomes Analyst** reads the comparison and recommends the champion; the
developer accepts or chooses another member at the champion gate.

For the champion in force the stage then does what it always did: embeds the model as an IMPACT
derived field and runs ``EntityPipeline`` (transparent built-in fallback when IMPACT is absent),
and reports the SR 11-7 outcomes analysis, the portfolio simulation and stress when enabled.
Trading-strategy metrics (PBO, Deflated Sharpe) are computed only in an explicit returns mode,
since they presume a P&L series. See ADR-0005.

Every member scored on the test sample is counted (``test_evaluations``): that sample is used
to choose, so only a sealed holdout, scored at validation, gives a number nobody chose on.
"""

from __future__ import annotations

import numpy as np

from .. import pool as pool_mod
from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..integrations.impact_adapter import score_with_impact
from ..modeling.backtest_stats import deflated_sharpe_ratio, pbo_cscv
from ..modeling.credit_metrics import credit_outcomes
from ..modeling.metrics import metric_direction, score
from ..runtime.score import score_frame
from .base import Stage, attach_recommendation, register_stage

_OUTCOME_KEYS = ("gini", "ks", "auc", "expected_calibration_error", "psi", "psi_label")
MAX_SERIES_POINTS = 2000


@register_stage
class BacktestStage(Stage):
    name = "backtest"
    requires = ("model",)
    description = "Compare the candidate pool on the test sample; recommend and analyse the champion."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        stage_model = ctx.require("model").payload
        members = list(stage_model.get("pool") or [])
        if not members:
            raise RuntimeError("The model stage left no candidate pool; re-run it.")
        metric = stage_model["metric"]
        is_clf = stage_model["is_classification"]
        plan_ev = (stage_model.get("plan") or {}).get("evaluation") or {}
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        # The comparison sample: the test sample when the plan holds one back. Out of time when
        # the plan cut it by date — exactly what credit backtesting requires.
        has_test = (ctx.run_dir / "data/test.parquet").exists()
        eval_df = ctx.load_df("data/test.parquet") if has_test else ctx.load_df("data/train.parquet")
        dev_df = ctx.load_df("data/train.parquet")
        is_oot = has_test and plan_ev.get("test") == "out_of_time"
        sample = "out_of_time" if is_oot else ("test" if has_test else "development")
        has_target = cfg.data.target in eval_df.columns
        y_eval = coerce_target(eval_df, cfg) if has_target else None
        if not has_test:
            res.add_finding(Finding(
                id="no-test-sample", severity=Severity.MEDIUM, category="sanity",
                message="The plan holds back no test sample: the pool is compared on the rows "
                        "the models were fitted on, which favours the model with more factors.",
                suggestion="Hold back a test sample in the plan at the design gate."))

        # --- every pool member, measured the same way ------------------------------
        metrics = list(dict.fromkeys([metric, *(r["metric"] for r in plan_ev.get("rank_by") or [])]))
        factors = stage_model.get("factors") or []
        scenarios = (cfg.stress.scenarios or []) if cfg.stress.enabled else []
        ead = (eval_df[cfg.portfolio.ead_column].to_numpy(dtype=float)
               if cfg.portfolio.ead_column and cfg.portfolio.ead_column in eval_df.columns else None)
        candidates, series = [], {}
        for m in members:
            entry = {k: m.get(k) for k in ("id", "name", "label", "family", "n_features", "features",
                                           "role", "interpretable", "methods", "violations")}
            entry["development"] = (m.get("scores") or {})
            entry["development_score"] = m.get("score")
            try:
                scores = np.asarray(score_frame(eval_df, m["scorer_path"]), dtype=float)
            except Exception as exc:  # a member that cannot be scored is reported, not fatal
                entry.update(error=f"{type(exc).__name__}: {exc}"[:200], metrics={}, flags=[])
                candidates.append(entry)
                continue
            series[m["id"]] = scores
            entry["metrics"] = self._metrics(metrics, y_eval, scores, is_clf)
            entry["flags"] = []
            if has_target and is_clf:
                oa = credit_outcomes(y_eval, scores, dev_scores=self._scores(dev_df, m["scorer_path"]))
                entry["outcomes"] = {k: oa.get(k) for k in _OUTCOME_KEYS}
                entry["outcomes"]["observed_rate"] = float(np.mean(y_eval))
                entry["outcomes"]["mean_predicted"] = float(np.mean(scores))
                entry["flags"] = [f["message"] for f in self._outcome_flags(oa)]
                entry["_full"] = oa
            elif has_target:
                entry["outcomes"] = self._forecast_outcomes(y_eval, scores)
            if scenarios:
                try:
                    from ..modeling.simulate import stress_scenarios

                    st = stress_scenarios(eval_df, m["scorer_path"], scenarios,
                                          lgd=cfg.portfolio.lgd, ead=ead, factors=factors,
                                          datetime_col=cfg.data.datetime_col,
                                          history=dev_df if has_test else None)
                    entry["scenarios"] = [{k: sc.get(k) for k in ("name", "mean", "delta")}
                                          for sc in st["scenarios"]]
                    entry["scenario_baseline"] = st["baseline"]["mean"]
                    entry["_stress"] = st
                except Exception as exc:
                    entry["scenarios_error"] = f"{type(exc).__name__}: {exc}"[:200]
            candidates.append(entry)

        # Best first on the run's metric on the comparison sample; ties to the simpler model.
        maximize = metric_direction(metric) == "maximize"

        def order(c: dict):
            v = (c.get("metrics") or {}).get(metric)
            if v is None or not np.isfinite(v):
                return (1, 0.0, c.get("n_features") or 0)
            return (0, -v if maximize else v, c.get("n_features") or 0)
        candidates.sort(key=order)
        if all(c.get("error") for c in candidates):
            raise RuntimeError("No pool member could be scored: "
                               + "; ".join(str(c.get("error")) for c in candidates))
        prev = ctx.get("backtest")
        test_evaluations = (int((prev.payload or {}).get("test_evaluations") or 0) if prev else 0) \
            + (len(series) if has_test else 0)
        forecasts = self._save_series(ctx, cfg, eval_df, y_eval, series, sample)
        forecasts_path = forecasts.path if forecasts is not None else None
        if forecasts is not None:
            res.add_artifact(forecasts)

        # --- the Outcomes Analyst's recommendation (engine-checked; cites facts) -----
        public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates]
        comparison = {"metric": metric, "direction": metric_direction(metric), "sample": sample,
                      "n_rows": int(len(eval_df)), "n_candidates": len(candidates)}
        sealed = bool(stage_model.get("n_holdout"))
        res.payload = {"evaluation_sample": sample, "candidates": public, "comparison": comparison}
        lead = next((m["scorer_path"] for c in candidates if not c.get("error")
                     for m in members if m["id"] == c["id"]), None)
        out = self._analyst(ctx, res, public, comparison, sealed, stage_model, lead)
        scoreable = {c["id"] for c in candidates if not c.get("error")}
        champion_id = out.champion if out.champion in scoreable else next(
            c["id"] for c in candidates if not c.get("error"))
        champion_source = "engine" if ctx.runner.kind == "heuristic" else "agent"
        if ctx.overrides.champion in scoreable:
            champion_id, champion_source = ctx.overrides.champion, "human"
        chosen = next(c for c in candidates if c["id"] == champion_id)

        # --- the champion in force: the model every later stage reads ----------------
        model = pool_mod.view(ctx.run_dir, stage_model, {"champion_id": champion_id,
                                                         "champion_source": champion_source})
        scorer_path = model["scorer_path"]
        features = [c for c in dict.fromkeys([*stage_model["raw_features"],
                                              *model["champion"]["features"]])
                    if c in eval_df.columns]

        # --- IMPACT scoring of the comparison sample --------------------------------
        impact_res = score_with_impact(
            eval_df, model_path=scorer_path, raw_features=features,
            work_dir=ctx.stage_dir("backtest"), entity_name=cfg.impact.entity_name,
            primary_key=cfg.impact.primary_key, prefer_impact=cfg.impact.enabled,
        )
        ctx.save_df("stages/backtest/scored.parquet", impact_res.scored_df)
        if impact_res.config_path:
            res.add_artifact(ctx.save_text(
                "stages/backtest/impact_entity.yaml", open(impact_res.config_path, encoding="utf-8").read(), kind="text"))
        if impact_res.validation.get("error_count", 0):
            res.add_finding(Finding(id="impact-validation", severity=Severity.MEDIUM, category="data-quality",
                                    message=f"IMPACT reported {impact_res.validation['error_count']} validation error(s)."))

        scores = impact_res.scored_df["score"].to_numpy()
        oos_metric = None
        if has_target:
            oos_metric = (score(metric, y_eval, (scores >= 0.5).astype(int), y_proba=scores)
                          if is_clf else score(metric, y_eval, scores))

        # --- outcomes analysis (default for a PD/scoring model) --------------------
        outcomes = chosen.get("_full")
        if outcomes is not None:
            for f in self._outcome_flags(outcomes):
                res.add_finding(Finding(id=f["id"], severity=f["severity"], category=f["category"],
                                        message=f["message"], confidence=f["confidence"]))
        forecast = chosen.get("outcomes") if not is_clf else None

        # --- walk-forward stability (generic) -------------------------------------
        wf = model.get("cv_fold_scores") or []
        walk_forward = {
            "scheme": plan_ev.get("validation") or "none",
            "fold_scores": [float(s) for s in wf],
            "mean": float(np.mean(wf)) if wf else None,
            "std": float(np.std(wf)) if wf else None,
        }

        # --- portfolio simulation (opt-in): Vasicek loss distribution + IRB capital -
        portfolio = None
        if is_clf and cfg.portfolio.enabled:
            from ..modeling.simulate import vasicek_loss_simulation

            pc = cfg.portfolio
            ead = (eval_df[pc.ead_column].to_numpy(dtype=float)
                   if pc.ead_column and pc.ead_column in eval_df.columns else None)
            portfolio = vasicek_loss_simulation(
                scores, lgd=pc.lgd, ead=ead, rho=pc.asset_correlation,
                confidence=pc.confidence, n_sims=pc.n_sims, seed=pc.seed,
            )
            res.add_artifact(ctx.save_json("stages/backtest/portfolio.json", portfolio))

        # --- macro-scenario stress (opt-in): shock covariates, re-score -------------
        stress = chosen.get("_stress")
        if stress is not None:
            res.add_artifact(ctx.save_json("stages/backtest/stress.json", stress))
            for sc in stress["scenarios"]:
                if sc.get("missing_columns"):
                    res.add_finding(Finding(
                        id=f"stress-missing-{sc['name']}", severity=Severity.LOW, category="stress",
                        message=f"Scenario '{sc['name']}' shocks unknown column(s): "
                                f"{', '.join(sc['missing_columns'])}.",
                    ))

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
            "evaluation_sample": sample,
            "candidates": public,
            "comparison": comparison,
            "champion_id": champion_id,
            "champion_label": chosen["label"],
            "champion_name": chosen.get("name") or chosen["label"],
            "champion_source": champion_source,
            "champion_rationale": out.champion_rationale,
            "recommended_id": out.champion,
            "test_evaluations": test_evaluations,
            "sealed_holdout": sealed,
            "forecasts_path": forecasts_path,
            "oos_metric": oos_metric,
            "oos_metric_name": metric,
            "outcomes_analysis": outcomes,
            "forecast_analysis": forecast,
            "portfolio_analysis": portfolio,
            "stress_testing": stress,
            "walk_forward": walk_forward,
            "trading": trading,
            "n_trials": model.get("n_candidates_tried"),
        }
        sev = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}
        for f in out.findings:
            res.add_finding(Finding(id=f"oa-{f.id}", severity=sev[f.severity], category=f.category,
                                    message=f.message, location="outcomes_analyst",
                                    suggestion="Evidence: " + ", ".join(f.evidence)))
        payload["interpretation"] = {"discrimination": out.discrimination,
                                     "calibration": out.calibration, "stability": out.stability}
        attach_recommendation(payload, ctx, "outcomes_analyst", out)
        res.add_artifact(ctx.save_json("stages/backtest/backtest.json", payload))
        res.payload = payload
        res.metrics = {"oos_metric": oos_metric, "used_impact": impact_res.used_impact,
                       "gini": (outcomes or {}).get("gini"), "psi": (outcomes or {}).get("psi"),
                       "champion": chosen["family"], "n_candidates": len(candidates)}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS

        impact_word = "IMPACT" if impact_res.used_impact else "built-in scorer"
        bits = [f"Compared {len(candidates)} candidate(s) on the {sample.replace('_', ' ')} "
                f"sample; champion {chosen.get('name') or champion_id} [{champion_id}] "
                f"{chosen['label']} ({champion_source})",
                f"scored {len(impact_res.scored_df)} rows via {impact_word}"]
        if oos_metric is not None:
            bits.append(f"OOS {metric}={oos_metric:.4f}")
        if forecast:
            bits.append(f"bias={forecast['bias']:+.4f}")
        if outcomes:
            bits.append(f"Gini={outcomes['gini']:.3f}, KS={outcomes['ks']:.3f}, "
                        f"PSI={outcomes['psi']:.3f} ({outcomes['psi_label']})")
        if portfolio:
            bits.append(f"portfolio EL={portfolio['expected_loss']:.4f}, "
                        f"VaR{portfolio['confidence']:.3f}={portfolio['var']:.4f}")
        if stress:
            worst = max(stress["scenarios"], key=lambda s: s["mean_pd"], default=None)
            if worst:
                bits.append(f"stress worst '{worst['name']}' mean={worst['mean_pd']:.4f} "
                            f"(Δ{worst['delta_pd']:+.4f})")
        res.summary = "; ".join(bits) + "."
        return res

    # --- helpers ----------------------------------------------------------------
    def _analyst(self, ctx: RunContext, res: StageResult, candidates: list[dict],
                 comparison: dict, sealed: bool, stage_model: dict, lead_scorer: str | None):
        """Ask the Outcomes Analyst — or reuse its last answer when only the developer's choice
        of champion changed (the same pool, compared on the same sample)."""
        from ..agents.contracts import OutcomesAnalystOutput

        prev = ctx.get("backtest")
        prev_rec = ((prev.payload or {}).get("recommendation") or {}) if prev else {}
        same_pool = prev is not None and [c["id"] for c in (prev.payload or {}).get(
            "candidates") or []] == [c["id"] for c in candidates]
        if (ctx.overrides.champion and same_pool and prev_rec.get("output")
                and not ctx.challenges_for("backtest")):
            try:
                out = OutcomesAnalystOutput.model_validate(prev_rec["output"])
                ctx.runner.last["outcomes_analyst"] = {
                    **{k: v for k, v in prev_rec.items() if k != "output"}, "reused": True}
                return out
            except Exception:  # recorded under an older contract: ask again
                pass
        tool_runs = ctx.consult_tools("outcomes_analyst", {
            "evaluation_sample": comparison["sample"],
            "candidates": [c["id"] for c in candidates],
            "engine_findings": [f.line() for f in res.findings]}, res=res,
            # a tool that needs a model is given the pool's leading candidate on this sample
            inputs={"model": lead_scorer} if lead_scorer else None)
        return ctx.recommend("outcomes_analyst", {
            "tool_runs": tool_runs,
            "evaluation_sample": comparison["sample"],
            "metric": comparison["metric"],
            "metric_direction": comparison["direction"],
            "interpretability": ctx.config.design.interpretability,
            "sealed_holdout": sealed,
            "candidates": candidates,
            "rank_by": ((stage_model.get("plan") or {}).get("evaluation") or {}).get("rank_by"),
            "engine_findings": [f.line() for f in res.findings],
        }, fresh={"backtest": res})

    @staticmethod
    def _metrics(names: list[str], y, scores, is_clf: bool) -> dict[str, float]:
        out: dict[str, float] = {}
        if y is None:
            return out
        for name in names:
            try:
                out[name] = float(score(name, y, (scores >= 0.5).astype(int), y_proba=scores)
                                  if is_clf else score(name, y, scores))
            except Exception:  # a metric that does not apply to this sample is left out
                continue
        return out

    @staticmethod
    def _forecast_outcomes(y, pred) -> dict[str, float]:
        """Forecast error for a continuous target: size, direction and fit."""
        err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
        level = float(np.mean(np.abs(y))) or 1.0
        return {"rmse": score("rmse", y, pred), "mae": score("mae", y, pred),
                "r2": score("r2", y, pred), "bias": float(np.mean(err)),
                "bias_share": float(np.mean(err)) / level,
                "max_abs_error": float(np.max(np.abs(err))),
                "mean_actual": float(np.mean(y)), "mean_forecast": float(np.mean(pred)),
                "n_scored": int(len(err))}

    @staticmethod
    def _save_series(ctx: RunContext, cfg, eval_df, y_eval, series: dict, sample: str):
        """Actual and forecast per row of the comparison sample, per pool member: the picture
        behind "similar statistics, different forecasts". Kept on disk, drawn by the workbench."""
        if not series or len(eval_df) > MAX_SERIES_POINTS:
            return None
        col = cfg.data.datetime_col
        x = (eval_df[col].astype(str).tolist() if col and col in eval_df.columns
             else list(range(len(eval_df))))
        return ctx.save_json("stages/backtest/forecasts.json", {
            "sample": sample, "x": x, "x_label": col or "row",
            "actual": [float(v) for v in y_eval] if y_eval is not None else None,
            "forecast": {cid: [float(v) for v in vals] for cid, vals in series.items()}})

    @staticmethod
    def _scores(frame, scorer_path: str):
        """Scores on the development sample (for PSI dev->comparison). None when unavailable."""
        try:
            return score_frame(frame, scorer_path)
        except Exception:
            return None

    @staticmethod
    def _outcome_flags(outcomes: dict) -> list[dict]:
        out = []
        if outcomes["psi_label"] == "significant shift":
            out.append({"id": "psi-shift", "severity": Severity.MEDIUM, "category": "stability",
                        "message": f"Population shift dev->OOT: PSI={outcomes['psi']:.3f} (>0.25).",
                        "confidence": 0.8})
        ece = outcomes["expected_calibration_error"]
        if ece is not None and not np.isnan(ece) and ece > 0.10:
            out.append({"id": "calibration", "severity": Severity.MEDIUM, "category": "calibration",
                        "message": f"Poor calibration: mean |observed-predicted|={ece:.3f} (>0.10).",
                        "confidence": 0.7})
        g = outcomes["gini"]
        if g is not None and not np.isnan(g) and g < 0.2:
            out.append({"id": "weak-discrimination", "severity": Severity.LOW,
                        "category": "discrimination",
                        "message": f"Weak discrimination: Gini={g:.3f} (<0.2).", "confidence": 0.7})
        return out

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
