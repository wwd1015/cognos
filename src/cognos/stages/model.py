"""Stage 3 — Search: from the factor library to a pool of candidate models (ADR-0016).

The stage follows the development plan confirmed at the design gate (``modeling/plan.py``):

1. **Samples.** The data are cut into development, test and (when the plan keeps one) a sealed
   holdout. The holdout is written to disk and not read again by this stage.
2. **Factor library.** Explored columns are widened into their transformations.
3. **Single-factor analysis.** Every factor is fitted alone; the plan's rules cut the library
   to a shortlist, and every dropped factor keeps the rule and the reason.
4. **Multi-factor selection.** Each method of the plan (forward, backward, stepwise, best
   subset, grid) builds models from the shortlist; every model evaluated lands in one ledger.
5. **Candidate pool.** The ledger is ranked by the plan's ranking. The **Modeler** recommends
   which of the leaders go forward; the developer adds or removes members at the gate.

No champion is chosen here. Every pool member is fitted on the development sample, tested
and saved as a scorer, and the outcomes analysis (backtest) compares them. The search is
cached by a fingerprint of its inputs, so a changed pool re-finalizes without searching again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis import consult
from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..modeling import factors as factors_mod
from ..modeling import fit_full, greedy_ensemble, score, selection, splits
from ..modeling import plan as plan_mod
from ..modeling.fit import HAZARD_FAMILIES, Candidate
from ..modeling.metrics import metric_direction
from ..runtime.score import save_scorer
from . import stat_tests
from .base import Stage, attach_recommendation, register_stage

_TREES = {"random_forest", "gradient_boosting"}
MAX_LEADERS = 12


@register_stage
class ModelStage(Stage):
    name = "model"
    requires = ("explore",)
    description = "Screen factors, build models by the plan's methods, and form the candidate pool."

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        df = ctx.load_dataset()
        profile = ctx.profile()  # human-excluded columns are already gone
        features = list(profile["features"])
        metric = cfg.metric.name
        is_clf = cfg.task.is_classification
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        # --- structural (Merton) augmentation: deterministic solver, target-hidden ---
        structural_info, structural_spec = None, None
        protected: list[str] = []  # engineered factors the screen never drops
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
                features.append(structural_spec.dd_feature)  # hybrid mode: DD feeds the models
            protected.append(structural_spec.dd_feature)
            structural_info = {"spec": structural_spec.to_dict(), **s_info}

        # --- the plan in force ------------------------------------------------------
        families = ctx.get("ideate").payload.get("families") if ctx.has("ideate") else None
        if ctx.overrides.slate:  # the human's slate edit at gate_design wins
            families = list(dict.fromkeys(h["family"] for h in ctx.overrides.slate))
        families = families or (cfg.search.model_families or None)
        plan = self.plan_in_force(ctx, df, features, families)
        ev = plan["evaluation"]
        shape = plan["shape"]["shape"]
        is_ts = cfg.task.value == "timeseries" or shape == "series"
        for note in plan.get("ignored") or []:
            res.add_finding(Finding(id="plan-ignored", severity=Severity.LOW, category="plan",
                                    message=note))

        # --- the factor library (a function of feature columns only) -----------------
        frame, library, lib_notes = factors_mod.build_library(
            df, features, plan["factors"], shape=shape, datetime_col=cfg.data.datetime_col,
            protected=tuple(protected))
        n_burn = len(df) - len(frame)

        # --- samples: the holdout is cut, written and not read again here -------------
        train_df, test_df, holdout_df = splits.split_samples(
            frame, ev, datetime_col=cfg.data.datetime_col, random_state=cfg.search.random_state)

        # --- rating-migration augmentation: matrix FITTED on the development sample ---
        # Unlike the deterministic Merton solver above, the transition matrix is estimated, so it
        # must never see the test sample or the sealed holdout. The horizon PD it implies replaces
        # the raw rating as a factor (they are exact functions of each other — keeping both would
        # make the K-1 inference design perfectly collinear).
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
            if len(test_df):
                test_df, _ = migration_augment(test_df, migration_model)
            if len(holdout_df):
                holdout_df, _ = migration_augment(holdout_df, migration_model)
            features = [f for f in features if f != mc.rating_col]
            library = [f for f in library if f.source != mc.rating_col]
            if migration_spec.pd_feature not in features:
                features.append(migration_spec.pd_feature)  # hybrid mode: migration PD is a factor
                library.append(factors_mod.Factor(name=migration_spec.pd_feature,
                                                  source=migration_spec.pd_feature))
            protected.append(migration_spec.pd_feature)
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
        for name, part in (("test", test_df), ("holdout", holdout_df)):
            if len(part):
                ctx.save_df(f"data/{name}.parquet", part)
            else:  # a plan without this sample must not leave an earlier run's file behind
                ctx.resolve(f"data/{name}.parquet").unlink(missing_ok=True)

        y_train = coerce_target(train_df, cfg)
        y_test = coerce_target(test_df, cfg) if len(test_df) else None

        # --- discrete-time hazard metadata (survival framework) ---------------------
        # The event-time column rides along in the search frame as metadata (never a factor);
        # hazard candidates panel-expand inside their estimator, so CV stays obligor-level.
        etc = cfg.data.event_time_col
        hazard_meta = None
        if etc and etc in train_df.columns:
            observed = pd.to_numeric(train_df[etc], errors="coerce")
            horizon = int(cfg.data.horizon_periods or max(1.0, float(np.nanmax(observed.to_numpy()))
                                                          if observed.notna().any() else 1.0))
            hazard_meta = {"event_time_col": etc, "horizon": horizon}

        # --- the search: single factors, the screen, the methods ----------------------
        rank_by = ev["rank_by"]
        rank_metric = rank_by[0]["metric"]
        expected = self._expected_signs(profile, plan)
        factor_names = [f.name for f in library]
        fingerprint = self._fingerprint(cfg, train_df, y_train, factor_names, families,
                                        hazard_meta, plan, expected)
        splitter, split_note = splits.make_splitter(ev, n=len(train_df),
                                                    random_state=cfg.search.random_state)
        groups = (train_df[ev["group_col"]].to_numpy()
                  if ev["validation"] == "group_kfold" and ev.get("group_col") in train_df.columns
                  else None)

        def new_evaluator() -> selection.Evaluator:
            return selection.Evaluator(
                train_df, y_train, task=cfg.task.value, metric=rank_metric,
                is_classification=is_clf, splitter=splitter, groups=groups,
                metrics=tuple(dict.fromkeys([metric, *(r["metric"] for r in rank_by)])),
                test=test_df, y_test=y_test,
                score_test=any(r["sample"] == "test" for r in rank_by),
                random_state=cfg.search.random_state, time_budget_s=cfg.search.time_budget_s)

        found = self._cached_search(ctx, fingerprint, lambda: self._search(
            new_evaluator(), cfg, plan, library, expected, families, hazard_meta,
            protected, is_clf, shape))
        evaluator = new_evaluator()  # carries the cached evaluations; fits nothing again
        evaluator.cache = {e.id: e for e in found["evaluations"]}
        evaluator.order = [e.id for e in found["evaluations"]]
        rows, shortlist, paths = found["single_factor"], found["shortlist"], found["paths"]
        ranked = selection.rank(found["evaluations"], rank_by,
                                complexity_penalty=cfg.search.complexity_penalty,
                                n_total=len(shortlist))
        if not ranked:
            raise RuntimeError("Search produced no viable candidate (all crashed). Check data/config.")
        ledger = selection.ledger(evaluator, ranked, rank_by)
        header = "idx\tlabel\tfamily\tn_features\tmetric\tmetric_value\tcv_std\trank\tstatus\tdescription"
        ctx.save_text("stages/model/ledger.tsv",
                      "\n".join([header, *(r.tsv_row() for r in ledger)]) + "\n", kind="tsv")
        from dataclasses import asdict

        ctx.save_json("stages/model/ledger.json", [asdict(r) for r in ledger])
        res.add_artifact(ctx.save_json("stages/model/single_factor.json", {
            "rows": rows, "screen": found["screen"], "shortlist": shortlist,
            "expected_signs": expected}))
        res.add_artifact(ctx.save_json("stages/model/selection_paths.json", paths))

        # --- ensemble of survivors (Caruana) ---------------------------------------
        # The deployed model is always one interpretable model (ADR-0007). An ensemble, when
        # explicitly enabled, is computed only as a labelled predictive-ceiling *challenger benchmark*
        # ("how much accuracy would a blend buy?") — it is reported, never silently shipped.
        challenger_benchmark = None
        validated = [e for e in ranked if e.cv is not None]
        if cfg.search.ensemble and len(validated) >= 2 and rank_by[0]["sample"] == "validation":
            top = validated[:8]
            ens = greedy_ensemble([e.cv.oof_pred for e in top], y_train,
                                  metric=rank_metric, is_classification=is_clf)
            if ens is not None:
                challenger_benchmark = {
                    "kind": "ensemble (Caruana)",
                    "deployed": False,
                    "note": "Exploratory predictive-ceiling benchmark only; the deployed model is one single model.",
                    "n_members": len(set(ens.member_indices)),
                    "weights": {top[i].candidate.label(): w for i, w in ens.weights.items()},
                    "benchmark_score": ens.ensemble_score,
                    "champion_single_score": ens.best_single_score,
                    "headroom_vs_champion": ens.improved,
                }

        # --- persist per-candidate OOF performance for an honest PBO ----------------
        # PBO must be computed over the full set of configurations tried (Bailey/López de Prado),
        # not a hand-picked few. We save the per-sample performance matrix of every evaluated
        # candidate so the backtest stage can compute PBO over the real search library.
        oof_perf_path = self._save_oof_perf(ctx, validated, y_train, is_clf)

        # --- agent-guided refinement (ADR-0001 stage B; opt-in; the agent proposes) ---
        guided_info, guided_entry = None, None
        lead = ranked[0]
        # Guided refinement refits via the generic path; a hazard leader has its own fit contract.
        if (cfg.search.guided and ctx.runner.kind != "heuristic" and lead.cv is not None
                and lead.candidate.family not in HAZARD_FAMILIES):
            from ..modeling.guided import guided_search
            from .ideate import _engine_families

            explored = [f.name for f in library if f.transform == "level" and not f.lag]
            gr = guided_search(
                lambda c: ctx.recommend("experiment", c).model_dump(),
                train_df[explored], y_train, columns=explored, champion=lead.candidate,
                champion_cv=lead.cv, metric=rank_metric, direction=metric_direction(rank_metric),
                is_classification=is_clf,
                allowed_families=[f for f in _engine_families(cfg) if f not in HAZARD_FAMILIES],
                is_timeseries=is_ts, folds=ev["folds"],
                random_state=cfg.search.random_state, rounds=cfg.search.guided_rounds,
            )
            guided_info = {"rounds": len(gr.rounds), "accepted": sum(1 for x in gr.rounds if x.accepted),
                           "improved": gr.improved, "transforms": [t.to_dict() for t in gr.transforms],
                           "history": [{"round": r.idx, "accepted": r.accepted, "score": r.score,
                                        "note": r.note} for r in gr.rounds]}
            if gr.improved:  # engine verified the agent-proposed candidate beats the incumbent
                guided_entry = selection.Evaluation(
                    id=selection.candidate_id(gr.champion, gr.transforms), candidate=gr.champion,
                    cv=gr.champion_cv, scores={"validation": {rank_metric: gr.champion_cv.mean}},
                    methods=["guided"])
                guided_entry.info["transforms"] = list(gr.transforms)
                guided_entry.info["base_features"] = explored

        # --- the leaders and the pool (no champion is chosen here) ---------------------
        leaders, lookup = self._leaders(evaluator, ranked, cfg, plan, expected, found["corr"],
                                        guided_entry, train_df, y_train, is_clf)
        rec = self._modeler_choice(ctx, leaders, plan, len(ledger), fingerprint)
        pool_ids, pool_source = [i for i in rec.pool if i in lookup], "agent"
        if ctx.runner.kind == "heuristic":
            pool_source = "engine"
        asked = [i for i in ctx.overrides.pool or []]
        if asked:
            everyone = {e.id: e for e in found["evaluations"] if not e.error}
            known = [i for i in asked if i in lookup or i in everyone]
            if known:
                pool_ids, pool_source = known, "human"
                for i in known:
                    lookup.setdefault(i, everyone.get(i))
            if len(known) < len(asked):
                res.add_finding(Finding(
                    id="pool-unknown", severity=Severity.LOW, category="plan",
                    message=f"{len(asked) - len(known)} pool member(s) chosen at the gate are not "
                            "in this search (the plan or the data changed since) and were left "
                            "out."))
        pool = []
        for cid in dict.fromkeys(pool_ids):
            pool.append(self._finalize(
                ctx, lookup[cid], train_df, y_train, cfg, library, hazard_meta, etc,
                structural_spec, migration_model, migration_spec, is_clf, is_ts, rank_by,
                leaders, expected))
        for member in pool:
            member["source"] = pool_source

        # Pure-structural / pure-migration challenger benchmarks: the PD scored directly on the
        # test sample — labelled reference points, never a pool member.
        if len(test_df) and is_clf:
            for info, spec in ((structural_info, structural_spec), (migration_info, migration_spec)):
                if info is None or spec is None:
                    continue
                pd_alone = test_df[spec.pd_feature].to_numpy(dtype=float)
                auc = score("roc_auc", y_test, (pd_alone >= 0.5).astype(int), y_proba=pd_alone)
                info["benchmark"] = {
                    "kind": ("pure structural Merton PD (challenger benchmark)"
                             if info is structural_info
                             else "pure rating-migration PD (challenger benchmark)"),
                    "deployed": False, "sample": "test",
                    "test_roc_auc": auc, "test_gini": 2 * auc - 1,
                }

        # --- migration expected-loss forecast (reported; never feeds selection) -----
        if migration_model is not None and migration_spec is not None:
            from ..modeling.migration import expected_loss_forecast

            # Forecast on the out-of-time book when one exists (the closest thing to the current
            # portfolio); otherwise the development book.
            book = test_df if len(test_df) else train_df
            ead = (book[mc.ead_column].to_numpy(dtype=float)
                   if mc.ead_column and mc.ead_column in book.columns else None)
            migration_info["loss_forecast"] = expected_loss_forecast(
                book[migration_spec.rating_col], migration_model, lgd=mc.lgd, ead=ead)
            migration_info["loss_forecast"]["book"] = (
                ("out_of_time_test" if ev["test"] == "out_of_time" else "test")
                if len(test_df) else "development")

        # --- findings ---------------------------------------------------------------
        for f in migration_findings:
            res.add_finding(f)
        for f in consult.findings(consult.recorded(ctx, "model")):
            res.add_finding(f)
        if found["screen_fallback"]:
            res.add_finding(Finding(
                id="screen-empty", severity=Severity.MEDIUM, category="factors",
                message="The screen left no factor. The strongest "
                        f"{len(shortlist)} by single-factor fit were kept so a model could be "
                        "built; loosen the rules or widen the library.",
                suggestion="Review the screen at the design gate."))
        worst = "LOW"
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        for member in pool:
            failed = member["diagnostics_failed"]
            if not failed:
                continue
            sev = member["diagnostics_worst"] or "LOW"
            worst = max(worst, sev, key=lambda s: order.get(s, 0))
            res.add_finding(Finding(
                id=f"diag-{member['id']}", severity=Severity(sev), category="diagnostic",
                location=member["id"],
                message=f"{member['id']} ({member['label']}): {len(failed)} of "
                        f"{member['diagnostics_run']} diagnostic test(s) failed "
                        f"({', '.join(failed[:6])})."))
        if not any(not m["violations"] for m in pool):
            res.add_finding(Finding(
                id="pool-constraints", severity=Severity.MEDIUM, category="selection",
                message="No pool member passes every constraint of the plan (significance, "
                        "sign, correlation between factors).",
                suggestion="Loosen a constraint at the design gate, or widen the factor library."))

        best = ranked[0]
        payload = {
            "plan": plan,
            "plan_summary": plan_mod.summary(plan),
            "samples": {
                "development": int(len(train_df)), "test": int(len(test_df)),
                "holdout": int(len(holdout_df)), "dropped_no_history": int(n_burn),
                "validation_blocks": splits.blocks(splitter, len(train_df), groups,
                                                   y_train if is_clf else None),
                "validation_note": split_note,
            },
            "library": {
                "n_factors": len(library), "n_sources": len({f.source for f in library}),
                "transforms": plan["factors"]["transforms"], "notes": lib_notes,
            },
            "single_factor": [r for r in rows if r["status"] == "kept"][:40],
            "screen": [{k: line[k] for k in ("rule", "setting", "before", "after")}
                       for line in found["screen"]],
            "shortlist": shortlist,
            "expected_signs": expected,
            "paths": paths,
            "leaders": leaders,
            "pool": pool,
            "pool_ids": [m["id"] for m in pool],
            "evaluated_ids": [e.id for e in ranked],
            "pool_source": pool_source,
            "search_fingerprint": fingerprint,
            "metric": metric,
            "direction": metric_direction(metric),
            "rank_by": rank_by,
            "rank_metric": rank_metric,
            "best_id": best.id,
            "best_score": best.value(rank_metric, rank_by[0]["sample"]),
            "n_candidates_tried": len(ledger),
            "n_configs_for_deflation": len(ledger),
            "test_evaluations": int(found["test_evaluations"]),
            "oof_perf_path": oof_perf_path,
            "n_search_strategies": len(validated),
            "challenger_benchmark": challenger_benchmark,
            "structural": structural_info,
            "migration": migration_info,
            "guided": guided_info,
            "raw_features": features,
            "factors": [f.to_dict() for f in library if f.transform != "level" or f.lag],
            "is_classification": is_clf,
            "is_timeseries": is_ts,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_holdout": int(len(holdout_df)),
        }
        attach_recommendation(payload, ctx, "modeler", rec)
        if migration_info is not None:
            res.add_artifact(ctx.save_json("stages/model/migration.json", migration_info))
        res.add_artifact(ctx.save_json("stages/model/summary.json", payload))
        res.payload = payload
        # ``best_score`` claims the run's direction in comparisons, so it is reported under that
        # name only when the ranking metric points the same way as the run's metric.
        aligned = metric_direction(rank_metric) == metric_direction(metric)
        res.metrics = {"best_score" if aligned else "rank_score": payload["best_score"],
                       "n_candidates_tried": len(ledger),
                       "n_factors": len(library), "n_shortlisted": len(shortlist),
                       "n_pool": len(pool)}
        if oof_perf_path:
            res.add_artifact(ArtifactRef(name="oof_perf", kind="json", path=oof_perf_path,
                                         description="Per-candidate OOF performance matrix for PBO."))
        plan_trouble = any(f.id in ("screen-empty", "pool-constraints") for f in res.findings)
        res.verdict = (Verdict.WARN if res.findings and (worst in ("MEDIUM", "HIGH") or plan_trouble)
                       else Verdict.PASS)
        res.summary = (
            f"{len(library)} factors screened to {len(shortlist)}; "
            f"{len(ledger)} models evaluated by {' + '.join(plan['selection']['methods'])}; "
            f"pool of {len(pool)} ({pool_source}): {', '.join(m['id'] for m in pool)}. "
            f"Best {rank_metric} ({rank_by[0]['sample'].replace('_', ' ')}) "
            f"{payload['best_score']:.4f}."
        )
        return res

    # --- the plan ------------------------------------------------------------------------
    @staticmethod
    def plan_in_force(ctx: RunContext, df: pd.DataFrame, features: list[str],
                      families: list[str] | None) -> dict:
        """The plan this run works to: the developer's edit, the profile, the Design Lead's
        proposal (recorded by ideate), the engine's recommendation, in that order."""
        from ..modeling.fit import DEFAULT_FAMILIES

        cfg = ctx.config
        families = families or DEFAULT_FAMILIES.get(cfg.task.value, DEFAULT_FAMILIES["regression"])
        ideate = ctx.get("ideate")
        proposed = ((ideate.payload or {}).get("plan_proposed") if ideate is not None else None)
        return plan_mod.resolve(cfg, plan_mod.shape_of(df, cfg), families=list(families),
                                columns=list(df.columns), proposed=proposed,
                                human=ctx.overrides.plan)

    @staticmethod
    def _expected_signs(profile: dict, plan: dict) -> dict[str, str]:
        """The direction each explored column is expected to act in: the Data Analyst's
        reading, overruled by what the developer states in the plan."""
        out = {c["column"]: c["relationship"] for c in profile.get("feature_candidates") or []
               if c.get("relationship") in ("+", "-")}
        out.update(plan["factors"].get("expected_signs") or {})
        return out

    # --- the search (cached by fingerprint) ---------------------------------------------
    def _search(self, evaluator: selection.Evaluator, cfg, plan: dict, library: list,
                expected: dict[str, str], families: list[str] | None, hazard_meta: dict | None,
                protected: list[str], is_clf: bool, shape: str) -> dict:
        from ..modeling.fit import DEFAULT_FAMILIES

        dev, y = evaluator.dev, evaluator.y
        rank_metric = evaluator.metric
        base_family = "logit" if is_clf else "ols"
        rs = cfg.search.random_state

        def alone(name: str) -> float | None:
            ev = evaluator.evaluate(Candidate(family=base_family, features=[name],
                                              hyperparams={"random_state": rs},
                                              description=f"{base_family}/single_factor"),
                                    "single_factor")
            return ev.value(rank_metric, "validation")

        period = plan["factors"].get("period")
        rows = factors_mod.single_factor(
            dev, y, library, metric=rank_metric, is_classification=is_clf,
            hac_lags=int(period or 4) if shape == "series" and not is_clf else None,
            validation=alone if evaluator.splitter is not None else None)
        numeric = [r["name"] for r in rows if r["numeric"]]
        corr = dev[numeric].astype(float).corr() if len(numeric) >= 2 else None
        carried = tuple(dict.fromkeys([*protected, *(plan["selection"].get("must_include") or [])]))
        shortlist, log = factors_mod.screen(
            rows, plan["factors"], corr=corr, expected=expected,
            maximize=metric_direction(rank_metric) == "maximize", protected=carried)
        fallback = False
        if not shortlist:  # a screen that leaves nothing still has to produce a model
            usable = [r for r in rows if not r["note"]]
            maximize = metric_direction(rank_metric) == "maximize"
            usable.sort(key=lambda r: (factors_mod.strength(r) is None,
                                       -(factors_mod.strength(r) or 0.0) if maximize
                                       else (factors_mod.strength(r) or 0.0)))
            shortlist = [r["name"] for r in usable[:3]]
            for r in usable[:3]:
                r.update(status="kept", reason="kept so a model could be built")
            fallback = True
        by_factor = {r["name"]: factors_mod.expected_sign(r, expected) for r in rows}
        # Hybrid mode: an engineered factor (Merton DD, the migration PD) is in every model.
        rules = {**plan["selection"], "must_include": list(dict.fromkeys(
            [*(plan["selection"].get("must_include") or []), *protected]))}
        families = families or DEFAULT_FAMILIES.get(cfg.task.value, DEFAULT_FAMILIES["regression"])
        paths = selection.run_methods(
            evaluator, families=list(families), factors=shortlist, rules=rules,
            expected={k: v for k, v in by_factor.items() if v}, corr=corr,
            all_columns=shortlist, max_candidates=cfg.search.max_candidates,
            max_features=cfg.search.max_features_per_candidate, hazard_meta=hazard_meta,
            random_state=rs)
        return {"evaluations": [evaluator.cache[i] for i in evaluator.order],
                "single_factor": rows, "screen": log, "shortlist": shortlist,
                "screen_fallback": fallback, "paths": [p.to_dict() for p in paths],
                "corr": corr, "factor_signs": {k: v for k, v in by_factor.items() if v},
                "test_evaluations": evaluator.test_evaluations}

    @staticmethod
    def _fingerprint(cfg, train_df, y_train, factor_names, families, hazard_meta, plan,
                     expected) -> str:
        import hashlib
        import json

        cols = [c for c in factor_names if c in train_df.columns]
        data_hash = int(pd.util.hash_pandas_object(train_df[cols], index=False).sum()) & 0xFFFFFFFF
        key = {"factors": factor_names, "families": families, "hazard": hazard_meta,
               "search": cfg.search.model_dump(exclude={"guided", "guided_rounds"}),
               "plan": {s: plan[s] for s in plan_mod.SECTIONS}, "expected": expected,
               "task": cfg.task.value, "metric": cfg.metric.name, "rows": len(train_df),
               "data": data_hash, "y": float(np.nansum(np.asarray(y_train, dtype=float)))}
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @staticmethod
    def _cached_search(ctx: RunContext, fingerprint: str, search):
        import joblib

        path = ctx.resolve("stages/model/search_cache.joblib")
        if path.exists():
            try:
                cached = joblib.load(path)
                if cached.get("fingerprint") == fingerprint and "found" in cached:
                    return cached["found"]
            except Exception:  # a stale/corrupt cache just means searching again
                pass
        found = search()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"fingerprint": fingerprint, "found": found}, path)
        return found

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

    # --- leaders, the Modeler's pool, and finalizing a member ---------------------------
    def _leaders(self, evaluator, ranked, cfg, plan: dict, expected: dict, corr,
                 guided_entry, train_df, y_train, is_clf: bool) -> tuple[list[dict], dict]:
        """The top of the ranking, one entry per distinct specification, with what the Modeler
        and the developer need to pick a pool: scores, signs, and any constraint it breaks."""
        ev = plan["evaluation"]
        rank_by = ev["rank_by"]
        first = rank_by[0]
        maximize = metric_direction(first["metric"]) == "maximize"
        required = cfg.design.interpretability == "required"
        pool_size = max(1, int(ev["pool_size"]))
        want = min(MAX_LEADERS, max(2 * pool_size, pool_size + 4))
        signs_by_factor = {r: s for r, s in (expected or {}).items()}

        chosen, seen = [], set()
        for e in ranked:
            key = (e.candidate.family, frozenset(e.candidate.features))
            if key in seen:
                continue  # the same factors in the same family: the best setting stands for it
            seen.add(key)
            chosen.append(e)
            if len(chosen) >= want:
                break
        if required and all(e.candidate.family in _TREES for e in chosen):
            interp = next((e for e in ranked if e.candidate.family not in _TREES), None)
            if interp is not None:
                chosen.append(interp)
        if guided_entry is not None:
            chosen.insert(0, guided_entry)

        best = chosen[0]
        best_v = best.value(first["metric"], first["sample"])
        best_sd = best.cv.std if (best.cv is not None and first["sample"] == "validation") else None
        leaders, lookup = [], {}
        for place, e in enumerate(chosen, 1):
            cand = e.candidate
            transforms = e.info.get("transforms") or []
            signs: list[dict] = []
            bad: list[str] = []
            if not transforms and cand.family in selection.STEP_FAMILIES:
                bad = selection.violations(e, evaluator, plan["selection"],
                                           self._factor_signs(expected, cand), corr)
                info = evaluator.inference(e)
                signs = self._signs(info.get("coefficients"), list(cand.features))
            elif cand.family not in _TREES and cand.family not in HAZARD_FAMILIES:
                try:
                    X_fit = train_df
                    if transforms:
                        from ..modeling.transforms import apply_transforms

                        X_fit, _, _ = apply_transforms(
                            train_df[e.info["base_features"]], transforms)
                    fitted = fit_full(cand, X_fit, y_train, task=cfg.task.value,
                                      is_classification=is_clf)
                    signs = self._signs(fitted.coefficients(), list(cand.features))
                except Exception:  # signs are advisory; a failed refit just omits them
                    signs = []
            for s in signs:
                s["expected"] = signs_by_factor.get(self._source_of(s["feature"]), "none")
            v = e.value(first["metric"], first["sample"])
            within = None
            if best_sd is not None and v is not None and best_v is not None:
                within = bool(v >= best_v - best_sd if maximize else v <= best_v + best_sd)
            role = "challenger" if (cand.family in _TREES and required) else "candidate"
            leaders.append({
                "id": e.id, "rank": place, "label": cand.label(), "family": cand.family,
                "n_features": len(cand.features), "features": list(cand.features),
                "methods": list(e.methods), "scores": e.scores,
                "score": v,
                "cv_mean": round(float(e.cv.mean), 6) if e.cv is not None else None,
                "cv_std": round(float(e.cv.std), 6) if e.cv is not None else None,
                "within_one_se": within,
                "role": role, "interpretable": cand.family not in _TREES,
                "transforms": [t.to_dict() for t in transforms],
                "coefficient_signs": signs, "violations": bad,
                "engine_pick": False,
            })
            lookup[e.id] = e
        self._engine_pool(leaders, pool_size, required)
        return leaders, lookup

    @staticmethod
    def _engine_pool(leaders: list[dict], pool_size: int, required: bool) -> None:
        """The engine's default pool. Models that rank together but differ are what the
        outcomes analysis is for, so: first the leaders that differ (see ``alike``), are within
        one standard error of the best and break no constraint; then their near-copies; and
        only to reach two members, the next best outside that band."""
        def eligible(a: dict) -> bool:
            return not (required and a["role"] == "challenger")

        def close(a: dict) -> bool:
            return a["within_one_se"] is not False

        def alike(a: dict, b: dict) -> bool:
            """Would these two tell the developer the same thing? The same factors in two
            linear families do; so do two shrinkage fits where one's factors contain the
            other's (the penalty decides what survives, not the list). Two tested models that
            differ by a factor do not: whether that factor earns its place is the question."""
            if (a["family"] in _TREES) != (b["family"] in _TREES):
                return False
            fa, fb = set(a["features"]), set(b["features"])
            if fa == fb:
                return True
            untested = not (a["family"] in selection.STEP_FAMILIES
                            or b["family"] in selection.STEP_FAMILIES)
            return untested and (fa <= fb or fb <= fa)

        picked: list[dict] = []
        passes = (
            lambda a: eligible(a) and close(a) and not a["violations"]
            and not any(alike(a, p) for p in picked),
            lambda a: eligible(a) and close(a) and not a["violations"],
            lambda a: eligible(a) and close(a),
        )
        for ok in passes:
            for a in leaders:
                if len(picked) >= pool_size:
                    break
                if a not in picked and ok(a):
                    picked.append(a)
        for a in leaders:  # a pool of one compares nothing
            if len(picked) >= min(2, pool_size):
                break
            if a not in picked and eligible(a):
                picked.append(a)
        for a in picked or leaders[:1]:
            a["engine_pick"] = True

    @staticmethod
    def _source_of(name: str) -> str:
        return name.split("__", 1)[0]

    @staticmethod
    def _factor_signs(expected: dict, cand: Candidate) -> dict[str, str]:
        """Expected sign per factor of one model (a transformation that keeps the direction of
        its column inherits the column's sign)."""
        out = {}
        for name in cand.features:
            source, _, rest = name.partition("__")
            if name in expected:
                out[name] = expected[name]
            elif source in expected and not rest.startswith("square"):
                out[name] = expected[source]
        return out

    @staticmethod
    def _modeler_choice(ctx: RunContext, leaders: list[dict], plan: dict, n_tried: int,
                        fingerprint: str):
        """Ask the Modeler — or reuse its last answer when only the developer's pool changed."""
        from ..agents.contracts import ModelerOutput

        prev = ctx.get("model")
        prev_rec = (prev.payload.get("recommendation") or {}) if prev else {}
        if (ctx.overrides.pool and prev is not None
                and prev.payload.get("search_fingerprint") == fingerprint
                and prev_rec.get("output") and not ctx.challenges_for("model")):
            try:
                out = ModelerOutput.model_validate(prev_rec["output"])
                ctx.runner.last["modeler"] = {**{k: v for k, v in prev_rec.items() if k != "output"},
                                              "reused": True}
                return out
            except Exception:  # a recommendation recorded under an older contract: ask again
                pass
        ev = plan["evaluation"]
        brief = [{k: a[k] for k in ("id", "label", "family", "n_features", "score")}
                 for a in leaders]
        # Tools at this stage are given the development sample only.
        tool_runs = ctx.consult_tools("modeler", {"rank_by": ev["rank_by"], "leaders": brief})
        return ctx.recommend("modeler", {
            "tool_runs": tool_runs,
            "rank_by": ev["rank_by"],
            "pool_size": ev["pool_size"],
            "interpretability": ctx.config.design.interpretability,
            "plan": {"evaluation": {k: ev[k] for k in ("test", "validation", "holdout_fraction")},
                     "selection": plan["selection"]},
            "leaders": leaders,
            "n_candidates_tried": n_tried,
        })

    def _finalize(self, ctx: RunContext, e, train_df, y_train, cfg, library, hazard_meta, etc,
                  structural_spec, migration_model, migration_spec, is_clf: bool, is_ts: bool,
                  rank_by: list[dict], leaders: list[dict], expected: dict) -> dict:
        """Fit one pool member on the development sample, test it and save its scorer."""
        cand = e.candidate
        transforms = e.info.get("transforms") or []
        base_features = e.info.get("base_features") or list(cand.features)
        hazard_info = None
        if cand.family in HAZARD_FAMILIES:
            from ..modeling.hazard import fit_full_hazard, term_structure

            X_fit = train_df
            fitted = fit_full_hazard(cand, train_df, y_train, event_time=train_df[etc],
                                     horizon=hazard_meta["horizon"], task=cfg.task.value)
            hazard_info = {
                "horizon_periods": hazard_meta["horizon"],
                "event_time_col": etc,
                "term_structure": term_structure(fitted.pipeline, train_df[cand.features]),
            }
        else:
            if transforms:
                from ..modeling.transforms import apply_transforms

                X_fit, _, _ = apply_transforms(train_df[base_features], transforms)
            else:
                X_fit = train_df
            fitted = fit_full(cand, X_fit, y_train, task=cfg.task.value, is_classification=is_clf)
        fitted.transforms = transforms
        fitted.base_features = base_features
        used = set(base_features) | set(cand.features)
        fitted.factors = [f.to_dict() for f in library
                          if f.name in used and (f.transform != "level" or f.lag)]
        fitted.datetime_col = cfg.data.datetime_col
        if structural_spec is not None and structural_spec.dd_feature in used:
            # Serving recomputes DD from the market observables, so the scorer's base features are
            # the raw inputs (market columns in, engineered DD out) — mirrors the transform contract.
            market = [structural_spec.equity_value_col, structural_spec.equity_vol_col,
                      structural_spec.debt_col]
            fitted.base_features = list(dict.fromkeys(
                [f for f in base_features if f != structural_spec.dd_feature] + market))
            fitted.structural = structural_spec.to_dict()
        if (migration_model is not None and migration_spec is not None
                and migration_spec.pd_feature in used):
            # Serving recomputes migration_pd from the rating column via the fitted matrix, so the
            # scorer's base features carry the raw rating in and the engineered PD out.
            current = list(fitted.base_features or base_features)
            fitted.base_features = list(dict.fromkeys(
                [f for f in current if f != migration_spec.pd_feature] + [migration_spec.rating_col]))
            fitted.migration = {k: migration_model[k] for k in ("spec", "pd_map", "fill")}
        diagnostics = stat_tests.run_battery(fitted, X_fit, y_train,
                                             is_classification=is_clf, is_timeseries=is_ts)

        directory = ctx.models_dir / "candidates"
        directory.mkdir(parents=True, exist_ok=True)
        scorer_file = directory / f"{e.id}.joblib"
        save_scorer(str(scorer_file.resolve()), fitted)
        coefficients = fitted.coefficients()
        pvalues = fitted.pvalues()
        detail_path = f"stages/model/candidates/{e.id}.json"
        ctx.resolve(detail_path).parent.mkdir(parents=True, exist_ok=True)
        ctx.save_json(detail_path, {
            "id": e.id, "candidate": cand.to_dict(), "coefficients": coefficients,
            "pvalues": pvalues, "feature_importances": fitted.feature_importances(),
            "diagnostics": diagnostics, "hazard": hazard_info,
            "transforms": [t.to_dict() for t in transforms], "base_features": base_features,
            "cv_fold_scores": [float(s) for s in e.cv.fold_scores] if e.cv is not None else [],
        })
        entry = next((a for a in leaders if a["id"] == e.id), None)
        if entry is None:  # added at the gate from below the leaders
            signs = self._signs(coefficients, list(cand.features))
            for s in signs:
                s["expected"] = expected.get(self._source_of(s["feature"]), "none")
            entry = {"rank": None, "methods": list(e.methods), "coefficient_signs": signs,
                     "violations": [], "within_one_se": None, "engine_pick": False,
                     "role": ("challenger" if cand.family in _TREES
                              and cfg.design.interpretability == "required" else "candidate"),
                     "interpretable": cand.family not in _TREES}
        first = rank_by[0]
        return {
            "id": e.id, "label": cand.label(), "family": cand.family,
            "features": list(cand.features), "n_features": len(cand.features),
            "hyperparams": {k: v for k, v in cand.hyperparams.items() if k != "random_state"},
            "rank": entry["rank"], "methods": entry["methods"],
            "scores": e.scores, "score": e.value(first["metric"], first["sample"]),
            "cv_mean": float(e.cv.mean) if e.cv is not None else None,
            "cv_std": float(e.cv.std) if e.cv is not None else None,
            "within_one_se": entry["within_one_se"], "engine_pick": entry["engine_pick"],
            "role": entry["role"], "interpretable": entry["interpretable"],
            "coefficient_signs": entry["coefficient_signs"], "violations": entry["violations"],
            "max_p_value": (max((v for k, v in pvalues.items() if k != "const"), default=None)
                            if pvalues else None),
            "diagnostics_run": diagnostics["n_run"], "diagnostics_passed": diagnostics["n_passed"],
            "diagnostics_failed": list(diagnostics["failed_tests"]),
            "diagnostics_worst": diagnostics["max_failed_severity"],
            "n_transforms": len(transforms),
            "scorer_path": str(scorer_file.resolve()),
            "detail_path": detail_path,
        }

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
    def _save_oof_perf(ctx: RunContext, validated: list, y_train, is_clf: bool) -> str | None:
        """Save an (n_train x n_candidates) per-sample performance matrix (higher = better)."""
        if not validated:
            return None
        y = np.asarray(y_train, dtype=float)
        cols, labels = [], []
        for e in validated:
            pred = np.asarray(e.cv.oof_pred, dtype=float)
            if is_clf:
                p = np.clip(pred, 1e-6, 1 - 1e-6)
                perf = np.where(np.isnan(pred), np.nan, y * np.log(p) + (1 - y) * np.log(1 - p))
            else:
                perf = -((y - pred) ** 2)
            cols.append(perf)
            labels.append(e.candidate.label())
        mat = np.column_stack(cols) if cols else np.zeros((len(y), 0))
        relpath = "stages/model/oof_perf.npz"
        np.savez(ctx.resolve(relpath), perf=mat, y=y, labels=np.array(labels, dtype=object))
        return relpath
