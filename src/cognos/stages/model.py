"""Stage 3 — Search: from the factor library to a pool of candidate models (ADR-0016).

The stage follows the development plan confirmed at the design gate (``modeling/plan.py``):

1. **Samples.** The data are cut into development, test and (when the plan keeps one) a sealed
   holdout. The holdout is written to disk and not read again by this stage.
2. **Factor library.** Explored columns are widened into their transformations.
3. **Single-factor analysis.** Every factor is fitted alone; the plan's rules cut the library
   to a shortlist, and every dropped factor keeps the rule and the reason.
4. **Multi-factor selection.** Each method of the plan (forward, backward, stepwise, best
   subset, grid) builds models from the shortlist; every model evaluated lands in one ledger.
5. **Candidate pool.** The ledger is ranked by the plan's ranking. The **Modeling agent** recommends
   which of the leaders go forward; the developer adds or removes members at the gate.

No champion is chosen here. Every pool member is fitted on the development sample, tested
and saved as a scorer, and the outcomes analysis (backtest) compares them. The search is
cached by a fingerprint of its inputs, so a changed pool re-finalizes without searching again.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..analysis import consult
from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target
from ..modeling import column_transforms, greedy_ensemble, selection, splits
from ..modeling import factors as factors_mod
from ..modeling import plan as plan_mod
from ..modeling.fit import HAZARD_FAMILIES, Candidate
from ..modeling.metrics import metric_direction
from . import pool_builder
from .base import Stage, attach_recommendation, register_stage
from .pool_builder import _TREES, FUNNEL_MODELS, MAX_LEADERS  # noqa: F401 (still reachable here)


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
        families = self.families(ctx)
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

        # --- each column's own transformation, fitted on the development sample -----------
        # (ADR-0024) A cap, a knee or a smooth curve is placed with the outcome in view, so it
        # is fitted here, after the split, on train_df alone; test and holdout get the same
        # frozen points. The library already left out a column not decided to enter as it is.
        expected = self._expected_signs(profile, plan)
        shaped, shape_report, shape_notes = column_transforms.library(
            train_df, coerce_target(train_df, cfg), features, plan["factors"].get("columns") or {},
            expected=expected, classification=is_clf, period=plan["factors"].get("period"))
        if shaped:
            train_df = factors_mod.apply_library(train_df, shaped)
            test_df = factors_mod.apply_library(test_df, shaped) if len(test_df) else test_df
            holdout_df = (factors_mod.apply_library(holdout_df, shaped) if len(holdout_df)
                          else holdout_df)
            library = [*library, *shaped]
        if shape_report:
            res.add_artifact(ctx.save_json("stages/model/transforms.json", shape_report))
        lib_notes = [*lib_notes, *shape_notes]

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
        factor_names = [f.name for f in library]
        fingerprint = self._fingerprint(cfg, train_df, y_train, factor_names, families,
                                        hazard_meta, plan, expected)
        splitter, split_note = splits.make_splitter(ev, n=len(train_df),
                                                    random_state=cfg.search.random_state)
        groups = (train_df[ev["group_col"]].to_numpy()
                  if ev["validation"] == "group_kfold" and ev.get("group_col") in train_df.columns
                  else None)

        # validation results the design gate's looks (or an earlier search) already computed
        cv_cache = self.cv_cache(ctx)

        def new_evaluator() -> selection.Evaluator:
            return selection.Evaluator(
                train_df, y_train, task=cfg.task.value, metric=rank_metric,
                is_classification=is_clf, splitter=splitter, groups=groups,
                metrics=tuple(dict.fromkeys([metric, *(r["metric"] for r in rank_by)])),
                test=test_df, y_test=y_test,
                score_test=any(r["sample"] == "test" for r in rank_by),
                random_state=cfg.search.random_state, time_budget_s=cfg.search.time_budget_s,
                cv_cache=cv_cache)

        found = self._cached_search(ctx, fingerprint, lambda: self._search(
            new_evaluator(), cfg, plan, library, expected, families, hazard_meta,
            protected, is_clf, shape))
        cv_cache.save()
        evaluator = new_evaluator()  # carries the cached evaluations; fits nothing again
        evaluator.cache = {e.id: e for e in found["evaluations"]}
        evaluator.order = [e.id for e in found["evaluations"]]
        rows, shortlist, paths = found["single_factor"], found["shortlist"], found["paths"]
        ranked = selection.rank(found["evaluations"], rank_by,
                                complexity_penalty=cfg.search.complexity_penalty,
                                n_total=len(shortlist), mode=ev.get("rank_mode", "order"))
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

            # every explored column, a transformed one too: the search builds on raw columns
            explored = list(dict.fromkeys(f.source for f in library
                                          if f.source in train_df.columns))
            gr = guided_search(
                lambda c: ctx.recommend("modeling_agent_search", c).model_dump(),
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

        # --- names a person can say; they stay with a model for the life of the run ------
        names = self._names(ctx, [*([guided_entry] if guided_entry is not None else []), *ranked])
        ctx.save_json("stages/model/ranking.json", [
            {"id": e.id, "name": names[e.id], "rank": i, "family": e.candidate.family,
             "features": list(e.candidate.features), "methods": list(e.methods),
             "score": e.value(rank_metric, rank_by[0]["sample"])}
            for i, e in enumerate(ranked, 1)])

        # --- the leaders and the pool (no champion is chosen here) ---------------------
        must = list(dict.fromkeys([*(plan["selection"].get("must_include") or []), *protected]))
        funnel = pool_builder.model_funnel(evaluator, ranked, plan, expected, found["corr"],
                                           must, names)
        res.add_artifact(ctx.save_json("stages/model/model_funnel.json", funnel))
        benchmark = set((ctx.get("ideate").payload or {}).get("benchmark_families") or []
                        if ctx.has("ideate") else [])
        leaders, lookup = pool_builder.leaders(evaluator, ranked, cfg, plan, expected,
                                               found["corr"], guided_entry, train_df, y_train,
                                               is_clf, must, benchmark)
        for entry in leaders:
            entry["name"] = names[entry["id"]]
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
            pool.append(pool_builder.finalize(
                ctx, lookup[cid], train_df, y_train, cfg, library, hazard_meta, etc,
                structural_spec, migration_model, migration_spec, is_clf, is_ts, rank_by,
                leaders, expected))
        for member in pool:
            member["source"] = pool_source
            member["name"] = names.get(member["id"], member["label"])

        # Pure-structural / pure-migration challenger benchmarks: the PD scored directly on the
        # test sample — labelled reference points, never a pool member.
        benchmark_looks = pool_builder.score_benchmarks(  # each is a look at the test sample
            test_df, y_test, is_clf, structural_info, structural_spec, migration_info,
            migration_spec)

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
            "transforms": {col: {"kind": r["kind"], "factor": r["factor"]["name"],
                                 "caption": r["caption"], "spec": r["spec"]}
                           for col, r in shape_report.items()},
            "single_factor": [r for r in rows if r["status"] == "kept"][:40],
            "screen": [{k: line[k] for k in ("rule", "setting", "before", "after")}
                       for line in found["screen"]],
            "exceptions": [{"factor": r["name"], "exception": r["exception"],
                            "reason": r["reason"], "overruled": r.get("overruled") or []}
                           for r in rows if r.get("exception")],
            "model_funnel": {"lines": [{k: v for k, v in line.items() if k != "dropped"}
                                       for line in funnel["lines"]],
                             "kept": funnel["kept"][:40], "n_models": funnel["n_models"],
                             "n_ranked": funnel["n_ranked"]},
            "rank_mode": plan["evaluation"].get("rank_mode", "order"),
            "blend": {e.id: e.info["blend"] for e in ranked[:40] if "blend" in e.info},
            "shortlist": shortlist,
            "expected_signs": expected,
            "paths": paths,
            "leaders": leaders,
            "pool": pool,
            "pool_ids": [m["id"] for m in pool],
            "evaluated_ids": [e.id for e in ranked],
            "names": {i: names[i] for i in {*(a["id"] for a in leaders), *(m["id"] for m in pool)}},
            "ranking_path": "stages/model/ranking.json",
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
            "test_evaluations": int(found["test_evaluations"]) + benchmark_looks,
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
        attach_recommendation(payload, ctx, "modeling_agent", rec)
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
            f"pool of {len(pool)} ({pool_source}): {', '.join(m['name'] for m in pool)}. "
            f"Best {rank_metric} ({rank_by[0]['sample'].replace('_', ' ')}) "
            f"{payload['best_score']:.4f}."
        )
        return res

    # --- the plan ------------------------------------------------------------------------
    @staticmethod
    def plan_in_force(ctx: RunContext, df: pd.DataFrame, features: list[str],
                      families: list[str] | None, *,
                      exceptions: dict[str, dict] | None = None) -> dict:
        """The plan this run works to: the developer's edit, the profile, the Design agent's
        proposal (recorded by ideate), the engine's recommendation, in that order.
        ``exceptions`` stands in for the screen exceptions in force (a preview's)."""
        from ..modeling.fit import DEFAULT_FAMILIES

        cfg = ctx.config
        families = families or DEFAULT_FAMILIES.get(cfg.task.value, DEFAULT_FAMILIES["regression"])
        ideate = ctx.get("ideate")
        proposed = ((ideate.payload or {}).get("plan_proposed") if ideate is not None else None)
        plan = plan_mod.resolve(cfg, plan_mod.shape_of(df, cfg), families=list(families),
                                columns=list(df.columns), proposed=proposed,
                                human=ctx.overrides.plan)
        # each column's transformation, as decided at the gates, wins (ADR-0024), and so do
        # the developer's exceptions to the screen (ADR-0025)
        column_transforms.lay_over(plan, ctx.overrides.transforms)
        return plan_mod.lay_exceptions(
            plan, ctx.overrides.screen_exceptions if exceptions is None else exceptions)

    @staticmethod
    def families(ctx: RunContext) -> list[str] | None:
        """The model families in force: the developer's slate, else ideate's, else the profile's."""
        families = ctx.get("ideate").payload.get("families") if ctx.has("ideate") else None
        if ctx.overrides.slate:  # the human's slate edit at gate_design wins
            families = list(dict.fromkeys(h["family"] for h in ctx.overrides.slate))
        return families or (ctx.config.search.model_families or None)

    @staticmethod
    def development(ctx: RunContext, plan: dict) -> tuple[pd.DataFrame, Any, dict[str, str]]:
        """The development sample exactly as this stage cuts it under ``plan`` (the library
        computed, then the split), its outcome and the expected signs. The test sample and the
        holdout are cut and dropped unread: the design gate fits its previews on this (ADR-0024)."""
        cfg = ctx.config
        df = ctx.load_dataset()
        profile = ctx.profile()
        features = list(profile["features"])
        frame, _, _ = factors_mod.build_library(
            df, features, plan["factors"], shape=plan["shape"]["shape"],
            datetime_col=cfg.data.datetime_col)
        dev, _, _ = splits.split_samples(frame, plan["evaluation"],
                                         datetime_col=cfg.data.datetime_col,
                                         random_state=cfg.search.random_state)
        return dev, coerce_target(dev, cfg), ModelStage._expected_signs(profile, plan)

    @staticmethod
    def development_library(ctx: RunContext, plan: dict
                            ) -> tuple[pd.DataFrame, Any, dict[str, str], list]:
        """The development sample with every factor the model stage would screen on it: the
        library, then the split, then each column's own transformation fitted on this sample
        alone. The engineered factors of the structural and migration frameworks join at the
        model stage itself."""
        cfg = ctx.config
        df = ctx.load_dataset()
        profile = ctx.profile()
        features = list(profile["features"])
        frame, library, _ = factors_mod.build_library(
            df, features, plan["factors"], shape=plan["shape"]["shape"],
            datetime_col=cfg.data.datetime_col)
        dev, _, _ = splits.split_samples(frame, plan["evaluation"],
                                         datetime_col=cfg.data.datetime_col,
                                         random_state=cfg.search.random_state)
        expected = ModelStage._expected_signs(profile, plan)
        y = coerce_target(dev, cfg)
        shaped, _, _ = column_transforms.library(
            dev, y, features, plan["factors"].get("columns") or {}, expected=expected,
            classification=cfg.task.is_classification, period=plan["factors"].get("period"))
        if shaped:
            dev = factors_mod.apply_library(dev, shaped)
            library = [*library, *shaped]
        return dev, y, expected, library

    @staticmethod
    def screen_preview(ctx: RunContext, plan: dict) -> dict:
        """The single-factor analysis the model stage will run under ``plan``, on the same
        development sample, for the design gate (ADR-0025). The test sample and the holdout are
        cut and dropped unread. The model stage runs it again and its result is the record."""
        cfg = ctx.config
        dev, y, expected, library = ModelStage.development_library(ctx, plan)
        ev = plan["evaluation"]
        splitter, _ = splits.make_splitter(ev, n=len(dev), random_state=cfg.search.random_state)
        groups = (dev[ev["group_col"]].to_numpy()
                  if ev["validation"] == "group_kfold" and ev.get("group_col") in dev.columns
                  else None)
        cv_cache = ModelStage.cv_cache(ctx)
        evaluator = selection.Evaluator(
            dev, y, task=cfg.task.value, metric=ev["rank_by"][0]["metric"],
            is_classification=cfg.task.is_classification, splitter=splitter, groups=groups,
            random_state=cfg.search.random_state, cv_cache=cv_cache)
        found = ModelStage.screen_factors(
            evaluator, cfg, plan, library, expected, [], cfg.task.is_classification,
            plan["shape"]["shape"])
        cv_cache.save()  # the model stage, and the next look, start from these
        keep = ("name", "source", "transform", "lag", "status", "dropped_by", "reason",
                "overruled", "exception", "sign", "p_value", "missing", "in_sample",
                "validation", "ranked_on", "note")
        return {"rows": [{k: r.get(k) for k in keep} for r in found["rows"]],
                "screen": found["screen"], "shortlist": found["shortlist"],
                "fallback": found["fallback"], "metric": evaluator.metric,
                "n_rows": int(len(dev)), "rules": plan_mod.screen_rules(plan),
                "engineered": bool(cfg.structural.enabled or cfg.migration.enabled)}

    @staticmethod
    def cv_cache(ctx: RunContext):
        """The run's memo of validation results (modeling/cv_cache.py): shared by every look
        at the single-factor analysis and by the search, so the same fit on the same rows is
        computed once. It changes how long they take, never what they find."""
        from ..modeling.cv_cache import CVCache

        return CVCache(ctx.resolve("stages/model/cv_cache.joblib"))

    @staticmethod
    def _names(ctx: RunContext, ranked: list) -> dict[str, str]:
        """Every evaluated model's name. Names given earlier in the run are kept (the register
        on disk), new models take the next free letter, and a name the developer gave wins."""
        path = ctx.resolve("stages/model/names.json")
        try:
            import json

            known = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            known = {}
        names = selection.assign_names(
            [(e.id, e.candidate.family, len(e.candidate.features)) for e in ranked], known)
        ctx.save_json("stages/model/names.json", names)
        return {**names, **{k: v for k, v in (ctx.overrides.names or {}).items() if k in names}}

    @staticmethod
    def _expected_signs(profile: dict, plan: dict) -> dict[str, str]:
        """The direction each explored column is expected to act in: the Data agent's
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

        rs = cfg.search.random_state
        screened = self.screen_factors(evaluator, cfg, plan, library, expected, protected,
                                       is_clf, shape)
        rows, log, shortlist = screened["rows"], screened["screen"], screened["shortlist"]
        fallback, corr, by_factor = (screened["fallback"], screened["corr"],
                                     screened["factor_signs"])
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
    def screen_factors(evaluator: selection.Evaluator, cfg, plan: dict, library: list,
                       expected: dict[str, str], protected: list[str], is_clf: bool,
                       shape: str) -> dict:
        """Every factor fitted alone on the development sample, then the screen. One function
        for the model stage and the design gate's preview, so the two cannot disagree
        (ADR-0025)."""
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
        for r in rows:  # the sample a factor's own strength is read on (the plan's choice)
            r["ranked_on"] = plan["factors"].get("criterion") or "validation"
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
        return {"rows": rows, "screen": log, "shortlist": shortlist, "fallback": fallback,
                "corr": corr, "factor_signs": by_factor}

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

    # --- the Modeling agent's pool (the leaders and finalizing a member: pool_builder.py) ---
    @staticmethod
    def _modeler_choice(ctx: RunContext, leaders: list[dict], plan: dict, n_tried: int,
                        fingerprint: str):
        """Ask the Modeling agent — or reuse its last answer when only the developer's pool changed."""
        from ..agents.contracts import ModelingAgentOutput

        prev = ctx.get("model")
        prev_rec = (prev.payload.get("recommendation") or {}) if prev else {}
        if ((ctx.overrides.pool or ctx.overrides.names) and prev is not None
                and prev.payload.get("search_fingerprint") == fingerprint
                and prev_rec.get("output") and not ctx.challenges_for("model")):
            try:
                out = ModelingAgentOutput.model_validate(prev_rec["output"])
                ctx.runner.last["modeling_agent"] = {**{k: v for k, v in prev_rec.items() if k != "output"},
                                              "reused": True}
                return out
            except Exception:  # a recommendation recorded under an older contract: ask again
                pass
        ev = plan["evaluation"]
        brief = [{k: a[k] for k in ("id", "label", "family", "n_features", "score")}
                 for a in leaders]
        # Tools at this stage are given the development sample only.
        tool_runs = ctx.consult_tools("modeling_agent", {"rank_by": ev["rank_by"], "leaders": brief})
        return ctx.recommend("modeling_agent", {
            "tool_runs": tool_runs,
            "rank_by": ev["rank_by"],
            "pool_size": ev["pool_size"],
            "interpretability": ctx.config.design.interpretability,
            "plan": {"evaluation": {k: ev[k] for k in ("test", "validation", "holdout_fraction")},
                     "selection": plan["selection"]},
            "leaders": leaders,
            "n_candidates_tried": n_tried,
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
