"""The end of the model stage: from the ranked ledger to the candidate pool (ADR-0016).

The leaders (the top of the ranking, one entry per distinct specification), the models through
the plan's rules, the engine's default pool, finalizing a pool member (fitted on the development
sample, the diagnostics battery, its scorer saved) and the challenger benchmarks of the
structural and migration frameworks. ``ModelStage`` (``stages/model.py``) calls these in order;
no champion is chosen here.
"""

from __future__ import annotations

from ..context import RunContext
from ..modeling import fit_full, score, selection
from ..modeling.fit import HAZARD_FAMILIES, Candidate
from ..modeling.metrics import metric_direction
from ..runtime.score import save_scorer
from . import stat_tests

_TREES = {"random_forest", "gradient_boosting"}
MAX_LEADERS = 20


FUNNEL_MODELS = 300  # distinct models the funnel follows, best ranked first


def leaders(evaluator, ranked, cfg, plan: dict, expected: dict, corr,
            guided_entry, train_df, y_train, is_clf: bool,
            must: list[str] | None = None,
            benchmark: set[str] | None = None) -> tuple[list[dict], dict]:
    """The top of the ranking, one entry per distinct specification, with what the Modeling agent
    and the developer need to pick a pool: scores, signs, and any constraint it breaks. A
    family whose framework is a benchmark only is fitted and ranked, never a champion."""
    benchmark = benchmark or set()
    ev = plan["evaluation"]
    rank_by = ev["rank_by"]
    first = rank_by[0]
    maximize = metric_direction(first["metric"]) == "maximize"
    required = cfg.design.interpretability == "required"
    pool_size = max(1, int(ev["pool_size"]))
    want = MAX_LEADERS  # the page shows these; the whole ranking is one click further
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
    def bench(family: str) -> bool:
        return (family in _TREES and required) or family in benchmark

    if all(bench(e.candidate.family) for e in chosen):
        interp = next((e for e in ranked if not bench(e.candidate.family)), None)
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
                                       _factor_signs(expected, cand), corr)
            info = evaluator.inference(e)
            signs = _signs(info.get("coefficients"), list(cand.features))
        elif cand.family not in _TREES and cand.family not in HAZARD_FAMILIES:
            try:
                X_fit = train_df
                if transforms:
                    from ..modeling.transforms import apply_transforms

                    X_fit, _, _ = apply_transforms(
                        train_df[e.info["base_features"]], transforms)
                fitted = fit_full(cand, X_fit, y_train, task=cfg.task.value,
                                  is_classification=is_clf)
                signs = _signs(fitted.coefficients(), list(cand.features))
            except Exception:  # signs are advisory; a failed refit just omits them
                signs = []
        for s in signs:
            s["expected"] = signs_by_factor.get(_source_of(s["feature"]), "none")
        missing = [f for f in must or [] if f not in cand.features and not transforms]
        if missing:
            bad = [*bad, f"does not carry {', '.join(missing)}, which every model must"]
        v = e.value(first["metric"], first["sample"])
        within = None
        if best_sd is not None and v is not None and best_v is not None:
            within = bool(v >= best_v - best_sd if maximize else v <= best_v + best_sd)
        role = "challenger" if bench(cand.family) else "candidate"
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
    engine_pool(leaders, pool_size, required)
    return leaders, lookup


def model_funnel(evaluator, ranked, plan: dict, expected: dict, corr,
                 must: list[str], names: dict[str, str]) -> dict:
    """The models through the plan's rules, best ranked first (ADR-0025): one entry per
    distinct specification, at most ``FUNNEL_MODELS`` of them. A family without
    coefficient tests is checked on correlation, inflation and required factors only."""
    seen, models = set(), []
    for e in ranked:
        key = (e.candidate.family, frozenset(e.candidate.features))
        if key in seen:
            continue
        seen.add(key)
        cand = e.candidate
        if e.info.get("transforms"):
            bad = []
        elif cand.family in selection.STEP_FAMILIES:
            bad = selection.violations(e, evaluator, plan["selection"],
                                       _factor_signs(expected, cand), corr)
        else:
            bad = selection.violations(e, evaluator, {
                k: plan["selection"].get(k) for k in ("max_correlation", "max_vif")},
                {}, corr)
        missing = [f for f in must if f not in cand.features and not e.info.get("transforms")]
        if missing:
            bad = [*bad, f"does not carry {', '.join(missing)}, which every model must"]
        models.append({"id": e.id, "name": names.get(e.id, e.id), "violations": bad})
        if len(models) >= FUNNEL_MODELS:
            break
    out = selection.model_funnel(models, {**plan["selection"],
                                          "must_include": must or None})
    return {**out, "n_models": len(models), "n_ranked": len(ranked)}


def engine_pool(leaders: list[dict], pool_size: int, required: bool) -> None:
    """The engine's default pool. Models that rank together but differ are what the
    outcomes analysis is for, so: first the leaders that differ (see ``alike``), are within
    one standard error of the best and break no constraint; then their near-copies; and
    only to reach two members, the next best outside that band."""
    def eligible(a: dict) -> bool:
        return a["role"] != "challenger"

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
    )  # a model that breaks a constraint is never added just to fill the pool
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


def _source_of(name: str) -> str:
    return name.split("__", 1)[0]


def _factor_signs(expected: dict, cand: Candidate) -> dict[str, str]:
    """Expected sign per factor of one model (a transformation that keeps the direction of
    its column inherits the column's sign)."""
    out = {}
    for name in cand.features:
        source, _, rest = name.partition("__")
        if name in expected:
            out[name] = expected[name]
        elif rest.startswith(("knee", "smooth")):
            out[name] = "+"  # turned to rise with the outcome (modeling/transforms.py)
        elif source in expected and not rest.startswith("square"):
            out[name] = expected[source]
    return out


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


def finalize(ctx: RunContext, e, train_df, y_train, cfg, library, hazard_meta, etc,
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
        signs = _signs(coefficients, list(cand.features))
        for s in signs:
            s["expected"] = expected.get(_source_of(s["feature"]), "none")
        entry = {"rank": None, "methods": list(e.methods), "coefficient_signs": signs,
                 "violations": [], "within_one_se": None, "engine_pick": False,
                 "role": ("challenger" if (cand.family in _TREES
                                           and cfg.design.interpretability == "required")
                          or cand.family in ((ctx.get("ideate").payload or {}).get(
                              "benchmark_families") or [] if ctx.has("ideate") else [])
                          else "candidate"),
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


def score_benchmarks(test_df, y_test, is_clf: bool, structural_info, structural_spec,
                     migration_info, migration_spec) -> int:
    """Pure-structural / pure-migration challenger benchmarks: the PD scored directly on the
    test sample — labelled reference points, never a pool member. Returns the looks at the
    test sample they took."""
    benchmark_looks = 0  # each is a look at the test sample, counted like any other
    if len(test_df) and is_clf:
        for info, spec in ((structural_info, structural_spec), (migration_info, migration_spec)):
            if info is None or spec is None:
                continue
            benchmark_looks += 1
            pd_alone = test_df[spec.pd_feature].to_numpy(dtype=float)
            auc = score("roc_auc", y_test, (pd_alone >= 0.5).astype(int), y_proba=pd_alone)
            info["benchmark"] = {
                "kind": ("pure structural Merton PD (challenger benchmark)"
                         if info is structural_info
                         else "pure rating-migration PD (challenger benchmark)"),
                "deployed": False, "sample": "test",
                "test_roc_auc": auc, "test_gini": 2 * auc - 1,
            }
    return benchmark_looks
