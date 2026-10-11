"""The development plan: how to split the data, how to shorten the factor library, and how
single factors become models (ADR-0016).

``recommend`` reads the shape of the data and the intent and returns a complete plan with a
reason for every choice and the options the data allow. The Design Lead may choose differently
among those options, the developer confirms or edits at the design gate, and ``resolve`` is the
one place that puts the layers together:

    developer's edit (``overrides.plan``)  >  the profile  >  the Design Lead  >  the engine

Nothing here measures anything. The plan says what will be done; ``splits``, ``factors`` and
``selection`` do it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import factors as factors_mod
from . import selection, splits
from .metrics import MAXIMIZE

SECTIONS = ("evaluation", "factors", "selection")
SAMPLES = ("validation", "in_sample", "test")
METRICS = {
    True: ("roc_auc", "average_precision", "log_loss", "brier", "accuracy", "f1"),
    False: ("rmse", "mae", "r2", "direction_accuracy"),
}
SEALED_MIN_ROWS = 2000  # below this a third sample costs more than it tells
MAX_ORIGINS = 12  # rolling forecast origins, at most


def shape_of(df: pd.DataFrame, cfg) -> dict[str, Any]:
    """What kind of sample this is: one row per period, repeated periods, or no time at all."""
    col = cfg.data.datetime_col
    n = int(len(df))
    out: dict[str, Any] = {"shape": "cross_section", "n_rows": n, "n_periods": None,
                           "period": None, "datetime_col": col}
    if col and col in df.columns:
        periods = int(df[col].nunique())
        out["n_periods"] = periods
        out["period"] = factors_mod.infer_period(df[col])
        out["shape"] = "series" if periods >= 0.98 * n else "panel"
    elif cfg.task is not None and cfg.task.value == "timeseries":
        out["shape"] = "series"  # declared a series; without a date column the row order is time
    return out


def options(shape: dict[str, Any], cfg, columns: list[str]) -> dict[str, list[str]]:
    """What this data allows. The gate and the Design Lead choose among these."""
    has_date = bool(shape.get("datetime_col"))
    is_clf = bool(cfg.task and cfg.task.is_classification)
    tests = ["random", "none"] if not has_date else ["out_of_time", "random", "none"]
    if shape["shape"] == "series":
        tests = ["out_of_time", "none"]
        validations = ["rolling_origin", "sliding_window", "none"]
    else:
        validations = ["stratified_kfold"] if is_clf else []
        validations += ["kfold", "none"]
        group = cfg.evaluation.group_col
        if group and group in columns:
            validations.insert(0, "group_kfold")
    return {"test": tests, "validation": validations, "methods": list(selection.METHODS),
            "factor_criterion": ["validation", "in_sample"],
            "criterion": list(selection.CRITERIA), "sample": list(SAMPLES),
            "metric": list(METRICS[is_clf]),
            "transforms": ["none", *(factors_mod.ALL_OPS if shape["shape"] == "series"
                                     else factors_mod.POINT_OPS)]}


def recommend(cfg, shape: dict[str, Any], *, families: list[str], columns: list[str]) -> dict:
    """The engine's plan for this data and this intent, with a reason for every choice."""
    ev_cfg, fc, sc = cfg.evaluation, cfg.factors, cfg.selection
    is_clf = bool(cfg.task and cfg.task.is_classification)
    kind, n = shape["shape"], int(shape["n_rows"])
    opts = options(shape, cfg, columns)
    reasons: dict[str, str] = {}
    sources: dict[str, str] = {}

    def choose(field: str, stated: Any, auto: Any, why: str) -> Any:
        if stated not in (None, "auto", [], ["auto"]):
            sources[field] = "profile"
            reasons[field] = "Stated in the profile."
            return stated
        sources[field] = "engine"
        reasons[field] = why
        return auto

    # --- the samples ------------------------------------------------------------------
    if kind == "series":
        test = choose("test", ev_cfg.test, "out_of_time",
                      "One row per period: a forecast is judged on periods after the ones it "
                      "was fitted on, so the latest periods are held back.")
        validation = choose("validation", ev_cfg.validation, "rolling_origin",
                            "A series cannot be shuffled. Each block is forecast from the "
                            "history before it, and the origin rolls forward.")
    elif kind == "panel":
        test = choose("test", ev_cfg.test, "out_of_time",
                      "The rows carry dates: the model will be used on periods after the ones "
                      "it was built on, so the latest ones are held back.")
        grouped = "group_kfold" in opts["validation"]
        validation = choose(
            "validation", ev_cfg.validation,
            "group_kfold" if grouped else ("stratified_kfold" if is_clf else "kfold"),
            (f"Rows of one {ev_cfg.group_col} stay in one fold, so a model is never validated "
             "on an entity it was fitted on.") if grouped else
            ("Folds keep the event rate. Rows of the same entity can sit on both sides of a "
             "fold: name evaluation.group_col to prevent it." if is_clf else
             "Shuffled folds inside the development periods."))
    else:
        test = choose("test", ev_cfg.test, "random",
                      "No date column: rows are held back at random.")
        validation = choose("validation", ev_cfg.validation,
                            "stratified_kfold" if is_clf else "kfold",
                            "Folds keep the event rate." if is_clf else "Shuffled folds.")
    test_fraction = ev_cfg.test_fraction if ev_cfg.test_fraction is not None \
        else cfg.search.holdout_fraction
    if test == "none":
        test_fraction = 0.0
    sealed_ok = n >= SEALED_MIN_ROWS and kind != "series" and test != "none"
    holdout_fraction = choose(
        "holdout_fraction", ev_cfg.holdout_fraction, round(test_fraction, 4) if sealed_ok else 0.0,
        (f"{n:,} rows are enough to seal a final sample. The candidates are compared on the "
         "test sample, so only a sample nobody chose on gives an unbiased last number.")
        if sealed_ok else
        (f"{n:,} {'periods' if kind == 'series' else 'rows'}: too few to hold back a third "
         "sample. The champion's test result is then also what it was chosen on, and "
         "validation says so."))
    folds = int(ev_cfg.folds or cfg.search.cv_folds)
    horizon = ev_cfg.horizon
    if validation in splits.ORDERED and not horizon:
        horizon = int(shape.get("period") or 0) or max(1, n // 20)
        reasons["horizon"] = ("One year per validation block." if shape.get("period")
                              else "About a twentieth of the sample per validation block.")
    if validation in splits.ORDERED and not ev_cfg.folds:
        # A handful of forecast origins all sit in the latest, similar years. Enough of them to
        # reach back over half the development sample covers more than one regime.
        n_dev = n * (1.0 - float(test_fraction) - float(holdout_fraction))
        folds = int(min(MAX_ORIGINS, max(folds, round(0.5 * n_dev / max(1, horizon)))))
        sources["folds"] = "engine"
        reasons["folds"] = (f"{folds} forecast origins, {horizon} period(s) each: together they "
                            "cover about the later half of the development sample, so a model "
                            "is not judged on one stretch of the cycle.")

    metric = cfg.metric.name
    rank_by = [{"metric": metric if r.metric == "auto" else r.metric, "sample": r.sample}
               for r in ev_cfg.rank_by]
    if rank_by:
        sources["rank_by"], reasons["rank_by"] = "profile", "Stated in the profile."
    else:
        first = "validation" if validation != "none" else "in_sample"
        rank_by = [{"metric": metric, "sample": first}]
        sources["rank_by"] = "engine"
        reasons["rank_by"] = (
            f"{metric} on the validation sample: every model is scored on rows it was not "
            "fitted on, and the test sample stays unused until the pool is compared."
            if first == "validation" else
            f"{metric} in sample: the plan has no validation sample.")

    # --- the factor library -----------------------------------------------------------
    transforms = factors_mod.resolve_transforms(list(fc.transforms), kind)
    stated_tf = "auto" not in fc.transforms
    sources["transforms"] = "profile" if stated_tf else "engine"
    reasons["transforms"] = "Stated in the profile." if stated_tf else (
        "One row per period: a driver may act with a delay or through its change, so every "
        "column is also tried lagged, as a change and as a year-on-year move."
        if kind == "series" else
        "Amounts with a long right tail are also tried as logarithms.")

    factor_criterion = choose(
        "factors.criterion", fc.criterion, "validation" if validation != "none" else "in_sample",
        "A factor fitted alone is ranked on rows it was not fitted on, so a factor that only "
        "fits its own sample does not lead the shortlist." if validation != "none" else
        "No validation sample: a factor is ranked on its in-sample fit.")

    # --- from single factors to models --------------------------------------------------
    steppable = [f for f in families if selection.steppable(f)]
    if not steppable:
        auto_methods, why = ["grid"], (
            "No family on the slate has coefficient tests, so a stepwise path has nothing to "
            "test: every family is fitted on the strongest factors and on all of them.")
    elif kind == "series":
        auto_methods, why = ["forward"], (
            "Few periods and many transformations: start from the best single factor and add one "
            "at a time while the model improves, so every factor's contribution is on record.")
    else:
        auto_methods, why = ["stepwise"], (
            "Add the factor that helps most, then drop any that no longer earns its place.")
    if steppable:
        why += (" One method is recommended. Choose another, or several: every model any of "
                "them evaluates is ranked in the same list, so the leaders are the best across "
                "the methods you ran.")
        if len(steppable) < len(families):
            why += (" Families on the slate without coefficient tests cannot be stepped; each "
                    "is fitted on the shortlist so it still has a candidate.")
    methods = choose("methods", [m for m in sc.methods if m != "auto"] or None, auto_methods, why)
    criterion = choose(
        "criterion", sc.criterion, "validation" if validation != "none" else "bic",
        "A step is kept when the ranking metric improves on rows the model was not fitted on."
        if validation != "none" else
        "No validation sample: a step is kept when BIC falls, which charges for every factor.")
    max_p = sc.max_p_value
    if max_p is None and kind == "series":
        max_p = 0.10
        reasons["max_p_value"] = ("With few periods a factor that is not significant is noise: "
                                  "every coefficient of an accepted model must have p ≤ 0.10.")

    plan = {
        "evaluation": {
            "test": test, "test_fraction": round(float(test_fraction), 4),
            "holdout_fraction": round(float(holdout_fraction), 4),
            "validation": validation, "folds": folds, "group_col": ev_cfg.group_col,
            "horizon": horizon, "window": ev_cfg.window,
            "rank_by": rank_by, "pool_size": int(ev_cfg.pool_size),
        },
        "factors": {
            "transforms": transforms, "lags": list(fc.lags), "windows": list(fc.windows),
            "period": fc.period or shape.get("period"),
            "criterion": factor_criterion,
            "max_missing": fc.max_missing, "max_p_value": fc.max_p_value,
            "enforce_sign": fc.enforce_sign, "expected_signs": dict(fc.expected_signs),
            "min_strength": fc.min_strength, "best_per_source": fc.best_per_source,
            "max_correlation": fc.max_correlation, "top_n": fc.top_n,
            "force_in": list(fc.force_in), "force_out": list(fc.force_out),
        },
        "selection": {
            "methods": list(methods), "criterion": criterion,
            "max_factors": int(sc.max_factors), "must_include": list(sc.must_include),
            "max_p_value": max_p,
            "enforce_sign": sc.enforce_sign, "max_correlation": sc.max_correlation,
            "max_models": int(sc.max_models),
        },
        "shape": shape, "options": opts, "reasons": reasons, "sources": sources,
    }
    return plan


def merge(plan: dict, patch: dict | None, source: str) -> dict:
    """Lay ``patch`` (sections of fields) over a plan, stamping where each field came from."""
    if not patch:
        return plan
    out = {**plan, **{s: dict(plan[s]) for s in SECTIONS},
           "sources": dict(plan.get("sources") or {}), "reasons": dict(plan.get("reasons") or {})}
    for section in SECTIONS:
        for field, value in (patch.get(section) or {}).items():
            if field not in out[section] or value == out[section][field]:
                continue
            out[section][field] = value
            # the screen and the model constraints share field names (max_p_value, ...)
            out["sources"][f"factors.{field}" if section == "factors" else field] = source
    return out


def problems(plan: dict) -> list[str]:
    """Why a plan cannot be run on this data (empty when it can)."""
    ev, sel, opts = plan["evaluation"], plan["selection"], plan.get("options") or {}
    shape = plan.get("shape") or {}
    out: list[str] = []
    if ev["test"] not in (opts.get("test") or splits.TESTS):
        out.append(f"test split {ev['test']!r} does not fit this data; choose from "
                   f"{opts.get('test')}")
    if ev["validation"] not in (opts.get("validation") or splits.VALIDATIONS):
        out.append(f"validation {ev['validation']!r} does not fit this data; choose from "
                   f"{opts.get('validation')}")
    if ev["validation"] in splits.ORDERED and ev["test"] == "random":
        out.append("a rolling forecast origin needs the rows in time order; a random test "
                   "sample breaks it")
    if ev["validation"] == "group_kfold" and not ev.get("group_col"):
        out.append("grouped k-fold needs evaluation.group_col")
    total = float(ev.get("test_fraction") or 0) + float(ev.get("holdout_fraction") or 0)
    if not 0 <= total < 0.8:
        out.append("the test and holdout samples together must leave at least a fifth of the "
                   "rows to build on")
    if ev.get("holdout_fraction") and shape.get("n_rows") and \
            int(shape["n_rows"] * ev["holdout_fraction"]) < 5:
        out.append("the sealed holdout would have fewer than 5 rows")
    if not ev.get("rank_by"):
        out.append("the ranking needs at least one line")
    known = set(METRICS[True]) | set(METRICS[False]) | MAXIMIZE
    for rule in ev.get("rank_by") or []:
        if rule.get("metric") not in known:
            out.append(f"unknown metric {rule.get('metric')!r}")
        if rule.get("sample") not in SAMPLES:
            out.append(f"unknown sample {rule.get('sample')!r}")
        if rule.get("sample") == "validation" and ev["validation"] == "none":
            out.append("the ranking reads the validation sample, but the plan has none")
        if rule.get("sample") == "test" and ev["test"] == "none":
            out.append("the ranking reads the test sample, but the plan has none")
    if int(ev.get("pool_size") or 0) < 1:
        out.append("the pool needs at least one candidate")
    methods = sel.get("methods") or []
    if not methods:
        out.append("choose at least one selection method")
    for m in methods:
        if m not in selection.METHODS:
            out.append(f"unknown selection method {m!r}")
    if sel.get("criterion") not in selection.CRITERIA:
        out.append(f"unknown criterion {sel.get('criterion')!r}")
    if plan["factors"].get("criterion") == "validation" and ev["validation"] == "none":
        out.append("single factors are ranked on the validation sample, but the plan has none")
    if sel.get("criterion") == "validation" and ev["validation"] == "none":
        out.append("the criterion reads the validation sample, but the plan has none")
    if int(sel.get("max_factors") or 0) < 1:
        out.append("a model needs at least one factor")
    return out


def resolve(cfg, shape: dict[str, Any], *, families: list[str], columns: list[str],
            proposed: dict | None = None, human: dict | None = None) -> dict:
    """The plan in force. A layer that cannot be run on this data is dropped and reported
    under ``ignored``; the developer's edit is checked at the gate, before it gets here."""
    plan = recommend(cfg, shape, families=families, columns=columns)
    profile_set = {f for f, src in plan["sources"].items() if src == "profile"}
    ignored: list[str] = []
    for patch, source in ((proposed, "agent"), (human, "human")):
        if not patch:
            continue
        if source == "agent":  # the profile outranks the Design Lead
            patch = {s: {k: v for k, v in (patch.get(s) or {}).items() if k not in profile_set}
                     for s in SECTIONS}
        trial = merge(plan, patch, source)
        bad = problems(trial)
        if bad:
            ignored.append(f"the {'Design Lead' if source == 'agent' else 'developer'}'s plan "
                           f"was not applied: {'; '.join(bad)}")
            continue
        plan = trial
    plan["ignored"] = ignored
    return plan


def summary(plan: dict) -> str:
    """One line a person can read."""
    ev, sel = plan["evaluation"], plan["selection"]
    parts = [f"test: {ev['test'].replace('_', ' ')}"
             + (f" {ev['test_fraction']:.0%}" if ev["test"] != "none" else "")]
    parts.append(f"validation: {ev['validation'].replace('_', ' ')}"
                 + (f" ×{ev['folds']}" if ev["validation"] != "none" else ""))
    parts.append(f"sealed holdout: {ev['holdout_fraction']:.0%}" if ev["holdout_fraction"]
                 else "no sealed holdout")
    parts.append("rank by " + ", then ".join(
        f"{r['metric']} ({r['sample'].replace('_', ' ')})" for r in ev["rank_by"]))
    parts.append("selection: " + " + ".join(m.replace("_", " ") for m in sel["methods"]))
    return "; ".join(parts)
