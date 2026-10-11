"""The development plan's mechanics (ADR-0016): cutting the samples, the factor library and its
screen, and the methods that turn single factors into multi-factor models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cognos import synth
from cognos.config import CognosConfig
from cognos.modeling import factors, plan, selection, splits
from cognos.modeling.fit import Candidate


# --- samples --------------------------------------------------------------------------------
def _dated(n=100):
    return pd.DataFrame({"date": pd.date_range("2000-03-31", periods=n, freq="QE"),
                         "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})


def test_three_samples_are_cut_from_the_end_in_time_order():
    df = _dated(100).sample(frac=1.0, random_state=3)  # arrives shuffled
    ev = {"test": "out_of_time", "test_fraction": 0.2, "holdout_fraction": 0.1,
          "validation": "rolling_origin"}
    dev, test, holdout = splits.split_samples(df, ev, datetime_col="date", random_state=1)
    assert (len(dev), len(test), len(holdout)) == (70, 20, 10)
    assert dev["date"].max() < test["date"].min() <= test["date"].max() < holdout["date"].min()
    assert dev["date"].is_monotonic_increasing


def test_a_plan_without_a_test_or_holdout_sample_cuts_none():
    ev = {"test": "none", "test_fraction": 0.2, "holdout_fraction": 0.0, "validation": "kfold"}
    dev, test, holdout = splits.split_samples(_dated(50), ev, datetime_col=None, random_state=1)
    assert (len(dev), len(test), len(holdout)) == (50, 0, 0)


def test_a_random_split_is_seeded_and_shuffled():
    ev = {"test": "random", "test_fraction": 0.25, "holdout_fraction": 0.0, "validation": "kfold"}
    a = splits.split_samples(_dated(80), ev, datetime_col="date", random_state=7)
    b = splits.split_samples(_dated(80), ev, datetime_col="date", random_state=7)
    assert a[1]["x"].tolist() == b[1]["x"].tolist() and len(a[1]) == 20
    assert not a[0]["date"].is_monotonic_increasing


def test_a_rolling_origin_always_validates_on_rows_after_the_ones_it_fitted():
    ev = {"validation": "rolling_origin", "folds": 6, "horizon": 4}
    splitter, note = splits.make_splitter(ev, n=80, random_state=1)
    blocks = splits.blocks(splitter, 80)
    assert len(blocks) == 6 and not note
    assert all(b["validate_rows"] == 4 and b["validate_from"] == b["fit_rows"] for b in blocks)
    assert [b["fit_rows"] for b in blocks] == sorted(b["fit_rows"] for b in blocks)  # expanding
    # a sliding window keeps a fixed-length history
    ev = {"validation": "sliding_window", "folds": 4, "horizon": 4, "window": 20}
    sliding = splits.blocks(splits.make_splitter(ev, n=80, random_state=1)[0], 80)
    assert {b["fit_rows"] for b in sliding} == {20}


def test_a_rolling_origin_shrinks_to_a_short_sample_and_says_so():
    splitter, note = splits.make_splitter({"validation": "rolling_origin", "folds": 10,
                                           "horizon": 8}, n=40, random_state=1)
    blocks = splits.blocks(splitter, 40)
    assert 2 <= len(blocks) < 10 and "forecast origin" in note
    assert splits.make_splitter({"validation": "none"}, n=40, random_state=1) == (None, "")


def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(20), 5)
    splitter, _ = splits.make_splitter({"validation": "group_kfold", "folds": 4}, n=100,
                                       random_state=1)
    for tr, va in splitter.split(np.zeros(100), None, groups):
        assert not set(groups[tr]) & set(groups[va])


# --- the plan -------------------------------------------------------------------------------
def _cfg(task="regression", **extra) -> CognosConfig:
    return CognosConfig.from_dict({"name": "p", "task": task,
                                   "data": {"path": "x.csv", "target": "y", **extra.pop("data", {})},
                                   **extra})


def test_the_plan_follows_the_shape_of_the_data():
    series = _dated(96)
    cfg = _cfg("timeseries", data={"datetime_col": "date"})
    shape = plan.shape_of(series, cfg)
    assert shape["shape"] == "series" and shape["period"] == 4
    p = plan.recommend(cfg, shape, families=["ols", "ridge"], columns=list(series.columns))
    ev, sel = p["evaluation"], p["selection"]
    assert (ev["test"], ev["validation"], ev["holdout_fraction"]) == ("out_of_time",
                                                                      "rolling_origin", 0.0)
    assert ev["horizon"] == 4 and ev["folds"] > cfg.search.cv_folds  # reaches back over the cycle
    assert sel["methods"] == ["forward"] and sel["max_p_value"] == 0.10  # one method, by default
    assert "ranked in the same list" in p["reasons"]["methods"]
    assert p["factors"]["criterion"] == "validation" and p["reasons"]["factors.criterion"]
    assert "lag" in p["factors"]["transforms"] and "kfold" not in p["options"]["validation"]
    assert all(p["reasons"][k] for k in ("test", "validation", "holdout_fraction", "methods"))
    assert plan.problems(p) == []

    # dated rows of many entities: out of time, stratified folds, grouped when a group is named
    panel = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-31", periods=10, freq="ME"), 300),
                          "obligor": np.tile(np.arange(300), 10), "y": 0})
    cfg = _cfg("classification", data={"datetime_col": "date"})
    p = plan.recommend(cfg, plan.shape_of(panel, cfg), families=["logit"], columns=list(panel))
    assert (p["shape"]["shape"], p["evaluation"]["test"]) == ("panel", "out_of_time")
    assert p["evaluation"]["validation"] == "stratified_kfold"
    assert p["evaluation"]["holdout_fraction"] > 0  # 3,000 rows can spare a sealed sample
    cfg = _cfg("classification", data={"datetime_col": "date"}, evaluation={"group_col": "obligor"})
    p = plan.recommend(cfg, plan.shape_of(panel, cfg), families=["logit"], columns=list(panel))
    assert p["evaluation"]["validation"] == "group_kfold"

    # no dates and no coefficient tests on the slate: a random test sample and the grid
    flat = pd.DataFrame({"x": np.arange(300.0), "y": np.arange(300.0)})
    cfg = _cfg("ml_regression")
    p = plan.recommend(cfg, plan.shape_of(flat, cfg), families=["random_forest"], columns=list(flat))
    assert (p["evaluation"]["test"], p["evaluation"]["validation"]) == ("random", "kfold")
    assert p["selection"]["methods"] == ["grid"] and p["factors"]["transforms"] == ["log"]


def test_plan_layers_the_profile_outranks_the_agent_and_the_developer_outranks_both():
    df = synth.make_regression_dataset(300)
    cfg = _cfg(evaluation={"validation": "none", "rank_by": [{"metric": "r2", "sample": "in_sample"}]},
               selection={"criterion": "bic"})
    shape = plan.shape_of(df, cfg)
    kw = {"families": ["ols"], "columns": list(df.columns)}
    agent = {"evaluation": {"validation": "kfold", "test": "none", "test_fraction": 0.0},
             "selection": {"methods": ["best_subset"]}}
    p = plan.resolve(cfg, shape, proposed=agent, **kw)
    assert p["evaluation"]["validation"] == "none" and p["sources"]["validation"] == "profile"
    assert p["evaluation"]["test"] == "none" and p["sources"]["test"] == "agent"
    assert p["selection"]["methods"] == ["best_subset"] and not p["ignored"]
    human = {"evaluation": {"validation": "kfold",
                            "rank_by": [{"metric": "rmse", "sample": "validation"}]}}
    p = plan.resolve(cfg, shape, proposed=agent, human=human, **kw)
    assert p["evaluation"]["validation"] == "kfold" and p["sources"]["validation"] == "human"
    # a layer the data cannot run is dropped and reported, never half-applied
    p = plan.resolve(cfg, shape, proposed={"evaluation": {"test": "out_of_time"}}, **kw)
    assert p["evaluation"]["test"] == "random" and "Design Lead" in p["ignored"][0]


@pytest.mark.parametrize(("patch", "why"), [
    ({"evaluation": {"validation": "rolling_origin"}}, "does not fit this data"),
    ({"evaluation": {"validation": "none"}}, "the plan has none"),
    ({"evaluation": {"test": "none"}, "_rank": "test"}, "the test sample, but the plan has none"),
    ({"evaluation": {"rank_by": [{"metric": "nope", "sample": "validation"}]}}, "unknown metric"),
    ({"evaluation": {"test_fraction": 0.6, "holdout_fraction": 0.3}}, "at least a fifth"),
    ({"selection": {"methods": []}}, "at least one selection method"),
    ({"selection": {"methods": ["magic"]}}, "unknown selection method"),
    ({"evaluation": {"pool_size": 0}}, "at least one candidate"),
])
def test_a_plan_that_cannot_be_run_says_why(patch, why):
    df = synth.make_regression_dataset(300)
    cfg = _cfg()
    p = plan.recommend(cfg, plan.shape_of(df, cfg), families=["ols"], columns=list(df.columns))
    if patch.pop("_rank", None):
        patch["evaluation"]["rank_by"] = [{"metric": "rmse", "sample": "test"}]
    assert any(why in x for x in plan.problems(plan.merge(p, patch, "human")))


# --- the factor library -----------------------------------------------------------------------
def test_the_library_widens_a_series_into_lags_changes_and_growth():
    df = _dated(60).assign(x=100 + np.arange(60.0) ** 2, flag=[0, 1] * 30,
                           sector=["a", "b", "c"] * 20)
    fc = {"transforms": ["lag", "diff", "yoy_diff", "yoy_pct", "ma"], "lags": [1, 2],
          "windows": [4]}
    frame, lib, notes = factors.build_library(df, ["x", "flag", "sector"], fc, shape="series",
                                              datetime_col="date")
    names = {f.name for f in lib}
    assert {"x", "x__lag2", "x__diff", "x__diff__lag1", "x__yoy_diff", "x__yoy_pct__lag2",
            "x__ma4", "flag", "sector"} <= names
    assert not [n for n in names if n.startswith(("flag__", "sector__"))]  # a flag enters as it is
    assert len(frame) == 60 - 6 and any("first 6 period(s)" in n for n in notes)  # yoy + lag 2
    assert frame[[f.name for f in lib if f.name != "sector"]].notna().all().all()
    row = frame.iloc[10]
    assert row["x__lag2"] == frame.iloc[8]["x"]
    assert row["x__diff"] == pytest.approx(row["x"] - frame.iloc[9]["x"])
    assert row["x__yoy_diff"] == pytest.approx(row["x"] - frame.iloc[6]["x"])  # four quarters back
    assert row["x__yoy_pct__lag2"] == pytest.approx(frame.iloc[8]["x"] / frame.iloc[4]["x"] - 1)

    # a growth rate of a series that crosses zero is not computed, and the library says so
    crossing = _dated(40).assign(x=np.linspace(-5, 5, 40))
    _, lib, notes = factors.build_library(crossing, ["x"], {"transforms": ["yoy_pct", "log"]},
                                          shape="series", datetime_col="date")
    assert [f.name for f in lib] == ["x"] and "not computable" in notes[0]


def test_a_cross_section_gets_logs_of_skewed_amounts_and_no_lags():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"revenue": np.exp(rng.normal(10, 1.2, 400)), "ratio": rng.normal(0, 1, 400)})
    assert factors.resolve_transforms(["auto"], "panel") == ["log"]
    assert factors.resolve_transforms(["lag", "diff", "sqrt"], "cross_section") == ["sqrt"]
    assert factors.resolve_transforms(["none"], "series") == []
    frame, lib, _ = factors.build_library(df, list(df.columns), {"transforms": ["auto"]},
                                          shape="cross_section", datetime_col=None)
    assert [f.name for f in lib] == ["revenue", "ratio", "revenue__log"] and len(frame) == 400


def test_serving_recomputes_a_factor_after_its_column_changed():
    df = _dated(30).assign(x=np.arange(30.0) + 10)
    spec = [factors.Factor("x__diff__lag1", "x", "diff", 1, 1).to_dict(),
            factors.Factor("x__log", "x", "log").to_dict()]
    first = factors.apply_library(df, spec, datetime_col="date")
    assert first["x__diff__lag1"].iloc[5] == 1.0 and np.isnan(first["x__diff__lag1"].iloc[0])
    assert factors.apply_library(first, spec, datetime_col="date") is first  # nothing missing
    shocked = first.assign(x=first["x"] * 2)
    again = factors.apply_library(shocked, spec, datetime_col="date", replace=True)
    assert again["x__diff__lag1"].iloc[5] == 2.0
    assert again["x__log"].iloc[3] == pytest.approx(np.log(26.0))


# --- single-factor analysis and the screen ----------------------------------------------------
def _screen_data(n=300, seed=4):
    rng = np.random.default_rng(seed)
    strong, noise, wrong = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    twin = strong + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"strong": strong, "twin": twin, "noise": noise, "wrong": wrong,
                       "strong__log": strong * 0.9 + rng.normal(0, 0.3, n),
                       "strong__sq": strong + rng.normal(0, 0.6, n),
                       "gappy": np.where(rng.random(n) < 0.4, np.nan, strong)})
    y = 2.0 * strong - 0.8 * wrong + rng.normal(0, 0.5, n)
    lib = [factors.Factor(c, c) for c in ("strong", "twin", "noise", "wrong", "gappy")]
    lib += [factors.Factor("strong__log", "strong", "log"),
            factors.Factor("strong__sq", "strong", "square")]
    return df, y, lib


def test_every_factor_is_fitted_alone_and_every_rule_says_what_it_dropped():
    df, y, lib = _screen_data()
    rows = factors.single_factor(df, y, lib, metric="rmse", is_classification=False)
    by = {r["name"]: r for r in rows}
    assert by["strong"]["sign"] == "+" and by["strong"]["p_value"] < 0.001
    assert by["wrong"]["sign"] == "-" and by["noise"]["p_value"] > 0.10
    assert by["gappy"]["missing"] == pytest.approx(df["gappy"].isna().mean())
    assert by["strong"]["in_sample"] < by["noise"]["in_sample"]  # rmse: lower is stronger

    rules = {"max_missing": 0.2, "max_p_value": 0.10, "enforce_sign": True,
             "best_per_source": 2, "max_correlation": 0.8, "top_n": 2, "force_out": []}
    numeric = df.drop(columns=["gappy"]).corr()
    kept, log = factors.screen(rows, rules, corr=numeric, expected={"wrong": "+"}, maximize=False)
    dropped = {d["factor"]: line["rule"] for line in log for d in line["dropped"]}
    assert dropped["gappy"] == "max_missing" and dropped["noise"] == "max_p_value"
    assert dropped["wrong"] == "enforce_sign"
    assert dropped["strong__sq"] == "best_per_source"  # the third transformation of one column
    assert dropped["twin"] == "max_correlation"
    assert kept == ["strong"] or kept[0] == "strong"
    assert [line["rule"] for line in log] == ["not_fitted", "max_missing", "max_p_value",
                                              "enforce_sign", "best_per_source",
                                              "max_correlation", "top_n"]
    assert all(line["before"] - line["after"] == len(line["dropped"]) for line in log)
    assert by["twin"]["status"] == "dropped" and "strong" in by["twin"]["reason"]
    # a square does not inherit its column's expected sign
    assert factors.expected_sign(by["strong__sq"], {"strong": "-"}) is None
    assert factors.expected_sign(by["strong__log"], {"strong": "-"}) == "-"


def test_the_developer_forces_a_factor_in_or_out_of_the_shortlist():
    df, y, lib = _screen_data()
    rows = factors.single_factor(df, y, lib, metric="rmse", is_classification=False)
    rules = {"max_p_value": 0.10, "top_n": 5, "force_in": ["noise"], "force_out": ["strong"]}
    kept, log = factors.screen(rows, rules, corr=None, expected={}, maximize=False)
    assert "noise" in kept and "strong" not in kept
    assert log[0]["rule"] == "force_out" and log[0]["dropped"][0]["factor"] == "strong"
    # a rule set to None is off
    kept, log = factors.screen(rows, {"max_p_value": None, "top_n": None}, corr=None,
                               expected={}, maximize=False)
    assert len(kept) == len(lib) and [line["rule"] for line in log] == ["not_fitted"]


def test_a_category_is_tested_as_a_whole():
    rng = np.random.default_rng(2)
    sector = rng.choice(["a", "b", "c"], 400)
    y = (rng.random(400) < np.where(sector == "a", 0.6, 0.2)).astype(int)
    df = pd.DataFrame({"sector": sector, "x": rng.normal(size=400)})
    rows = factors.single_factor(df, y, [factors.Factor("sector", "sector"),
                                         factors.Factor("x", "x")],
                                 metric="roc_auc", is_classification=True)
    cat = rows[0]
    assert cat["sign"] == "n/a" and cat["p_value"] < 0.001 and cat["in_sample"] > 0.6


# --- from single factors to models ------------------------------------------------------------
def _selection_data(n=240, seed=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    X["a_twin"] = X["a"] + rng.normal(0, 0.05, n)
    y = 3.0 * X["a"] + 2.0 * X["b"] + 1.0 * X["c"] + rng.normal(0, 0.6, n)
    return X, y.to_numpy()


def _evaluator(X, y, **kw):
    from sklearn.model_selection import KFold

    return selection.Evaluator(X, y, task="regression", metric="rmse", is_classification=False,
                               splitter=KFold(4, shuffle=True, random_state=0), **kw)


RULES = {"criterion": "validation", "max_factors": 4, "max_p_value": 0.05, "enforce_sign": True,
         "max_correlation": 0.8, "max_models": 400}
ORDER = ["a", "a_twin", "b", "c", "d", "e", "f"]  # the shortlist arrives strongest first


def _run(method, X, y, rules=RULES, expected=None, factors_=ORDER):
    ev = _evaluator(X, y)
    run = selection._Run(ev, "ols", factors_, rules, expected or {}, X.corr(), {"random_state": 0})
    return ev, selection._RUNNERS[method](run)


def test_forward_selection_adds_one_factor_at_a_time_while_it_helps():
    X, y = _selection_data()
    ev, path = _run("forward", X, y)
    assert [s["action"] for s in path.steps] == ["add", "add", "add"]
    assert [s["factor"] for s in path.steps][0] in ("a", "a_twin")
    assert set(path.steps[-1]["factors"]) - {"a_twin"} >= {"b", "c"}
    assert not {"a", "a_twin"} <= set(path.steps[-1]["factors"])  # two factors too alike
    assert all(s["change"] < 0 for s in path.steps[1:])  # rmse fell at every step kept
    assert path.steps[1]["rejected_by_constraints"] >= 1  # the twin was tried and refused
    assert "no further factor improves" in path.note and path.final == path.steps[-1]["model"]
    assert ev.cache[path.final].candidate.features == path.steps[-1]["factors"]


def test_backward_elimination_drops_what_does_not_earn_its_place():
    X, y = _selection_data()
    _, path = _run("backward", X, y, factors_=["a", "b", "c", "d", "e", "f"])
    assert path.steps[0]["action"] == "start" and path.steps[0]["n_factors"] == 6
    assert {s["factor"] for s in path.steps[1:]} <= {"d", "e", "f"}
    assert set(path.steps[-1]["factors"]) >= {"a", "b", "c"} and path.steps[-1]["n_factors"] <= 4


def test_stepwise_and_best_subset_land_on_the_true_factors():
    X, y = _selection_data()
    _, stepwise = _run("stepwise", X, y)
    ev, best = _run("best_subset", X, y, rules={**RULES, "max_factors": 3})
    for path in (stepwise, best):
        final = set(path.steps[-1]["factors"])
        assert final - {"a_twin"} | {"a"} == {"a", "b", "c"}
    assert [s["n_factors"] for s in best.steps] == [1, 2, 3]  # the best model of each size
    assert best.steps[0]["tried"] == 7 and best.steps[1]["rejected_by_constraints"] >= 1
    assert "best model of each size" in best.note


def test_a_specification_is_fitted_once_however_many_methods_reach_it():
    X, y = _selection_data()
    ev = _evaluator(X, y)
    paths = selection.run_methods(
        ev, families=["ols", "ridge"], factors=ORDER, expected={}, corr=X.corr(),
        rules={**RULES, "methods": ["forward", "best_subset"], "max_factors": 2},
        all_columns=ORDER, max_candidates=6, max_features=None, hazard_meta=None, random_state=0)
    assert [p.method for p in paths] == ["forward", "best_subset", "grid"]
    assert paths[-1].family == "ridge"  # no coefficient tests: the grid covers it
    assert len(ev.order) == len(set(ev.order)) == len(ev.cache)
    shared = [e for e in ev.cache.values() if {"forward", "best_subset"} <= set(e.methods)]
    assert shared, "the forward path lies inside the subsets searched"
    ranked = selection.rank(list(ev.cache.values()), [{"metric": "rmse", "sample": "validation"}])
    assert ranked[0].value("rmse", "validation") <= ranked[-1].value("rmse", "validation")
    ledger = selection.ledger(ev, ranked, [{"metric": "rmse", "sample": "validation"}])
    assert len(ledger) == len(ev.order) and ledger[0].description.startswith(ev.order[0])
    assert sum(r.status == "keep" for r in ledger) == 1


def test_a_candidate_id_follows_the_specification_not_its_place_in_the_search():
    a = Candidate("ols", ["x", "y"], {"random_state": 1})
    assert selection.candidate_id(a) == selection.candidate_id(
        Candidate("ols", ["y", "x"], {"random_state": 9}))
    assert selection.candidate_id(a) != selection.candidate_id(Candidate("ridge", ["x", "y"]))
    assert selection.candidate_id(a) != selection.candidate_id(Candidate("ols", ["x"]))


def test_constraints_reject_a_wrong_sign_and_an_insignificant_coefficient():
    X, y = _selection_data()
    ev = _evaluator(X, y)
    wrong = ev.evaluate(Candidate("ols", ["a", "d"], {}))
    bad = selection.violations(wrong, ev, RULES, {"a": "-"}, X.corr())
    assert any("a has sign +, expected -" in b for b in bad)
    assert any(b.startswith("d has p-value") for b in bad)
    alike = ev.evaluate(Candidate("ols", ["a", "a_twin"], {}))
    assert any("are correlated" in b for b in selection.violations(alike, ev, RULES, {}, X.corr()))
    clean = ev.evaluate(Candidate("ols", ["a", "b", "c"], {}))
    assert selection.violations(clean, ev, RULES, {"a": "+"}, X.corr()) == []
    assert ev.inference(clean)["bic"] < ev.inference(wrong)["bic"]


def test_an_information_criterion_can_drive_the_steps_without_a_validation_sample():
    X, y = _selection_data()
    ev = selection.Evaluator(X, y, task="regression", metric="rmse", is_classification=False)
    run = selection._Run(ev, "ols", ORDER, {**RULES, "criterion": "bic"}, {}, X.corr(), {})
    path = selection.forward(run)
    assert set(path.steps[-1]["factors"]) - {"a_twin"} | {"a"} == {"a", "b", "c"}
    assert all(s["change"] < 0 for s in path.steps[1:])  # BIC fell
    assert ev.cache[path.final].cv is None and "in_sample" in ev.cache[path.final].scores


def test_factors_every_model_must_carry_start_every_path():
    X, y = _selection_data()
    rules = {**RULES, "must_include": ["f"], "max_p_value": None, "max_factors": 3}
    _, path = _run("forward", X, y, rules=rules)
    assert path.steps[0]["action"] == "start" and path.steps[0]["factors"] == ["f"]
    assert all("f" in s["factors"] for s in path.steps)
    _, best = _run("best_subset", X, y, rules=rules)
    assert all("f" in s["factors"] for s in best.steps) and best.steps[0]["n_factors"] == 1


def test_the_test_sample_is_scored_only_when_the_ranking_reads_it_and_every_look_is_counted():
    X, y = _selection_data()
    quiet = _evaluator(X.iloc[:180], y[:180], test=X.iloc[180:], y_test=y[180:])
    quiet.evaluate(Candidate("ols", ["a"], {}))
    assert quiet.test_evaluations == 0 and "test" not in quiet.cache[quiet.order[0]].scores
    loud = _evaluator(X.iloc[:180], y[:180], test=X.iloc[180:], y_test=y[180:], score_test=True,
                      metrics=("r2",))
    for feats in (["a"], ["a", "b"], ["a"]):
        ev = loud.evaluate(Candidate("ols", feats, {}))
    assert loud.test_evaluations == 2  # a specification already scored is not scored again
    assert set(ev.scores) == {"validation", "in_sample", "test"} and "r2" in ev.scores["test"]


def test_every_model_gets_a_name_it_keeps():
    ranked = [("c1", "logit", 4), ("c2", "logit", 4), ("c3", "probit", 3), ("c4", "ridge_logit", 4)]
    names = selection.assign_names(ranked, {})
    assert names == {"c1": "Logit 4F-A", "c2": "Logit 4F-B", "c3": "Probit 3F-A",
                     "c4": "Ridge logit 4F-A"}
    # a re-search that ranks them differently and finds a new model changes no existing name
    again = selection.assign_names([("c9", "logit", 4), ("c2", "logit", 4), ("c1", "logit", 4)],
                                   names)
    assert again["c1"] == "Logit 4F-A" and again["c2"] == "Logit 4F-B"
    assert again["c9"] == "Logit 4F-C" and again["c3"] == "Probit 3F-A"
    many = selection.assign_names([(f"m{i}", "ols", 2) for i in range(28)], {})
    assert many["m25"] == "OLS 2F-Z" and many["m26"] == "OLS 2F-AA"
