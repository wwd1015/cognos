"""Each column's transformation, decided in two steps (ADR-0024): whether at the data gate,
from a review read on every row explore has; how at the design gate, fitted on the development
sample alone. The model stage fits again on its own development sample and freezes the points."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cognos import service
from cognos.engine import GateError
from cognos.modeling import column_transforms as ct
from cognos.modeling import factors


def _bent(n: int = 6000, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Risk falls with coverage until 3, then stops falling; noise does nothing."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(1.0, 0.6, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.0 - 1.2 * np.minimum(x, 3.0))))).astype(float)
    return pd.DataFrame({"coverage": x, "noise": rng.normal(size=n)}), y


def test_a_knee_is_found_where_the_risk_stops_falling():
    df, y = _bent()
    fit = factors.fit_shape(df["coverage"], y, knees=2, direction="-")
    assert fit["knees_used"] == 1 and abs(fit["knots"][0] - 3.0) < 0.6
    assert fit["monotonic"] and fit["origin"] == "fitted"
    given = factors.fit_shape(df["coverage"], y, knees=0, knots=[2.0], direction="-")
    assert given["origin"] == "given" and given["knots"] == [2.0]


def test_the_review_recommends_a_kind_by_the_model_s_intent_never_a_parameter():
    df, y = _bent()
    rv = ct.review(df, y, ["coverage", "noise"], intent="rank", expected={"coverage": "-"})
    by = {c["column"]: c for c in rv["columns"]}
    assert by["coverage"]["recommended"]["kind"] == "knee"
    assert by["noise"]["recommended"]["kind"] == "as_is"  # flat: a shape would fit noise
    assert set(rv["kinds"]) == {"as_is", "cap", "knee", "smooth", "log", "custom"}
    assert "knots" not in by["coverage"] and "points" not in by["coverage"]  # 0/1: no cloud
    rng = np.random.default_rng(2)
    amount = rng.lognormal(0, 1.2, 3000)
    loss = 0.3 * np.log(amount) + rng.normal(0, 0.2, 3000)
    lin = ct.review(pd.DataFrame({"amount": amount}), loss, ["amount"], intent="linear",
                    expected={})
    assert lin["columns"][0]["recommended"]["kind"] == "log"
    assert [ct.form_key(f) for f in lin["columns"][0]["recommended"]["forms"]] == ["as_is", "log"]
    assert lin["columns"][0]["points"]  # a cloud of rows for an outcome that is not 0/1
    t = np.arange(200.0)
    drift = np.cumsum(rng.normal(0.5, 1, 200))
    ts = ct.review(pd.DataFrame({"hpi": drift}), np.diff(drift, prepend=0) + rng.normal(0, .1, 200),
                   ["hpi"], intent="stationary", expected={})
    rec = ts["columns"][0]["recommended"]
    assert rec["kind"] == "library" and "drifts" in rec["why"] and t.size
    assert [ct.form_key(f) for f in rec["forms"]][:6] == [
        "as_is", "as_is@1", "as_is@2", "diff", "diff@1", "diff@2"]
    assert ts["grid"]["lags"] == [0, 1, 2, 3, 4] and "pct" in ts["grid"]["bases"]


def test_a_kind_can_be_previewed_before_it_is_decided():
    df, y = _bent()
    knee = ct.sketch(df, y, "coverage", "knee", direction="-", classification=True)
    assert knee["curve"] and "a preview" in knee["caption"]
    plain = ct.sketch(df, y, "coverage", "as_is", direction="-", classification=True)
    assert plain["curve"][0][1] > plain["curve"][-1][1]  # the outcome falls with coverage
    assert ct.sketch(df, y, "coverage", "library", direction="-", classification=True) is None


def test_a_custom_expression_is_a_function_of_its_column_alone():
    df, y = _bent()
    assert ct.check_expr(" np.minimum(x,  3) ", "coverage") == "np.minimum(x, 3)"
    assert ct.check_expr("np.minimum(coverage, 3)", "coverage")  # its own name works too
    for bad in ("noise * 2", "y", "__import__('os')", "np.random.rand()", "open('f')", ""):
        with pytest.raises(ValueError):
            ct.check_expr(bad, "coverage")
    seen = ct.sketch(df, y, "coverage", "custom", direction="-", classification=True,
                     expr="np.minimum(x, 3)")
    assert seen["expr"] == "np.minimum(x, 3)" and seen["curve"][-1][1] == seen["curve"][-2][1]
    frame, lib, _ = factors.build_library(
        df, ["coverage"], {"columns": {"coverage": {"kind": "custom", "expr": "np.minimum(x, 3)"}}},
        shape="cross_section", datetime_col=None)
    f = next(f for f in lib if f.transform == "custom")
    assert f.name == "coverage__custom" and "coverage" not in {g.name for g in lib}
    assert np.allclose(frame[f.name], np.minimum(df["coverage"], 3))
    again = factors.apply_library(pd.DataFrame({"coverage": [1.0, 9.0, np.nan]}), [f.to_dict()])
    assert again[f.name].iloc[1] == 3.0 and np.isnan(again[f.name].iloc[2])


def test_fitted_factors_read_their_points_never_an_outcome():
    df, y = _bent()
    made, report, notes = ct.library(
        df, y, ["coverage", "noise"],
        {"coverage": {"kind": "knee"}, "noise": {"kind": "smooth", "span": 0.4}},
        expected={"coverage": "-"}, classification=True)
    assert {f.transform for f in made} == {"knee", "smooth"} and notes
    smooth = next(f for f in made if f.transform == "smooth")
    values = [p[1] for p in smooth.points]
    assert values == sorted(values) or values == sorted(values, reverse=True)
    other = pd.DataFrame({"coverage": [0.1, 2.0, 50.0, np.nan], "noise": [0, 1, 2, 3]})
    out = factors.apply_library(other, [f.to_dict() for f in made])  # no outcome column
    knee = next(f for f in made if f.transform == "knee")
    assert out[knee.name].isna().iloc[3] and out[knee.name].iloc[2] == knee.points[-1][1]
    assert factors.expected_sign({"name": knee.name, "transform": "knee", "source": "coverage"},
                                 {"coverage": "-"}) == "+"


def test_a_column_s_own_log_replaces_its_level_in_the_library():
    df, _ = _bent()
    frame, lib, _ = factors.build_library(
        df, ["coverage", "noise"], {"transforms": ["log"], "columns": {"coverage": {"kind": "log"},
                                                                       "noise": {"kind": "as_is"}}},
        shape="cross_section", datetime_col=None)
    names = {f.name for f in lib}
    assert "coverage__log" in names and "coverage" not in names  # replaced
    assert names >= {"noise"} and not any(n.startswith("noise__") for n in names)  # as it is


def test_a_column_enters_in_every_form_decided_each_a_variable_of_its_own():
    rng = np.random.default_rng(4)
    n = 80
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n, freq="QS"),
                       "hpi": np.cumsum(rng.normal(1, 1, n)) + 100,
                       "rate": rng.normal(5, 1, n)})
    forms = [ct.form_spec("as_is"), ct.form_spec("as_is", 1), ct.form_spec("diff"),
             ct.form_spec("diff", 2), ct.form_spec("log", 1)]
    frame, lib, _ = factors.build_library(
        df, ["hpi", "rate"], {"transforms": ["auto"], "lags": [1, 2],
                              "columns": {"hpi": {"forms": forms}, "rate": {"kind": "as_is"}}},
        shape="series", datetime_col="date")
    hpi = sorted(f.name for f in lib if f.source == "hpi")
    assert len(hpi) == 5 and "hpi" in hpi  # as it is is one of the forms decided
    assert {f.name for f in lib if f.source == "rate"} == {"rate"}  # nothing else for it
    lagged = next(f for f in lib if f.source == "hpi" and f.transform == "log")
    assert lagged.lag == 1 and np.isclose(frame[lagged.name].iloc[5],
                                          np.log(frame["hpi"].iloc[4]))
    # one form decided the old way reads as that form; the plan's lags as a driver's forms
    assert [ct.form_key(f) for f in ct.forms_of("log")] == ["log"]
    assert len(ct.forms_of({"kind": "library"})) == 6
    assert ct.decided({"x": {"forms": [ct.form_spec("as_is"), {"kind": "knee"}]}}) == {
        "x::knee": {**ct.normalize("knee"), "column": "x"}}


def _at_data_gate(runs_dir) -> str:
    cfg = service.template_config("built-in:pd_logistic_scorecard", "pd", root=runs_dir)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_intent", "accept", reason="t", root=runs_dir,
                        background=False)
    service.run_until_idle(run_id, runs_dir)
    info = service.data_info(run_id, runs_dir, with_catalog=True)
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(info["preselected"], root=runs_dir),
                        root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    return run_id


def test_two_steps_whether_at_the_data_gate_how_at_the_design_gate(runs_dir):
    run_id = _at_data_gate(runs_dir)
    review = service.results(run_id, runs_dir)["explore"].payload["transform_review"]
    rec = {c["column"]: c["recommended"]["kind"] for c in review["columns"]}
    assert rec["interest_coverage"] == "knee"
    forms = {c["column"]: [ct.form_key(f) for f in c["recommended"]["forms"]]
             for c in review["columns"]}
    assert forms["drawn"] == ["as_is", "log"]  # skewed: its log competes in the screen
    with pytest.raises(GateError, match="decide every variable"):
        service.submit_gate(run_id, "gate_data", "edit", {"transforms": {"interest_coverage":
                                                                         "knee"}},
                            root=runs_dir, background=False)
    with pytest.raises(GateError, match="not a form this model offers"):
        service.submit_gate(run_id, "gate_data", "edit", {"transforms": {**rec, "leverage":
                                                                         "change"}},
                            root=runs_dir, background=False)
    with pytest.raises(GateError, match="current_ratio"):
        service.submit_gate(run_id, "gate_data", "edit", {"transforms": {
            **rec, "current_ratio": {"kind": "custom", "expr": "leverage * 2"}}},
            root=runs_dir, background=False)
    with pytest.raises(GateError, match="at least one form"):
        service.submit_gate(run_id, "gate_data", "edit", {"transforms": {
            **rec, "leverage": {"forms": []}}}, root=runs_dir, background=False)
    service.submit_gate(run_id, "gate_data", "edit", {"transforms": {
        **rec, "leverage": {"forms": [{"kind": "as_is"}, {"kind": "cap"}, {"kind": "log"}]},
        "current_ratio": {"kind": "custom", "expr": "np.minimum(x, 1.3)"}}},
        root=runs_dir, by="dana", background=False)
    st = service.state(run_id, runs_dir)
    assert [ct.form_key(f) for f in st.overrides.transforms["leverage"]["forms"]] == [
        "as_is", "cap", "log"]
    assert st.last_decision("gate_data").by == "dana"
    service.run_until_idle(run_id, runs_dir)

    ideate = service.results(run_id, runs_dir)["ideate"].payload
    proposed = ideate["transforms"]  # only the forms that need fitting or testing
    assert proposed["interest_coverage::knee"]["knots"]
    assert proposed["leverage::cap"]["kind"] == "cap" and "leverage::log" not in proposed
    assert "current_ratio::custom" in proposed
    names = {r["name"] for r in ideate["screen"]["rows"]}
    assert {"leverage", "leverage__cap", "leverage__log"} <= names  # three forms, three variables
    entry = service.preview_transform(run_id, "interest_coverage::knee",
                                      {"kind": "knee", "knots": [2.5]}, runs_dir)
    assert entry["origin"] == "given" and entry["knots"] == [2.5]
    with pytest.raises(GateError, match="set every transformed variable"):
        service.submit_gate(run_id, "gate_design", "edit",
                            {"transform_params": {"interest_coverage::knee": {"knots": [2.5]}}},
                            root=runs_dir, background=False)
    every = {c: {"kind": e["kind"]} for c, e in proposed.items()}
    with pytest.raises(GateError, match="decided at the data gate"):  # one place decides kinds
        service.submit_gate(run_id, "gate_design", "edit", {"transform_params": {
            **every, "leverage::cap": {"kind": "smooth"}}}, root=runs_dir, background=False)
    with pytest.raises(GateError, match="inside the capped range"):
        service.submit_gate(run_id, "gate_design", "edit", {"transform_params": {
            **every, "interest_coverage::knee": {"kind": "knee", "knots": [9999.0]}}},
            root=runs_dir, background=False)
    service.submit_gate(run_id, "gate_design", "edit", {"transform_params": {
        **every, "interest_coverage::knee": {"kind": "knee", "knots": [2.5]}}},
        root=runs_dir, by="dana", background=False)
    knee = ct.forms_of(service.state(run_id, runs_dir).overrides.transforms["interest_coverage"])
    assert knee[0]["knots"] == [2.5]
    service.run_until_idle(run_id, runs_dir)

    model = service.results(run_id, runs_dir)["model"].payload
    screened = json.loads(service.read_artifact(run_id, "stages/model/single_factor.json",
                                                runs_dir))
    assert "current_ratio__custom" in {r["name"] for r in screened["rows"]}
    name = model["transforms"]["interest_coverage::knee"]["factor"]
    assert "(given)" in model["transforms"]["interest_coverage::knee"]["caption"]
    run = service.runs_root(runs_dir) / run_id
    train = pd.read_parquet(run / "data/train.parquet")
    test = pd.read_parquet(run / "data/test.parquet")
    assert name in train.columns and name in test.columns
    report = json.loads(service.read_artifact(run_id, "stages/model/transforms.json", runs_dir))
    knee = factors.Factor(**report["interest_coverage::knee"]["factor"])
    again = factors.apply_library(test.drop(columns=[name]), [knee])
    assert np.allclose(again[name], test[name], equal_nan=True)  # frozen: the same points


def test_accepting_takes_the_engine_s_kind_for_every_variable(runs_dir):
    run_id = _at_data_gate(runs_dir)
    review = service.results(run_id, runs_dir)["explore"].payload["transform_review"]
    service.submit_gate(run_id, "gate_data", "accept", root=runs_dir, background=False)
    st = service.state(run_id, runs_dir)
    assert {c: [ct.form_key(f) for f in s["forms"]]
            for c, s in st.overrides.transforms.items()} == {
        c["column"]: [ct.form_key(f) for f in c["recommended"]["forms"]]
        for c in review["columns"] if c["column"] not in st.overrides.exclude_columns}
