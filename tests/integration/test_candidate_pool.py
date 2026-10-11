"""The development plan end to end (ADR-0016): the samples the plan cuts, the pool the search
ends on, the outcomes analysis that tells the pool apart, and what stays sealed until the
champion is chosen."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cognos import service, synth
from cognos.agents import slices
from cognos.config import CognosConfig
from cognos.engine import Engine
from cognos.engine.graph import GATES
from cognos.orchestrator import Orchestrator
from cognos.runtime.score import load_scorer


def _accept_all(eng: Engine, until: str | None = None):
    for _ in range(24):
        state = eng.run_until_idle()
        waiting = [g for g in GATES if state.status_of(g) == "awaiting"]
        if not waiting or waiting[0] == until:
            return state
        eng.submit_gate(waiting[0], "accept", reason="looks right")
    return eng.state


# --- a quarterly loss series: the whole method on one page ------------------------------------
@pytest.fixture(scope="module")
def loss_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("loss")
    cfg = service.demo_config("loss_forecast", root)
    eng = Engine(cfg, runs_root=str(root))
    state = eng.run_until_idle()
    return eng, state


def test_a_series_gets_a_rolling_origin_and_a_factor_library(loss_run):
    eng, state = loss_run
    assert state.status == "completed"
    model = eng.results()["model"].payload
    ev = model["plan"]["evaluation"]
    assert (ev["test"], ev["validation"], ev["holdout_fraction"]) == ("out_of_time",
                                                                      "rolling_origin", 0.0)
    assert model["plan"]["sources"]["validation"] == "engine"
    assert "cannot be shuffled" in model["plan"]["reasons"]["validation"]
    blocks = model["samples"]["validation_blocks"]
    assert len(blocks) == ev["folds"] > 5 and all(b["validate_rows"] == 4 for b in blocks)
    assert all(b["validate_from"] == b["fit_rows"] for b in blocks)  # never fitted on the future
    assert model["samples"]["dropped_no_history"] == 8  # a year-on-year change, lagged four
    assert not (eng.run_dir / "data" / "holdout.parquet").exists()

    # hundreds of transformations in, a handful out, every rule accounted for
    assert model["library"]["n_factors"] > 150 and model["library"]["n_sources"] == 8
    assert 3 <= len(model["shortlist"]) <= 12
    screen = model["screen"]
    assert screen[0]["before"] == model["library"]["n_factors"]
    assert screen[-1]["after"] == len(model["shortlist"])
    assert all(a["after"] == b["before"] for a, b in zip(screen, screen[1:], strict=False))
    record = json.loads((eng.run_dir / "stages/model/single_factor.json").read_text("utf-8"))
    dropped = [r for r in record["rows"] if r["status"] == "dropped"]
    assert len(record["rows"]) == model["library"]["n_factors"]
    assert dropped and all(r["dropped_by"] and r["reason"] for r in dropped)
    wrong = [r for r in dropped if r["dropped_by"] == "enforce_sign"]
    assert wrong and all("expected" in r["reason"] for r in wrong)


def test_models_are_built_one_factor_at_a_time_and_the_path_is_on_record(loss_run):
    eng, _ = loss_run
    model = eng.results()["model"].payload
    paths = {p["method"]: p for p in model["paths"]}
    assert {"forward", "best_subset"} <= set(paths)
    forward = paths["forward"]["steps"]
    assert [s["n_factors"] for s in forward] == [1, 2, 3]  # the plan allows three
    assert all(s["action"] == "add" and s["factor"] in s["factors"] for s in forward)
    assert all(s["change"] < 0 for s in forward[1:])  # validation RMSE fell with each addition
    assert forward[1]["rejected_by_constraints"] > 0  # significance, sign or correlation said no
    assert [s["n_factors"] for s in paths["best_subset"]["steps"]] == [1, 2, 3]
    # forward selection is checked against every combination: here they agree
    assert set(forward[-1]["factors"]) == set(paths["best_subset"]["steps"][-1]["factors"])
    # the three true drivers are found, each through a lag or a change, not its level
    sources = {f.split("__")[0] for f in forward[-1]["factors"]}
    assert sources == {"unemployment_rate", "bbb_spread", "hpi"}
    assert all("__" in f for f in forward[-1]["factors"])
    assert model["n_candidates_tried"] > 100 and model["test_evaluations"] == 0
    ledger = json.loads((eng.run_dir / "stages/model/ledger.json").read_text("utf-8"))
    assert len(ledger) == model["n_candidates_tried"]


def test_the_pool_is_told_apart_on_forecasts_and_scenarios(loss_run):
    eng, _ = loss_run
    model, backtest = eng.results()["model"].payload, eng.results()["backtest"].payload
    assert "champion" not in model and 2 <= len(model["pool"]) <= 4
    assert backtest["evaluation_sample"] == "out_of_time"
    cands = backtest["candidates"]
    assert [c["id"] for c in cands] != [] and {c["id"] for c in cands} == set(model["pool_ids"])
    rmse = [c["metrics"]["rmse"] for c in cands]
    assert rmse == sorted(rmse)  # best first on the run's metric
    for c in cands:
        assert {"rmse", "mae", "r2", "bias", "mean_actual", "mean_forecast"} <= set(c["outcomes"])
        adverse, severe = (next(s for s in c["scenarios"] if s["name"] == n)
                           for n in ("adverse", "severely_adverse"))
        assert severe["delta"] > adverse["delta"] > 0  # losses rise with the shock, more with more
    assert len({round(c["scenarios"][1]["delta"], 3) for c in cands}) > 1  # and not by the same
    champion = next(c for c in cands if c["id"] == backtest["champion_id"])
    assert champion["outcomes"]["r2"] > 0.8 and backtest["test_evaluations"] == len(cands)
    assert backtest["forecast_analysis"] == champion["outcomes"]
    # near-ties go to the simpler model: three factors, not the nine-factor lasso beside it
    assert champion["n_features"] == 3 and "fewer factors" in backtest["champion_rationale"]

    series = json.loads((eng.run_dir / backtest["forecasts_path"]).read_text("utf-8"))
    assert set(series["forecast"]) == set(model["pool_ids"])
    assert len(series["actual"]) == len(series["x"]) == model["n_test"]
    train = pd.read_parquet(eng.run_dir / "data" / "train.parquet")
    assert min(series["x"]) > str(train["date"].max())  # every forecast is out of time


def test_the_champion_scores_raw_columns_by_recomputing_its_factors(loss_run):
    eng, _ = loss_run
    view = eng.context().champion()
    assert view["champion_decided"]
    bundle = load_scorer(view["scorer_path"])
    assert bundle.factors and {f["name"] for f in bundle.factors} <= set(view["champion"]["features"])
    test = pd.read_parquet(eng.run_dir / "data" / "test.parquet")
    raw = pd.read_parquet(eng.run_dir / "data" / "dataset.parquet")  # explored columns only
    assert not set(view["champion"]["features"]) & set(raw.columns)
    scores = bundle.score_frame(raw.iloc[20:])
    assert np.isnan(scores[0]) and np.isfinite(scores[-len(test):]).all()  # no history: no score
    np.testing.assert_allclose(scores[-len(test):], bundle.score_frame(test), rtol=1e-9)


def test_validation_says_what_no_sealed_sample_means_and_the_paper_records_the_method(loss_run):
    eng, _ = loss_run
    validate = eng.results()["validate"]
    assert validate.verdict.value in ("PASS", "WARN")
    finding = next(f for f in validate.findings if f.id == "no-holdout")
    assert finding.severity.value == "LOW" and "also what it was chosen on" in finding.message
    assert validate.payload["holdout_metric"] is None and validate.payload["holdout_scored"] == []
    method = (eng.run_dir / "docs" / "methodology.md").read_text(encoding="utf-8")
    for heading in ("## The development plan", "## Single-factor analysis",
                    "## Multi-factor selection", "## Candidate pool"):
        assert heading in method
    assert "rolling origin" in method and "Forward" in method
    assert eng.results()["backtest"].payload["champion_id"] in method
    assert eng.results()["review"].verdict.value == "PASS"  # every code anchor still resolves


# --- a sealed holdout: scored once, after the choice ------------------------------------------
@pytest.fixture
def sealed_config(tmp_path):
    df = synth.make_cni_portfolio_dataset(n=2400)
    csv = tmp_path / "cni.csv"
    df.to_csv(csv, index=False)
    return CognosConfig.from_dict({
        "name": "sealed", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome", "default_quarter"]},
        "search": {"max_candidates": 6, "cv_folds": 3, "model_families": ["logit"]},
    })


def test_the_sealed_holdout_is_scored_once_after_the_choice_and_a_revised_choice_is_counted(
        sealed_config, runs_dir):
    eng = Engine(sealed_config, runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    model, backtest = eng.results()["model"].payload, eng.results()["backtest"].payload
    assert model["plan"]["evaluation"]["holdout_fraction"] == 0.2
    assert "2,400 rows are enough" in model["plan"]["reasons"]["holdout_fraction"]
    assert (model["n_train"], model["n_test"], model["n_holdout"]) == (1440, 480, 480)
    holdout = pd.read_parquet(eng.run_dir / "data" / "holdout.parquet")
    test = pd.read_parquet(eng.run_dir / "data" / "test.parquet")
    assert test["vintage"].max() <= holdout["vintage"].min()  # the latest rows are the sealed ones
    assert backtest["sealed_holdout"] and backtest["scored_rows"] == 480
    assert eng.state.status_of("validate") == "pending"  # nothing has opened the holdout yet
    ctx = eng.context()
    for agent in ("modeler", "outcomes_analyst"):  # nobody who chooses can cite it
        facts = slices.build(ctx, agent, {})["facts"]
        assert not [k for k in facts if k.startswith("validate.") or "holdout_metric" in k]
    assert not [k for k in slices.build(ctx, "modeler", {})["facts"] if k.startswith("backtest.")]

    eng.submit_gate("gate_champion", "accept", reason="agreed")
    eng.run_until_idle()
    first = eng.results()["validate"].payload
    assert 0.5 < first["holdout_metric"] <= 1.0 and first["holdout_evaluations"] == 1
    assert first["holdout_scored"] == [backtest["champion_id"]]
    assert not [f for f in eng.results()["validate"].findings if f.id in ("no-holdout",
                                                                         "holdout-reuse")]
    assert eng.context().champion()["holdout_metric"] == first["holdout_metric"]

    # choosing again after the holdout was opened is allowed, visible and counted
    other = next(c["id"] for c in backtest["candidates"] if c["id"] != backtest["champion_id"])
    eng.reopen("gate_champion")
    eng.submit_gate("gate_champion", "override", {"champion": other}, reason="simpler to explain")
    _accept_all(eng, until="gate_validation")
    second = eng.results()["validate"]
    assert second.payload["holdout_scored"] == [backtest["champion_id"], other]
    assert second.payload["holdout_evaluations"] == 2
    assert any(f.id == "holdout-reuse" for f in second.findings)
    assert eng.results()["backtest"].payload["test_evaluations"] == 2 * len(backtest["candidates"])


# --- other ways to slice ---------------------------------------------------------------------
def test_a_clean_train_test_split_with_no_validation_ranks_in_sample(make_config, runs_dir):
    cfg = make_config("regression",
                      evaluation={"validation": "none", "test": "random", "holdout_fraction": 0,
                                  "rank_by": [{"metric": "r2", "sample": "in_sample"}]},
                      selection={"methods": ["forward"], "criterion": "adj_r2"})
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run(["explore", "ideate", "model", "backtest", "validate"])
    model = orch.ctx.require("model").payload
    assert model["plan"]["sources"]["validation"] == "profile"
    assert model["samples"]["validation_blocks"] == [] and model["oof_perf_path"] is None
    assert model["rank_metric"] == "r2" and model["rank_by"][0]["sample"] == "in_sample"
    assert all(m["cv_mean"] is None and "validation" not in m["scores"] for m in model["pool"])
    assert all(m["scores"]["in_sample"]["r2"] > 0.5 for m in model["pool"])
    path = model["paths"][0]
    assert path["criterion"] == "adj_r2" and all(s["change"] > 0 for s in path["steps"][1:])
    assert "rank_score" in orch.ctx.require("model").metrics  # r2 and rmse point opposite ways
    assert orch.ctx.require("backtest").payload["oos_metric"] is not None


def test_ranking_on_the_test_sample_is_allowed_and_every_look_is_counted(make_config, runs_dir):
    cfg = make_config("regression", evaluation={
        "rank_by": [{"metric": "rmse", "sample": "test"}, {"metric": "rmse", "sample": "validation"}]})
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run(["explore", "ideate", "model", "backtest"])
    model = orch.ctx.require("model").payload
    assert model["test_evaluations"] == model["n_candidates_tried"] > 0
    assert all("rmse" in m["scores"]["test"] for m in model["pool"])
    scores = [a["score"] for a in model["leaders"]]
    assert scores == sorted(scores)


def test_a_plan_without_a_test_sample_is_run_and_flagged(make_config, runs_dir):
    cfg = make_config("regression", evaluation={"test": "none"})
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run(["explore", "ideate", "model", "backtest", "validate"])
    assert orch.ctx.require("model").payload["n_test"] == 0
    assert not (orch.ctx.run_dir / "data" / "test.parquet").exists()
    backtest = orch.ctx.require("backtest")
    assert backtest.payload["evaluation_sample"] == "development"
    assert backtest.payload["test_evaluations"] == 0
    assert any(f.id == "no-test-sample" for f in backtest.findings)
    finding = next(f for f in orch.ctx.require("validate").findings if f.id == "no-holdout")
    assert finding.severity.value == "MEDIUM"


def test_an_agents_plan_is_checked_against_what_the_data_allow(make_config, runs_dir, replay_dir):
    first = Orchestrator(make_config("regression"), runs_root=runs_dir)
    first.run(["explore", "ideate"])
    answer = first.ctx.require("ideate").payload["recommendation"]["output"]
    assert answer["plan"] is None  # the deterministic Design Lead accepts the engine's plan

    def lead(plan):
        return {**answer, "plan": plan}

    good = {"test": "random", "validation": "kfold", "sealed_holdout": False,
            "methods": ["best_subset"], "criterion": "bic", "rationale": "small, linear, explainable"}
    bad = {**good, "validation": "rolling_origin"}
    cfg = make_config("regression", agents={"replay_dir": replay_dir({"design_lead": [lead(bad),
                                                                                    lead(good)]})})
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run(["explore", "ideate", "model"])
    audit = [json.loads(x) for x in (orch.ctx.agents_dir / "audit.jsonl").read_text(
        encoding="utf-8").splitlines() if '"design_lead"' in x]
    assert [a["status"] for a in audit] == ["invalid", "ok"]
    assert "does not fit this data" in audit[0]["error"]
    plan = orch.ctx.require("ideate").payload["plan"]
    assert plan["selection"]["methods"] == ["best_subset"] and plan["sources"]["methods"] == "agent"
    assert plan["selection"]["criterion"] == "bic"
    assert orch.ctx.require("model").payload["paths"][0]["method"] == "best_subset"


def test_fair_lending_is_measured_out_of_sample_without_a_sealed_holdout(make_config, runs_dir):
    """800 rows seal nothing: the disparate-impact test reads the test sample, not nothing."""
    orch = Orchestrator(make_config("credit"), runs_root=runs_dir)
    orch.run(["explore", "ideate", "model", "backtest", "validate", "comply"])
    assert orch.ctx.require("model").payload["n_holdout"] == 0
    fair = orch.ctx.require("comply").payload["fair_lending"]
    assert fair["disparate_impact"] is not None and fair["group_rates"]
    assert not [f for f in orch.ctx.require("comply").findings if f.id == "fair-lending-unavailable"]
