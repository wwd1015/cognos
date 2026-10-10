"""Linked sources: the engine measures how tables connect, validates a join plan and runs it
without ever changing the base table's rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cognos import linking
from cognos.agents import checks, heuristic
from cognos.agents.contracts import CONTRACTS
from cognos.config import CognosConfig


@pytest.fixture
def tables():
    rng = np.random.default_rng(0)
    loans = pd.DataFrame({
        "loan_id": range(1000, 1400), "obligor_id": rng.integers(1, 120, 400),
        "amount": rng.normal(100, 10, 400), "default_flag": rng.integers(0, 2, 400),
        "term": rng.integers(1, 60, 400)})
    # the same key under another spelling and stored as text; ten obligors are missing
    obligors = pd.DataFrame({
        "OBLIGOR_ID": [str(i) for i in range(1, 110)], "leverage": rng.normal(3, 1, 109),
        "sector": rng.choice(list("abc"), 109), "age": rng.integers(1, 60, 109)})
    # several rows per obligor, the key stored as a float
    financials = pd.DataFrame({
        "obligor_id": np.repeat(np.arange(1, 100), 3).astype(float),
        "year": np.tile([2021, 2022, 2023], 99), "ebitda": rng.normal(5, 1, 297)})
    return {"loans": loans, "obligors": obligors, "financials": financials}


def _link(prof, a, b):
    return next(lk for lk in prof["links"] if {lk["a"], lk["b"]} == {a, b})


def test_profile_finds_keys_across_types_and_ignores_coincidence(tables):
    prof = linking.profile(tables)
    lo = _link(prof, "loans", "obligors")
    assert (lo["a_on"], lo["b_on"]) == (["obligor_id"], ["OBLIGOR_ID"])
    assert lo["relation"] == "many-to-one" and lo["same_name"] and 0.85 < lo["a_in_b"] < 0.95
    assert _link(prof, "obligors", "financials")["relation"] == "one-to-many"
    # loans.term and obligors.age overlap as small integers; neither identifies anything
    assert not any("term" in lk["a_on"] + lk["b_on"] or "age" in lk["a_on"] + lk["b_on"]
                   for lk in prof["links"])


def test_join_keeps_the_base_rows_and_reduces_many_to_one(tables):
    prof = linking.profile(tables)
    plan = linking.suggest(prof, target="default_flag")
    assert plan["base"] == "loans" and not plan["left_out"]
    keep = [{**j, "unmatched": "keep"} for j in plan["joins"]]
    drop = [{**j, "unmatched": "drop"} for j in plan["joins"]]
    steps, problems = linking.resolve(plan["base"], keep, prof)
    assert not problems
    df, report = linking.apply(tables, plan["base"], steps)
    assert len(df) == 400 and list(df["loan_id"]) == list(tables["loans"]["loan_id"])
    fin = next(s for s in report["steps"] if s["right"] == "financials")
    assert fin["reduced"] == "aggregate" and "financials__rows" in df.columns
    assert df["financials__rows"].dropna().eq(3).all()
    obl = next(s for s in report["steps"] if s["right"] == "obligors")
    assert obl["reduced"] == "none" and obl["match_rate"] == pytest.approx(
        tables["loans"]["obligor_id"].le(109).mean(), abs=1e-4)
    assert report["column_origin"]["leverage"] == "obligors"
    # the same plan dropping unmatched rows: fewer rows, every one counted, none added
    dropped, rep = linking.apply(tables, plan["base"],
                                 linking.resolve(plan["base"], drop, prof)[0])
    assert len(dropped) == 400 - sum(s["dropped_rows"] for s in rep["steps"]) < 400
    assert rep["base_rows"] == 400 and dropped[["leverage", "ebitda"]].notna().all().all()
    assert dropped["loan_id"].is_monotonic_increasing


def test_a_bad_plan_is_explained_not_run(tables):
    prof = linking.profile(tables)
    many = _link(prof, "loans", "financials")["id"]
    _, problems = linking.resolve("loans", [{"link": many, "many": "none"}, {"link": "L99"}], prof)
    assert any("several rows" in p for p in problems) and any("unknown link" in p for p in problems)
    assert linking.resolve("nowhere", [], prof)[1]
    steps, _ = linking.resolve("loans", [{"link": many, "many": "aggregate"}], prof)
    steps[0]["many"] = "refuse"
    with pytest.raises(linking.LinkError, match="several rows"):
        linking.apply(tables, "loans", steps)
    # two links to the same table: one link per added table
    both = [{"link": lk["id"], "many": "aggregate"} for lk in prof["links"]]
    assert any("already joined" in p for p in linking.resolve("loans", both, prof)[1])


def test_composite_key_is_offered_when_no_single_column_identifies_a_row():
    panel = pd.DataFrame({"firm": np.repeat(np.arange(50), 4), "year": np.tile(range(2020, 2024), 50),
                          "y": 1.0})
    ratios = pd.DataFrame({"firm": np.repeat(np.arange(50), 4), "year": np.tile(range(2020, 2024), 50),
                           "x": 2.0})
    prof = linking.profile({"panel": panel, "ratios": ratios})
    both = next(lk for lk in prof["links"] if len(lk["a_on"]) == 2)
    assert both["relation"] == "one-to-one"
    plan = linking.suggest(prof)
    steps, problems = linking.resolve(plan["base"], plan["joins"], prof)
    df, _ = linking.apply({"panel": panel, "ratios": ratios}, plan["base"], steps)
    assert not problems and len(df) == 200 and df["x"].notna().all()


def test_the_deterministic_analyst_passes_the_engine_checks(tables):
    prof = linking.profile(tables)
    sl = {"tables": prof["tables"], "links": prof["links"], "target": "default_flag",
          "base_table": "", "facts": {}, "challenges": []}
    raw = heuristic.data_linker(sl)
    out = CONTRACTS["data_linker"].model_validate(raw)
    assert checks.run_checks("data_linker", out, sl) == []
    # an agent that leaves the target's table out, or a table unexplained, is sent back
    bad = CONTRACTS["data_linker"].model_validate({**raw, "base_table": "obligors", "joins": []})
    errs = checks.run_checks("data_linker", bad, sl)
    assert any("target" in e for e in errs)


def test_config_accepts_several_sources_and_refuses_a_mixed_one():
    base = {"name": "m", "data": {"sources": [
        {"name": "loans", "kind": "snowflake", "table": "RISK.LOANS"},
        {"name": "fin", "kind": "file", "path": "fin.csv"}],
        "join": [{"left": "loans", "left_on": "id", "right": "fin", "right_on": "id"}]}}
    cfg = CognosConfig.from_dict(base)
    assert linking.is_linked(cfg) and cfg.data.join[0].left_on == ["id"]
    for broken in ({"path": "x.csv"}, {"base": "nowhere"},
                   {"sources": base["data"]["sources"] + [{"name": "fin", "kind": "file"}]}):
        with pytest.raises(ValueError):
            CognosConfig.from_dict({"name": "m", "data": {**base["data"], **broken}})


def test_named_sources_from_files_and_snowflake_lines():
    spec = linking.named_sources(["/tmp/2024 loans.csv"], "fin = RISK.FIN\nr = SELECT a FROM b")
    assert [s["name"] for s in spec["sources"]] == ["t_2024_loans", "fin", "r"]
    assert spec["sources"][1]["table"] == "RISK.FIN" and spec["sources"][2]["query"]
    with pytest.raises(linking.LinkError):
        linking.named_sources(["only_one.csv"])


def _book(n=2000, lost=0.05, risky=False, seed=1):
    """A loan book and a table of financials that misses ``lost`` of the obligors: at random,
    or (``risky``) mostly the ones that default."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    p_lost = np.where(y == 1, lost * 6, lost * 0.56) if risky else np.full(n, lost)
    gone = rng.random(n) < p_lost
    loans = pd.DataFrame({"obligor_id": np.arange(n), "default": y, "amount": rng.normal(size=n)})
    fin = pd.DataFrame({"obligor_id": np.arange(n)[~gone], "leverage": rng.normal(size=(~gone).sum())})
    return {"loans": loans, "financials": fin}


@pytest.mark.parametrize("kwargs, expect, says", [
    (dict(lost=0.04), "drop", "within chance"),             # few, and like the rest
    (dict(lost=0.04, risky=True), "keep", "would bias"),    # few, but the ones that default
    (dict(lost=0.30), "keep", "too many"),                  # many
])
def test_unmatched_rows_are_dropped_only_when_that_is_low_risk(kwargs, expect, says):
    tables = _book(**kwargs)
    prof = linking.profile(tables, target="default")
    lk = prof["links"][0]
    assert lk["a_outcome"]["unmatched_rows"] == 2000 - len(tables["financials"])
    plan = linking.suggest(prof, target="default")
    assert plan["joins"][0]["unmatched"] == expect and says in plan["joins"][0]["unmatched_reason"]
    sl = {"tables": prof["tables"], "links": prof["links"], "target": "default",
          "base_table": "", "facts": {}, "challenges": []}
    out = CONTRACTS["data_linker"].model_validate(heuristic.data_linker(sl))
    assert checks.run_checks("data_linker", out, sl) == []
    assert (expect == "keep") == any("missing values" in c for c in out.concerns)


def test_without_a_target_the_advice_says_the_outcome_was_not_compared():
    prof = linking.profile(_book(lost=0.04))
    assert prof["links"][0]["a_outcome"] is None
    choice = linking.suggest(prof)["joins"][0]
    assert choice["unmatched"] == "drop" and "could not be compared" in choice["unmatched_reason"]
