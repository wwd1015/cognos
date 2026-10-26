"""The data catalog (ADR-0020): the tables a team keeps, what each is for and how they join.
A run chooses from it once the intent is confirmed, adds data of its own, and the Data Analyst
proposes the join from measured links, the catalog's documented ones among them."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from cognos import catalog, linking, service
from cognos.config import JoinStep
from cognos.engine import GateError


def test_the_default_catalog_is_in_force_and_covers_every_template(runs_dir):
    cat = catalog.load(runs_dir)
    assert catalog.in_force(runs_dir) == (catalog.DEFAULT, "default")
    assert not catalog.team_file(runs_dir).exists()  # read from the package, never copied
    assert (catalog.folder(runs_dir) / catalog.DEFAULT_DATA / "obligor_quarters.parquet").exists()
    listed = {t["id"]: t for t in catalog.listing(runs_dir)["tables"]}
    for t in listed.values():  # every table measured, every key a key, every column described
        m = t["summary"]
        assert m["measured"] and m["n_rows"] > 0 and m["keys_unique"], t["id"]
        assert not m["undescribed"] and not m["missing_columns"], t["id"]
    assert set(cat.defaults) <= set(listed)
    for tpl in service.list_templates(runs_dir):
        cfg = service.template_config(tpl["ref"], "m", root=runs_dir)
        assert cfg.data.catalog and set(cfg.data.catalog) <= set(listed), tpl["id"]
        assert catalog.preselected(cfg, runs_dir) == cfg.data.catalog
    scratch = service.template_config(None, "m", root=runs_dir)
    assert catalog.preselected(scratch, runs_dir) == cat.defaults


def test_a_selection_becomes_the_runs_data(runs_dir, tmp_path):
    one = catalog.data_spec(["macro_quarterly"], root=runs_dir)
    assert one["source"]["catalog"] == "macro_quarterly" and one["datetime_col"] == "as_of_date"
    assert "nco_rate" in one["source"]["columns"]  # the column meanings travel with the run
    extra = tmp_path / "facilities.csv"
    pd.DataFrame({"obligor_id": ["OB00001"], "on_watch": [1]}).to_csv(extra, index=False)
    many = catalog.data_spec(["obligor_quarters", "facilities"],
                             [{"name": "facilities", "kind": "file", "path": str(extra)}],
                             root=runs_dir)
    assert [s["name"] for s in many["sources"]] == ["obligor_quarters", "facilities",
                                                    "facilities_2"]
    assert many["base"] == "obligor_quarters"
    assert [(lk["left"], lk["right"]) for lk in many["links"]] == [
        ("obligor_quarters", "facilities")]  # only links between the chosen tables
    with pytest.raises(KeyError):
        catalog.data_spec(["no_such_table"], root=runs_dir)
    with pytest.raises(GateError, match="choose a table"):
        service.data_payload([], root=runs_dir)


def test_a_documented_link_is_offered_and_measured_never_imposed():
    a = pd.DataFrame({"id": [1, 2, 3, 4], "when": ["q1"] * 4, "y": [0, 1, 0, 1]})
    b = pd.DataFrame({"code": [1, 2, 3, 9], "x": [1.0, 2.0, 3.0, 4.0]})
    documented = [JoinStep(left="a", left_on=["id"], right="b", right_on=["code"],
                           note="b is keyed on the same customer number"),
                  JoinStep(left="a", left_on=["gone"], right="b", right_on=["code"])]
    prof = linking.profile({"a": a, "b": b}, target="y", documented=documented,
                           described={"a": "the base"})
    link = next(lk for lk in prof["links"] if lk["a_on"] == ["id"])
    assert link["documented"] and link["note"].startswith("b is keyed")
    assert link["a_in_b"] == 0.75  # measured like any other
    assert "a.gone" in prof["unusable"][0]  # an out-of-date entry is said, not raised
    assert prof["tables"][0]["description"] == "the base"
    plan = linking.suggest(prof, target="y")
    assert plan["base"] == "a" and plan["joins"][0]["link"] == link["id"]


def test_a_run_takes_catalog_tables_and_the_choice_is_kept(runs_dir):
    cfg = service.template_config("built-in:loss_forecast_timeseries", "nco", root=runs_dir)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    info = service.data_info(run_id, runs_dir, with_catalog=True)
    assert info["preselected"] == ["macro_quarterly"]
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(["obligor_quarters", "macro_quarterly"],
                                             root=runs_dir),
                        root=runs_dir, by="dana", background=False)
    st = service.state(run_id, runs_dir)
    decision = st.last_decision("gate_source")
    assert decision.payload["catalog"] == ["obligor_quarters", "macro_quarterly"]
    assert decision.by == "dana"
    assert "2 sources (obligor_quarters, macro_quarterly)" in service.data_info(
        run_id, runs_dir)["given"]
    # revising the decision starts from the last choice
    service.reopen(run_id, "gate_source", root=runs_dir)
    again = service.data_info(run_id, runs_dir, with_catalog=True)
    assert again["preselected"] == ["obligor_quarters", "macro_quarterly"]


def test_explore_joins_catalog_tables_on_documented_links(runs_dir):
    cfg = service.template_config(None, "pd", root=runs_dir)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_intent", "accept", reason="test", root=runs_dir,
                        background=False)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(["obligor_quarters", "obligors"], root=runs_dir),
                        root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    p = service.results(run_id, runs_dir)["explore"].payload
    assert p["target"] == "default_12m"
    plan = p["linking"]["plan"]
    assert plan["base"] == "obligor_quarters" and len(plan["steps"]) == 1
    used = next(lk for lk in p["linking"]["links"] if lk["id"] == plan["steps"][0]["link"])
    assert used["documented"]
    assert "documents" in p["linking"]["rationale"]["joins"][used["id"]]


def test_a_team_catalog_replaces_the_default(runs_dir, tmp_path):
    folder = catalog.folder(runs_dir)
    folder.mkdir(parents=True)
    data = tmp_path / "loans.csv"
    pd.DataFrame({"loan_id": [1, 2], "bad": [0, 1]}).to_csv(data, index=False)
    (folder / "catalog.yaml").write_text(yaml.safe_dump({
        "title": "Our data", "defaults": ["loans"],
        "tables": [{"id": "loans", "title": "Loans", "keys": ["loan_id"],
                    "source": {"kind": "file", "path": str(data)},
                    "columns": {"loan_id": "Loan number."}}]}), encoding="utf-8")
    listed = catalog.listing(runs_dir)
    assert listed["title"] == "Our data" and [t["id"] for t in listed["tables"]] == ["loans"]
    assert listed["origin"] == "team"
    assert listed["tables"][0]["summary"]["undescribed"] == ["bad"]
    assert not (folder / catalog.DEFAULT_DATA).exists()  # the default's data is not made
    with pytest.raises(FileExistsError):
        catalog.init(runs_dir)  # never overwrites a team's catalog


def test_init_starts_a_team_catalog_from_the_default(runs_dir):
    path = catalog.init(runs_dir)
    assert path == catalog.team_file(runs_dir)
    assert catalog.in_force(runs_dir) == (path, "team")
    listed = catalog.listing(runs_dir)  # the copy reads the default's synthetic tables
    assert all(t["summary"]["measured"] for t in listed["tables"])


def test_the_catalog_names_the_outcome_and_the_identifiers_are_left_out(runs_dir):
    cfg = service.template_config("built-in:lgd_regression", "lgd", root=runs_dir)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    info = service.data_info(run_id, runs_dir, with_catalog=True)
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(info["preselected"], root=runs_dir),
                        root=runs_dir, background=False)
    st = service.run_until_idle(run_id, runs_dir)
    p = service.results(run_id, runs_dir)["explore"].payload
    # the base table's declared outcome, not the biggest number on the page
    assert p["target"] == "realized_lgd"
    assert "data catalog names it an outcome" in p["target_candidates"][0]["why"]
    assert {"facility_id", "obligor_id"} <= set(p["identifiers"])
    assert {"facility_id", "obligor_id"} <= set(st.overrides.exclude_columns)
    assert st.status == "completed"
