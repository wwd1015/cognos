"""Templates and the data step (ADR-0018): a run starts from a template or from nothing, the
developer gives the data once the intent is confirmed, and a finished run can be kept as a team
template."""

from __future__ import annotations

import pytest

from cognos import service, synth
from cognos.engine import GateError


def _csv(tmp_path, name: str, n: int = 240) -> str:
    path = tmp_path / f"{name}.csv"
    synth.GENERATORS["classification"](n=n).to_csv(path, index=False)
    return str(path)


def _small(cfg):
    """A template's setup with a search small enough for a test."""
    raw = cfg.model_dump(mode="json")
    raw["search"] = {**raw["search"], "max_candidates": 6, "cv_folds": 3}
    raw["backtest"] = {**raw["backtest"], "n_splits": 3}
    return type(cfg).from_dict(raw)


def test_builtin_templates_are_commercial_credit_setups_without_data(runs_dir):
    builtin = [t for t in service.list_templates(runs_dir) if t["origin"] == "built-in"]
    assert {t["id"] for t in builtin} == {
        "pd_logistic_scorecard", "pd_structural_simulation", "loss_forecast_timeseries",
        "rating_migration", "lgd_regression", "ead_ccf_regression"}
    for t in builtin:
        assert t["title"] and t["guide"].strip()  # what a person picks, and what data it needs
        assert dict(t["setup"])["For"] and "Governance" in dict(t["setup"])  # what it sets up
        cfg = service.template_config(t["ref"], "my model", root=runs_dir)
        assert not cfg.data.given and cfg.name == "my_model"
        assert cfg.template.id == t["id"] and cfg.template.origin == "built-in"
    with pytest.raises(KeyError):
        service.template_config("built-in:../../etc/passwd", root=runs_dir)


def test_a_run_from_a_template_asks_for_the_data_after_the_intent(tmp_path, runs_dir):
    cfg = _small(service.template_config("built-in:pd_logistic_scorecard", "pd", root=runs_dir))
    run_id = service.create_run(cfg, mode="autonomous", root=runs_dir)
    state = service.run_until_idle(run_id, runs_dir)
    # autonomous mode accepts the intent, but nobody can give the data for the developer
    assert state.status_of("gate_intent") == "done"
    assert state.status_of("gate_source") == "awaiting"
    assert state.status_of("explore") == "pending"

    with pytest.raises(GateError, match="give the data"):
        service.provide_data(run_id, {}, root=runs_dir, background=False)
    with pytest.raises(GateError, match="two or more"):
        service.provide_data(run_id, {"sources": [{"name": "a", "kind": "file",
                                                   "path": _csv(tmp_path, "a")}]},
                             root=runs_dir, background=False)

    first = _csv(tmp_path, "first")
    service.provide_data(run_id, {"source": {"kind": "file", "path": first}}, target="target",
                         root=runs_dir, by="dana", background=False)
    state = service.state(run_id, runs_dir)
    decision = state.last_decision("gate_source")
    assert decision.by == "dana" and decision.seat == "developer"
    assert decision.payload["data"] == {"source": {"kind": "file", "path": first}}
    assert state.overrides.data and state.overrides.target == "target"
    state = service.run_until_idle(run_id, runs_dir)
    assert state.status_of("explore") == "done"
    assert service.results(run_id, runs_dir)["explore"].payload["target"] == "target"


def test_new_data_starts_the_data_work_over(tmp_path, runs_dir, make_config):
    cfg = service.template_config(None, "blank", root=runs_dir)
    assert cfg.template is None and not cfg.data.given
    run_id = service.create_run(cfg, mode="interactive", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_intent", "accept", reason="test brief", root=runs_dir,
                        background=False)
    service.run_until_idle(run_id, runs_dir)
    service.provide_data(run_id, {"source": {"kind": "file", "path": _csv(tmp_path, "one")}},
                         root=runs_dir, background=False)
    state = service.run_until_idle(run_id, runs_dir)
    assert state.status_of("gate_data") == "awaiting"
    run_dir = service.runs_root(runs_dir) / run_id
    assert (run_dir / "data" / "dataset.parquet").exists()
    service.submit_gate(run_id, "gate_data", "accept", root=runs_dir, background=False)

    service.reopen(run_id, "gate_source", root=runs_dir)
    service.provide_data(run_id, {"source": {"kind": "file", "path": _csv(tmp_path, "two", 300)}},
                         root=runs_dir, background=False)
    state = service.state(run_id, runs_dir)
    assert state.status_of("explore") == "stale"
    assert "Provide the data" in state.steps["explore"].rerun_reason
    assert state.overrides.target is None and state.overrides.exclude_columns == []
    assert not (run_dir / "data" / "dataset.parquet").exists()  # the old snapshot went
    service.run_until_idle(run_id, runs_dir)
    assert service.results(run_id, runs_dir)["explore"].payload["n_rows"] == 300


def test_a_finished_run_is_kept_as_a_team_template(make_config, runs_dir, tmp_path):
    run_id = service.create_run(make_config("classification"), mode="autonomous", root=runs_dir)
    with pytest.raises(GateError, match="whole process"):
        service.save_template(run_id, "Too early", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    ref = service.save_template(run_id, "Our PD setup", "What we use for small books",
                                root=runs_dir, by="dana")
    assert ref == "team:our_pd_setup"
    listed = {t["ref"]: t for t in service.list_templates(runs_dir)}
    assert listed[ref]["origin"] == "team" and listed[ref]["saved_by"] == "dana"
    assert listed[ref]["from_run"] == run_id and run_id in listed[ref]["guide"]
    cfg = service.template_config(ref, "next", root=runs_dir)
    assert not cfg.data.given  # the data stays with the run it belongs to
    assert cfg.task.value == "classification" and cfg.metric.name == "roc_auc"
    assert not cfg.engagement.intent
    with pytest.raises(GateError, match="already has a template"):
        service.save_template(run_id, "Our PD setup", root=runs_dir)
