"""Re-opening an earlier decision. The work after it is on hold, not undone: the same decision
again leaves it standing, a changed one sends it back to a person with the earlier choices
offered where they still apply. Express preparation never replaces a person's decision."""

from __future__ import annotations

import json

from cognos import service
from cognos.engine.gates import carried_forward
from cognos.engine.process import shown_status, waits_on
from cognos.ui.panels import stage_panel

FIRST = ["obligor_quarters", "obligors"]
SECOND = ["obligor_quarters", "facilities", "obligors"]


def _give(run_id, runs_dir, tables, by="dana"):
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(tables, root=runs_dir),
                        root=runs_dir, by=by, background=False)
    service.run_until_idle(run_id, runs_dir)


def _past_the_data_gate(runs_dir, mode="interactive") -> tuple[str, str]:
    cfg = service.template_config("built-in:pd_logistic_scorecard", "pd", root=runs_dir)
    run_id = service.create_run(cfg, mode=mode, provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    if mode == "interactive":
        service.submit_gate(run_id, "gate_intent", "accept", reason="t", root=runs_dir,
                            background=False)
        service.run_until_idle(run_id, runs_dir)
    _give(run_id, runs_dir, FIRST)
    features = service.results(run_id, runs_dir)["explore"].payload["features"]
    gone = "current_ratio"  # a column of the base table: both data sets keep it
    assert gone in features
    if mode == "autonomous":  # a person revises what express prepared
        service.reopen(run_id, "gate_data", root=runs_dir, by="dana")
    service.submit_gate(run_id, "gate_data", "edit", {"exclude_columns": [gone]},
                        root=runs_dir, by="dana", background=False)
    service.run_until_idle(run_id, runs_dir)
    return run_id, gone


def test_re_opening_the_data_puts_the_data_gate_on_hold_and_the_same_data_keeps_it(runs_dir):
    run_id, gone = _past_the_data_gate(runs_dir)
    decided = service.state(run_id, runs_dir).last_decision("gate_data").id

    service.reopen(run_id, "gate_source", root=runs_dir, by="dana")
    st = service.state(run_id, runs_dir)
    assert st.status_of("gate_data") == "done"  # nothing is undone by re-opening
    assert waits_on(st, "gate_data") == "gate_source"
    assert shown_status(st, "gate_data") == "on_hold"
    assert shown_status(st, "ideate") == "on_hold"
    page = json.dumps(stage_panel("explore", service.results(run_id, runs_dir)["explore"], st,
                                  "light").__repr__())
    assert "is open again" in page

    _give(run_id, runs_dir, FIRST)  # the same data: the data gate stands
    st = service.state(run_id, runs_dir)
    assert st.status_of("gate_data") == "done" and waits_on(st, "gate_data") is None
    assert st.last_decision("gate_data").id == decided
    assert st.overrides.exclude_columns == [gone]


def test_new_data_sends_the_data_gate_back_with_the_earlier_choices_offered(runs_dir):
    run_id, gone = _past_the_data_gate(runs_dir)
    service.reopen(run_id, "gate_source", root=runs_dir, by="dana")
    _give(run_id, runs_dir, SECOND)
    st = service.state(run_id, runs_dir)
    assert st.status_of("gate_data") == "awaiting"
    assert st.overrides.exclude_columns == []  # offered, never in force by itself
    carried = carried_forward(st)
    assert carried["exclude_columns"] == [gone]
    res = service.results(run_id, runs_dir)["explore"]
    page = repr(stage_panel("explore", res, st, "light"))
    assert "Switches start from your earlier choices" in page

    service.submit_gate(run_id, "gate_data", "edit", {"exclude_columns": [gone]},
                        root=runs_dir, by="dana", background=False)
    st = service.state(run_id, runs_dir)
    assert carried_forward(st) is None and st.overrides.exclude_columns == [gone]


def test_express_preparation_never_replaces_a_persons_decision(runs_dir):
    run_id, _ = _past_the_data_gate(runs_dir, mode="autonomous")
    st = service.state(run_id, runs_dir)
    assert st.last_decision("gate_data").actor == "human"
    service.reopen(run_id, "gate_source", root=runs_dir, by="dana")
    _give(run_id, runs_dir, SECOND)
    st = service.state(run_id, runs_dir)
    assert st.status_of("gate_data") == "awaiting"  # back to a person, not express
    assert st.last_decision("gate_data").actor == "human"
