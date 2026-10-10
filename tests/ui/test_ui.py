"""The Dash workbench: every page and stage panel renders for real runs; gate forms map to the
engine's decisions. (Browser-free: components are built and serialized, callbacks' helpers called.)"""

from __future__ import annotations

import json

import plotly
import pytest

pytest.importorskip("dash_mantine_components")

from cognos import service  # noqa: E402
from cognos.engine.graph import STAGES  # noqa: E402
from cognos.ui import app as ui  # noqa: E402
from cognos.ui.panels import stage_panel  # noqa: E402


def _serialize(component) -> str:
    return json.dumps(component, cls=plotly.utils.PlotlyJSONEncoder)


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "runs"
    monkeypatch.setenv("COGNOS_RUNS_DIR", str(r))
    return r


@pytest.fixture
def completed_run(root):
    cfg = service.demo_config("cni", root, n=800, search_budget=6)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic")
    service.run_until_idle(run_id)
    return run_id


def test_app_builds_and_routes(root, completed_run):
    app = ui.create_app(str(root))
    assert _serialize(app.layout)
    assert "Model development runs" in _serialize(ui.runs_page())
    assert completed_run in _serialize(ui.runs_table())
    assert _serialize(ui.workspace_page(completed_run))


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_stage_panel_renders_for_a_completed_run(completed_run, scheme):
    st = service.state(completed_run)
    results = service.results(completed_run)
    for stage in STAGES:
        out = _serialize(stage_panel(stage, results.get(stage), st, scheme))
        assert out and "Traceback" not in out
        if stage in ("model", "backtest", "validate"):  # charts render, not their placeholders
            assert '"type": "Graph"' in out, stage
            assert "No ledger." not in out and "no per-feature effects" not in out
    assert _serialize(ui.run_header(st)) and _serialize(ui.rail(st, "model"))
    assert _serialize(ui.activity(completed_run))
    assert "Decision log" in _serialize(ui.questions_tab(st))
    assert "data_analyst" in _serialize(ui.audit_tab(completed_run))


def test_interactive_gate_forms_round_trip(root):
    cfg = service.demo_config("commercial", root, n=600, search_budget=6)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    st = service.run_until_idle(run_id)

    # the intent gate: the interview is a form, an answer becomes a gap answer
    assert ui.auto_step(st) == "intake"
    panel = _serialize(stage_panel("intake", service.results(run_id)["intake"], st, "light"))
    assert "Confirm the intent" in panel and "Submit answers" in panel and "blocking" in panel
    assert "answer::design-horizon" in panel and "Engagement brief" in panel
    payload, _ = ui.gate_payload("gate_intent", "edit", {"answer::design-horizon": "12-month",
                                                         "answer::design-segment": " "}, run_id)
    assert payload == {"answers": {"design-horizon": "12-month"}}
    service.submit_gate(run_id, "gate_intent", "edit", payload, background=False)
    st = service.run_until_idle(run_id)
    panel = _serialize(stage_panel("intake", service.results(run_id)["intake"], st, "light"))
    assert "answer::design-horizon" not in panel and '"answered"' in panel
    assert '"c": null' not in panel  # an explicit None style prop breaks the page in the browser
    service.submit_gate(run_id, "gate_intent", "accept", reason="the rest is with the sponsor",
                        background=False)
    st = service.run_until_idle(run_id)

    assert ui.auto_step(st) == "explore"
    panel = _serialize(stage_panel("explore", service.results(run_id)["explore"], st, "light"))
    assert '"gate-act"' in panel and "Accept recommendation" in panel

    # the data gate: an edit maps to exclude_columns
    feature = service.results(run_id)["explore"].payload["features"][0]
    payload, reason = ui.gate_payload("gate_data", "edit", {"exclude": [feature], "reason": "late"},
                                      run_id)
    assert payload == {"exclude_columns": [feature]} and reason == "late"
    service.submit_gate(run_id, "gate_data", "edit", payload, reason, background=False)
    st = service.run_until_idle(run_id)
    assert ui.auto_step(st) == "ideate"

    # the design gate: dropping a spec becomes a slate edit; an answer becomes a gap answer
    hyps = service.results(run_id)["ideate"].payload["hypotheses"]
    open_gap = next(g for g in st.gaps if g.status == "open")
    fields = {"keep": [h["id"] for h in hyps[:-1]], f"answer::{open_gap.id}": "origination"}
    payload, _ = ui.gate_payload("gate_design", "edit", fields, run_id)
    assert len(payload["slate"]) == len(hyps) - 1
    assert payload["answers"] == {open_gap.id: "origination"}

    # send-back carries the message; the validation gate also carries its target
    payload, reason = ui.gate_payload("gate_validation", "send_back",
                                      {"message": "check signs", "target": "ideate"}, run_id)
    assert payload == {"message": "check signs", "target": "ideate"} and reason == "check signs"


def test_failed_step_offers_retry(root):
    from cognos.engine.state import RunState

    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic")
    st = service.state(run_id)
    st.set_step("explore", "failed", "boom")
    assert '"retry"' in _serialize(stage_panel("explore", None, st, "light"))
    assert isinstance(st, RunState)


def test_compare_page_and_rerun_note_render(root):
    from cognos.ui.panels import compare_page, rerun_note

    cfg = service.demo_config("commercial", root, n=600, search_budget=6)
    first = service.create_run(cfg, mode="autonomous", provider="heuristic")
    service.run_until_idle(first)
    second = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(second)
    service.submit_gate(second, "gate_intent", "accept", reason="as written", background=False)
    service.run_until_idle(second)
    prof = service.results(second)["explore"].payload
    drop = [c for c in prof["features"] if c not in prof.get("recommended_exclusions", [])][0]
    service.submit_gate(second, "gate_data", "edit", {"exclude_columns": [drop]}, "what if", background=False)
    service.run_until_idle(second)

    page = _serialize(compare_page(service.compare(first, second)))
    assert "Compare runs" in page and first in page and second in page
    assert "Decided differently" in page and "Excluded columns" in page and drop in page
    assert "has not finished" in page  # the second run is waiting at a gate, and the page says so
    assert f"/compare/{second}/{first}" in page  # swap
    # the picker offers both runs; the workspace links to the previous run and offers the export
    assert first in _serialize(ui.compare_picker()) and second in _serialize(ui.compare_picker())
    actions = _serialize(ui.run_actions(second))
    assert f"/compare/{first}/{second}" in actions and "export-btn" in actions
    assert "/compare/" not in _serialize(ui.run_actions(first))  # nothing earlier to compare with

    # revise the decision: the rail and the panel say why the stage is out of date...
    service.submit_gate(second, "gate_design", "accept", background=False)
    service.run_until_idle(second)
    service.reopen(second, "gate_data")
    service.submit_gate(second, "gate_data", "accept", background=False)
    st = service.state(second)
    assert st.status_of("model") == "stale"
    assert "Review data decisions: recommended exclusions accepted" in _serialize(ui.rail(st, "model"))
    panel = _serialize(stage_panel("model", service.results(second)["model"], st, "light"))
    assert "This stage will re-run because" in panel and "recommended exclusions accepted" in panel
    assert rerun_note("model", st, None) is None  # not while it is still stale
    # ...and once it has re-run, what that changed
    service.run_until_idle(second)
    service.submit_gate(second, "gate_design", "accept", background=False)
    st = service.run_until_idle(second)
    changes = service.step_changes(second, "model")
    panel = _serialize(stage_panel("model", service.results(second)["model"], st, "dark", changes))
    assert "What the re-run changed" in panel and "Ran again because" in panel
    assert rerun_note("explore", st, None) is None  # explore ran once


def test_new_run_form_asks_for_what_the_development_mode_needs(root):
    page = _serialize(ui.runs_page())
    assert "New model development" in page and "Model update" in page
    assert "new-intent" in page and "new-prior" in page and "Download the template" in page

    demo = service.demo_config("commercial", root, n=200, search_budget=4)
    bare = demo.model_copy(update={"engagement": demo.engagement.model_copy(update={"intent": None})})
    assert ui.engagement_problem("new", demo, False, False) is None  # a demo brings its own intent
    assert "business intent document" in ui.engagement_problem("new", bare, False, False)
    assert ui.engagement_problem("new", bare, True, False) is None
    assert "update request" in ui.engagement_problem("update", demo, False, True)
    assert "existing model" in ui.engagement_problem("update", demo, True, False)
    assert ui.engagement_problem("update", demo, True, True) is None

    data_url = "data:text/plain;base64,aGVsbG8="
    assert [open(p, encoding="utf-8").read() for p in ui.saved_uploads(data_url, "a.txt")] == ["hello"]
    assert len(ui.saved_uploads([data_url, data_url], ["a.txt", "b.txt"])) == 2
    assert ui.saved_uploads(None, None) == [] and ui.upload_names(None) is None
    assert "a.txt" in _serialize(ui.upload_names(["a.txt", "b.md"]))


def test_intake_panel_shows_the_update_request(root, tmp_path):
    from cognos import engagement as eg

    cfg = service.demo_config("commercial", root, n=600, search_budget=6)
    request = tmp_path / "request.md"
    request.write_text(eg.render_template("update", {
        "objective": "Keep the PD model accurate.", "prior_model": "PD v1",
        "update_reason": "Calibration drift.",
        "requested_changes": "- Recalibrate to the long-run default rate"}), encoding="utf-8")
    paper = tmp_path / "whitepaper.md"
    paper.write_text("PD v1 is a logistic regression scorecard.", encoding="utf-8")
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", engagement={
        "kind": "update", "intent": str(request), "prior_artifacts": [str(paper)]})
    st = service.run_until_idle(run_id)
    panel = _serialize(stage_panel("intake", service.results(run_id)["intake"], st, "light"))
    assert "Model update" in panel and "Update request" in panel and "recalibration" in panel
    assert "The existing model" in panel and "logit" in panel and "whitepaper.md" in panel
    assert "update" in _serialize(ui.runs_table())
