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
from cognos.ui.panels import chip_class, stage_panel  # noqa: E402


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
    assert "Model development" in _serialize(ui.runs_page())
    assert completed_run in _serialize(ui.runs_table())
    # a second run from the same profile is a second model, not a second run of the first
    again = service.create_run(service.demo_config("cni", root, n=800, search_budget=6),
                               mode="interactive", provider="heuristic")
    service.run_until_idle(again)
    listed = _serialize(ui.runs_table())
    # one model, one main run: each is listed under its own heading, told apart by its start
    assert listed.count("Main run") == 2 and listed.count("started 2") == 2
    assert "Run 1 of" not in listed and "same model" not in listed
    assert "waiting at" in listed and "Confirm the intent" in listed
    assert "finished, not signed off" in listed and "one development of one model" in _serialize(
        ui.runs_page())
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


def test_analysis_charts_draw_every_spec_kind_in_both_themes():
    from cognos.ui import charts

    x = ["a", "b", "c"]
    specs = [
        {"kind": "bar", "x": x, "series": [{"name": "rate", "y": [0.1, 0.2, 0.3]}], "y_format": "%"},
        {"kind": "line", "x": [1, 2, 3], "series": [{"name": "m", "y": [1, 2, 3]},
                                                    {"name": "n", "y": [3, 2, 1]}]},
        {"kind": "scatter", "x": [1, 2, 3], "series": [{"name": "m", "y": [1, None, 3]}]},
        {"kind": "histogram", "x": [0.5, 1.5, 2.5], "series": [{"name": "rows", "y": [5, 9, 2]}]},
        {"kind": "heatmap", "x": x, "y": x, "z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    ]
    for scheme in ("light", "dark"):
        for spec in specs:
            fig = charts.from_spec(spec, scheme)
            assert fig is not None and len(fig.data) >= 1, spec["kind"]
    assert charts.from_spec(specs[1], "light").layout.showlegend is True  # two series: a legend
    assert charts.from_spec(specs[0], "light").layout.yaxis.tickformat == ".0%"
    assert charts.from_spec(None, "light") is None


def test_own_data_form_builds_a_source_and_explore_shows_charts_and_code(root, tmp_path, monkeypatch):
    from cognos import synth

    page = _serialize(ui.runs_page())
    # the drawer asks what to start from, never for the data or a demo (ADR-0018)
    assert "new-start" in page and "logistic scorecard" in page and "From scratch" in page
    assert "The data this kind of model is built on" in page and "Governance" in page
    assert "new-data" not in page and "new-sf-table" not in page and "Synthetic demos" not in page

    data = tmp_path / "loans.csv"
    synth.GENERATORS["commercial"](n=500).to_csv(data, index=False)
    cfg = service.config_from_data("My PD model", {"kind": "file", "path": str(data)})
    assert cfg.name == "my_pd_model" and cfg.data.target == ""
    assert "business intent document" in ui.engagement_problem("new", cfg, False, False)
    scout = {"target_column": "default", "target_rationale": "the default flag", "done": True,
             "notes": "", "requests": [{"purpose": "Sector mix", "tool": "", "params": [], "code": (
                 "by = df.groupby('sector')[TARGET].mean()\n"
                 "emit_chart('bar', by.index, {'rate': by.values}, title='Rate by sector')")}]}
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    (recorded / "data_scout.json").write_text(json.dumps(scout), encoding="utf-8")
    monkeypatch.setenv("COGNOS_PROVIDER", "replay")
    cfg = service.load_config({**cfg.model_dump(mode="json"),
                               "agents": {"replay_dir": str(recorded)}})
    run_id = service.create_run(cfg, mode="interactive")
    service.run_until_idle(run_id)
    service.submit_gate(run_id, "gate_intent", "accept", reason="as given", background=False)
    st = service.run_until_idle(run_id)
    panel = _serialize(stage_panel("explore", service.results(run_id)["explore"], st, "light"))
    assert "Agent-written script" in panel and "Code to review" in panel and "groupby" in panel
    assert "Rate by sector" in panel and '"type": "Graph"' in panel
    assert "proposed by the Data Analyst" in panel and "loans.csv" in panel
    assert "analysis script(s) written by the Data Analyst" in panel  # the gate says what is accepted
    assert '"c": null' not in panel


def test_every_stage_page_shows_the_tools_its_agent_ran_and_the_ones_it_could_not(root, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples" / "plugins"))
    cfg = service.demo_config("cni", root, n=800, search_budget=6)
    cfg.plugins = ["validation_tools"]
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic")
    service.run_until_idle(run_id)
    st, results = service.state(run_id), service.results(run_id)
    panel = _serialize(stage_panel("validate", results["validate"], st, "light"))
    assert "Tools run (1)" in panel and "score_band_monotonicity" in panel
    assert "Plugin tool (validation_tools)" in panel and "monotonic" in panel
    assert "Tools not run" in panel and "impact_test_suite" in panel
    assert '"c": null' not in panel
    assert "events_per_feature" in _serialize(stage_panel("ideate", results["ideate"], st, "dark"))
    # a stage with no tool keeps its page as it was
    assert "Tools run" not in _serialize(stage_panel("comply", results["comply"], st, "light"))


def test_delete_runs_warns_before_anything_is_removed(root, completed_run):
    page = _serialize(ui.runs_page())
    assert "Delete runs" in page and "del-sure" in page
    assert completed_run in json.dumps(ui.delete_options())
    assert ui.deletion_blocked([]) and not ui.deletion_blocked([completed_run])
    warning = _serialize(ui.deletion_warning([completed_run]))
    assert "cannot be undone" in warning and "1 run," in warning
    assert "red" in _serialize(ui.deletion_warning(["../elsewhere"]))
    assert (root / completed_run).exists()


def test_linked_sources_show_the_join_and_let_the_developer_change_it(root):
    cfg = service.demo_config("linked", root, n=500, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    service.submit_gate(run_id, "gate_intent", "accept", background=False)
    service.run_until_idle(run_id)
    st = service.state(run_id)
    page = _serialize(stage_panel("explore", service.results(run_id)["explore"], st, "light"))
    assert "Sources and join" in page and "loans.obligor_id = financials.obligor_id" in page
    assert "join_links" in page and "Base rows with no match" in page and "dropped" in page
    # the join is drawn as a canvas: a tile per table, a chip per measured link; the chips
    # write join_links and the tiles' buttons write join_base, so the decision reads the same
    assert "jc-canvas" in page and '"type": "jc-tile"' in page and '"type": "jc-base"' in page
    linked = service.results(run_id)["explore"].payload["linking"]
    for lk in linked["links"]:
        assert f'"link": "{lk["id"]}"' in page
    assert chip_class(True, True) == "jc-chip chosen suggested"
    assert chip_class(False, True) == "jc-chip proposed"
    assert chip_class(False, False) == "jc-chip measured"
    payload, _ = ui.gate_payload("gate_data", "edit", {
        "join_base": "loans", "join_links": ["L1"], "join_many": "first",
        "join_unmatched": "keep", "exclude": []}, run_id)
    assert payload["join"] == {"base": "loans", "links": ["L1"], "many": "first",
                               "unmatched": "keep"}
    assert "of the rows have no match" in page  # why the unmatched rows are dropped, from the measurements
    # applying the form untouched ("as recommended") is not a change to the join
    plan = service.results(run_id)["explore"].payload["linking"]["plan"]
    same, _ = ui.gate_payload("gate_data", "edit", {
        "join_base": plan["base"], "join_links": [s["link"] for s in plan["steps"]],
        "join_many": "aggregate", "join_unmatched": "plan", "exclude": []}, run_id)
    st = service.submit_gate(run_id, "gate_data", "edit", same, background=False)
    assert st.status_of("gate_data") == "done" and st.status_of("explore") == "done"


def test_the_record_names_the_person(root, monkeypatch):
    monkeypatch.setenv("COGNOS_USER", "maria")
    cfg = service.demo_config("cni", root, n=500, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    service.submit_gate(run_id, "gate_intent", "accept", background=False)
    st = service.state(run_id)
    assert "maria" in _serialize(ui.questions_tab(st, "developer"))
    assert "maria" in _serialize(ui.header()) and "maria" in _serialize(ui.run_header(st))


def test_the_plan_the_pool_and_the_champion_are_decided_on_forms(root):
    """ADR-0016: the plan is edited at the design gate, the pool at the candidate gate, and the
    champion is chosen on the page that compares the candidates' forecasts."""
    cfg = service.demo_config("loss_forecast", root)
    cfg.factors.lags = [1, 2]  # a smaller library: the forms are what is under test
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    for gate in ("gate_intent", "gate_data"):
        service.submit_gate(run_id, gate, "accept", reason="ok", background=False)
        st = service.run_until_idle(run_id)

    # the design gate shows the plan, who set each choice and why, and lets the developer edit it
    ideate = service.results(run_id)["ideate"]
    plan = ideate.payload["plan"]
    panel = _serialize(stage_panel("ideate", ideate, st, "light"))
    assert "Development plan" in panel and "rolling origin" in panel and "cannot be shuffled" in panel
    assert "plan::evaluation.validation" in panel and "plan::selection.methods" in panel
    assert "plan::factors.max_p_value" in panel and "plan::rank.0.metric" in panel
    assert "Your modelling criteria" in panel and "recommended by the engine" in panel
    assert "Primary metric" in panel and "plan::factors.criterion" in panel
    assert '"c": null' not in panel
    untouched = {"plan::evaluation.test": plan["evaluation"]["test"],
                 "plan::evaluation.folds": float(plan["evaluation"]["folds"]),
                 "plan::evaluation.holdout_fraction": 0, "plan::factors.enforce_sign": "yes",
                 "plan::factors.transforms": list(plan["factors"]["transforms"]),
                 "plan::selection.methods": ["forward"], "plan::factors.criterion": "validation",
                 "plan::factors.min_strength": "", "plan::selection.max_p_value": 0.1,
                 "plan::rank.0.metric": "rmse", "plan::rank.0.sample": "validation",
                 "plan::rank.1.metric": None, "plan::rank.1.sample": "in_sample"}
    assert ui.gate_payload("gate_design", "edit", dict(untouched), run_id)[0] == {}
    # the developer picks their own criteria: the primary metric, the single-factor ranking,
    # the multi-factor method and what a step must improve
    edited = {**untouched, "plan::evaluation.pool_size": 3.0,
              "plan::rank.0.metric": "mae", "plan::rank.1.metric": "r2",
              "plan::factors.criterion": "in_sample", "plan::factors.enforce_sign": "no",
              "plan::factors.top_n": "", "plan::selection.methods": ["best_subset"],
              "plan::selection.criterion": "bic"}
    payload, _ = ui.gate_payload("gate_design", "edit", edited, run_id)
    assert payload == {"plan": {
        "evaluation": {"pool_size": 3, "rank_by": [{"metric": "mae", "sample": "validation"},
                                                   {"metric": "r2", "sample": "in_sample"}]},
        "selection": {"methods": ["best_subset"], "criterion": "bic"},
        "factors": {"criterion": "in_sample", "enforce_sign": False, "top_n": None}}}
    service.submit_gate(run_id, "gate_design", "edit", payload, "one method, three candidates",
                        background=False)
    st = service.run_until_idle(run_id)
    ideate = service.results(run_id)["ideate"]
    plan = ideate.payload["plan"]
    assert plan["sources"]["pool_size"] == plan["sources"]["factors.criterion"] == "human"
    panel = _serialize(stage_panel("ideate", ideate, st, "light"))
    assert '"developer"' in panel and "set by the developer" in panel  # who set what
    service.submit_gate(run_id, "gate_design", "accept", reason="ok", background=False)
    st = service.run_until_idle(run_id)

    # the candidate gate: the screen, each method's path, the leaders, and the pool as a form
    assert ui.auto_step(st) == "model"
    model = service.results(run_id)["model"]
    panel = _serialize(stage_panel("model", model, st, "dark"))
    for text in ("Review the candidate pool", "Single-factor screen", "Multi-factor selection",
                 "Best subset", "the method chosen in the plan", "Leaders", "Candidate pool",
                 "Accept the pool", "best per source", "as many as you want to compare"):
        assert text in panel, text
    assert '"type": "Graph"' in panel and "No ledger." not in panel and '"c": null' not in panel
    assert model.payload["rank_metric"] == "mae"  # the developer's primary metric is in force
    ids = model.payload["pool_ids"]
    names = model.payload["names"]
    assert len(ids) <= 3 and all(i in panel and names[i] in panel for i in ids)
    assert all(n.startswith("OLS ") or n.split()[0] in ("Lasso", "Ridge") for n in names.values())
    leaders = {a["id"] for a in model.payload["leaders"]}
    outside = next(a["id"] for a in model.payload["leaders"] if a["id"] not in ids)
    below = next(i for i in model.payload["evaluated_ids"] if i not in leaders)
    assert f"name::{ids[0]}" in panel and "Add from the rest of the ranking" in panel
    assert below in panel  # every ranked model can be picked, not only the leaders shown
    # more than the engine proposed, one from below the leaders, and a name of the developer's own
    fields = {"pool": [*ids, outside], "extra": [below], "reason": "wider comparison",
              **{f"name::{a['id']}": a["name"] for a in model.payload["leaders"]},
              f"name::{ids[0]}": "  Base   case "}
    payload, reason = ui.gate_payload("gate_candidates", "edit", fields, run_id)
    assert payload == {"pool": [*ids, outside, below], "names": {ids[0]: "Base case"}}
    service.submit_gate(run_id, "gate_candidates", "edit", payload, reason, background=False)
    st = service.run_until_idle(run_id)
    model = service.results(run_id)["model"]
    assert model.payload["pool_ids"] == [*ids, outside, below]
    assert model.payload["names"][ids[0]] == "Base case"
    assert model.payload["names"][outside] == names[outside]  # the others keep their names
    panel = _serialize(stage_panel("model", model, st, "light"))
    assert "Base case" in panel
    service.submit_gate(run_id, "gate_candidates", "accept", reason="ok", background=False)
    st = service.run_until_idle(run_id)

    # the champion gate sits on the comparison, not on the search
    assert ui.auto_step(st) == "backtest"
    backtest = service.results(run_id)["backtest"]
    panel = _serialize(stage_panel("backtest", backtest, st, "light"))
    for text in ("Choose the champion", "Candidates side by side", "Forecasts against outcomes",
                 "Scenarios, candidate by candidate", "severely_adverse", "Accept the champion",
                 "no sealed holdout was reserved"):
        assert text in panel, text
    assert '"type": "Graph"' in panel and '"actual"' in panel and '"c": null' not in panel
    assert "Base case" in panel and len(backtest.payload["candidates"]) == len(ids) + 2
    other = next(c["id"] for c in backtest.payload["candidates"]
                 if c["id"] != backtest.payload["champion_id"])
    payload, _ = ui.gate_payload("gate_champion", "override", {"champion": other}, run_id)
    assert payload == {"champion": other}
    # another seat reads the page and cannot decide on it
    locked = _serialize(stage_panel("backtest", backtest, st, "light", seat="reviewer"))
    assert "Accept the champion" not in locked and "belongs to the model developer" in locked
    # the decision is one card: what accepting means and the engine's figures first; choosing
    # another candidate folds away, and its button is only the owning seat's
    assert "gate-proposal" in panel and "gate-evidence" in panel
    assert "Choose another candidate" in panel and "Choose the selected candidate" in panel
    assert "Choose the selected candidate" not in locked


def test_a_stage_shows_only_its_own_questions_and_the_record_covers_the_run(root):
    cfg = service.demo_config("commercial", root, n=500, search_budget=4)  # a blank design brief
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    st = service.state(run_id)
    core = [g for g in st.gaps if g.design_field in ("use_case", "horizon")]
    assert core and all(g.stage == "intake" for g in core)  # intent questions are intake's
    results = service.results(run_id)
    # while the intent gate is open the interview is the one place to answer
    intake = _serialize(stage_panel("intake", results["intake"], st, "light"))
    assert "Questions from this stage" not in intake and "Interview" in intake
    rail = _serialize(ui.rail(st, "intake"))
    assert "open question" in rail and "required" in rail
    assert '"disableRightSectionRotation": true' in rail  # the status label stays level
    assert '"active": true' not in _serialize(ui.rail(st, None))  # a run-wide view: no stage
    record = _serialize(ui.questions_tab(st))
    assert "Run record" in record and "Raised at" in record and "business owner" in record
    assert "must be answered before sign-off" in record
    assert "Agent calls" in _serialize(ui.audit_tab(run_id))
    page = _serialize(ui.workspace_page(run_id))
    assert page.index('"ws-tab"') < page.index('"ws-rail"')  # the views sit above the stages


def test_assumptions_and_limitations_reach_validation_and_the_white_paper(root):
    cfg = service.demo_config("cni", root, n=500, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic")
    st = service.run_until_idle(run_id)
    assert st.status_of("validate") == "done"
    from cognos.engine.state import Gap

    eng = service.engine(run_id)
    with eng.lock:  # two data questions, as the Data Analyst raises them
        state = eng.state
        state.gaps += [Gap(id="data-t1", question="Is utilization known at origination?",
                           category="data", stage="explore", reentry="explore"),
                       Gap(id="data-t2", question="Is collateral revalued?", category="data",
                           stage="explore", reentry="explore")]
        eng._save(state)
    a, b = "data-t1", "data-t2"
    explore = _serialize(stage_panel("explore", service.results(run_id)["explore"],
                                     service.state(run_id), "light"))
    assert "Flag as limitation" in explore and "What each choice does" in explore
    service.answer_gap(run_id, a, "Taken as known.", assume=True, background=False)
    st = service.answer_gap(run_id, b, "Not covered.", limitation=True, background=False)
    assert st.status_of("explore") == "done"  # the stage that asked does not run again
    assert st.status_of("validate") == "stale" and "assumption" in st.steps["validate"].rerun_reason
    st = service.run_until_idle(run_id)
    found = {f.id: f for f in service.results(run_id)["validate"].findings}
    assert "sensitiv" in found[f"assumption-{a}"].suggestion
    assert "mitigated" in found[f"limitation-{b}"].suggestion
    assert service.results(run_id)["validate"].verdict.value != "BLOCK"
    page = service.read_artifact(run_id, "docs/limitations.md")
    assert "Assumptions accepted" in page and "Taken as known." in page
    assert page.index("Limitations recorded") < page.index("Not covered.")
    assert "limitation" in _serialize(ui.questions_tab(st))


def test_trial_runs_are_made_listed_and_kept_on_the_pages(root):
    """ADR-0017: the control is on the run, the runs page says which run is which, the run
    record lists every run of the model, and a deleted trial run keeps its line."""
    from cognos.engine.graph import GATES

    def to(run_id, until=None):
        for _ in range(24):
            st = service.run_until_idle(run_id)
            waiting = [g for g in GATES if st.status_of(g) == "awaiting"]
            if not waiting or waiting[0] == until:
                return st
            service.submit_gate(run_id, waiting[0], "accept", reason="ok", background=False)

    cfg = service.demo_config("regression", root, n=220, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    to(run_id, "gate_champion")
    actions = _serialize(ui.run_actions(run_id))
    assert "Try a different decision" in actions and "Make this the main run" not in actions
    assert "only run of this model" in _serialize(ui.model_runs_section(service.state(run_id)))
    modal = _serialize(ui.trial_modal())
    assert "cannot open the sealed holdout" in modal and "One line stays" in modal
    assert [o["value"] for o in ui.trial_gate_options(run_id)][-1] == "gate_champion"

    trial = service.trial_run(run_id, "gate_champion", "another champion")
    runs = _serialize(ui.runs_table())
    assert "Main run" in runs and "Trial run" in runs and "another champion" in runs
    assert "1 trial run" in runs and "same model" in runs and "Run 1 of" not in runs
    header = _serialize(ui.run_header(service.state(trial)))
    assert "Make this the main run" in header and f"/compare/{run_id}/{trial}" in header
    assert "Compare with the run it was made from" in header and f"of {run_id}" in header
    assert "1 trial run on record" in _serialize(ui.run_header(service.state(run_id)))

    to(trial)
    assert "This trial run ends here" in _serialize(ui.run_header(service.state(trial)))
    assert "does not go to validation" in _serialize(ui.runs_table())
    record = _serialize(ui.questions_tab(service.state(run_id)))
    assert "Runs of this model (2)" in record and trial in record and "Test sample used" in record
    assert "Made as a trial run of" in _serialize(ui.questions_tab(service.state(trial)))

    to(run_id, "gate_validation")
    res = service.results(run_id)["validate"]
    page = _serialize(stage_panel("validate", res, service.state(run_id), "light"))
    assert "Runs of this model (2)" in page and "another champion" in page

    assert "One line stays" in _serialize(ui.deletion_warning([trial]))
    assert "Cannot be deleted now" in _serialize(ui.deletion_warning([run_id]))
    service.delete_runs([trial])
    assert "Deleted trial run" in _serialize(ui.runs_table())
    assert "files deleted" in _serialize(ui.questions_tab(service.state(run_id)))
    assert "Make this the main run" in _serialize(ui.promote_modal())


def test_the_new_run_form_offers_the_providers_models(root):
    providers = service.provider_list()
    data, value = ui.model_choices(providers, "claude_cli")
    assert value == "opus" and [d["value"] for d in data][:3] == ["opus", "sonnet", "haiku"]
    assert ui.model_choices(providers, "heuristic") == ([], None)
    assert "CLI" in ui.model_note(providers, "claude_cli")
    assert "new-model" in _serialize(ui.runs_page())

    cfg = service.demo_config("cni", root, n=200, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", model="sonnet")
    assert service.engine(run_id).config.agents.model == "sonnet"
    assert service.agent_model(run_id) is None  # the deterministic agents have no model


def test_focus_hides_both_side_columns_on_every_view():
    assert ui.column_layout("workflow", None) == (True, 13, True)
    assert ui.column_layout("workflow", True) == (False, 24, False)
    assert ui.column_layout("audit", False) == (True, 18, False)
    assert ui.column_layout("questions", True) == (False, 24, False)
    assert ui.column_layout("workflow", {"left": False}) == (True, 13, True)  # an old choice


def test_the_data_is_given_on_the_explore_page_after_the_intent(root, tmp_path):
    from cognos import synth

    cfg = service.template_config("built-in:pd_logistic_scorecard", "cni pd")
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    service.submit_gate(run_id, "gate_intent", "accept", reason="test", background=False)
    st = service.run_until_idle(run_id)
    assert ui.auto_step(st) == "explore"  # the page that asks for it
    info = service.data_info(run_id, with_catalog=True)
    assert "default flag" in info["guide"] and info["given"] == ""
    # the data catalog, with the template's tables checked and what each holds
    assert info["preselected"] == ["obligor_quarters", "facilities", "obligors", "macro_quarterly"]
    page = _serialize(stage_panel("explore", None, st, "light", data_info=info))
    assert "Provide the data" in page and "Use this data" in page and "default flag" in page
    assert "Starts once the data is given" in page
    assert "Obligor financials by quarter" in page and "rows" in page and "Add data" in page
    assert "How the tables join, as documented" in page
    assert "The default catalog shipped with COGNOS" in page  # which catalog is in force
    rail = _serialize(ui.rail(st, "explore"))
    assert "Provide the data" in rail
    # the form, read back as the engine's payload
    with pytest.raises(ui.GateError, match="choose a table"):
        ui.gate_payload("gate_source", "accept", {"catalog": [], "add_files": ""}, run_id)
    payload, _ = ui.gate_payload("gate_source", "accept", {"catalog": ["obligor_quarters"]},
                                 run_id)
    assert payload["catalog"] == ["obligor_quarters"]
    assert payload["data"]["source"]["catalog"] == "obligor_quarters"
    assert payload["data"]["datetime_col"] == "as_of_date"  # the catalog's, unless given
    data = tmp_path / "loans.csv"
    synth.GENERATORS["commercial"](n=300).to_csv(data, index=False)
    payload, _ = ui.gate_payload("gate_source", "accept", {
        "catalog": [], "add_files": json.dumps([str(data)]), "target": "default",
        "datetime_col": "vintage"}, run_id)
    assert payload == {"catalog": [], "target": "default", "data": {
        "source": {"kind": "file", "path": str(data)}, "datetime_col": "vintage"}}
    service.submit_gate(run_id, "gate_source", "accept", payload, background=False)
    st = service.run_until_idle(run_id)
    page = _serialize(stage_panel("explore", service.results(run_id)["explore"], st, "light",
                                  data_info=service.data_info(run_id)))
    assert "Data in use" in page and "loans.csv" in page
    assert service.results(run_id)["explore"].payload["target"] == "default"
