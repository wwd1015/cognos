"""The data stage end to end: a database source, a target the Data Analyst proposes, analyses
it requests through tools and plugins, and Python it writes — kept, reviewed, delivered to
validation and printed in the white paper."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from cognos import service, synth
from cognos.analysis import sandbox
from cognos.cli import main
from cognos.engine import Engine, GateError

EXAMPLES = str(Path(__file__).resolve().parents[2] / "examples" / "plugins")

SCRIPT = '''# Does the default rate differ by sector once leverage is high?
high = df[df["leverage"] > df["leverage"].median()]
by = high.groupby("sector")[TARGET].agg(["mean", "size"]).reset_index()
emit_table("high-leverage default rate by sector", by)
emit_value("spread", by["mean"].max() - by["mean"].min())
emit_chart("bar", by["sector"], {"event rate": by["mean"]},
           title="High-leverage default rate by sector", y_format="%")
'''


def _scout(*requests, done=True, target="") -> dict:
    return {"target_column": target, "target_rationale": "the 90+ DPD flag" if target else "",
            "done": done, "notes": "", "requests": [
                {"purpose": r.get("purpose", "why"), "tool": r.get("tool", ""),
                 "code": r.get("code", ""),
                 "params": [{"name": k, "value": v} for k, v in r.get("params", {}).items()]}
                for r in requests]}


@pytest.fixture
def book(tmp_path) -> str:
    """A loan book in a database, as a sponsor's data would arrive."""
    db = tmp_path / "book.db"
    con = sqlite3.connect(db)
    synth.GENERATORS["commercial"](n=800).to_sql("loans", con, index=False)
    con.close()
    return str(db)


def _open_config(book: str, runs_dir: str, **extra):
    return service.config_from_data("pd", {"kind": "sqlite", "path": book, "table": "loans"},
                                    root=runs_dir, search={"max_candidates": 6, "cv_folds": 3},
                                    **extra)


def test_target_is_proposed_from_a_database_source_and_confirmed_at_the_gate(book, runs_dir,
                                                                             confirm_intent):
    cfg = _open_config(book, runs_dir, data={"source": {"kind": "sqlite", "path": book,
                                                        "table": "loans"},
                                             "datetime_col": "vintage"})
    assert cfg.task is None and cfg.data.target == ""
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    state = confirm_intent(eng)
    assert state.status_of("gate_data") == "awaiting"
    p = eng.results()["explore"].payload
    assert (p["target"], p["task"], p["target_source"]) == ("default", "classification", "agent")
    assert p["target_candidates"][0]["column"] == "default" and p["target_rationale"]
    assert [c["column"] for c in p["feature_candidates"]] and "default" not in p["features"]
    # where the data came from is recorded, and every later stage reads the snapshot
    assert p["source"]["kind"] == "sqlite" and p["source"]["n_rows"] == 800
    src = json.loads((eng.run_dir / "data" / "source.json").read_text(encoding="utf-8"))
    assert src["table"] == "loans" and len(src["snapshot_sha256"]) == 64
    # the standard visuals ran as tool calls, each a kept artifact with a chart
    tools = [a["tool"] for a in p["analyses"]]
    assert tools[:2] == ["target_distribution", "time_trend"] and "correlation_matrix" in tools
    assert all(a["status"] == "ok" and a["kind"] == "tool" for a in p["analyses"])
    first = json.loads((eng.run_dir / p["analyses"][0]["artifact"]).read_text(encoding="utf-8"))
    assert first["charts"][0]["kind"] in ("bar", "histogram") and first["summary"]["n_events"] > 0
    assert state.overrides.target is None  # nothing is decided until the developer decides

    # another target re-runs the exploration against it; the gate comes back
    eng.submit_gate("gate_data", "edit", {"target": "leverage"})
    state = eng.run_until_idle()
    p = eng.results()["explore"].payload
    assert (p["target"], p["task"], p["target_source"]) == ("leverage", "regression", "decision")
    assert state.status_of("gate_data") == "awaiting" and "leverage" not in p["features"]
    assert "dependent variable set to leverage" in state.steps["explore"].rerun_reason
    with pytest.raises(GateError, match="unknown column"):
        eng.submit_gate("gate_data", "edit", {"target": "nope"})
    eng.submit_gate("gate_data", "edit", {"target": "default"})
    eng.run_until_idle()
    eng.submit_gate("gate_data", "accept")
    state = eng.run_until_idle()
    assert (state.overrides.target, state.overrides.task) == ("default", "classification")
    ctx = eng.context()
    assert (ctx.config.data.target, ctx.config.task.value, ctx.config.metric.name) == (
        "default", "classification", "roc_auc")
    assert state.status_of("gate_design") == "awaiting"


def test_open_target_run_completes_unattended_and_documents_the_data(book, runs_dir):
    eng = Engine(_open_config(book, runs_dir), runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "completed" and state.overrides.target == "default"
    assert eng.summary().champion_metric_name == "roc_auc"
    doc = (eng.run_dir / "docs" / "analysis.md").read_text(encoding="utf-8")
    assert "SQLite database" in doc and "proposed by the Data Analyst" in doc
    assert "## Analyses run" in doc and "target_relationship" in doc
    assert "analysis" in eng.results()["document"].payload["concepts"]


def test_a_fixed_target_cannot_be_changed_at_the_gate(make_config, runs_dir, confirm_intent):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    confirm_intent(eng)
    assert eng.results()["explore"].payload["target_source"] == "profile"
    with pytest.raises(GateError, match="profile fixes the dependent variable"):
        eng.submit_gate("gate_data", "edit", {"target": "x1"})
    eng.submit_gate("gate_data", "accept")
    assert eng.state.overrides.target is None


def test_agent_written_code_is_kept_reviewed_validated_and_delivered(make_config, runs_dir,
                                                                     replay_dir, tmp_path):
    rejected = _scout({"purpose": "peek", "code": "import os\nemit_value('n', len(os.listdir('.')))"})
    recorded = _scout({"purpose": "Does sector matter once leverage is high?", "code": SCRIPT},
                      {"purpose": "How is leverage distributed?", "tool": "distribution",
                       "params": {"column": "leverage"}},
                      {"purpose": "broken on purpose", "code": "emit_value('x', df['nope'].sum())"})
    cfg = make_config("commercial", agents={"replay_dir": replay_dir(
        {"data_scout": [rejected, recorded]})})
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    eng.run_until_idle()
    eng.submit_gate("gate_intent", "accept", reason="as written")
    state = eng.run_until_idle()

    # the engine refused the first answer (it reached for the OS) and the agent was asked again
    audit = [a for a in service.audit(eng.run_id, runs_dir) if a["agent"] == "data_scout"]
    assert [a["status"] for a in audit] == ["invalid", "ok"] and "import of 'os'" in audit[0]["error"]

    res = eng.results()["explore"]
    a1, a2, a3 = res.payload["analyses"]
    assert (a1["kind"], a1["status"], a2["kind"], a3["status"]) == ("code", "ok", "tool", "error")
    assert "KeyError" in a3["error"] and any(f.id == "analysis-failed" for f in res.findings)
    # the script is an artifact: the code as written, under a header saying who wrote it and why
    saved = (eng.run_dir / a1["code_path"]).read_text(encoding="utf-8")
    assert saved.startswith("# COGNOS analysis script") and "Written by: the Data Analyst agent" in saved
    assert SCRIPT.strip() in saved and a1["code_sha256"] == sandbox.sha256(SCRIPT.strip())
    result = json.loads((eng.run_dir / a1["artifact"]).read_text(encoding="utf-8"))
    assert result["charts"][0]["title"] == "High-leverage default rate by sector"
    assert result["tables"][0]["columns"] == ["sector", "mean", "size"]
    # what it found is citable, like any engine number
    from cognos.agents import slices
    facts = slices.build(eng.context(), "design_lead", {})["facts"]
    assert facts["explore.analysis.a1.spread"] == round(a1["summary"]["spread"], 4)
    assert facts["explore.n_code_analyses"] == 1

    # the data gate records exactly which code the developer accepted
    eng.submit_gate("gate_data", "accept", reason="read both scripts")
    decision = eng.state.last_decision("gate_data")
    assert decision.payload["reviewed_analyses"] == [
        {"id": "a1", "sha256": a1["code_sha256"]}, {"id": "a3", "sha256": a3["code_sha256"]}]
    for gate in ("gate_design", "gate_champion"):
        eng.run_until_idle()
        eng.submit_gate(gate, "accept", reason="ok")
    state = eng.run_until_idle()
    assert state.status_of("gate_validation") == "awaiting"

    # validation re-ran the script from the saved file and got the recorded result
    review = {c["id"]: c for c in eng.results()["validate"].payload["analysis_code"]}
    assert review["a1"]["reproduced"] and review["a1"]["ran"] and not review["a3"]["ran"]
    sent = json.loads(next((eng.run_dir / "agents").glob("*validator*.input.json")).read_text(
        encoding="utf-8"))["context"]["analysis_code"]
    assert "high.groupby" in sent[0]["code"] and sent[0]["reproduced"] is True

    eng.submit_gate("gate_validation", "accept", reason="ok")
    eng.run_until_idle()
    paper = (eng.run_dir / "stages/document/whitepaper.md").read_text(encoding="utf-8")
    assert "## Analysis code" in paper and 'high.groupby("sector")' in paper
    assert "reproduced at validation" in paper and a1["code_sha256"][:16] in paper
    assert eng.results()["review"].verdict.value != "BLOCK"
    path = service.export_run(eng.run_id, tmp_path / "out", runs_dir)
    with zipfile.ZipFile(path) as z:
        names = {n.split("/", 1)[1] for n in z.namelist()}
    assert {a1["code_path"], a1["artifact"], "docs/analysis.md"} <= names

    # a script that no longer matches its record is a validation finding
    target = eng.run_dir / a1["code_path"]
    target.write_text(saved.replace("> df[", ">= df["), encoding="utf-8")
    again = eng.run_stage("validate")
    finding = next(f for f in again.findings if f.id == "analysis-not-reproduced-a1")
    assert "hash differs" in finding.message and finding.severity.value == "MEDIUM"


def test_code_can_be_switched_off_and_a_failed_request_round_does_not_fail_the_stage(
        make_config, runs_dir, replay_dir):
    cfg = make_config("commercial", analysis={"allow_code": False}, agents={
        "max_retries": 1, "replay_dir": replay_dir({"data_scout": _scout({"code": SCRIPT})})})
    eng = Engine(cfg, runs_root=runs_dir)
    assert eng.run_until_idle().status == "completed"
    res = eng.results()["explore"]
    assert res.payload["analyses"] == []
    finding = next(f for f in res.findings if f.id == "analysis-requests-failed")
    assert "switched off" in finding.message


def test_plugin_tool_is_requested_by_name_and_shown_with_its_origin(make_config, runs_dir,
                                                                    replay_dir, monkeypatch):
    monkeypatch.syspath_prepend(EXAMPLES)
    cfg = make_config("commercial", plugins=["credit_tools"], agents={"replay_dir": replay_dir({
        "data_scout": _scout({"purpose": "How well does leverage separate defaults?",
                              "tool": "information_value", "params": {"column": "leverage"}})})})
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    eng.run_until_idle()
    eng.submit_gate("gate_intent", "accept", reason="as written")
    state = eng.run_until_idle()
    (a,) = eng.results()["explore"].payload["analyses"]
    assert (a["tool"], a["origin"], a["status"]) == ("information_value", "credit_tools", "ok")
    assert a["summary"]["information_value"] > 0 and a["summary"]["strength"]

    pytest.importorskip("dash_mantine_components")
    import plotly

    from cognos.ui import app as ui
    from cognos.ui.panels import stage_panel

    monkeypatch.setenv("COGNOS_RUNS_DIR", runs_dir)
    panel = json.dumps(stage_panel("explore", eng.results()["explore"], state, "light"),
                       cls=plotly.utils.PlotlyJSONEncoder)
    assert "plugin: credit_tools" in panel and '"type": "Graph"' in panel
    assert "Weight of evidence by band of leverage" in panel or "Weight of evidence" in panel
    assert "Dependent variable" in panel and "Features to consider" in panel
    assert '"c": null' not in panel
    assert ui.gate_payload("gate_data", "edit", {"exclude": [], "target": "default"}, eng.run_id)[0] == {
        "exclude_columns": [], "target": "default"}


def test_cli_starts_from_a_data_file_alone_and_lists_plugins(tmp_path, capsys, monkeypatch):
    data = tmp_path / "loans.csv"
    synth.GENERATORS["commercial"](n=500).to_csv(data, index=False)
    runs = tmp_path / "runs"
    assert main(["run", "--data", str(data), "--runs-dir", str(runs), "--run-id", "r1"]) == 0
    p = service.results("r1", runs)["explore"].payload
    assert p["target"] == "default" and p["source"]["location"] == "loans.csv"
    assert service.state("r1", runs).overrides.target == "default"
    assert main(["run", "--runs-dir", str(runs)]) == 1  # neither a profile nor data
    capsys.readouterr()
    monkeypatch.setenv("COGNOS_PLUGINS", "credit_tools,missing_plugin")
    monkeypatch.syspath_prepend(EXAMPLES)
    from cognos import plugins
    plugins.registry(refresh=True)
    assert main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "target_relationship(column, bins)" in out and "information_value" in out
    assert "snowflake" in out and "fixed_width" in out and "! missing_plugin" in out
    assert main(["init", "-o", str(tmp_path / "cognos.yaml")]) == 0
    assert "allow_code" in (tmp_path / "cognos.yaml").read_text(encoding="utf-8")
