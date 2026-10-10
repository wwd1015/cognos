"""Plugin tools at every stage, end to end: the stage's agent requests them, the engine runs
them with exactly the inputs they declared, and the results are kept, cited, shown and delivered."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from cognos import plugins, service
from cognos.engine import Engine
from cognos.integrations import impact_tools

EXAMPLES = str(Path(__file__).resolve().parents[2] / "examples" / "plugins")

PLUGIN = '''
from cognos.plugins import Tool

SEEN = {}

def _see(name):
    def fn(df, params, env):
        SEEN[name] = {"df": df is not None, "env": sorted(env), "params": dict(params),
                      "results": sorted(env.get("results", {})),
                      "prior": "prior" in env.get("results", {}).get("intake", {})}
        if name == "explode":
            raise RuntimeError("the test library is down")
        out = {"title": name, "summary": {"n": 7}}
        if name == "hard_test":
            out["checks"] = [{"name": "limit", "passed": False, "severity": "critical",
                              "detail": "over the limit"}]
        if name == "doc_scan":
            out["summary"] = {"n_documents": len(env["documents"])}
        if name == "train_only":
            out["summary"] = {"n_rows": len(env["train"])}
        return out
    return fn

def register(r):
    r.add_tool(Tool("doc_scan", "Scan the documents.", _see("doc_scan"), stages=("intake",),
                    needs=("documents",)))
    r.add_tool(Tool("train_only", "Look at the development sample.", _see("train_only"),
                    stages=("model",), needs=("train", "results")))
    r.add_tool(Tool("hard_test", "A custom test that fails.", _see("hard_test"),
                    {"limit": "the limit"}, stages=("validate",), needs=("model", "holdout")))
    r.add_tool(Tool("explode", "A tool that raises.", _see("explode"), stages=("validate",),
                    needs=()))
    r.add_tool(Tool("readiness", "A readiness check.", _see("readiness"),
                    stages=("comply", "document"), needs=("results",)))
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    (tmp_path / "bank_tests.py").write_text(PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "bank_tests"
    plugins._cache.clear()
    sys.modules.pop("bank_tests", None)


def _tool_calls(eng) -> list[str]:
    return [a["agent"] for a in service.audit(eng.run_id, eng.run_dir.parent)
            if a["agent"].endswith("_tools")]


def test_without_tools_for_a_stage_no_agent_is_asked(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    assert eng.run_until_idle().status == "completed"
    assert _tool_calls(eng) == []  # the step costs nothing unless a tool is registered
    vp = eng.results()["validate"].payload
    assert vp["tool_runs"] == []
    # ...and the validation record still says which registered test was not run, and why
    assert vp["tools_unavailable"] == [{
        "name": impact_tools.NAME, "origin": "cognos", "note": impact_tools.NOT_READY,
        "description": vp["tools_unavailable"][0]["description"]}]
    doc = (eng.run_dir / "docs" / "analysis.md").read_text(encoding="utf-8")
    assert "## Tools registered but not run" in doc and impact_tools.NAME in doc


def test_example_plugin_runs_custom_tests_at_design_outcomes_and_validation(
        make_config, runs_dir, monkeypatch):
    monkeypatch.syspath_prepend(EXAMPLES)
    eng = Engine(make_config("commercial", plugins=["validation_tools"]), runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "completed"
    results = eng.results()
    assert set(_tool_calls(eng)) == {"design_lead_tools", "outcomes_analyst_tools",
                                     "validator_tools"}
    assert [r["tool"] for r in results["ideate"].payload["tool_runs"]] == ["events_per_feature"]
    run = results["validate"].payload["tool_runs"][0]
    assert (run["id"], run["tool"], run["status"], run["origin"], run["version"]) == (
        "t1", "score_band_monotonicity", "ok", "validation_tools", "0.1")
    assert run["needs"] == ["model", "holdout"] and run["checks"][0]["name"] == "monotonic"
    kept = json.loads((eng.run_dir / run["artifact"]).read_text(encoding="utf-8"))
    assert kept["charts"][0]["kind"] == "bar" and kept["tables"][0]["columns"][0] == "band"
    # the validator was handed the result as facts it can cite
    call = next(p for p in sorted((eng.run_dir / "agents").glob("*_validator_a1_*.input.json")))
    ctx = json.loads(call.read_text(encoding="utf-8"))["context"]
    assert ctx["tool_runs"][0]["tool"] == "score_band_monotonicity"
    assert "tools.validate.t1.reversals" in ctx["facts"]
    assert ctx["facts"]["tools.validate.t1.check.monotonic"] in ("passed", "failed")
    assert "tools.ideate.t1.events_per_feature" in ctx["facts"]
    # delivered: white paper and export
    doc = (eng.run_dir / "docs" / "analysis.md").read_text(encoding="utf-8")
    assert "## Tools run by stage" in doc and "plugin (validation_tools 0.1)" in doc
    with zipfile.ZipFile(service.export_run(eng.run_id, root=runs_dir)) as z:
        assert any(n.endswith("stages/validate/analyses/t1.json") for n in z.namelist())


def test_each_stage_tool_gets_only_what_it_declared(make_config, runs_dir, plugin, tmp_path):
    intent = tmp_path / "intent.md"
    intent.write_text("## Use case\norigination underwriting\n", encoding="utf-8")
    cfg = make_config("commercial", plugins=[plugin],
                      engagement={"kind": "new", "intent": str(intent)})
    eng = Engine(cfg, runs_root=runs_dir)
    state = eng.run_until_idle()
    results = eng.results()
    import bank_tests

    seen = bank_tests.SEEN
    # intake: the documents, never the dataset
    assert seen["doc_scan"]["df"] is False and "documents" in seen["doc_scan"]["env"]
    assert results["intake"].payload["tool_runs"][0]["summary"] == {"n_documents": 1}
    # model: the development sample and upstream results only; the choice stays blind
    env = seen["train_only"]["env"]
    assert "train" in env and not {"holdout", "score", "model_path"} & set(env)
    assert seen["train_only"]["results"] == ["explore", "ideate", "intake"]
    assert seen["train_only"]["prior"] is False
    call = next(p for p in sorted((eng.run_dir / "agents").glob("*_modeler_a1_*.input.json")))
    facts = json.loads(call.read_text(encoding="utf-8"))["context"]["facts"]
    assert "tools.model.t1.n_rows" in facts
    assert not [k for k in facts if k.startswith(("model.", "backtest.", "tools.validate."))]
    # validation: the champion and the sealed holdout
    assert {"holdout", "score", "model_path"} <= set(seen["hard_test"]["env"])

    # a failed check is a finding; HIGH fails the validation; a tool can never block
    va = results["validate"]
    finding = next(f for f in va.findings if f.id == "tool-t1-limit")
    assert finding.severity.value == "HIGH" and finding.category == "tool/hard_test"
    assert va.verdict.value == "FAIL" and "over the limit" in finding.message
    assert state.status == "completed"  # autonomous preparation carries on; nothing blocked
    # a tool that raises is a recorded failure, not a failed stage
    runs = {r["tool"]: r for r in va.payload["tool_runs"]}
    assert runs["explode"]["status"] == "error" and "test library is down" in runs["explode"]["error"]
    assert "tools.validate.t2.n" not in json.dumps(va.payload)
    # one tool can serve several stages
    assert results["comply"].payload["tool_runs"][0]["tool"] == "readiness"
    assert results["document"].payload["tool_runs"][0]["tool"] == "readiness"
    doc = (eng.run_dir / "docs" / "analysis.md").read_text(encoding="utf-8")
    assert "FAILED: limit" in doc and "| model | t1 | train_only | plugin (bank_tests)" in doc


def test_a_live_agents_requests_are_checked_retried_and_run_with_parameters(
        make_config, runs_dir, plugin, replay_dir):
    def ask(tool, done=True, **params):
        return {"requests": [{"purpose": "Is the limit respected?", "tool": tool,
                              "params": [{"name": k, "value": v} for k, v in params.items()]}],
                "done": done, "notes": ""}

    cfg = make_config("classification", plugins=[plugin], agents={
        "max_retries": 1, "replay_dir": replay_dir({
            "validator_tools": [ask("score_everything"), ask("hard_test", done=False, limit="3"),
                                {"requests": [], "done": True, "notes": "enough"}]})})
    eng = Engine(cfg, runs_root=runs_dir)
    eng.run_until_idle()
    audit = [a for a in service.audit(eng.run_id, runs_dir) if a["agent"] == "validator_tools"]
    assert [a["status"] for a in audit] == ["invalid", "ok", "ok"]
    assert "not a tool offered" in audit[0]["error"]
    runs = eng.results()["validate"].payload["tool_runs"]
    assert [(r["tool"], r["params"]) for r in runs] == [("hard_test", {"limit": "3"})]
    import bank_tests

    assert bank_tests.SEEN["hard_test"]["params"] == {"limit": "3"}


def test_stage_tools_can_be_switched_off(make_config, runs_dir, plugin):
    eng = Engine(make_config("classification", plugins=[plugin],
                             analysis={"stage_tools": False}), runs_root=runs_dir)
    eng.run_until_idle()
    assert _tool_calls(eng) == []
    assert all(not (r.payload or {}).get("tool_runs") for r in eng.results().values())
