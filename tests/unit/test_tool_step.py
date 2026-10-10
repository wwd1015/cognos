"""Stage tools: what a plugin may declare, the tool-request step every stage agent has, and the
IMPACT placeholder."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognos import plugins
from cognos.agents import checks, heuristic, runner, slices
from cognos.agents.contracts import AGENT_STAGE, CONTRACTS, TOOL_USERS, tool_step_of
from cognos.analysis import run as analysis
from cognos.integrations import impact_tools
from cognos.plugins import Registry, Tool

EXAMPLES = str(Path(__file__).resolve().parents[2] / "examples" / "plugins")


def _tool(**kw) -> Tool:
    return Tool("t", "a tool", lambda df, params, env: {}, **kw)


# --- what a tool may declare ----------------------------------------------------------------
def test_a_tool_cannot_ask_for_an_input_its_stage_does_not_have():
    reg = Registry()
    reg.add_tool(_tool(stages=("validate", "backtest"), needs=("model", "holdout")))
    reg.add_tool(_tool(stages=("model",), needs=("train", "data", "results")))
    # nothing before backtest is handed the sealed holdout or the fitted champion
    for stage in ("intake", "explore", "ideate", "model"):
        for need in ("holdout", "model"):
            with pytest.raises(ValueError, match="not available that early"):
                reg.add_tool(_tool(stages=(stage,), needs=(need,)))
    with pytest.raises(ValueError, match="unknown stage"):
        reg.add_tool(_tool(stages=("deploy",)))
    with pytest.raises(ValueError, match="unknown input"):
        reg.add_tool(_tool(needs=("secrets",)))
    # a tool written against the first plugin API is an explore tool over the dataset
    old = plugins.AnalysisTool("old", "d", lambda df, params, env: {}, {"column": "c"}, True)
    assert (old.stages, old.needs) == (("explore",), ("data",))


def test_tools_are_offered_per_stage_and_an_unavailable_one_is_listed_not_offered():
    reg = Registry()
    reg.add_tool(_tool(stages=("validate",), needs=()))
    reg.add_tool(Tool("gone", "needs a library", lambda *a: {}, stages=("validate",), needs=(),
                      available=lambda: (False, "library not installed")))
    reg.add_tool(Tool("broken", "probe raises", lambda *a: {}, stages=("validate",), needs=(),
                      available=lambda: 1 / 0))
    assert [t.name for t in reg.tools_for("validate")] == ["t"] and reg.tools_for("model") == []
    assert {t["name"]: t["note"] for t in reg.unavailable_for("validate")} == {
        "gone": "library not installed", "broken": "ZeroDivisionError: division by zero"}


def test_impact_is_a_registered_placeholder_a_plugin_can_replace(monkeypatch, tmp_path):
    reg = plugins.registry(refresh=True)
    tool = reg.tools[impact_tools.NAME]
    assert tool.stages == ("validate",) and tool.status() == (False, impact_tools.NOT_READY)
    assert impact_tools.NAME not in [t.name for t in reg.tools_for("validate")]
    listed = {t["name"]: t for t in plugins.public_list()["tools"]}[impact_tools.NAME]
    assert listed["available"] is False and listed["stages"] == ["validate"]
    # the day IMPACT ships its tests, a plugin registers under the same name
    (tmp_path / "impact_plugin.py").write_text(
        "from cognos.plugins import Tool\n"
        "def register(r):\n"
        "    r.add_tool(Tool('impact_test_suite', 'real', lambda df, p, e: {}, "
        "stages=('validate',), needs=('model', 'holdout')))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = plugins.registry(["impact_plugin"], refresh=True)
    assert reg.tools[impact_tools.NAME].origin == "impact_plugin"
    assert impact_tools.NAME in [t.name for t in reg.tools_for("validate")]


def test_a_plugin_with_a_badly_declared_tool_is_reported_not_loaded(monkeypatch, tmp_path):
    (tmp_path / "early_plugin.py").write_text(
        "from cognos.plugins import Tool\n"
        "def register(r):\n"
        "    r.add_tool(Tool('peek', 'd', lambda df, p, e: {}, stages=('model',), "
        "needs=('holdout',)))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = plugins.registry(["early_plugin"], refresh=True)
    assert "peek" not in reg.tools and "not available that early" in reg.problems[0]["error"]


def test_example_plugin_serves_three_stages(monkeypatch):
    monkeypatch.syspath_prepend(EXAMPLES)
    reg = plugins.registry(["validation_tools"], refresh=True)
    assert reg.problems == []
    assert [t.name for t in reg.tools_for("validate")] == ["score_band_monotonicity"]
    assert [t.name for t in reg.tools_for("backtest")] == ["score_band_monotonicity"]
    assert [t.name for t in reg.tools_for("ideate")] == ["events_per_feature"]
    # explore keeps to the tools registered for it
    assert "score_band_monotonicity" not in [t.name for t in reg.tools_for("explore")]


def test_a_tool_can_fail_a_check_but_never_block():
    got = analysis._checks([{"name": "a", "passed": False, "severity": "critical", "detail": "x"},
                            {"name": "b", "passed": 1, "severity": "nonsense"}, "junk"])
    assert [(c["name"], c["passed"], c["severity"]) for c in got] == [
        ("a", False, "high"), ("b", True, "medium")]


# --- the tool-request step ------------------------------------------------------------------
def _slice(**kw) -> dict:
    return {"tools": [{"name": "band_test", "description": "d", "params": {"bands": "n"},
                       "origin": "p"},
                      {"name": "other", "description": "d", "params": {}, "origin": "cognos"}],
            "max_requests": 2, "tool_runs": [], "facts": {}, "challenges": [], **kw}


def _checked(agent: str, sl: dict, requests: list[dict], done: bool = True) -> list[str]:
    out = CONTRACTS[agent].model_validate({"requests": requests, "done": done})
    return checks.run_checks(agent, out, sl)


def _req(tool="band_test", purpose="why", **params) -> dict:
    return {"purpose": purpose, "tool": tool,
            "params": [{"name": k, "value": v} for k, v in params.items()]}


@pytest.mark.parametrize("role", TOOL_USERS)
def test_every_stage_agent_has_a_tool_step_and_the_heuristic_passes_its_checks(role):
    agent = f"{role}_tools"
    assert tool_step_of(agent) == role and AGENT_STAGE[agent] == AGENT_STAGE[role]
    sl = _slice()
    out = heuristic.recommend(agent, sl)
    assert [r["tool"] for r in out["requests"]] == ["band_test", "other"] and out["done"]
    assert _checked(agent, sl, out["requests"]) == []
    # nothing is run twice
    ran = _slice(tool_runs=[{"tool": "band_test", "params": {}}, {"tool": "other", "params": {}}])
    assert heuristic.recommend(agent, ran)["requests"] == []
    # the guidance is COGNOS's own: the role prompt plus the shared tool-use skill
    prompt = runner.system_prompt(agent)
    assert runner.system_prompt(role).split("\n")[0] in prompt
    assert "requesting tool runs" in prompt and "context.tools" in runner.task_of(agent)


def test_engine_rejects_requests_it_cannot_run():
    sl = _slice()
    assert any("not a tool offered" in e for e in _checked("validator_tools", sl, [_req("nope")]))
    assert any("no parameter" in e for e in _checked("validator_tools", sl, [_req(depth="3")]))
    assert any("purpose" in e for e in _checked("validator_tools", sl, [_req(purpose=" ")]))
    assert any("at most 2" in e for e in _checked(
        "validator_tools", sl, [_req(bands="3"), _req(bands="4"), _req("other")]))
    assert any("already" in e for e in _checked("validator_tools", sl, [_req(), _req()]))
    ran = _slice(tool_runs=[{"tool": "band_test", "params": {"bands": "5"}}])
    assert any("already" in e for e in _checked("validator_tools", ran, [_req(bands="5")]))
    assert _checked("validator_tools", ran, [_req(bands="10")]) == []
    assert _checked("validator_tools", sl, [], done=True) == []  # asking for nothing is valid


def test_tool_results_follow_the_independence_scopes():
    modeler = slices.fact_scope("modeler")
    assert "tools.model." in modeler and "tools.explore." in modeler
    assert not any(p in modeler for p in ("model.", "backtest.", "tools.backtest.",
                                          "tools.validate.", "prior.", "tools.prior."))
    assert slices.fact_scope("modeler_tools") == modeler
    validator = slices.fact_scope("validator")
    assert {"tools.model.", "tools.backtest.", "tools.validate."} <= set(validator)
    assert "tools.comply." not in validator
    assert "tools.document." in slices.fact_scope("writer")
