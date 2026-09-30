"""The agent layer: contracts, engine checks, facts, independence slices, runner, backends."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cognos.agents import checks, facts, heuristic, providers
from cognos.agents.contracts import (
    CONTRACTS,
    DataAnalystOutput,
    ModelerOutput,
    WriterOutput,
)
from cognos.agents.runner import AgentRunError, AgentRunner, output_schema
from cognos.engine import Engine

# --- facts ------------------------------------------------------------------------------


def test_render_replaces_placeholders_and_flags_bare_metrics():
    f = {"model.cv_mean": 0.781234, "model.champion_family": "logit"}
    text = "A {{fact:model.champion_family}} model with AUC {{fact:model.cv_mean}}."
    assert facts.render(text, f) == "A logit model with AUC 0.7812."
    assert facts.bare_metrics(text) == []
    assert facts.bare_metrics("AUC of 0.78 on 12.5% of the book") == ["0.78", "12.5%"]
    assert facts.bare_metrics("SR 11-7, 5 folds, 2019 vintages") == []


# --- engine checks ------------------------------------------------------------------------


def test_default_check_rejects_unknown_facts_and_missing_challenge_answers():
    out = DataAnalystOutput.model_validate({
        "summary": "s", "column_decisions": [],
        "data_quality": [{"statement": "x", "evidence": ["explore.nope"]}]})
    sl = {"facts": {"explore.n_rows": 10}, "challenges": [{"id": "ch-1"}]}
    errs = checks.default(out, sl)
    assert any("explore.nope" in e for e in errs)
    assert any("ch-1" in e for e in errs)


def test_data_analyst_must_decide_every_suspect_and_never_drop_the_target():
    sl = {"columns": ["a", "b", "y"], "features": ["a", "b"], "leakage_suspects": ["b"],
          "project": {"target": "y"}}
    out = DataAnalystOutput.model_validate({"summary": "s", "column_decisions": [
        {"column": "y", "decision": "exclude", "reason": "r"}]})
    errs = checks.data_analyst(out, sl)
    assert any("'b' has no keep/exclude" in e for e in errs)
    assert any("target" in e for e in errs)


def test_modeler_must_pick_from_admissible_set_and_respect_interpretability():
    sl = {"interpretability": "required", "admissible_set": [
        {"id": "c1", "family": "logit", "role": "candidate", "features": ["a"]},
        {"id": "c2", "family": "gradient_boosting", "role": "challenger", "features": ["a"]}]}
    bad = ModelerOutput.model_validate({"summary": "s", "champion": "c9", "rationale": "r"})
    assert "not in admissible_set" in checks.modeler(bad, sl)[0]
    tree = ModelerOutput.model_validate({"summary": "s", "champion": "c2", "rationale": "r"})
    assert "challenger" in checks.modeler(tree, sl)[0]
    ok = ModelerOutput.model_validate({"summary": "s", "champion": "c1", "rationale": "r",
                                       "sign_checks": [{"feature": "a", "expected": "+",
                                                        "observed": "+", "assessment": "consistent"}]})
    assert checks.modeler(ok, sl) == []


def test_writer_cannot_type_numbers_or_cite_unknown_facts():
    sections = [{"section": s, "markdown": "ok"} for s in
                ("methodology_rationale", "alternatives_considered", "limitations",
                 "use_and_monitoring")]
    out = WriterOutput.model_validate({"summary": "s", "sections": [
        {"section": "executive_summary", "markdown": "AUC 0.81 and {{fact:model.nope}}"},
        *sections]})
    errs = checks.writer(out, {"facts": {"model.cv_mean": 0.81}})
    assert any("0.81" in e for e in errs)
    assert any("model.nope" in e for e in errs)


def test_output_schemas_are_strict_grammar_friendly():
    for contract in CONTRACTS.values():
        schema = output_schema(contract)
        text = json.dumps(schema)
        assert '"minimum"' not in text and '"maximum"' not in text

        def closed(o):
            if isinstance(o, dict):
                if o.get("type") == "object" and "properties" in o:
                    assert o["additionalProperties"] is False
                for v in o.values():
                    closed(v)
            elif isinstance(o, list):
                for v in o:
                    closed(v)

        closed(schema)


def test_sign_priors():
    assert heuristic.sign_prior("debt_to_ebitda") == "+"
    assert heuristic.sign_prior("interest_coverage") == "-"
    assert heuristic.sign_prior("zip_code") == "none"


# --- independence ---------------------------------------------------------------------------


def test_modeler_never_sees_holdout_and_validator_never_sees_modeler_rationale(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    eng.run_until_idle()
    inputs = {}
    for p in sorted((eng.run_dir / "agents").glob("*.input.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        inputs[rec["agent"]] = rec["context"]
    assert not [k for k in inputs["modeler"]["facts"] if k.startswith(("model.", "backtest."))]
    assert "holdout" not in json.dumps(inputs["modeler"]).lower().replace("holdout_", "")
    modeler_rationale = eng.results()["model"].payload["recommendation"]["output"]["rationale"]
    assert modeler_rationale not in json.dumps(inputs["validator"])


# --- runner ---------------------------------------------------------------------------------


def _modeler_slice_data():
    return {"admissible_set": [{"id": "c1", "family": "logit", "role": "candidate",
                                "features": ["a"], "coefficient_signs": [], "label": "logit(1f)"}],
            "interpretability": "required", "metric": "roc_auc"}


def test_runner_retries_with_errors_then_succeeds(make_config, runs_dir):
    cfg = make_config("classification")
    eng = Engine(cfg, runs_root=runs_dir)
    answers = iter([
        {"summary": "s", "champion": "c7", "rationale": "r"},
        {"summary": "s", "champion": "c1", "rationale": "r"},
    ])
    seen_prompts = []

    def backend(*, prompt, **_):
        seen_prompts.append(prompt)
        return next(answers), {"model": "fake", "cost_usd": 0.01}

    runner = AgentRunner(providers.get("heuristic"), eng.run_dir, backend=backend)
    ctx = eng.context()
    out = runner.recommend(ctx, "modeler", _modeler_slice_data())
    assert out.champion == "c1"
    assert "previous answer was rejected" in seen_prompts[1] and "c7" in seen_prompts[1]
    audit = [json.loads(x) for x in (eng.run_dir / "agents" / "audit.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert [a["status"] for a in audit] == ["invalid", "ok"]
    assert len(list((eng.run_dir / "agents").glob("*modeler*.input.json"))) == 2


def test_runner_gives_up_after_retries(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    runner = AgentRunner(providers.get("heuristic"), eng.run_dir, max_retries=1,
                         backend=lambda **_: ({"summary": "s", "champion": "zz", "rationale": "r"}, {}))
    with pytest.raises(AgentRunError) as err:
        runner.recommend(eng.context(), "modeler", _modeler_slice_data())
    assert "not in admissible_set" in str(err.value)


def test_budget_is_enforced_for_paid_providers(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    (eng.run_dir / "agents").mkdir(exist_ok=True)
    (eng.run_dir / "agents" / "audit.jsonl").write_text(json.dumps({"cost_usd": 6.0}) + "\n",
                                                         encoding="utf-8")
    runner = AgentRunner(providers.get("anthropic"), eng.run_dir, budget_usd=5.0,
                         backend=lambda **_: ({}, {}))
    with pytest.raises(AgentRunError, match="budget"):
        runner.recommend(eng.context(), "modeler", _modeler_slice_data())


# --- providers ------------------------------------------------------------------------------


def test_auto_falls_back_to_heuristic_and_unavailable_names_alternatives(monkeypatch):
    monkeypatch.delenv("COGNOS_PROVIDER", raising=False)
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY",
                "OLLAMA_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(providers.shutil, "which", lambda _: None)
    assert providers.resolve("auto")["id"] == "heuristic"
    with pytest.raises(providers.ProviderUnavailable, match="Available: heuristic, replay"):
        providers.resolve("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert providers.resolve("auto")["id"] == "anthropic"
    assert providers.resolve("anthropic", "claude-sonnet-5-5")["prices"]["input"] == 2.0


# --- backends (no network) --------------------------------------------------------------------


def test_claude_cli_argv_isolates_the_agent():
    from cognos.agents.backends import claude_cli

    argv = claude_cli.argv("sys.md", {"type": "object"}, {"model": "opus"})
    joined = " ".join(argv)
    for flag in ("-p", "--system-prompt-file", "--json-schema", "--setting-sources",
                 "--strict-mcp-config", "--no-session-persistence", "--tools"):
        assert flag in argv, flag
    assert argv[argv.index("--tools") + 1] == ""
    assert "--output-format json" in joined


class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.message


def test_anthropic_backend_uses_structured_output_and_prices_usage(tmp_path):
    from cognos.agents.backends import anthropic_api

    captured = {}
    message = SimpleNamespace(
        stop_reason="end_turn", model="claude-opus-5-5",
        content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0,
                              cache_read_input_tokens=0),
        to_dict=lambda: {"ok": True})

    def stream(**kw):
        captured.update(kw)
        return _FakeStream(message)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)),
                             messages=SimpleNamespace(stream=stream))
    prov = providers.get("anthropic")
    answer, meta = anthropic_api.call("sys", "prompt", {"type": "object"}, prov, None,
                                      tmp_path / "t.json", client=client)
    assert answer == {"summary": "ok"}
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["fallbacks"] == "default"
    assert meta["cost_usd"] == pytest.approx((1000 * 4 + 500 * 20) / 1e6)


def test_anthropic_backend_surfaces_refusals(tmp_path):
    from cognos.agents.backends import anthropic_api

    message = SimpleNamespace(stop_reason="refusal", stop_details="cyber", content=[],
                              usage=None, to_dict=lambda: {})
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **kw: _FakeStream(message))))
    with pytest.raises(RuntimeError, match="declined"):
        anthropic_api.call("s", "p", {}, providers.get("anthropic"), None, tmp_path / "t.json",
                           client=client)


def test_openai_compat_reads_submit_answer_call(tmp_path):
    from cognos.agents.backends import openai_compat

    call = SimpleNamespace(function=SimpleNamespace(name="submit_answer",
                                                    arguments='{"summary": "ok"}'))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        model_dump=lambda: {})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: resp)))
    prov = dict(providers.get("xai"))
    answer, meta = openai_compat.call("s", "p", {"$defs": {}, "type": "object"}, None, prov, None,
                                      tmp_path / "t.json", client=client)
    assert answer == {"summary": "ok"} and meta["turns"] == 1
    assert meta["cost_usd"] == pytest.approx((100 * 2 + 50 * 6) / 1e6)
