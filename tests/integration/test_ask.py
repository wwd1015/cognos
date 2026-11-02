"""Questions from a page (ADR-0021): the page's agent answers in its own scope, numbers come from
the record, and asking changes nothing in the run."""

from __future__ import annotations

import zipfile

import pytest

from cognos import ask as ask_mod
from cognos import service
from cognos.agents import checks, slices
from cognos.agents.contracts import ASKABLE, FRIENDLY, STAGE_AGENT, AskAnswer


@pytest.fixture(scope="module")
def at_data_gate(tmp_path_factory):
    root = tmp_path_factory.mktemp("ask") / "runs"
    cfg = service.demo_config("commercial", root, n=600, search_budget=6)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.run_until_idle(run_id, root)
    service.submit_gate(run_id, "gate_intent", "accept", reason="the sponsor is still deciding",
                        root=root, background=False)
    st = service.run_until_idle(run_id, root)
    assert st.status_of("gate_data") == "awaiting"
    return root, run_id


def test_the_page_agent_answers_from_the_record_and_the_run_is_unchanged(at_data_gate):
    root, run_id = at_data_gate
    before = service.state(run_id, root).model_dump_json()
    turn = service.ask(run_id, "explore", "How many rows and columns does the dataset have?",
                       root=root, by="alan")
    assert turn["agent"] == "data_analyst" and turn["by"] == "alan"
    assert "{{fact:" not in turn["answer"] and turn["cites"]
    assert all(c["id"].startswith(("intake.", "explore.", "tools.")) for c in turn["cites"])
    assert service.state(run_id, root).model_dump_json() == before  # asking decides nothing
    calls = [a for a in service.audit(run_id, root) if a["agent"] == "data_analyst_ask"]
    assert calls and calls[-1]["call_id"] == turn["call_id"] and calls[-1]["status"] == "ok"
    service.ask(run_id, "explore", "and the target?", root=root, by="sam")
    assert [t["by"] for t in service.ask_thread(run_id, "explore", root)] == ["alan", "sam"]


def test_a_thread_travels_in_the_export(at_data_gate, tmp_path):
    root, run_id = at_data_gate
    service.ask(run_id, "explore", "What is the target?", root=root, by="alan")
    with zipfile.ZipFile(service.export_run(run_id, tmp_path, root)) as z:
        assert any(n.endswith("asks/explore.json") for n in z.namelist())


def test_pages_without_an_agent_and_empty_questions_are_refused(at_data_gate):
    root, run_id = at_data_gate
    with pytest.raises(ValueError, match="no agent"):
        service.ask(run_id, "review", "anything?", root=root)
    with pytest.raises(ValueError, match="question"):
        service.ask(run_id, "explore", "   ", root=root)


@pytest.mark.parametrize("agent", ASKABLE)
def test_a_question_step_keeps_its_agents_walls(agent):
    assert slices.fact_scope(f"{agent}_ask") == slices.fact_scope(agent)
    assert FRIENDLY[f"{agent}_ask"].startswith(FRIENDLY[agent])
    assert set(ASKABLE) == set(STAGE_AGENT.values())


def test_the_page_never_hands_one_agent_anothers_recommendation():
    class Res:
        payload = {"recommendation": {"agent": "modeler", "output": {"rationale": "mine"}}}

    class Ctx:
        def get(self, stage):
            return Res()

    assert ask_mod._page(Ctx(), "model", "done", "modeler")["recommendation"] == {"rationale": "mine"}
    assert ask_mod._page(Ctx(), "model", "done", "validator")["recommendation"] is None


def test_answers_type_no_numbers_and_cite_only_known_facts():
    sl = {"facts": {"explore.n_rows": 600}}

    def errs(**kw):
        return checks.run_checks("data_analyst_ask", AskAnswer(**{"answer": "ok", **kw}), sl)

    assert errs(answer="There are {{fact:explore.n_rows}} rows.", cites=["explore.n_rows"]) == []
    assert any("directly" in e for e in errs(answer="The AUC is 0.78."))
    assert any("unknown fact" in e for e in errs(answer="{{fact:model.cv_mean}}"))
    assert any("unknown fact" in e for e in errs(cites=["validate.verdict"]))
    assert any("empty" in e for e in errs(answer=" "))
    assert any("placeholders" in e for e in errs(suggested_challenge="{{fact:explore.n_rows}}"))


def test_the_window_shows_examples_then_the_thread_and_offers_a_send_back_only_at_an_open_gate():
    from cognos.ui import ask as ask_ui

    def text(parts):
        return str(parts)

    empty = text(ask_ui.thread([], "data_analyst", "explore", gate_open=True))
    assert "ask-example" in empty and "Why is a column excluded?" in empty
    turn = {"question": "Why is dpd out?", "answer": "It is recorded after default.", "by": "alan",
            "at": "2026-11-02T10:15:00Z", "cites": [{"id": "explore.n_rows", "value": 600}],
            "suggested_challenge": "Exclude days_past_due."}
    open_ = text(ask_ui.thread([turn], "data_analyst", "explore", gate_open=True))
    assert "DATA AGENT" in open_ and "explore.n_rows = 600" in open_ and "ask-use" in open_
    closed = text(ask_ui.thread([turn], "data_analyst", "explore", gate_open=False))
    assert "ask-use" not in closed and "not open" in closed
    assert ask_ui.heading("validator", "validate")[0] == "Validation agent"
