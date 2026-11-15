"""Seats, the design brief, and the sealed package. Mechanical: no agent judgment."""

from __future__ import annotations

import json

import pytest

from cognos.engine import Engine, GateError
from cognos.engine.gates import apply_answers
from cognos.engine.process import (
    digest_of,
    journal,
    material,
    missing_design,
    next_action,
    preparation_of,
    resolve_seat,
)
from cognos.engine.state import Gap, GateDecision, PackageSeal, RunState


def test_a_seat_only_acts_on_its_own_gate():
    assert resolve_seat("gate_data", None) == "developer"
    assert resolve_seat("gate_validation", "reviewer") == "reviewer"
    assert resolve_seat("gate_signoff", "") == "approver"
    with pytest.raises(GateError, match="model developer"):
        resolve_seat("gate_champion", "approver")
    with pytest.raises(GateError, match="approver"):
        resolve_seat("gate_signoff", "developer")


def test_core_design_cannot_be_assumed():
    state = RunState(run_id="r")
    state.gaps.append(Gap(id="design-use_case", question="Use?", design_field="use_case"))
    state.gaps.append(Gap(id="data-x", question="Is this late?", category="data"))
    with pytest.raises(GateError, match="cannot be accepted as an assumption"):
        apply_answers(state, {"design-use_case": ""}, assume=True)
    assert state.gap("design-use_case").status == "open"
    assert apply_answers(state, {"data-x": ""}, assume=True) == ["validate"]
    assert state.gap("data-x").status == "assumed"
    # a limitation is the third way to close a question; the core four take an answer only
    state.gaps.append(Gap(id="data-y", question="Is the collateral revalued?", category="data"))
    with pytest.raises(GateError, match="or a limitation"):
        apply_answers(state, {"design-use_case": "n/a"}, limitation=True)
    assert apply_answers(state, {"data-y": "Not known at origination."}, limitation=True) == [
        "validate"]
    gap = state.gap("data-y")
    assert (gap.status, gap.answer) == ("limitation", "Not known at origination.")
    assert state.overrides.design == {}  # neither fills the design brief


def test_express_preparation_is_not_a_human_signature():
    state = RunState(run_id="r", status="completed", mode="autonomous")
    state.decisions.append(GateDecision(gate="gate_data", action="accept", actor="auto",
                                        seat="express"))
    assert preparation_of(state) == "express"
    assert next_action(state)["kind"] == "prepared"
    state.decisions.append(GateDecision(gate="gate_design", action="accept", actor="human",
                                        seat="developer"))
    assert preparation_of(state) == "human"


def test_journal_orders_people_agents_and_the_engine():
    state = RunState(run_id="r")
    state.decisions.append(GateDecision(gate="gate_data", action="accept", actor="human",
                                        seat="developer", reason="keep them", at="2026-01-02T00:00:00+00:00"))
    rows = journal(state, {})
    assert rows[0]["who"] == "person" and rows[0]["name"] == "developer"
    assert "keep them" in rows[0]["what"]


def test_journal_lists_reopens_retries_and_takeovers():
    from cognos.engine.state import StepAction

    state = RunState(run_id="r")
    assert journal(state, {}) == []  # a run recorded before actions were kept shows none
    state.actions += [
        StepAction(kind="retry", step="model", by="cy", note="failed", at="2026-01-03T00:00:00+00:00"),
        StepAction(kind="reopen", step="gate_data", by="bo", seat="developer",
                   at="2026-01-02T00:00:00+00:00"),
        StepAction(kind="takeover", step="explore", by="di", note="al@laptop",
                   at="2026-01-04T00:00:00+00:00")]
    rows = journal(RunState.model_validate_json(state.model_dump_json()), {})
    assert [r["name"] for r in rows] == ["bo (developer)", "cy", "di"]  # in the order they happened
    assert all(r["who"] == "person" for r in rows)
    assert "re-opened for revision" in rows[0]["what"]
    assert "asked to run again after it failed" in rows[1]["what"]
    assert "taken over from al@laptop" in rows[2]["what"]


def test_reset_supersedes_a_sealed_package(make_config, runs_dir):
    eng = Engine(make_config("regression"), runs_root=runs_dir, mode="interactive")
    state = eng.state
    state.status = "approved"
    state.package = PackageSeal(version=1, digest="abc", preparation="human",
                                 status="sealed", path="packages/v1.json")
    state.save(eng.run_dir)
    eng.reset()
    state = eng.state
    assert state.package is not None and state.package.status == "superseded"
    assert state.package.path == "packages/v1.json"
    assert state.status == "created"


def test_reviewer_cannot_take_the_developers_gate(make_config, runs_dir, confirm_intent):
    eng = Engine(make_config("regression"), runs_root=runs_dir, mode="interactive")
    confirm_intent(eng)
    assert eng.state.status_of("gate_data") == "awaiting"
    with pytest.raises(GateError, match="model developer"):
        eng.submit_gate("gate_data", "accept", seat="reviewer")
    assert eng.state.status_of("gate_data") == "awaiting"
    eng.submit_gate("gate_data", "accept", seat="developer")
    assert eng.state.last_decision("gate_data").seat == "developer"


def test_approve_seals_a_package_and_a_later_edit_does_not_rewrite_it(make_config, runs_dir, apply_brief):
    eng = Engine(apply_brief(make_config("classification")), runs_root=runs_dir, mode="interactive")
    for _ in range(20):
        state = eng.run_until_idle()
        waiting = [g for g in ("gate_intent", "gate_data", "gate_design", "gate_candidates",
                               "gate_champion", "gate_validation", "gate_signoff")
                   if state.status_of(g) == "awaiting"]
        if not waiting:
            break
        action = "approve" if waiting[0] == "gate_signoff" else "accept"
        if waiting[0] == "gate_signoff":
            bare = make_config("classification")
            assert missing_design(bare, RunState(run_id="x")) == [
                "use_case", "horizon", "default_definition", "segment"]
        eng.submit_gate(waiting[0], action, reason="signed" if action == "approve" else "ok")
    state = eng.state
    assert state.status == "approved" and state.package is not None
    assert state.package.status == "sealed" and state.package.preparation == "human"
    path = eng.run_dir / state.package.path
    original = path.read_text(encoding="utf-8")
    assert json.loads(original)["digest"] == state.package.digest
    body = material(eng.config, state, eng.results())
    assert digest_of(body) == state.package.digest
    # every decision is bound: the data, the target, the join, the forms, the model form
    assert {"data", "target", "task", "join", "exclude_columns", "column_forms", "model_form",
            "screen_exceptions", "slate", "plan", "pool", "champion_id"} <= set(body)
    assert body["target"] and body["task"]
    other = state.model_copy(deep=True)
    other.overrides.transforms = {"x": {"kind": "log"}}
    assert digest_of(material(eng.config, other, eng.results())) != state.package.digest
    eng.reopen("gate_signoff", seat="approver")
    assert eng.state.package.status == "superseded"
    assert eng.state.status != "approved"
    assert path.read_text(encoding="utf-8") == original  # the sealed file is never rewritten


_GIVEN = {"source": {"kind": "sql", "location": "loans", "options": {
    "warehouse": "W", "api_token": "abc123", "password_env": "MY_PW",
    "client_secret": "env:MY_SECRET"}}}


def test_a_package_binds_the_data_with_secret_named_literals_replaced(make_config, runs_dir, tmp_path):
    """A literal source option named like a secret is replaced in the material; the run's own
    overrides keep it (the engine reads them back); the sealed file verifies against itself."""
    from cognos.datasources import LEFT_OUT
    from cognos.engine.process import seal_package

    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    state = eng.state.model_copy(deep=True)
    state.overrides.data = json.loads(json.dumps(_GIVEN))
    body = material(eng.config, state, {})
    assert body["data"]["source"]["options"] == {
        "warehouse": "W", "api_token": LEFT_OUT, "password_env": "MY_PW",
        "client_secret": "env:MY_SECRET"}
    assert state.overrides.data == _GIVEN  # the state keeps the value
    seal = seal_package(state, eng.config, {}, tmp_path)
    sealed = json.loads((tmp_path / seal.path).read_text(encoding="utf-8"))
    assert "abc123" not in json.dumps(sealed)
    assert digest_of(sealed["material"]) == sealed["digest"] == seal.digest


def test_a_run_without_secret_named_options_binds_its_data_as_before(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    state = eng.state.model_copy(deep=True)
    for data in (None, {}, {"source": {"kind": "csv", "location": "x.csv",
                                       "options": {"sep": ";", "password_env": "PW"}},
                            "sources": [{"name": "a", "options": {"token": "env:T"}}]}):
        state.overrides.data = data
        body = material(eng.config, state, {})
        assert body["data"] == data and body["data"] is data
        assert digest_of(body) == digest_of({**body, "data": state.overrides.data})


def test_the_value_of_a_secret_is_not_a_recorded_fact(make_config, runs_dir):
    """Two runs that differ only in the literal value of a secret-named option bind the same
    digest: the secret's value is not a recorded fact. An ordinary option still counts."""
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")

    def digest(**options):
        state = eng.state.model_copy(deep=True)
        given = json.loads(json.dumps(_GIVEN))
        given["source"]["options"].update(options)
        state.overrides.data = given
        return digest_of(material(eng.config, state, {}))

    assert digest(api_token="abc123") == digest(api_token="other")
    assert digest(warehouse="W") != digest(warehouse="X")


def test_open_brief_cannot_be_signed(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    for _ in range(20):
        state = eng.run_until_idle()
        waiting = [g for g in ("gate_intent", "gate_data", "gate_design", "gate_candidates",
                               "gate_champion", "gate_validation", "gate_signoff")
                   if state.status_of(g) == "awaiting"]
        if not waiting:
            break
        if waiting[0] == "gate_signoff":
            with pytest.raises(GateError, match="design brief"):
                eng.submit_gate("gate_signoff", "approve", reason="too soon", seat="approver")
            assert eng.state.package is None and eng.state.status != "approved"
            return
        eng.submit_gate(waiting[0], "accept", reason="ok")
    pytest.fail("sign-off never opened")
