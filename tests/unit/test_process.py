"""Seats, the design brief, and the sealed package. Mechanical: no agent judgment."""

from __future__ import annotations

import json

import pytest

from cognos.engine import Engine, GateError
from cognos.engine.gates import apply_answers
from cognos.engine.process import (
    journal,
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
    apply_answers(state, {"data-x": ""}, assume=True)
    assert state.gap("data-x").status == "assumed"


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
        waiting = [g for g in ("gate_intent", "gate_data", "gate_design", "gate_champion",
                               "gate_validation", "gate_signoff")
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
    eng.reopen("gate_signoff", seat="approver")
    assert eng.state.package.status == "superseded"
    assert eng.state.status != "approved"
    assert path.read_text(encoding="utf-8") == original  # the sealed file is never rewritten


def test_open_brief_cannot_be_signed(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    for _ in range(20):
        state = eng.run_until_idle()
        waiting = [g for g in ("gate_intent", "gate_data", "gate_design", "gate_champion",
                               "gate_validation", "gate_signoff")
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
