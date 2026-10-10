"""The v1 workflow: agents recommend, humans decide at gates, the engine disposes.

Driven through :class:`cognos.engine.Engine` with the deterministic (heuristic) agents, or with
recorded outputs (replay) where a test needs a specific agent answer.
"""

from __future__ import annotations

import json

import pytest

from cognos.artifacts import Verdict
from cognos.engine import Engine, GateError
from cognos.engine.graph import GATES


def _accept_all(eng: Engine, until: str | None = None):
    """Accept every gate as it opens; stop when ``until`` is awaiting."""
    for _ in range(20):
        state = eng.run_until_idle()
        waiting = [g for g in GATES if state.status_of(g) == "awaiting"]
        if not waiting:
            return state
        if waiting[0] == until:
            return state
        action = "approve" if waiting[0] == "gate_signoff" else "accept"
        eng.submit_gate(waiting[0], action, reason="looks right")
    return eng.state


def test_autonomous_run_completes_with_auto_decisions(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "completed"
    assert all(state.status_of(s) == "done" for s in state.steps)
    assert {d.gate for d in state.decisions} == set(GATES)
    assert all(d.actor == "auto" and d.seat == "express" for d in state.decisions)
    assert state.package is None  # express preparation is not a signature
    # every agent call is audited; the white paper carries a decision log
    audit = (eng.run_dir / "agents" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(x)["agent"] for x in audit} >= {"data_analyst", "design_lead", "modeler",
                                                         "outcomes_analyst", "validator",
                                                         "risk_analyst", "writer"}
    assert (eng.run_dir / "docs" / "decisions.md").exists()
    narrative = (eng.run_dir / "docs" / "narrative.md").read_text(encoding="utf-8")
    assert "{{fact:" not in narrative  # every placeholder rendered by the engine


def test_interactive_run_pauses_at_each_gate_and_signs_off(make_config, runs_dir, apply_brief):
    eng = Engine(apply_brief(make_config("classification")), runs_root=runs_dir, mode="interactive")
    state = eng.run_until_idle()
    assert state.status == "awaiting" and state.status_of("gate_intent") == "awaiting"
    assert state.status_of("explore") == "pending"  # no data is touched before the intent stands
    eng.submit_gate("gate_intent", "accept")
    state = eng.run_until_idle()
    assert state.status == "awaiting" and state.status_of("gate_data") == "awaiting"
    assert state.status_of("ideate") == "pending"
    state = _accept_all(eng)
    assert state.status == "approved"
    assert [d.gate for d in state.decisions] == GATES
    assert all(d.actor == "human" for d in state.decisions)


def test_double_submit_is_rejected_cleanly(make_config, runs_dir, confirm_intent):
    eng = Engine(make_config("regression"), runs_root=runs_dir, mode="interactive")
    confirm_intent(eng)
    eng.submit_gate("gate_data", "accept")
    before = eng.state.version
    with pytest.raises(GateError, match="not awaiting"):
        eng.submit_gate("gate_data", "accept")
    assert eng.state.version == before


def test_send_back_routes_a_challenge_and_the_agent_answers(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    eng.submit_gate("gate_champion", "send_back",
                    {"message": "Prefer fewer features; check the leverage sign."},
                    reason="parsimony")
    state = eng.state
    assert state.status_of("model") == "stale" and state.status_of("gate_champion") == "stale"
    state = eng.run_until_idle()
    ch = state.challenges[0]
    assert ch.source == "human" and ch.target_stage == "model"
    assert ch.status == "answered" and ch.response
    assert state.status_of("gate_champion") == "awaiting"  # re-opened for the human


def test_champion_override_refits_without_researching(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    model = eng.results()["model"].payload
    others = [c["id"] for c in model["admissible_set"] if c["id"] != model["champion_id"]]
    if not others:
        pytest.skip("admissible set has a single candidate for this seed")
    with pytest.raises(GateError, match="reason"):
        eng.submit_gate("gate_champion", "override", {"champion": others[0]})
    with pytest.raises(GateError, match="admissible"):
        eng.submit_gate("gate_champion", "override", {"champion": "c999"}, reason="x")
    eng.submit_gate("gate_champion", "override", {"champion": others[0]}, reason="simpler")
    eng.run_until_idle()
    model = eng.results()["model"].payload
    assert model["champion_id"] == others[0] and model["champion_source"] == "human"
    assert model["holdout_evaluations"] == 2
    assert model["recommendation"].get("reused") is True  # the modeler was not re-asked


def test_edit_after_downstream_ran_marks_descendants_stale(make_config, runs_dir):
    cfg = make_config("classification")
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    eng.reopen("gate_data")
    feature = eng.results()["explore"].payload["features"][0]
    eng.submit_gate("gate_data", "edit", {"exclude_columns": [feature]}, reason="not at origination")
    state = eng.state
    for s in ("ideate", "gate_design", "model", "gate_champion"):
        assert state.status_of(s) == "stale", s
    assert state.status_of("explore") == "done"
    eng.run_until_idle()
    eng.submit_gate("gate_design", "accept")
    eng.run_until_idle()
    model = eng.results()["model"].payload
    assert feature not in model["raw_features"]


def test_step_invalidated_while_running_stays_stale(make_config, runs_dir, monkeypatch):
    eng = Engine(make_config("regression"), runs_root=runs_dir)
    original = Engine._run_stage

    def racing(self, stage):
        result = original(self, stage)
        if stage == "ideate":  # a human edit lands while ideate is running
            with self.lock:
                st = self.state
                st.invalidate(["ideate"])
                st.save(self.run_dir)
        return result

    monkeypatch.setattr(Engine, "_run_stage", racing)
    eng._execute("explore")
    eng._execute("gate_data")
    eng._execute("ideate")
    assert eng.state.status_of("ideate") == "stale"


def test_leakage_block_send_back_then_completes(leak_config, runs_dir, apply_brief):
    eng = Engine(apply_brief(leak_config), runs_root=runs_dir, mode="interactive")
    state = _accept_all(eng, until="gate_validation")
    assert state.status_of("validate") == "blocked"
    assert eng.results()["validate"].verdict == Verdict.BLOCK
    with pytest.raises(GateError, match="BLOCK cannot be accepted"):
        eng.submit_gate("gate_validation", "accept", reason="ship it")
    eng.submit_gate("gate_validation", "send_back",
                    {"target": "explore", "message": "leaky is recorded after the outcome; exclude leaky"},
                    reason="confirmed leak")
    state = eng.run_until_idle()
    assert state.status_of("gate_data") == "awaiting"
    assert "leaky" in eng.results()["explore"].payload["recommended_exclusions"]
    state = _accept_all(eng)
    assert state.status == "approved"
    assert eng.results()["validate"].verdict != Verdict.BLOCK
    assert "leaky" not in eng.results()["model"].payload["champion"]["features"]


def test_leakage_blocks_autonomous_run(leak_config, runs_dir):
    state = Engine(leak_config, runs_root=runs_dir).run_until_idle()
    assert state.status == "blocked"
    assert state.status_of("document") == "pending"


def test_answering_a_design_gap_reenters_ideate(tmp_path, runs_dir):
    from cognos import synth
    from cognos.config import CognosConfig

    df = synth.make_cni_portfolio_dataset(n=600)
    csv = tmp_path / "cni.csv"
    df.to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "gap", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome"]},
        "search": {"max_candidates": 4, "cv_folds": 3},
    })
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    state = _accept_all(eng, until="gate_design")
    gap_ids = {g.id for g in state.gaps if g.status == "open"}
    assert "design-use_case" in gap_ids
    eng.submit_gate("gate_design", "edit", {"answers": {"design-use_case": "CECL lifetime loss"}},
                    reason="sponsor answered")
    state = eng.run_until_idle()
    assert state.overrides.design["use_case"] == "CECL lifetime loss"
    assert state.gap("design-use_case").status == "answered"
    assert eng.results()["ideate"].payload["design"]["use_case"] == "CECL lifetime loss"
    assert state.status_of("gate_design") == "awaiting"


def test_validator_findings_loop_back_automatically(make_config, runs_dir, replay_dir):
    finding = {"id": "signs", "severity": "high", "category": "soundness",
               "message": "Leverage enters with the wrong sign.", "evidence": ["model.cv_mean"],
               "target_stage": "model", "remedy": "Refit without the collinear ratio."}
    rec = {"summary": "Send back.", "assessment": "Sign issue.", "findings": [finding],
           "recommendation": "send_back"}
    cfg = make_config("classification", agents={"replay_dir": replay_dir({"validator": rec})})
    state = Engine(cfg, runs_root=runs_dir).run_until_idle()
    assert state.loops["validator"] == 2  # bounded (workflow.auto_challenge_loops)
    val = [c for c in state.challenges if c.source == "validator"]
    assert len(val) == 2 and all(c.status == "answered" for c in val)
    assert state.status == "completed"


def test_invalid_agent_fails_step_visibly_and_retry_recovers(make_config, runs_dir, replay_dir):
    bad = {"summary": "x", "discrimination": "x", "calibration": "x", "stability": "x",
           "findings": [{"id": "f", "severity": "low", "category": "c", "message": "m",
                         "evidence": ["backtest.not_a_fact"]}]}
    d = replay_dir({"outcomes_analyst": bad})
    cfg = make_config("classification", agents={"replay_dir": d, "max_retries": 1})
    eng = Engine(cfg, runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "failed" and state.status_of("backtest") == "failed"
    assert "backtest.not_a_fact" in state.steps["backtest"].message
    good = dict(bad, findings=[])
    (eng.run_dir.parent.parent / "recorded" / "outcomes_analyst.json").write_text(
        json.dumps(good), encoding="utf-8")
    eng.retry("backtest")
    assert eng.run_until_idle().status == "completed"


def test_engine_rehydrates_from_run_dir(make_config, runs_dir, confirm_intent):
    eng = Engine(make_config("regression"), runs_root=runs_dir, mode="interactive")
    confirm_intent(eng)
    again = Engine.load(eng.run_dir)
    assert again.state.status_of("gate_data") == "awaiting"
    again.submit_gate("gate_data", "accept")
    assert again.run_until_idle().status_of("gate_design") == "awaiting"


def test_sponsor_answer_reaches_the_agent_and_changes_its_recommendation(leak_config, runs_dir,
                                                                           confirm_intent):
    eng = Engine(leak_config, runs_root=runs_dir, mode="interactive")
    state = confirm_intent(eng)
    assert "leaky" not in eng.results()["explore"].payload["recommended_exclusions"]
    gap = next(g for g in state.gaps if g.stage == "explore" and "leaky" in g.question)
    eng.answer_gap(gap.id, "No - leaky is recorded after the outcome window.")
    state = eng.run_until_idle()
    assert state.gap(gap.id).status == "answered"
    assert "leaky" in eng.results()["explore"].payload["recommended_exclusions"]
    # the re-run's context carried the answer (call ids share a second, so don't rely on name order)
    inputs = [p.read_text(encoding="utf-8")
              for p in (eng.run_dir / "agents").glob("*data_analyst*.input.json")]
    assert len(inputs) == 2
    assert sum("recorded after the outcome" in t for t in inputs) == 1
