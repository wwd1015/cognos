"""An agent's id is its display name in snake case (ADR-0023), and runs recorded with the old
ids still read: names, challenges, loop counts and replay recordings."""

from __future__ import annotations

import json

from cognos.agents.contracts import CONTRACTS, FRIENDLY, STAGE_AGENT
from cognos.agents.ids import LEGACY, canonical, legacy_names
from cognos.engine.state import RunState


def test_every_id_reads_as_its_name():
    for agent in CONTRACTS:
        role = agent.removesuffix("_tools").removesuffix("_ask")
        name = FRIENDLY[role].split(" (")[0]  # "Data agent (joining sources)" -> "Data agent"
        assert role.startswith(name.lower().replace(" ", "_")), (agent, name)
    assert set(STAGE_AGENT.values()) <= set(LEGACY.values())


def test_old_ids_still_read():
    assert canonical("data_analyst") == "data_agent" and canonical("validator_ask") == \
        "validation_agent_ask" and canonical("data_agent") == "data_agent"
    assert FRIENDLY["data_analyst"] == FRIENDLY["data_agent"] == "Data agent"
    assert FRIENDLY["modeler_tools"] == FRIENDLY["modeling_agent_tools"]
    assert legacy_names("writing_agent") == ["writer"]
    st = RunState.model_validate({
        "run_id": "r", "status": "running", "loops": {"validator": 2},
        "challenges": [{"source": "validator", "target_stage": "model", "message": "m"}]})
    assert st.loops == {"validation_agent": 2} and st.challenges[0].source == "validation_agent"


def test_a_replay_recorded_under_an_old_id_is_found(tmp_path):
    from cognos.agents.runner import AgentRunner

    (tmp_path / "writer.json").write_text(json.dumps({"summary": "old"}), encoding="utf-8")
    runner = AgentRunner.__new__(AgentRunner)
    runner.replay_dir, runner._calls = str(tmp_path), {}
    assert runner._replay("writing_agent") == {"summary": "old"}
