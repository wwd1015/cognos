"""Engine graph + run state: staleness, persistence, events."""

from __future__ import annotations

from cognos.engine import events
from cognos.engine.graph import DEPS, GATES, STEPS, descendants
from cognos.engine.state import Challenge, RunState


def test_graph_is_complete_and_ordered():
    assert set(DEPS) == set(STEPS)
    for step, deps in DEPS.items():
        for d in deps:
            assert STEPS.index(d) < STEPS.index(step)
    assert all(g in STEPS for g in GATES)


def test_descendants_of_data_gate_is_everything_after_it():
    assert descendants("gate_data") == STEPS[STEPS.index("gate_data") + 1:]
    assert descendants("gate_signoff") == []


def test_invalidate_marks_only_steps_that_produced_output():
    st = RunState(run_id="r")
    for s in ("explore", "gate_data", "ideate"):
        st.set_step(s, "done")
    st.set_step("gate_design", "awaiting")
    marked = st.invalidate(["ideate"])
    assert marked == ["ideate", "gate_design"]
    assert st.status_of("explore") == "done"
    assert st.status_of("model") == "pending"  # never ran: nothing to invalidate


def test_skipped_satisfies_dependencies():
    st = RunState(run_id="r")
    st.set_step("explore", "done")
    st.set_step("gate_data", "skipped")
    assert st.deps_satisfied("ideate")


def test_state_round_trips_and_bumps_version(tmp_path):
    st = RunState(run_id="r", project="p")
    st.challenges.append(Challenge(target_stage="model", message="try cloglog"))
    st.save(tmp_path)
    back = RunState.load(tmp_path)
    assert back.version == 1 and back.challenges[0].message == "try cloglog"
    assert back.open_challenges("model")[0].status == "open"
    back.save(tmp_path)
    assert RunState.load(tmp_path).version == 2


def test_events_append_and_filter_by_time(tmp_path):
    e1 = events.publish(tmp_path, "step_start", "one")
    events.publish(tmp_path, "step_done", "two")
    hist = events.history(tmp_path)
    assert [e["message"] for e in hist] == ["one", "two"]
    assert [e["message"] for e in events.history(tmp_path, since=e1["ts"])] == ["two"]
