"""A team on one runs folder: every state change is exclusive across processes, decisions
carry a name, and a step a dead process left running can be taken over."""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from cognos import service
from cognos.engine import GateError
from cognos.engine.state import RunBusy, RunLock, RunState


def _bump(run_dir: str, n: int) -> None:
    """Another process doing read-modify-write on the run's state, like a second workbench."""
    from pathlib import Path

    from cognos.engine.state import RunState, run_lock

    d = Path(run_dir)
    for _ in range(n):
        with run_lock(d):
            st = RunState.load(d)
            st.loops["bumps"] = st.loops.get("bumps", 0) + 1
            st.save(d)


def test_two_processes_never_lose_an_update(tmp_path):
    d = tmp_path / "runs" / "r1"
    RunState(run_id="r1").save(d)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_bump, args=(str(d), 25)) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    assert RunState.load(d).loops["bumps"] == 75
    assert not (tmp_path / "runs" / "_locks" / "r1.lock").exists()  # released


def test_lock_is_reentrant_and_takes_over_a_dead_holder(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "r1"
    lock = RunLock(d)
    with lock, lock:
        assert lock.path.exists()
    assert not lock.path.exists()

    lock.path.write_text("maria@desk pid=1 gone", encoding="utf-8")  # a holder that died
    monkeypatch.setattr(RunLock, "STALE_S", 0.2)
    started = time.monotonic()
    with lock:
        assert "maria" not in lock.path.read_text(encoding="utf-8")
    assert time.monotonic() - started >= 0.2  # it waited before taking over


def test_a_live_holder_is_not_robbed(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "r1"
    other = RunLock(d)
    monkeypatch.setattr(RunLock, "TIMEOUT_S", 0.3)
    with RunLock(d):
        with pytest.raises(RunBusy, match="is updating this run"):
            other.__enter__()


def test_decisions_and_answers_carry_the_person(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("COGNOS_USER", "alan")
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.run_until_idle(run_id, root)
    st = service.state(run_id, root)
    assert st.created_by == "alan" and st.steps["intake"].by.startswith("alan@")

    monkeypatch.setenv("COGNOS_USER", "maria")
    gap = st.open_gaps()[0].id
    st = service.answer_gap(run_id, gap, "origination", root=root, background=False)
    assert st.gap(gap).answered_by == "maria"
    service.run_until_idle(run_id, root)
    st = service.submit_gate(run_id, "gate_intent", "accept", reason="starting without them",
                             root=root, background=False)
    assert st.decisions[-1].by == "maria" and "maria" in st.steps["gate_intent"].message
    record = [r["name"] for r in __import__("cognos.engine.process", fromlist=["journal"]).journal(
        st, service.results(run_id, root)) if r["who"] == "person"]
    assert record == ["maria (developer)"]


def test_a_stuck_step_is_taken_over_only_when_asked(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    eng = service.engine(run_id, root)
    with eng.lock:  # a process started the step and died
        st = eng.state
        st.set_step("intake", "running", by="maria@desk")
        st.save(eng.run_dir)
    with pytest.raises(GateError, match="not failed"):
        service.retry(run_id, "intake", root, background=False)
    st = service.retry(run_id, "intake", root, background=False, stuck=True)
    assert st.status_of("intake") == "pending"
    assert "taken over from maria@desk" in st.steps["intake"].message
