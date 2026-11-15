"""The service layer (the UI/CLI boundary) and the v1 CLI commands."""

from __future__ import annotations

import json

import pytest

from cognos import service
from cognos.cli import main


def test_demo_run_through_the_service(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("commercial", root, n=600, search_budget=6)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    st = service.run_until_idle(run_id, root)
    assert st.status_of("gate_intent") == "awaiting" and st.status_of("explore") == "pending"
    service.submit_gate(run_id, "gate_intent", "accept", reason="the sponsor is still deciding",
                        root=root, background=False)
    st = service.run_until_idle(run_id, root)
    assert st.status_of("gate_data") == "awaiting"
    service.submit_gate(run_id, "gate_data", "accept", root=root, background=False)
    st = service.run_until_idle(run_id, root)
    assert st.status_of("gate_design") == "awaiting"
    assert service.results(run_id, root)["ideate"].payload["hypotheses"]
    kinds = {e["type"] for e in service.run_events(run_id, root=root)}
    assert {"run_created", "step_start", "step_done", "gate_waiting", "gate_decision",
            "agent_start", "agent_done"} <= kinds
    calls = service.audit(run_id, root)
    io = service.agent_io(run_id, calls[0]["call_id"], root)
    assert io["input"]["agent"] == "intake_agent" and "raw" in io["output"]


def test_background_advance_reaches_the_next_gate(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=200, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.start(run_id, root)
    eng = service.engine(run_id, root)
    assert eng.wait(timeout=120)
    assert service.state(run_id, root).status_of("gate_intent") == "awaiting"
    service.submit_gate(run_id, "gate_intent", "accept", reason="as written", root=root)
    assert eng.wait(timeout=120)  # the decision advanced the run in the background
    assert service.state(run_id, root).status_of("gate_data") == "awaiting"


def test_list_runs_includes_legacy_runs_read_only(tmp_path):
    root = tmp_path / "runs"
    legacy = root / "20250101T000000Z-abcdef0"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text(json.dumps({"project": "old", "mode": "autonomous",
                                                      "created_at": "2025-01-01"}), encoding="utf-8")
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    rows = service.list_runs(root)
    assert {r["status"] for r in rows} >= {"legacy", "created"}
    assert next(r for r in rows if r["legacy"])["project"] == "old"
    # every row carries the same keys, so a page can list a legacy run beside the others
    assert len({frozenset(r) for r in rows}) == 1
    with pytest.raises(KeyError):
        service.engine(legacy.name, root)


def test_read_artifact_is_confined_to_the_run(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    assert service.read_artifact(run_id, "config.yaml", root)
    assert service.read_artifact(run_id, "../../secret.txt", root) is None


def test_cli_interactive_gate_commands(tmp_path, capsys):
    root = str(tmp_path / "runs")
    cfg = service.demo_config("regression", root, n=200, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.run_until_idle(run_id, root)
    assert main(["status", "--run", run_id, "--runs-dir", root]) == 0
    assert "gate_data" in capsys.readouterr().out
    # the intent gate refuses a bare accept while the interview is open; an answer re-runs intake
    assert main(["gate", "gate_intent", "--run", run_id, "--action", "accept", "--runs-dir", root]) == 1
    assert "blocking interview question" in capsys.readouterr().out
    assert main(["answer", "--run", run_id, "--gap", "design-horizon", "--text", "12-month",
                 "--runs-dir", root]) == 0
    assert service.state(run_id, root).steps["intake"].runs == 2
    assert main(["gate", "gate_intent", "--run", run_id, "--action", "accept",
                 "--reason", "the rest is with the sponsor", "--runs-dir", root]) == 0
    assert main(["gate", "gate_data", "--run", run_id, "--action", "accept", "--runs-dir", root]) == 0
    assert service.state(run_id, root).status_of("gate_design") == "awaiting"
    # a refused decision is reported, not raised
    assert main(["gate", "gate_data", "--run", run_id, "--action", "accept", "--runs-dir", root]) == 1
    assert main(["gate", "gate_design", "--run", run_id, "--action", "send_back",
                 "--message", "rank the regularized families higher", "--runs-dir", root]) == 0
    st = service.state(run_id, root)
    assert st.challenges and st.challenges[0].status == "answered"


def test_cli_providers_and_agents(capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "heuristic" in out and "claude_cli" in out
    assert main(["agents"]) == 0
    assert "Validation agent" in capsys.readouterr().out


def test_delete_runs_removes_only_real_runs(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    keep = service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    gone = service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    (root / gone / "packages").mkdir()
    (root / gone / "packages" / "v1.json").write_text("{}", encoding="utf-8")

    preview = service.deletion_preview([gone], root)[0]
    assert preview["packages"] == 1 and preview["size_bytes"] > 0 and not preview["blocked"]
    assert (root / gone).exists()  # a preview deletes nothing

    for bad in ("../runs", "_demo_data", str(root / keep), "", "no-such-run"):
        with pytest.raises((KeyError, ValueError)):
            service.delete_runs([bad], root)
    assert (root / "_demo_data").exists() and (root / keep).exists()

    assert [r["run_id"] for r in service.delete_runs([gone], root)] == [gone]
    assert not (root / gone).exists()
    assert [r["run_id"] for r in service.list_runs(root)] == [keep]
    with pytest.raises(KeyError):  # the cached engine went with it
        service.engine(gone, root)


def test_delete_refuses_a_run_that_is_working(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=150, search_budget=4)
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=root)
    monkeypatch.setattr(service, "is_busy", lambda *a, **k: True)
    with pytest.raises(RuntimeError, match="a step is running"):
        service.delete_runs([run_id], root)
    assert (root / run_id).exists()


def test_every_demo_takes_n_and_n_is_the_obligors_of_the_migration_panel(tmp_path):
    import pandas as pd

    root = tmp_path / "runs"
    for preset in service.DEMO_PRESETS:
        if preset != "linked":  # the linked demo splits the cni frame; cni covers its n
            service.demo_config(preset, root, n=60)
    panel = pd.read_csv(root / "_demo_data" / "migration.csv")
    assert panel.groupby("asof")["obligor_id"].nunique().iloc[0] == 60


def test_reopen_and_retry_record_who_asked(tmp_path):
    from cognos.engine.graph import GATES

    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=200, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    for _ in range(4):
        st = service.run_until_idle(run_id, root)
        gate = next(g for g in GATES if st.status_of(g) == "awaiting")
        if gate == "gate_design":
            break
        service.submit_gate(run_id, gate, "accept", reason="as proposed", root=root,
                            background=False)

    st = service.reopen(run_id, "gate_data", root, by="bo@desk")
    assert st.status_of("gate_data") == "awaiting"
    assert st.steps["gate_data"].by == "bo@desk" and "bo@desk" in st.steps["gate_data"].message

    eng = service.engine(run_id, root)
    with eng.lock:
        state = eng.state
        state.set_step("explore", "failed", "boom")
        eng._save(state)
    st = service.retry(run_id, "explore", root, background=False, by="cy@desk")
    assert st.status_of("explore") == "pending" and "cy@desk" in st.steps["explore"].message
    events = service.run_events(run_id, root=root)
    assert any(e["type"] == "gate_waiting" and e.get("by") == "bo@desk" for e in events)
    assert any(e["type"] == "step_retry" and e.get("by") == "cy@desk" for e in events)

    # both are kept on the run and listed in the run record, in order, beside the decisions
    from cognos.engine.process import journal

    assert [(a.kind, a.step, a.by, a.seat) for a in st.actions] == [
        ("reopen", "gate_data", "bo@desk", "developer"), ("retry", "explore", "cy@desk", "")]
    record = journal(st, service.results(run_id, root))
    said = [(r["name"], r["what"]) for r in record if r["who"] == "person"]
    reopened = next(i for i, (n, w) in enumerate(said) if "re-opened for revision" in w)
    assert said[reopened][0] == "bo@desk (developer)"
    assert any(n == "cy@desk" and "asked to run again after it failed" in w for n, w in said)
    assert any("accept" in w for _, w in said[:reopened])  # the decisions before it still lead
