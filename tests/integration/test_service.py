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
    assert io["input"]["agent"] == "data_analyst" and "raw" in io["output"]


def test_background_advance_reaches_the_next_gate(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=200, search_budget=4)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.start(run_id, root)
    eng = service.engine(run_id, root)
    assert eng.wait(timeout=120)
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
    assert "Independent Validator" in capsys.readouterr().out
