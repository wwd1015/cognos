"""Opt-in live test: the full interactive workflow with real agents via ``claude -p``.

Skipped unless ``COGNOS_LIVE=1`` and the ``claude`` CLI is on PATH — the default suite stays offline
(CLAUDE.md non-negotiable #1). Run it locally with::

    COGNOS_LIVE=1 pytest tests/live -q        # ~5-7 min, about $1-2 with the default model (opus)

It exercises what the heuristic agents cannot: real recommendations passing the engine checks, a
human send-back answered by a live agent, and a completed, signed-off run.
"""

from __future__ import annotations

import os
import shutil

import pytest

from cognos import service
from cognos.engine.graph import GATES

pytestmark = pytest.mark.skipif(
    os.environ.get("COGNOS_LIVE") != "1" or shutil.which(os.environ.get("COGNOS_CLAUDE_BIN", "claude")) is None,
    reason="live test: set COGNOS_LIVE=1 with the claude CLI installed",
)


def _waiting(state) -> str | None:
    return next((g for g in GATES if state.status_of(g) == "awaiting"), None)


def test_interactive_run_with_live_claude_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("COGNOS_PROVIDER", "claude_cli")
    root = tmp_path / "runs"
    cfg = service.demo_config("commercial", root, n=800, search_budget=6)
    run_id = service.create_run(cfg, mode="interactive", provider="claude_cli", root=root)

    state = service.run_until_idle(run_id, root)
    assert _waiting(state) == "gate_data", state.steps["explore"].message
    service.submit_gate(run_id, "gate_data", "accept", reason="live test", root=root, background=False)

    state = service.run_until_idle(run_id, root)
    assert _waiting(state) == "gate_design", state.steps["ideate"].message
    service.submit_gate(run_id, "gate_design", "send_back",
                        {"message": "Rank the regularized logit families above plain logit and "
                                    "explain why for a thin-event sample."},
                        reason="live challenge", root=root, background=False)
    state = service.run_until_idle(run_id, root)
    challenge = state.challenges[0]
    assert challenge.status == "answered" and challenge.response
    rec = service.results(run_id, root)["ideate"].payload["recommendation"]
    assert rec["provider"] == "claude_cli"
    assert [r["challenge_id"] for r in rec["output"]["responses_to_challenges"]] == [challenge.id]

    for _ in range(8):
        gate = _waiting(state)
        if gate is None:
            break
        action = "approve" if gate == "gate_signoff" else "accept"
        service.submit_gate(run_id, gate, action, reason="live test (FAIL accepted if any)",
                            root=root, background=False)
        state = service.run_until_idle(run_id, root)
    assert state.status == "approved", (state.status, state.halted_reason)
    audit = service.audit(run_id, root)
    assert {a["agent"] for a in audit if a["status"] == "ok"} >= {
        "data_analyst", "design_lead", "modeler", "outcomes_analyst", "validator", "risk_analyst",
        "writer"}
    narrative = (root / run_id / "docs" / "narrative.md").read_text(encoding="utf-8")
    assert "{{fact:" not in narrative
