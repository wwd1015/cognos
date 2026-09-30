"""The recorded live run (real Claude agents) replays offline and still passes every engine check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cognos import service

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "live_commercial"


def _load_replay():
    spec = importlib.util.spec_from_file_location("live_replay", EXAMPLE / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_recording_replays_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("COGNOS_PROVIDER", raising=False)
    run_id, state = _load_replay().replay(tmp_path)
    assert state.status == "completed"
    results = service.results(run_id, tmp_path)
    for stage in ("explore", "ideate", "model", "backtest", "validate", "comply", "document"):
        rec = results[stage].payload["recommendation"]
        assert rec["provider"] == "replay" and rec["attempts"] == 1, stage
    # the live validator's challenge is reproduced verbatim through the engine
    assert any(f.id.startswith("val-") for f in results["validate"].findings)
    narrative = (tmp_path / run_id / "docs" / "narrative.md").read_text(encoding="utf-8")
    assert "{{fact:" not in narrative and "ridge_logit" in narrative
