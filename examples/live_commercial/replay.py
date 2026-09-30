"""Replay the recorded live run offline — no key, no network, same recommendations.

The run in ``run/`` was produced by real Claude agents (``claude -p``, Opus) on the synthetic
commercial PD demo. ``recorded/`` holds each agent's raw answer. This script regenerates the same
synthetic dataset, runs the full workflow with the ``replay`` provider (recorded answers, validated
by the engine exactly as the live ones were), and prints where the recommendations landed.

    python examples/live_commercial/replay.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replay(runs_root: str | Path) -> tuple[str, object]:
    from cognos import service

    cfg = service.demo_config("commercial", runs_root)
    cfg.agents.replay_dir = str(HERE / "recorded")
    run_id = service.create_run(cfg, mode="autonomous", provider="replay", root=runs_root)
    return run_id, service.run_until_idle(run_id, runs_root)


def main() -> int:
    from cognos import service

    root = Path(tempfile.mkdtemp(prefix="cognos-replay-"))
    run_id, state = replay(root)
    results = service.results(run_id, root)
    print(f"status: {state.status}")
    for stage, res in results.items():
        rec = (res.payload or {}).get("recommendation") or {}
        who = f"{rec.get('agent')} via {rec.get('provider')}" if rec else "engine only"
        print(f"  {stage:<9} {res.verdict.value:<5} {who}")
    print(f"\nnarrative: {root / run_id / 'docs' / 'narrative.md'}")
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
