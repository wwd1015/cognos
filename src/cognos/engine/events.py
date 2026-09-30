"""The run's activity feed: ``runs/<id>/events.jsonl``.

Append-only and file-backed, so any reader (the UI polling once a second, the CLI, a test) sees the
same history, and a browser refresh or a fresh process loses nothing.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def publish(run_dir: Path, type_: str, message: str, **extra: Any) -> dict:
    event = {"ts": time.time(), "type": type_, "message": message, **extra}
    path = Path(run_dir) / "events.jsonl"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
    return event


def history(run_dir: Path, since: float = 0.0, limit: int = 500) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("ts", 0) > since:
            out.append(ev)
    return out
