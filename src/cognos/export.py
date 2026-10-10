"""Export a run as one file to hand to a validator or a committee.

The bundle is everything a reader needs and nothing they must not have: the documents (OKF
bundle), each stage's result and its small evidence tables, the run state with every decision,
challenge and question, the effective config, and the agent audit log. **Never** the data
directory — the sealed holdout stays sealed — and no fitted models or binary caches.

``EXPORT.json`` lists every file with its SHA-256, so a recipient can check that what they are
reading is what was exported.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

# Top-level entries that are part of the record.
_FILES = ("state.json", "config.yaml", "summary.json", "summary.txt", "manifest.json", "events.jsonl")
_DIRS = ("docs", "stages", "packages")
# Readable evidence only: data, models and caches are referenced by the results, not shipped.
# (.py: the analysis scripts the Data Analyst wrote are part of the record a validator reviews)
_TEXT = {".json", ".md", ".tsv", ".csv", ".txt", ".yaml", ".yml", ".jsonl", ".py"}
NOT_INCLUDED = ("data/ (training data and the sealed holdout)", "models/ (fitted scorers)",
                "binary caches under stages/ (.parquet, .joblib, .npz)")


def _members(run_dir: Path, agent_io: bool) -> list[Path]:
    out = [run_dir / f for f in _FILES if (run_dir / f).is_file()]
    for d in _DIRS:
        out += sorted(p for p in (run_dir / d).rglob("*") if p.is_file() and p.suffix in _TEXT
                      and not p.name.startswith("."))
    agents = run_dir / "agents"
    if (agents / "audit.jsonl").is_file():
        out.append(agents / "audit.jsonl")
    if agent_io:  # full prompts and context slices: larger, and they quote the data's facts
        out += sorted(p for p in agents.glob("*.json") if p.is_file())
    return out


def export_run(run_dir: str | Path, dest: str | Path | None = None, *, agent_io: bool = False) -> Path:
    """Write ``<run_id>.zip`` and return its path. ``dest`` is a directory (default: the runs
    root's ``_exports/``) or a ``.zip`` path."""
    run_dir = Path(run_dir)
    if not (run_dir / "state.json").is_file():
        raise FileNotFoundError(f"no run at {run_dir}")
    dest = Path(dest) if dest else run_dir.parent / "_exports"
    target = dest if dest.suffix == ".zip" else dest / f"{run_dir.name}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = _members(run_dir, agent_io)
    listing = []
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                data = p.read_bytes()
                rel = p.relative_to(run_dir).as_posix()
                z.writestr(f"{run_dir.name}/{rel}", data)
                listing.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            manifest = {"run_id": run_dir.name, "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "cognos_version": __version__, "agent_io_included": agent_io,
                        "not_included": list(NOT_INCLUDED), "files": listing}
            z.writestr(f"{run_dir.name}/EXPORT.json", json.dumps(manifest, indent=1))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
