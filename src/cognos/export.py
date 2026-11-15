"""Export a run as one file to hand to a validator or a committee.

The bundle is everything a reader needs and nothing they must not have: the documents (OKF
bundle), each stage's result and its small evidence tables, the run state with every decision,
challenge and question, the effective config, and the agent audit log. **Never** the data
directory — the sealed holdout stays sealed — and no fitted models or binary caches. In the
exported ``config.yaml`` and ``state.json`` a literal source option named like a secret is
replaced (``datasources.redacted``); the run's own files keep it.

``EXPORT.json`` lists every file with its SHA-256, so a recipient can check that what they are
reading is what was exported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__, datasources

# Top-level entries that are part of the record.
_FILES = ("state.json", "config.yaml", "summary.json", "summary.txt", "manifest.json", "events.jsonl")
_DIRS = ("docs", "stages", "packages", "asks")
# Readable evidence only: data, models and caches are referenced by the results, not shipped.
# (.py: the analysis scripts the Data agent wrote are part of the record a validator reviews)
_TEXT = {".json", ".md", ".tsv", ".csv", ".txt", ".yaml", ".yml", ".jsonl", ".py"}
NOT_INCLUDED = ("data/ (training data and the sealed holdout)", "models/ (fitted scorers)",
                "binary caches under stages/ (.parquet, .joblib, .npz)")


def _redact_data(data: Any) -> bool:
    """Replace secret-named literal options in a ``data`` mapping (a profile's data section, or
    the data given at gate_source): its ``source`` and each of its ``sources``. In place; says
    whether anything was replaced."""
    if not isinstance(data, dict):
        return False
    changed = False
    for src in [data.get("source"), *(data.get("sources") or [])]:
        if isinstance(src, dict) and isinstance(src.get("options"), dict):
            new = datasources.redacted(src["options"])
            if new != src["options"]:
                src["options"], changed = new, True
    return changed


def redact_profile(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """A loaded ``config.yaml`` as it may leave the run, and whether anything was replaced."""
    out = copy.deepcopy(raw)
    return out, _redact_data(out.get("data"))


def redact_state(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """A loaded ``state.json`` as it may leave the run: the data given at gate_source sits in
    ``overrides.data`` and on each gate_source decision's payload."""
    out = copy.deepcopy(raw)
    changed = _redact_data((out.get("overrides") or {}).get("data"))
    for d in out.get("decisions") or []:
        if isinstance(d, dict) and d.get("gate") == "gate_source":
            changed = _redact_data((d.get("payload") or {}).get("data")) or changed
    return out, changed


def _exported_bytes(p: Path, rel: str) -> bytes:
    """The file as it goes into the export: config.yaml and state.json with secret-named
    literal source options replaced (re-serialised only when something was), all else as is."""
    data = p.read_bytes()
    if rel == "config.yaml":
        raw = yaml.safe_load(data.decode("utf-8"))
        if isinstance(raw, dict):
            out, changed = redact_profile(raw)
            if changed:
                return yaml.safe_dump(out, sort_keys=False, allow_unicode=True).encode("utf-8")
    elif rel == "state.json":
        out, changed = redact_state(json.loads(data.decode("utf-8")))
        if changed:
            return json.dumps(out, indent=1, ensure_ascii=False).encode("utf-8")
    return data


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
                rel = p.relative_to(run_dir).as_posix()
                data = _exported_bytes(p, rel)
                z.writestr(f"{run_dir.name}/{rel}", data)
                listing.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            from . import trials

            rec = trials.record_for(run_dir)
            if rec["n_runs"] > 1:  # the other runs of this model, one line each
                data = json.dumps(rec, indent=1, ensure_ascii=False, default=str).encode("utf-8")
                z.writestr(f"{run_dir.name}/MODEL_RECORD.json", data)
                listing.append({"path": "MODEL_RECORD.json", "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()})
            manifest = {"run_id": run_dir.name, "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "cognos_version": __version__, "agent_io_included": agent_io,
                        "not_included": list(NOT_INCLUDED), "files": listing}
            z.writestr(f"{run_dir.name}/EXPORT.json", json.dumps(manifest, indent=1))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
