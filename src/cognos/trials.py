"""Trial runs: deciding a gate differently without losing the result that was there (ADR-0017).

A model has one **main run** and any number of **trial runs**. A trial run is a copy of a run
made at a developer gate; the developer decides that gate differently and reads what it does.
Three rules keep that cheap to do and impossible to hide:

- a trial run never reaches validation, so it can never open the sealed holdout. To go on, the
  developer makes it the main run, and the run it replaces stays on record as a trial;
- every look at the test sample is counted for the model, across all of its runs;
- deleting a trial run removes its files and keeps its line.

Everything here is mechanical. A line is read from what a run already recorded (its state and
its stage results); the only things stored are what cannot be read again: who made what from
where (``Lineage`` on each run's state), the events of the model, and the lines of deleted
runs (``<runs>/_models/<model>.json``, the model's record).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .context import new_run_id
from .engine.graph import LABELS, STAGES, STEPS, TRIAL_STOPS_BEFORE
from .engine.process import SEAT_OF_GATE
from .engine.state import Lineage, RunState, read_json, run_lock, utcnow
from .fsutil import atomic_write

STOPS_BEFORE = TRIAL_STOPS_BEFORE  # the sealed holdout is scored there
BEYOND = STEPS[STEPS.index(STOPS_BEFORE):]
# The gates a trial run can be made from: the developer's, all before validation.
GATES = [g for g, seat in SEAT_OF_GATE.items() if seat == "developer"]
# Never copied: a package is one run's signature, and the rest is rewritten by the copy.
_NOT_COPIED = {"packages", "docs", "summary.json", "summary.txt", "manifest.json", "events.jsonl"}


class TrialError(ValueError):
    """A trial-run action the engine refuses, with the reason."""


# --- the model a run belongs to ----------------------------------------------------------------
def model_of(state: RunState) -> str:
    """The model's id: the first run of it. A run that never had a trial run is its own model."""
    return state.lineage.model if state.lineage is not None else state.run_id


def record_path(runs_root: Path, model: str) -> Path:
    return Path(runs_root) / "_models" / f"{model}.json"


def _lock(runs_root: Path, model: str):
    # The same lock as a run's, on a name no run can have.
    return run_lock(Path(runs_root) / f"_model-{model}")


def _stored(runs_root: Path, model: str) -> dict[str, Any]:
    return read_json(record_path(runs_root, model)) or {"model": model, "events": [],
                                                        "deleted": []}


def _store(runs_root: Path, model: str, rec: dict[str, Any]) -> None:
    path = record_path(runs_root, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(rec, indent=1, ensure_ascii=False, default=str))


def _event(runs_root: Path, model: str, what: str, run_id: str, by: str, **more: Any) -> None:
    with _lock(runs_root, model):
        rec = _stored(runs_root, model)
        rec["events"].append({"at": utcnow(), "what": what, "run_id": run_id, "by": by, **more})
        _store(runs_root, model, rec)


def members(runs_root: Path, model: str) -> list[RunState]:
    """Every run of the model still on disk, oldest first."""
    out = []
    for d in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        if not d.is_dir() or d.name.startswith(("_", ".")) or not RunState.exists(d):
            continue
        try:
            st = RunState.load(d)
        except Exception:
            continue
        if model_of(st) == model and (st.lineage is not None or st.run_id == model):
            out.append(st)
    return out


# --- one run's line ----------------------------------------------------------------------------
def _payload(run_dir: Path, stage: str, name: str = "result.json") -> dict[str, Any]:
    got = read_json(Path(run_dir) / "stages" / stage / name) or {}
    return got.get("payload") or {}


def counts(run_dir: Path) -> dict[str, Any]:
    """What a run's results say about its use of the test sample, as recorded."""
    model, backtest = _payload(run_dir, "model"), _payload(run_dir, "backtest")
    return {"search": int(model.get("test_evaluations") or 0),
            "search_fingerprint": str(model.get("search_fingerprint") or ""),
            "backtest": int(backtest.get("test_evaluations") or 0)}


def own_looks(state: RunState, run_dir: Path) -> int:
    """How many times this run scored a model on the test sample itself. A trial run starts
    as a copy, so what it copied is taken off: the comparisons its parent had made, and a
    search it did not run again (same inputs, same fingerprint)."""
    now = counts(run_dir)
    lin = state.lineage
    inherited = lin.inherited if lin is not None else {}
    searched = now["search"]
    if lin is not None and now["search_fingerprint"] == str(lin.base.get("search_fingerprint", "")):
        searched = 0
    return max(0, now["backtest"] - int(inherited.get("backtest", 0))) + searched


def holdout_opened(run_dir: Path) -> list[str]:
    """The champions this run scored on the sealed holdout (validation records each one)."""
    out: list[str] = []
    for name in ("result.json", "result.prev.json"):
        for cid in _payload(run_dir, "validate", name).get("holdout_scored") or []:
            if cid not in out:
                out.append(str(cid))
    return out


def changed(state: RunState) -> list[str]:
    """The decisions a trial run holds differently from the run it was copied from."""
    if state.lineage is None or not state.lineage.parent:
        return []  # the first run of the model was copied from nothing
    base = state.lineage.base.get("overrides") or {}
    now = state.overrides.model_dump(mode="json")
    return [k for k in now if now.get(k) != base.get(k)]


CHANGE_LABELS = {"data": "the data", "exclude_columns": "excluded columns", "target": "dependent variable",
                 "task": "task", "join": "join of the sources", "design": "design answers",
                 "compliance": "intended use", "slate": "model families", "plan": "the plan",
                 "pool": "candidate pool", "names": "model names", "champion": "champion"}


def stands(state: RunState) -> str:
    """Where a run is, in a few words (the model's record keeps this for a deleted run)."""
    if state.is_trial and at_end(state):
        return "trial finished at the champion"
    waiting = next((s for s in STEPS if state.status_of(s) == "awaiting"), None)
    if waiting:
        return f"waiting at “{LABELS[waiting]}”"
    return {"completed": "finished, not signed off", "approved": "signed off",
            "rejected": "rejected", "blocked": "blocked", "failed": "a step failed",
            "running": "working", "created": "not started"}.get(state.status, state.status)


def at_end(state: RunState) -> bool:
    """A trial run has gone as far as it may: everything before validation is settled."""
    before = STEPS[:STEPS.index(STOPS_BEFORE)]
    return all(state.status_of(s) in ("done", "skipped") for s in before)


def line(state: RunState, run_dir: Path) -> dict[str, Any]:
    """One run of a model on one line: what it was for, how far it went, what it found and
    how often it used the test sample. Read from the run; nothing judged."""
    lin = state.lineage
    backtest = _payload(run_dir, "backtest")
    names = _payload(run_dir, "model").get("names") or {}
    champion = state.overrides.champion or backtest.get("champion_id")
    return {
        "run_id": state.run_id,
        "role": "trial" if state.is_trial else "main",
        "parent": lin.parent if lin else None,
        "gate": lin.gate if lin else None,
        "note": lin.note if lin else "",
        "by": (lin.by if lin and lin.parent else "") or state.created_by,
        "at": (lin.at if lin and lin.parent else "") or state.created_at,
        "changed": changed(state),
        "stands": stands(state),
        "champion_id": champion if backtest else None,
        "champion": (names.get(champion) or backtest.get("champion_name")) if backtest else None,
        "metric": backtest.get("oos_metric_name"),
        "sample": backtest.get("evaluation_sample"),
        "test_metric": _test_metric(backtest, champion),
        "test_evaluations": own_looks(state, run_dir),
        "holdout_scored": holdout_opened(run_dir),
        "promoted_at": lin.promoted_at if lin else "",
        "replaced_at": lin.replaced_at if lin else "",
        "deleted_at": "",
    }


def _test_metric(backtest: dict[str, Any], champion: str | None) -> float | None:
    metric = backtest.get("oos_metric_name")
    for c in backtest.get("candidates") or []:
        if c.get("id") == champion:
            v = (c.get("metrics") or {}).get(metric)
            return float(v) if isinstance(v, int | float) else None
    v = backtest.get("oos_metric")
    return float(v) if isinstance(v, int | float) else None


# --- the model's record ------------------------------------------------------------------------
def record(runs_root: Path, model: str) -> dict[str, Any]:
    """Every run of a model, the deleted ones included, with the model's totals. Computed on
    read: live runs from their own results, deleted runs from the line kept when they went."""
    runs_root = Path(runs_root)
    stored = _stored(runs_root, model)
    live = [line(st, runs_root / st.run_id) for st in members(runs_root, model)]
    lines = sorted([*live, *stored.get("deleted", [])], key=lambda r: r.get("at") or "")
    opened: list[str] = []
    for r in lines:
        opened += [c for c in r.get("holdout_scored") or [] if c not in opened]
    main = next((r["run_id"] for r in live if r["role"] == "main"), None)
    return {
        "model": model,
        "main": main,
        "lines": lines,
        "n_runs": len(lines),
        "n_trials": sum(1 for r in lines if r["run_id"] != main),
        "n_deleted": sum(1 for r in lines if r.get("deleted_at")),
        "test_evaluations": sum(int(r.get("test_evaluations") or 0) for r in lines),
        "holdout_scored": opened,
        "events": stored.get("events", []),
    }


def record_for(run_dir: Path) -> dict[str, Any]:
    """The record of the model a run belongs to (one line, its own, when it never had a
    trial run)."""
    run_dir = Path(run_dir)
    return record(run_dir.parent, model_of(RunState.load(run_dir)))


def summary(rec: dict[str, Any], run_id: str) -> dict[str, Any]:
    """The part of a record a stage result carries: small, and without the event log."""
    others = [r for r in rec["lines"] if r["run_id"] != run_id]
    mine = next((r for r in rec["lines"] if r["run_id"] == run_id), {})
    # The run that carries this summary is the one being validated: say that, not the
    # status it happened to have while the stage ran.
    lines = [r | {"stands": "the run validated here"} if r["run_id"] == run_id else r
             for r in rec["lines"]]
    return {"model": rec["model"], "n_runs": rec["n_runs"], "n_other_runs": len(others),
            "n_deleted": rec["n_deleted"], "test_evaluations": rec["test_evaluations"],
            "test_evaluations_this_run": int(mine.get("test_evaluations") or 0),
            "holdout_scored": rec["holdout_scored"], "lines": lines}


def describe(r: dict[str, Any]) -> str:
    """One line of the record as a sentence (documents, the CLI, a finding)."""
    what = "main run" if r["role"] == "main" else "trial run"
    parts = [f"{r['run_id']} ({what}"
             + (f", from “{LABELS.get(r['gate'], r['gate'])}”" if r.get("gate") else "")
             + (f", the main run until {r['replaced_at'][:10]}"
                if r.get("replaced_at") and r["role"] == "trial" else "") + ")"]
    if r.get("note"):
        parts.append(f"to try: {r['note']}")
    if r.get("changed"):
        parts.append("changed " + ", ".join(CHANGE_LABELS.get(k, k) for k in r["changed"]))
    parts.append(r["stands"])
    if r.get("champion"):
        got = f"champion {r['champion']}"
        if r.get("test_metric") is not None and r.get("metric"):
            got += f", {r['metric']} {r['test_metric']:.4f} on the {r.get('sample') or 'test'} sample"
        parts.append(got)
    parts.append(f"test sample used {r.get('test_evaluations') or 0} time(s)")
    if r.get("deleted_at"):
        parts.append(f"files deleted {r['deleted_at'][:10]}"
                     + (f" by {r['deleted_by']}" if r.get("deleted_by") else ""))
    return "; ".join(parts) + "."


# --- making a trial run ------------------------------------------------------------------------
def can_branch(state: RunState, gate: str) -> str:
    """Why a trial run cannot be made from this gate now ('' when it can)."""
    if gate not in GATES:
        return (f"{LABELS.get(gate, gate)} is not a developer decision: a trial run is made "
                "from one of " + ", ".join(f"“{LABELS[g]}”" for g in GATES) + ".")
    if state.status_of(gate) not in ("done", "awaiting"):
        return f"This run has not reached “{LABELS[gate]}” yet."
    if any(s.status == "running" for s in state.steps.values()):
        return "A step is running; wait for the run to pause."
    return ""


def branch_gates(state: RunState) -> list[str]:
    return [g for g in GATES if not can_branch(state, g)]


def branch(runs_root: Path, run_id: str, gate: str, *, note: str, by: str) -> str:
    """Copy a run into a trial run that waits at ``gate``. Everything before the gate comes
    across as it was (results, data, the search cache), the later developer steps too, so an
    unchanged decision changes nothing; nothing from validation on is copied. Returns the
    trial run's id. The caller (``service.trial_run``) holds nothing while the files copy."""
    runs_root = Path(runs_root)
    src = runs_root / run_id
    note = " ".join(str(note or "").split())
    if not note:
        raise TrialError("Say what this trial run is for: the note is its line in the "
                         "model's record.")
    lock = run_lock(src)
    with lock:
        parent = RunState.load(src)
        why = can_branch(parent, gate)
        if why:
            raise TrialError(why)
        model = model_of(parent)
        if parent.lineage is None:  # the first trial run of this model: its run is the main run
            parent.lineage = Lineage(model=model, trial=False, by=parent.created_by,
                                     at=parent.created_at)
            parent.save(src)
        snapshot = parent.model_copy(deep=True)
        new_id = new_run_id()
        dest = runs_root / new_id
        # Copied under the parent's lock: its state cannot move while the files are read.
        # (Short: a run is a few megabytes. Stages never run while it is held, by the check above.)
        shutil.copytree(src, dest, ignore=_ignore(src))
    state = snapshot
    state.run_id = new_id
    state.created_by, state.created_at, state.version = by, utcnow(), 0
    state.package, state.halted_reason = None, None
    for step in BEYOND:
        if state.status_of(step) != "skipped":
            state.steps[step] = type(state.steps[step])()
    state.set_step(gate, "awaiting", "a trial run: decide this differently", by=by)
    seen = counts(dest)
    state.lineage = Lineage(
        model=model, trial=True, parent=run_id, gate=gate, note=note, by=by,
        base={"overrides": parent.overrides.model_dump(mode="json"),
              "search_fingerprint": seen["search_fingerprint"]},
        inherited={"backtest": seen["backtest"], "search": seen["search"]})
    state.status = "awaiting"
    state.save(dest)
    _event(runs_root, model, "trial", new_id, by, parent=run_id, gate=gate, note=note)
    return new_id


def _ignore(src: Path):
    beyond = {s for s in BEYOND if s in STAGES}

    def ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        if here == src:
            return {n for n in names if n in _NOT_COPIED}
        if here == src / "stages":
            return {n for n in names if n in beyond}
        return set()

    return ignore


# --- making a trial run the main run -----------------------------------------------------------
def can_promote(state: RunState) -> str:
    if not state.is_trial:
        return "This is already the main run."
    if any(s.status == "running" for s in state.steps.values()):
        return "A step is running; wait for the run to pause."
    return ""


def promote(runs_root: Path, run_id: str, *, note: str = "", by: str) -> str | None:
    """Make a trial run the main run of its model. The run it replaces becomes a trial run and
    keeps everything it recorded; a package it had sealed is no longer in force. Returns the
    replaced run's id (None when its files were already deleted)."""
    runs_root = Path(runs_root)
    target = runs_root / run_id
    state = RunState.load(target)
    why = can_promote(state)
    if why:
        raise TrialError(why)
    model = model_of(state)
    old = next((m for m in members(runs_root, model) if not m.is_trial), None)
    if old is not None:
        if any(s.status == "running" for s in old.steps.values()):
            raise TrialError(f"The main run {old.run_id} is running a step; wait for it to pause.")
        old_dir = runs_root / old.run_id
        with run_lock(old_dir):
            st = RunState.load(old_dir)
            from .engine.process import supersede

            supersede(st, f"{run_id} was made the main run by {by}")
            st.lineage.trial = True
            st.lineage.replaced_at, st.lineage.replaced_by = utcnow(), run_id
            st.save(old_dir)
    with run_lock(target):
        st = RunState.load(target)
        st.lineage.trial = False
        st.lineage.promoted_at, st.lineage.promoted_by = utcnow(), by
        st.lineage.promoted_note = " ".join(str(note or "").split())
        if st.status == "completed":  # validation is open to it now
            st.status = "running"
        st.save(target)
    _event(runs_root, model, "promoted", run_id, by, replaced=old.run_id if old else None,
           note=note)
    return old.run_id if old else None


# --- deleting --------------------------------------------------------------------------------
def can_delete(runs_root: Path, state: RunState) -> str:
    """Why a run of a model with trial runs cannot be deleted now ('' when it can)."""
    if state.lineage is None or state.is_trial:
        return ""
    others = [m.run_id for m in members(Path(runs_root), model_of(state))
              if m.run_id != state.run_id]
    if others:
        return ("it is the main run of a model with trial runs (" + ", ".join(others)
                + "): make one of them the main run, or delete them first")
    return ""


def keep_line(runs_root: Path, run_id: str, *, by: str) -> dict[str, Any] | None:
    """Before a run's files go: write its line into the model's record. Returns the line, or
    None for a run that never had a trial run (nothing to keep it in)."""
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id
    try:
        state = RunState.load(run_dir)
    except Exception:
        return None
    if state.lineage is None:
        return None
    model = model_of(state)
    kept = line(state, run_dir) | {"deleted_at": utcnow(), "deleted_by": by, "role": "trial"
                                   if state.is_trial else "main"}
    with _lock(runs_root, model):
        rec = _stored(runs_root, model)
        rec["deleted"] = [r for r in rec.get("deleted", []) if r["run_id"] != run_id] + [kept]
        rec["events"].append({"at": kept["deleted_at"], "what": "deleted", "run_id": run_id,
                              "by": by})
        _store(runs_root, model, rec)
    return kept
