"""Seats, the design brief, and the sealed decision package.

Borrowed in spirit from a credit process that keeps three moments apart: the analysis
getting finished, a person approving it, and anything that happens after. The engine
stays mechanical. A seat is a permission, not a judgment. A package is a hash of
recorded facts, written once.

Finishing a run is not a signature. Autonomous mode is express preparation: it may
accept every gate so the stages can run, and it records those decisions as
``express``. Only an approver's ``approve``, with the design brief answered, seals
a package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

from ..fsutil import atomic_write
from .graph import LABELS, STAGES, STEPS
from .state import PackageSeal, RunState, utcnow

# The four sponsor decisions a package cannot be sealed without. Data questions
# may be accepted as assumptions; these may not.
CORE_DESIGN = ("use_case", "horizon", "default_definition", "segment")

SEAT_OF_GATE = {
    "gate_intent": "developer",
    "gate_data": "developer",
    "gate_design": "developer",
    "gate_candidates": "developer",
    "gate_champion": "developer",
    "gate_validation": "reviewer",
    "gate_signoff": "approver",
}

SEAT_LABEL = {
    "developer": "model developer",
    "reviewer": "independent reviewer",
    "approver": "approver",
    "express": "express preparation",
}

# One map, in order. The graph in graph.py is still what runs. This is only how
# the workbench says who a page belongs to.
MAP = (
    ("Intent", ("intake", "gate_intent"), "developer"),
    ("Data & design", ("explore", "gate_data", "ideate", "gate_design"), "developer"),
    ("Model", ("model", "gate_candidates", "backtest", "gate_champion"), "developer"),
    ("Challenge", ("validate", "gate_validation"), "reviewer"),
    ("Decision", ("comply", "document", "review", "gate_signoff"), "approver"),
)


def _refuse(message: str) -> NoReturn:
    """Raised as ``GateError`` so every front end already catches it. Imported lazily:
    ``gates`` imports this module."""
    from .gates import GateError
    raise GateError(message)


def seat_label(seat: str) -> str:
    return SEAT_LABEL.get(seat, seat or "this seat")


def resolve_seat(gate: str, seat: str | None) -> str:
    """The seat a decision is recorded under. Omitting it means the seat that owns the gate
    (tests and the CLI). Naming a different seat is refused."""
    owner = SEAT_OF_GATE.get(gate)
    if owner is None:
        _refuse(f"{gate} is not a review gate")
    if seat in (None, "", owner):
        return owner
    _refuse(f"{LABELS.get(gate, gate)} belongs to the {seat_label(owner)}. "
            f"You are acting as the {seat_label(seat)}.")


def effective_design(config: Any, state: RunState) -> dict[str, str]:
    """Sponsor answers in force: the profile, then any override recorded on the run."""
    design = config.design
    out = {field: str(getattr(design, field, "") or "").strip() for field in CORE_DESIGN}
    for field, value in state.overrides.design.items():
        if field in CORE_DESIGN and str(value or "").strip():
            out[field] = str(value).strip()
    return out


def missing_design(config: Any, state: RunState) -> list[str]:
    design = effective_design(config, state)
    return [field for field in CORE_DESIGN if not design.get(field)]


def preparation_of(state: RunState) -> str:
    """``express`` when every developer gate was accepted without a person. One human
    review of those gates makes the preparation human."""
    developer_gates = [g for g, seat in SEAT_OF_GATE.items() if seat == "developer"]
    decisions = [d for d in state.decisions if d.gate in developer_gates]
    if decisions and all(d.actor == "auto" or d.seat == "express" for d in decisions):
        return "express"
    return "human"


def material(config: Any, state: RunState, results: dict[str, Any]) -> dict[str, Any]:
    """What an approval binds to. Facts the engine already recorded, nothing judged."""
    def payload_of(stage: str) -> dict[str, Any]:
        res = results.get(stage)
        return (res.payload if res is not None else None) or {}

    model, backtest, validate = payload_of("model"), payload_of("backtest"), payload_of("validate")

    def verdict(stage: str) -> str | None:
        res = results.get(stage)
        return res.verdict.value if res is not None else None

    trials = validate.get("trials") or {}
    tried = ({"trial_runs": sorted(r["run_id"] for r in trials.get("lines") or []
                                   if r["run_id"] != state.run_id),
              "test_evaluations_model": trials.get("test_evaluations")}
             if trials.get("n_other_runs") else {})
    return {
        **tried,
        "design": effective_design(config, state),
        "exclude_columns": sorted(state.overrides.exclude_columns),
        "slate": state.overrides.slate,
        "plan": state.overrides.plan,
        "pool": list(model.get("pool_ids") or []),
        "champion_id": backtest.get("champion_id"),
        "test_metric": backtest.get("oos_metric"),
        "holdout_metric": validate.get("holdout_metric"),
        "validation_verdict": verdict("validate"),
        "review_verdict": verdict("review"),
    }


def digest_of(body: dict[str, Any]) -> str:
    raw = json.dumps(body, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def seal_package(state: RunState, config: Any, results: dict[str, Any], run_dir: Path,
                 *, by: str = "") -> PackageSeal:
    """Write ``packages/vN.json`` once. A later decision supersedes the pointer in
    ``state.json``; it does not rewrite the file."""
    version = (state.package.version + 1) if state.package is not None else 1
    body = material(config, state, results)
    digest = digest_of(body)
    rel = f"packages/v{version}.json"
    path = Path(run_dir) / rel
    if path.exists():
        _refuse(f"package {rel} already exists and is not rewritten")
    sealed_at = utcnow()
    prep = preparation_of(state)
    snapshot = {
        "version": version,
        "digest": digest,
        "sealed_at": sealed_at,
        "seat": "approver",
        "by": by,
        "preparation": prep,
        "material": body,
        "decisions": [
            {"gate": d.gate, "action": d.action, "seat": d.seat, "actor": d.actor,
             "by": d.by, "reason": d.reason, "at": d.at}
            for d in state.decisions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(snapshot, indent=1, ensure_ascii=False, default=str))
    state.package = PackageSeal(
        version=version, digest=digest, sealed_at=sealed_at, preparation=prep, by=by,
        status="sealed", path=rel)
    return state.package


def supersede(state: RunState, why: str) -> None:
    """A sealed package stays on disk. The live pointer says it no longer matches the run,
    and an approval that pointed at it is no longer in force."""
    if state.package is None or state.package.status != "sealed":
        return
    state.package.status = "superseded"
    state.package.superseded_because = why
    if state.status == "approved":
        state.status = "running"


def next_action(state: RunState) -> dict[str, str]:
    """The single thing the map is waiting on. Computed on read."""
    for step in STEPS:
        if state.status_of(step) == "blocked":
            return {"kind": "blocked", "step": step, "seat": "",
                    "text": f"{LABELS.get(step, step)} is blocked."}
    for step in STEPS:
        if state.status_of(step) == "awaiting":
            seat = SEAT_OF_GATE.get(step, "")
            who = f" — waiting for the {seat_label(seat)}" if seat else ""
            return {"kind": "decision", "step": step, "seat": seat,
                    "text": f"{LABELS.get(step, step)}{who}."}
    if state.is_trial and state.status == "completed":
        return {"kind": "trial_end", "step": "", "seat": "developer",
                "text": ("This trial run ends here: it does not go to validation. Compare it "
                         "with the main run, then make it the main run or leave it on record.")}
    if state.package is not None and state.package.status == "sealed" and state.status == "approved":
        prep = " Express prepared the analysis." if state.package.preparation == "express" else ""
        return {"kind": "sealed", "step": "gate_signoff", "seat": "approver",
                "text": (f"Package v{state.package.version} is sealed.{prep} "
                         "Approval is not deployment.")}
    if state.status == "completed":
        express = state.mode == "autonomous" or preparation_of(state) == "express"
        text = ("Analysis finished. Express accepted the gates. Nobody has signed a package."
                if express else
                "Analysis finished. The gates were accepted. Nobody has signed a package.")
        return {"kind": "prepared", "step": "", "seat": "", "text": text}
    if state.status == "rejected":
        return {"kind": "rejected", "step": "", "seat": "",
                "text": state.halted_reason or "Rejected."}
    running = next((s for s in STEPS if state.status_of(s) == "running"), "")
    if running:
        return {"kind": "working", "step": running, "seat": "",
                "text": f"{LABELS.get(running, running)} is running."}
    return {"kind": "working", "step": "", "seat": "", "text": "Working through the stages."}


def journal(state: RunState, results: dict[str, Any]) -> list[dict[str, str]]:
    """Who did what, in order. People, agents, and the engine. Computed on read from
    stamps the run already carries. Never stored, never edited."""
    rows: list[dict[str, str]] = []
    for stage in STAGES:
        res = results.get(stage)
        rec = ((res.payload or {}).get("recommendation") if res is not None else None) or {}
        if not rec:
            continue
        summary = str((rec.get("output") or {}).get("summary") or "").strip()
        rows.append({
            "at": getattr(res, "started_at", "") or "",
            "who": "agent",
            "name": str(rec.get("agent") or stage),
            "what": summary or f"recommended at {stage}",
        })
    for decision in state.decisions:
        express = decision.seat == "express" or decision.actor == "auto"
        what = f"{LABELS.get(decision.gate, decision.gate)}: {decision.action.replace('_', ' ')}"
        if decision.reason:
            what += f" — {decision.reason}"
        rows.append({
            "at": decision.at,
            "who": "system" if express else "person",
            "name": (f"{decision.by} ({decision.seat})" if decision.by and not express
                     else decision.seat or decision.actor),
            "what": what,
        })
    lin = state.lineage
    if lin is not None and lin.parent:
        what = (f"Made as a trial run of {lin.parent}, to decide "
                f"“{LABELS.get(lin.gate or '', lin.gate)}” differently")
        rows.append({"at": lin.at, "who": "person", "name": f"{lin.by} (developer)",
                     "what": what + (f" — {lin.note}" if lin.note else "")})
    if lin is not None and lin.promoted_at:
        rows.append({"at": lin.promoted_at, "who": "person",
                     "name": f"{lin.promoted_by} (developer)",
                     "what": "Made the main run of the model"
                             + (f" — {lin.promoted_note}" if lin.promoted_note else "")})
    if lin is not None and lin.replaced_at:
        rows.append({"at": lin.replaced_at, "who": "system", "name": "engine",
                     "what": f"No longer the main run: {lin.replaced_by} was made the main run"})
    if state.package is not None:
        pkg = state.package
        what = f"Sealed package v{pkg.version}"
        if pkg.status == "superseded":
            what += " — no longer in force"
            if pkg.superseded_because:
                what += f" ({pkg.superseded_because})"
        elif pkg.preparation == "express":
            what += " (express prepared the analysis)"
        if pkg.by:
            what += f", approved by {pkg.by}"
        rows.append({"at": pkg.sealed_at, "who": "system", "name": "engine", "what": what})
    rows.sort(key=lambda row: row.get("at") or "")
    return rows
