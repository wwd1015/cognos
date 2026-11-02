"""Questions from a page (ADR-0021): a person asks the agent that owns a stage's page.

The agent answers as ``<agent>_ask``, with its stage's scope (the same facts and the same walls:
the validator never sees the modeler's reasoning, the modeler never a holdout fact), and every
number goes through a ``{{fact:<id>}}`` placeholder the engine renders. Asking changes nothing in
the run: no state, no result, no gap. What should change goes through a gate, where the answer
can suggest a send-back the person still has to make.

A stage's thread is kept in ``runs/<id>/asks/<stage>.json`` with who asked and the agent call
that answered (in the run's audit), so a reviewer can read what a developer was told.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .agents import facts as facts_mod
from .agents import slices
from .agents.contracts import STAGE_AGENT
from .fsutil import atomic_write

MAX_QUESTION = 2000
THREAD_SHOWN = 6  # earlier turns handed back to the agent with a new question


def agent_for(stage: str | None) -> str | None:
    """The agent that owns a stage's page, or None for a stage without one."""
    return STAGE_AGENT.get(stage or "")


def thread_path(run_dir: Path, stage: str) -> Path:
    return Path(run_dir) / "asks" / f"{stage}.json"


def thread(run_dir: Path, stage: str) -> list[dict[str, Any]]:
    path = thread_path(run_dir, stage)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _page(ctx, stage: str, status: str, agent: str) -> dict[str, Any]:
    """What the page shows the agent may be asked about: where the stage stands and the
    recommendation this agent recorded for it (never another agent's)."""
    res = ctx.get(stage)
    rec = (res.payload or {}).get("recommendation") if res is not None else None
    own = rec.get("output") if isinstance(rec, dict) and rec.get("agent") == agent else None
    return {"stage": stage, "status": status, "recommendation": own}


def ask(eng, stage: str, question: str, *, by: str) -> dict[str, Any]:
    """Ask the page's agent; return the turn as kept in the thread. Raises ``ValueError`` for a
    page without an agent or an empty question, and the runner's error when no valid answer
    came back."""
    agent = agent_for(stage)
    if agent is None:
        raise ValueError(f"no agent works on the {stage} page")
    q = " ".join(question.split())[:MAX_QUESTION]
    if not q:
        raise ValueError("write a question first")
    ctx = eng.context()
    step = f"{agent}_ask"
    prior = thread(eng.run_dir, stage)
    out = ctx.recommend(step, {
        "question": q,
        "thread": [{"question": t["question"], "answer": t["raw"]} for t in prior[-THREAD_SHOWN:]],
        "page": _page(ctx, stage, eng.state.status_of(stage), agent),
    })
    known = facts_mod.collect(ctx, prefixes=slices.fact_scope(step))
    used = dict.fromkeys([*facts_mod.placeholder_ids(out.answer), *out.cites])
    meta = ctx.runner.last.get(step, {})
    turn = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by": by,
        "agent": agent,
        "question": q,
        "raw": out.answer,
        "answer": facts_mod.render(out.answer, known),
        "cites": [{"id": k, "value": known[k]} for k in used if k in known],
        "suggested_challenge": out.suggested_challenge.strip(),
        "call_id": meta.get("call_id"),
        "provider": meta.get("provider"),
    }
    with eng.lock:  # two people may ask at once: re-read inside the lock, never across the call
        turns = [*thread(eng.run_dir, stage), turn]
        path = thread_path(eng.run_dir, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(turns, indent=1, default=str))
    return turn
