"""Gate handlers: turn a human (or auto) decision into state changes.

Each handler validates the decision against the reviewed stage's result and returns the steps to
invalidate. It may change ``state.overrides`` (the effective config), add a Challenge (send-back),
or answer Gaps. Handlers never run a stage — the engine re-runs whatever became stale.
"""

from __future__ import annotations

from typing import Any

from .graph import LABELS, SEND_BACK_TARGETS, STAGE_OF_GATE, descendants
from .process import CORE_DESIGN
from .state import Challenge, RunState

ACTIONS: dict[str, set[str]] = {
    "gate_intent": {"accept", "edit", "send_back"},
    "gate_data": {"accept", "edit", "send_back"},
    "gate_design": {"accept", "edit", "send_back"},
    "gate_candidates": {"accept", "edit", "send_back"},
    "gate_champion": {"accept", "override", "send_back"},
    "gate_validation": {"accept", "send_back", "reject"},
    "gate_signoff": {"approve", "reject", "accept"},
}


class GateError(ValueError):
    """A decision the engine refuses (wrong state, invalid payload, BLOCK override)."""


def _clip(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def why(gate: str, action: str, payload: dict[str, Any] | None, reason: str = "") -> str:
    """One line saying what a decision changed, recorded on every step it makes stale. Mechanical:
    it restates the decision, it does not judge it."""
    payload = payload or {}
    label = LABELS[gate]
    if action == "send_back":
        target = payload.get("target") or (
            "model" if gate in ("gate_validation", "gate_champion") else STAGE_OF_GATE[gate])
        message = payload.get("message") or reason
        return f"{label}: sent back to “{LABELS.get(target, target)}”" + (f" — {_clip(message)}" if message else "")
    if gate == "gate_intent":
        if payload.get("answers"):
            return f"{label}: {len(payload['answers'])} interview question(s) answered"
        return f"{label}: the brief was accepted"
    if gate == "gate_data":
        cols = payload.get("exclude_columns")
        if payload.get("join_changed"):
            return f"{label}: the join of the sources was changed"
        if payload.get("target"):
            return f"{label}: dependent variable set to {payload['target']}"
        if action == "edit" and cols is not None:
            return f"{label}: exclusions changed to {len(cols)} column(s)" + (f" ({_clip(', '.join(cols), 80)})" if cols else "")
        return f"{label}: recommended exclusions accepted"
    if gate == "gate_design":
        parts = []
        if payload.get("slate") is not None:
            parts.append(f"slate edited to {len(payload['slate'])} hypothesis(es)")
        if payload.get("plan"):
            fields = sorted(f for section in payload["plan"].values() for f in section)
            parts.append(f"development plan changed ({_clip(', '.join(fields), 80)})")
        if payload.get("answers"):
            parts.append(f"{len(payload['answers'])} design question(s) answered")
        return f"{label}: " + (" and ".join(parts) if parts else action.replace("_", " "))
    if gate == "gate_candidates" and action == "edit":
        return (f"{label}: pool set to {_clip(', '.join(payload.get('pool') or []), 80)}"
                + (f" — {_clip(reason)}" if reason else ""))
    if gate == "gate_champion" and action == "override":
        return f"{label}: champion set to {payload.get('champion')}" + (f" — {_clip(reason)}" if reason else "")
    return f"{label}: {action.replace('_', ' ')}"


def _changed_downstream(gate: str) -> list[str]:
    return descendants(gate)


def send_back(state: RunState, gate: str, payload: dict[str, Any], reason: str) -> list[str]:
    target = payload.get("target") or STAGE_OF_GATE[gate]
    # Outcomes analysis measures; what a reviewer wants reconsidered is the search behind it.
    if gate in ("gate_validation", "gate_champion") and not payload.get("target"):
        target = "model"
    if target not in SEND_BACK_TARGETS:
        raise GateError(f"cannot send work back to {target!r}; choose one of {SEND_BACK_TARGETS}")
    message = (payload.get("message") or reason or "").strip()
    if not message:
        raise GateError("a send-back needs a message telling the agent what to reconsider")
    state.challenges.append(Challenge(
        source="human", target_stage=target, severity=payload.get("severity", "high"),
        message=message, evidence=list(payload.get("evidence") or []),
        remedy=payload.get("remedy")))
    return [target]


def handle(state: RunState, gate: str, action: str, payload: dict[str, Any], reason: str,
           results: dict[str, Any]) -> tuple[list[str], bool]:
    """Apply a decision. Returns (steps to invalidate, whether the gate itself is now done)."""
    if action not in ACTIONS[gate]:
        raise GateError(f"{gate} does not accept {action!r}; allowed: {sorted(ACTIONS[gate])}")
    stage = STAGE_OF_GATE[gate]
    res = results.get(stage)
    verdict = res.verdict.value if res is not None else None
    payload = payload if payload is not None else {}

    if action == "send_back":
        return send_back(state, gate, payload, reason), False

    if gate == "gate_intent":
        brief = (res.payload if res is not None else None) or {}
        rerun = apply_answers(state, payload.get("answers") or {})
        if rerun:  # the interview continues: intake re-reads the brief with the answers
            return sorted(set(rerun)), False
        if action == "edit":
            raise GateError("nothing to apply: answer at least one interview question, or accept "
                            "the brief")
        blocking = [q for q in brief.get("questions") or []
                    if q.get("blocking") and (g := state.gap(q["id"])) and g.status == "open"]
        if blocking and not reason.strip():
            raise GateError(
                f"{len(blocking)} blocking interview question(s) are still open. Answer them, or "
                "give a reason for starting without them (they stay open; nobody can sign until "
                "use, horizon, default definition and segment are answered).")
        # What the sponsor's document states becomes part of the effective config. An answer
        # already recorded on the run wins over the document.
        changed = False
        for target, stated in ((state.overrides.design, brief.get("design_from_intent")),
                               (state.overrides.compliance, brief.get("compliance_from_intent"))):
            for field, value in (stated or {}).items():
                if value and field not in target:
                    target[field] = value
                    changed = True
        return (_changed_downstream(gate) if changed else []), True

    if gate == "gate_data":
        profile = res.payload if res is not None else {}
        # What the developer had in front of them: every script the analyst wrote, by hash.
        payload["reviewed_analyses"] = [
            {"id": a["id"], "sha256": a["code_sha256"]}
            for a in profile.get("analyses") or [] if a.get("kind") == "code"]
        linked = profile.get("linking") or {}
        plan = linked.get("plan") or {}
        if plan:  # several sources: the join the developer had in front of them
            payload["reviewed_join"] = plan.get("digest")
        asked = payload.get("join") if action == "edit" else None
        if asked and plan:
            from .. import linking

            # One answer for every link when the developer gives one; otherwise each link
            # keeps what the plan on the page says, and a link added here takes the rule's
            # advice (linking.unmatched_advice).
            current = {s["link"]: s for s in plan["steps"]}
            fixed = asked.get("unmatched") if asked.get("unmatched") in linking.UNMATCHED else None
            steps, problems = linking.resolve(
                asked.get("base") or plan["base"],
                [{"link": i, "many": asked.get("many") or "aggregate",
                  "unmatched": fixed or current.get(i, {}).get("unmatched") or "keep"}
                 for i in asked.get("links") or []], linked)
            if problems:
                raise GateError("the join cannot be run: " + "; ".join(problems))
            for step in steps:
                if not fixed and step["link"] not in current:
                    step["unmatched"] = linking.unmatched_advice(step)[0]
            base_table = asked.get("base") or plan["base"]
            if linking.plan_digest(base_table, steps) != plan.get("digest"):
                if plan.get("source") == "profile":
                    raise GateError("the profile states the join; change data.join there and "
                                    "start a new run")
                state.overrides.join = {"base": base_table, "steps": steps}
                payload["join_changed"] = True
        elif plan and plan.get("source") != "profile":
            state.overrides.join = {"base": plan["base"], "steps": plan["steps"]}
        pick = payload.get("target") if action == "edit" else None
        analysed = profile.get("target")
        if payload.get("join_changed"):
            if pick and pick != analysed and profile.get("target_source") != "profile":
                state.overrides.target, state.overrides.task = pick, None
            return ["explore"], False  # explore rebuilds the dataset from the new join
        if pick and pick != analysed:
            if profile.get("target_source") == "profile":
                raise GateError("the profile fixes the dependent variable; change data.target "
                                "there and start a new run")
            if pick not in (profile.get("dtypes") or {}):
                raise GateError(f"unknown column {pick!r}")
            state.overrides.target, state.overrides.task = pick, None
            return ["explore"], False  # explore re-reads the data against the new target
        if profile.get("target_source") != "profile" and analysed:
            state.overrides.target, state.overrides.task = analysed, profile.get("task")
        if action == "accept":
            new = list(profile.get("recommended_exclusions", []))
        else:
            new = list(payload.get("exclude_columns")
                       if payload.get("exclude_columns") is not None
                       else state.overrides.exclude_columns or
                       profile.get("recommended_exclusions", []))
            known = set(profile.get("dtypes", {}))
            unknown = [c for c in new if c not in known]
            if unknown:
                raise GateError(f"unknown column(s) {unknown}")
            if profile.get("target") in new:
                raise GateError("the target column cannot be excluded")
        changed = sorted(new) != sorted(state.overrides.exclude_columns)
        state.overrides.exclude_columns = sorted(set(new))
        return (_changed_downstream(gate) if changed else []), True

    if gate == "gate_design":
        answers = payload.get("answers") or {}
        slate = payload.get("slate")
        invalidate: list[str] = []
        if action == "edit" and slate is not None:
            if not slate:
                raise GateError("the slate cannot be empty")
            fittable = set((res.payload.get("fittable_families") if res else None) or [])
            for item in slate:
                if "family" not in item:
                    raise GateError("every slate item needs a family")
                if fittable and item["family"] not in fittable:
                    raise GateError(f"family {item['family']!r} is not engine-fittable here")
            state.overrides.slate = slate
            invalidate += _changed_downstream(gate)
        elif action == "accept" and state.overrides.slate is not None:
            pass  # accepting keeps whatever slate is in force
        patch = payload.get("plan") if action == "edit" else None
        replan = apply_plan(state, res, patch) if patch else []
        rerun = apply_answers(state, answers) + replan
        if rerun:  # the design stage restates the plan in force, and the gate re-opens on it
            return sorted(set(invalidate) | set(rerun)), False
        return invalidate, True

    if gate == "gate_candidates":
        model = (res.payload if res is not None else None) or {}
        current = list(model.get("pool_ids") or [])
        if action == "edit":
            pool = list(dict.fromkeys(payload.get("pool") or []))
            if not pool:
                raise GateError("the pool cannot be empty: outcomes analysis needs a candidate")
            known = ({a["id"] for a in model.get("leaders") or []} | set(current)
                     | set(model.get("evaluated_ids") or []))
            unknown = [i for i in pool if i not in known]
            if unknown:
                raise GateError(f"{unknown} were not evaluated in this search; add a candidate "
                                "by its id from the leaders or the ledger")
            if sorted(pool) == sorted(current):
                return [], True  # the pool on the page is the pool asked for
            if not reason.strip():
                raise GateError("changing the pool requires a reason")
            state.overrides.pool = pool
            return ["model"], False  # the new members are fitted and tested; the gate re-opens
        return [], True

    if gate == "gate_champion":
        backtest = (res.payload if res is not None else None) or {}
        pool = {c["id"]: c for c in backtest.get("candidates") or []}
        if action == "override":
            pick = payload.get("champion")
            if pick not in pool:
                raise GateError(f"champion {pick!r} is not in the candidate pool {sorted(pool)}")
            if not reason.strip():
                raise GateError("choosing a different champion requires a reason")
            if pick == backtest.get("champion_id"):
                return [], True  # choosing the current champion is an accept
            state.overrides.champion = pick
            return ["backtest"], False  # its outcomes analysis is written up; the gate re-opens
        return [], True

    if gate == "gate_validation":
        if action == "accept":
            if verdict == "BLOCK":
                raise GateError("a BLOCK cannot be accepted: confirmed leakage invalidates the model; "
                                "send the work back instead")
            if verdict == "FAIL" and not reason.strip():
                raise GateError("accepting a FAIL verdict requires a reason (accepted risk)")
            return [], True
        if action == "reject":
            state.status = "rejected"
            state.halted_reason = f"rejected at {gate}: {reason or 'no reason given'}"
            return [], True

    if gate == "gate_signoff":
        if action == "approve":
            if verdict == "BLOCK":
                raise GateError("the docs↔code review BLOCKed; the model cannot be signed off")
            state.status = "approved"
        elif action == "reject":
            state.status = "rejected"
            state.halted_reason = f"rejected at sign-off: {reason or 'no reason given'}"
        return [], True

    raise GateError(f"unhandled decision {gate}/{action}")


def apply_plan(state: RunState, res: Any, patch: dict[str, Any]) -> list[str]:
    """Lay the developer's edit over the plan on the page. Refused when the plan could not be
    run on this data; returns the steps to re-run (none when nothing changed)."""
    from ..modeling import plan as plan_mod

    current = ((res.payload if res is not None else None) or {}).get("plan")
    if not current:
        raise GateError("there is no development plan on this run to edit; re-run the design stage")
    patch = {s: dict(patch.get(s) or {}) for s in plan_mod.SECTIONS if patch.get(s)}
    unknown = [f"{s}.{f}" for s, fields in patch.items() for f in fields if f not in current[s]]
    if unknown:
        raise GateError(f"unknown plan field(s) {unknown}")
    trial = plan_mod.merge(current, patch, "human")
    bad = plan_mod.problems(trial)
    if bad:
        raise GateError("the plan cannot be run: " + "; ".join(bad))
    if all(trial[s] == current[s] for s in plan_mod.SECTIONS):
        return []
    merged = {s: dict((state.overrides.plan or {}).get(s) or {}) for s in plan_mod.SECTIONS}
    for section, fields in patch.items():
        merged[section].update(fields)
    state.overrides.plan = {s: v for s, v in merged.items() if v}
    return ["ideate"]


def apply_answers(state: RunState, answers: dict[str, str], *, assume: bool = False,
                  limitation: bool = False) -> list[str]:
    """Answer gaps. Design answers fill the effective config; returns stages to re-run.

    ``assume`` records the question as an assumption and ``limitation`` as a model limitation
    instead of answering it: the stage that raised it does not re-run, and validation (which
    challenges both) does when it has already run."""
    assume = assume or limitation
    rerun: list[str] = []
    from .state import utcnow

    for gap_id, text in (answers or {}).items():
        gap = state.gap(gap_id)
        if gap is None:
            raise GateError(f"unknown question {gap_id!r}")
        text = str(text).strip()
        if assume and gap.design_field in CORE_DESIGN:
            raise GateError(
                f"{gap.design_field.replace('_', ' ')} is a sponsor decision. "
                "Answer it; it cannot be accepted as an assumption or a limitation.")
        if not text and not assume:
            continue
        gap.status = "limitation" if limitation else "assumed" if assume else "answered"
        gap.answer = text or gap.answer
        gap.answered_at = utcnow()
        if gap.design_field and not assume:
            state.overrides.design[gap.design_field] = text
        if not assume:
            reentry = gap.reentry
            if (reentry == "intake" and gap.design_field
                    and state.status_of("gate_intent") == "done"):
                # The intent is confirmed: a design answer given later is a design decision and
                # re-enters at the design stage, not at the interview.
                reentry = "ideate"
            rerun.append(reentry)
        else:
            rerun.append("validate")
    return rerun
