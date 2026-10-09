"""Comparing results: one run against another, or a stage's re-run against what it replaced.

Pure functions over what is already on disk (``state.json``, ``stages/*/result.json``,
``config.yaml``). Nothing here is judgment and nothing is stored: a comparison restates what two
records say and subtracts. "Better" is only claimed where the direction is a property of the
metric (the run's own metric direction, or a quantity that is lower-is-better by definition);
everything else is reported as a difference without a verdict.
"""

from __future__ import annotations

import math
from typing import Any

from .artifacts import StageResult
from .engine.graph import GATES, LABELS, STAGES
from .engine.state import RunState

# Metrics whose good direction does not depend on the project's metric.
_HIGHER = {"gini", "overall_score", "ks"}
_LOWER = {"cv_std", "psi", "n_blockers", "brier", "ece"}
# Metrics reported in the project's own metric (direction comes from the model stage).
_PRIMARY = {"cv_mean", "holdout_metric", "oos_metric"}
METRIC_LABELS = {
    "cv_mean": "Cross-validated metric (mean)", "cv_std": "Cross-validated metric (std)",
    "holdout_metric": "Sealed holdout metric", "oos_metric": "Out-of-sample metric",
    "n_candidates_tried": "Candidates tried", "n_transforms": "Transforms kept", "gini": "Gini",
    "psi": "Population stability (PSI)", "overall_score": "Validation rubric score",
    "n_blockers": "Validation blockers",
}


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return None if math.isnan(v) or math.isinf(v) else float(v)


def _primary_direction(results: dict[str, StageResult]) -> str | None:
    model = results.get("model")
    d = (model.payload or {}).get("direction") if model is not None else None
    return {"maximize": "higher", "minimize": "lower"}.get(d)


def _direction(key: str, primary: str | None) -> str | None:
    if key in _PRIMARY:
        return primary
    return "higher" if key in _HIGHER else "lower" if key in _LOWER else None


def metric_rows(a: dict[str, StageResult], b: dict[str, StageResult], stages: list[str] | None = None) -> list[dict]:
    """Every metric either side reports, stage by stage, with the difference ``b - a``.

    ``better`` is ``"a"``, ``"b"``, ``"same"`` or ``None`` (no direction is claimed, or one side
    has no number)."""
    pa, pb = _primary_direction(a), _primary_direction(b)
    like_for_like = pa == pb  # a different metric direction means the primary metrics do not compare
    rows = []
    for stage in stages or STAGES:
        ma = a[stage].metrics if stage in a else {}
        mb = b[stage].metrics if stage in b else {}
        for key in [*ma, *[k for k in mb if k not in ma]]:
            va, vb = ma.get(key), mb.get(key)
            na, nb = _num(va), _num(vb)
            if na is None and nb is None and va == vb:
                continue  # an unchanged label (e.g. the same champion) is not a difference
            direction = _direction(key, pa if like_for_like else None)
            delta = nb - na if na is not None and nb is not None else None
            better = None
            if delta is not None and direction:
                better = "same" if delta == 0 else ("b" if (delta > 0) == (direction == "higher") else "a")
            elif delta == 0:
                better = "same"
            rows.append({"stage": stage, "key": key, "label": METRIC_LABELS.get(key, key.replace("_", " ")),
                         "a": va, "b": vb, "delta": delta, "direction": direction, "better": better})
    return rows


def _finding_key(stage: str, f) -> tuple[str, str]:
    return stage, f.id


def finding_changes(a: dict[str, StageResult], b: dict[str, StageResult], stages: list[str] | None = None) -> dict:
    """Findings only in ``b`` (new) and only in ``a`` (gone), by stage and finding id."""
    def index(results):
        return {_finding_key(s, f): f for s in (stages or STAGES) if s in results for f in results[s].findings}

    ia, ib = index(a), index(b)

    def rows(keys, src):
        out = [{"stage": k[0], "id": k[1], "severity": src[k].severity.value, "message": src[k].message} for k in keys]
        return sorted(out, key=lambda r: (-_RANK.get(r["severity"], 0), STAGES.index(r["stage"]), r["id"]))

    return {"new": rows([k for k in ib if k not in ia], ib), "gone": rows([k for k in ia if k not in ib], ia)}


_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


# Where a run lives says nothing about what it is.
_CONFIG_IGNORE = ("runs_dir", "data.path")


def config_changes(a: dict | None, b: dict | None) -> list[dict]:
    fa, fb = _flatten(a or {}), _flatten(b or {})
    return [{"key": k, "a": fa.get(k), "b": fb.get(k)} for k in sorted(set(fa) | set(fb))
            if fa.get(k) != fb.get(k) and k not in _CONFIG_IGNORE]


def _champion(results: dict[str, StageResult]) -> str | None:
    model = results.get("model")
    if model is None:
        return None
    return (model.payload or {}).get("champion_label") or model.metrics.get("champion")


def _metric_name(results: dict[str, StageResult]) -> str | None:
    model = results.get("model")
    return (model.payload or {}).get("metric") if model is not None else None


def _summary(state: RunState, results: dict[str, StageResult]) -> dict:
    return {"run_id": state.run_id, "project": state.project, "status": state.status, "mode": state.mode,
            "provider": state.provider, "created_at": state.created_at, "champion": _champion(results),
            "metric": _metric_name(results), "spend_usd": state.spend_usd}


def _decision_rows(sa: RunState, sb: RunState) -> list[dict]:
    rows = []
    for gate in GATES:
        da, db = sa.last_decision(gate), sb.last_decision(gate)
        if da is None and db is None:
            continue
        def view(d):
            return None if d is None else {"action": d.action, "actor": d.actor, "reason": d.reason}
        # who clicked (a person, or autonomous mode) is shown, but only a different action counts
        rows.append({"gate": gate, "label": LABELS[gate], "a": view(da), "b": view(db),
                     "different": (da.action if da else None) != (db.action if db else None)})
    return rows


def _override_rows(sa: RunState, sb: RunState) -> list[dict]:
    oa, ob = sa.overrides, sb.overrides
    rows = []
    ea, eb = set(oa.exclude_columns), set(ob.exclude_columns)
    if ea != eb:
        rows.append({"what": "Excluded columns", "a": sorted(ea - eb), "b": sorted(eb - ea),
                     "note": f"{len(ea & eb)} excluded in both"})
    for field in sorted(set(oa.design) | set(ob.design)):
        if oa.design.get(field) != ob.design.get(field):
            rows.append({"what": f"Design: {field.replace('_', ' ')}", "a": oa.design.get(field),
                         "b": ob.design.get(field), "note": None})
    if oa.slate != ob.slate:
        def fams(slate):
            return None if slate is None else sorted({i.get("family", "?") for i in slate})
        rows.append({"what": "Hypothesis slate", "a": fams(oa.slate), "b": fams(ob.slate),
                     "note": "None = the design lead's slate was accepted"})
    if oa.champion != ob.champion:
        rows.append({"what": "Champion override", "a": oa.champion, "b": ob.champion, "note": None})
    return rows


def compare_runs(sa: RunState, ra: dict[str, StageResult], ca: dict | None,
                 sb: RunState, rb: dict[str, StageResult], cb: dict | None) -> dict:
    """Run ``b`` against run ``a`` (differences read as "b relative to a")."""
    a, b = _summary(sa, ra), _summary(sb, rb)
    caveats = []
    if a["project"] != b["project"]:
        caveats.append(f"Different projects ({a['project']} vs {b['project']}): the runs may not share data or design.")
    if a["metric"] and b["metric"] and a["metric"] != b["metric"]:
        caveats.append(f"Different metrics ({a['metric']} vs {b['metric']}): the headline numbers are not like for like.")
    for s, r in ((a, ra), (b, rb)):
        missing = [LABELS[x] for x in STAGES if x not in r]
        if missing:
            caveats.append(f"{s['run_id']} has not finished: no result yet for {', '.join(missing)}.")
        state = sa if s is a else sb
        outdated = [LABELS[x] for x in STAGES if x in r and state.status_of(x) in ("stale", "running")]
        if outdated:
            caveats.append(f"{s['run_id']} is re-running: its results for {', '.join(outdated)} predate its latest decision.")
    verdicts = [{"stage": s, "label": LABELS[s],
                 "a": ra[s].verdict.value if s in ra else None, "b": rb[s].verdict.value if s in rb else None}
                for s in STAGES if s in ra or s in rb]
    return {"a": a, "b": b, "caveats": caveats, "metrics": metric_rows(ra, rb), "verdicts": verdicts,
            "decisions": _decision_rows(sa, sb), "overrides": _override_rows(sa, sb),
            "config": config_changes(ca, cb), "findings": finding_changes(ra, rb)}


def step_changes(stage: str, previous: StageResult | None, current: StageResult | None) -> dict | None:
    """What a re-run of one stage changed, or None when there is nothing to compare (first run,
    still running, or the previous attempt crashed)."""
    if previous is None or current is None or "ERROR" in (previous.verdict.value, current.verdict.value):
        return None
    pa, pb = {stage: previous}, {stage: current}
    rows = metric_rows(pa, pb, [stage])
    champion = None
    if stage == "model" and _champion(pa) != _champion(pb):
        champion = {"a": _champion(pa), "b": _champion(pb)}
    return {"stage": stage, "metrics": rows, "champion": champion,
            "verdict": {"a": previous.verdict.value, "b": current.verdict.value},
            "findings": finding_changes(pa, pb, [stage]),
            "unchanged": champion is None and previous.verdict == current.verdict
            and all(r["delta"] in (0, 0.0) for r in rows if r["delta"] is not None)
            and not any(r["delta"] is None for r in rows)}
