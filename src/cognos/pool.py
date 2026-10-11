"""The candidate pool and the champion in force (ADR-0016).

The model stage ends on a pool, not a champion. The champion is whichever pool member is in
force: the developer's choice at the champion gate, else the Outcomes Analyst's recommendation,
else (before outcomes analysis has run) the first member of the pool.

``view`` is how every later reader sees the model stage: the stage's payload with the champion's
own record laid over it under the names those readers already use (``champion``, ``cv_mean``,
``coefficients``, ``diagnostics``, ``scorer_path`` ...). Mechanical reads of what the run
recorded, nothing judged and nothing stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def champion_id(model: dict[str, Any] | None, backtest: dict[str, Any] | None = None,
                chosen: str | None = None) -> str | None:
    """The id of the champion in force, or None when there is no pool yet."""
    ids = list((model or {}).get("pool_ids") or [])
    if not ids:
        return None
    if chosen in ids:
        return chosen
    recommended = (backtest or {}).get("champion_id")
    return recommended if recommended in ids else ids[0]


def member(model: dict[str, Any] | None, cid: str | None) -> dict[str, Any] | None:
    return next((m for m in (model or {}).get("pool") or [] if m["id"] == cid), None)


def detail(run_dir: str | Path, entry: dict[str, Any] | None) -> dict[str, Any]:
    """A member's full record (coefficients, p-values, diagnostics), kept on disk."""
    if not entry or not entry.get("detail_path"):
        return {}
    try:
        return json.loads((Path(run_dir) / entry["detail_path"]).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def view(run_dir: str | Path, model: dict[str, Any] | None,
         backtest: dict[str, Any] | None = None, validate: dict[str, Any] | None = None,
         chosen: str | None = None) -> dict[str, Any]:
    """The model payload as the stages after the choice read it."""
    model = dict(model or {})
    cid = champion_id(model, backtest, chosen)
    entry = member(model, cid)
    if entry is None:
        return model
    rec = detail(run_dir, entry)
    decided = bool(backtest) and (backtest or {}).get("champion_id") == cid
    candidate = rec.get("candidate") or {
        "family": entry["family"], "features": entry["features"],
        "hyperparams": entry.get("hyperparams", {}), "description": entry["label"]}
    model.update({
        "champion": candidate,
        "champion_id": cid,
        "champion_label": entry["label"],
        "champion_source": ((backtest or {}).get("champion_source") if decided
                            else "provisional"),
        "champion_decided": decided,
        "admissible_set": model.get("pool") or [],
        "cv_mean": entry.get("cv_mean"),
        "cv_std": entry.get("cv_std"),
        "cv_fold_scores": rec.get("cv_fold_scores") or [],
        "development_score": entry.get("score"),
        "scores": entry.get("scores") or {},
        "coefficients": rec.get("coefficients"),
        "pvalues": rec.get("pvalues"),
        "feature_importances": rec.get("feature_importances"),
        "diagnostics": rec.get("diagnostics") or {},
        "hazard": rec.get("hazard"),
        "transforms": rec.get("transforms") or [],
        "base_features": rec.get("base_features") or entry["features"],
        "scorer_path": entry.get("scorer_path"),
        "test_metric": (backtest or {}).get("oos_metric") if decided else None,
        "holdout_metric": (validate or {}).get("holdout_metric"),
        "holdout_evaluations": int((validate or {}).get("holdout_evaluations") or 0),
    })
    return model


def from_results(run_dir: str | Path, results: dict[str, Any], chosen: str | None = None
                 ) -> dict[str, Any]:
    """``view`` from a dict of stage results (the engine, the workbench, comparisons)."""
    def payload(stage: str) -> dict[str, Any] | None:
        res = results.get(stage)
        return (res.payload if res is not None else None) or None

    return view(run_dir, payload("model"), payload("backtest"), payload("validate"), chosen)
