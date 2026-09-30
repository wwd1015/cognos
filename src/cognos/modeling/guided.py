"""Agent-guided search — the modeler agent as the mutation function (ADR-0001 stage B).

After the deterministic ratchet establishes a champion, the modeler agent proposes the *next*
experiment given the ledger so far — typically new feature engineering, sometimes a different
family/hyperparameters. The deterministic engine *disposes*: each proposal is applied target-hidden,
scored with the same leakage-safe CV, and kept only if it beats the incumbent on the frozen metric.
The agent can drive exploration without being able to hallucinate a result into the record.

``propose(context) -> dict`` is supplied by the model stage and routes through the agent runner,
so every proposal is validated, audited and logged like any other agent call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .fit import DEFAULT_FAMILIES, GLM_FAMILIES, LINEAR_FAMILIES, Candidate, make_fit_predict
from .metrics import CVResult, cv_score, is_better
from .transforms import SAFE_NP_FUNCS, TransformSpec, apply_transforms

KNOWN_FAMILIES = set(LINEAR_FAMILIES) | set(GLM_FAMILIES) | {
    f for fams in DEFAULT_FAMILIES.values() for f in fams
}


@dataclass
class GuidedRound:
    idx: int
    proposal: dict
    accepted: bool
    score: float | None
    note: str


@dataclass
class GuidedResult:
    champion: Candidate
    champion_cv: CVResult
    transforms: list[TransformSpec]
    rounds: list[GuidedRound] = field(default_factory=list)
    improved: bool = False


def _parse_value(text: str) -> Any:
    t = str(text).strip()
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            continue
    return {"none": None, "true": True, "false": False}.get(t.lower(), t)


def parse_proposal(obj: dict, champion: Candidate) -> tuple[str, dict, list[TransformSpec]]:
    family = obj.get("family") if obj.get("family") in KNOWN_FAMILIES else champion.family
    hp = obj.get("hyperparams") or []
    if isinstance(hp, dict):  # tolerate the map form
        hyperparams = dict(hp)
    else:
        hyperparams = {h["name"]: _parse_value(h.get("value", "")) for h in hp
                       if isinstance(h, dict) and h.get("name")}
    transforms = [TransformSpec(name=str(t["name"]), expr=str(t["expr"]))
                  for t in obj.get("transforms", []) or []
                  if isinstance(t, dict) and t.get("name") and t.get("expr")]
    return family, hyperparams, transforms


def guided_search(
    propose: Callable[[dict], dict],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    champion: Candidate,
    champion_cv: CVResult,
    metric: str,
    direction: str,
    is_classification: bool,
    allowed_families: list[str],
    is_timeseries: bool = False,
    folds: int = 5,
    random_state: int = 42,
    rounds: int = 6,
) -> GuidedResult:
    """Run up to ``rounds`` agent-proposed experiments; keep any that beat the incumbent."""
    base_features = list(champion.features)
    best_cand, best_cv, best_transforms = champion, champion_cv, []
    history: list[GuidedRound] = []

    for i in range(rounds):
        context = {
            "metric": metric, "direction": direction,
            "champion": {"family": best_cand.family, "n_features": len(best_cand.features),
                         "cv_mean": round(float(best_cv.mean), 6),
                         "transforms": [t.to_dict() for t in best_transforms]},
            "columns": base_features,
            "allowed_families": allowed_families,
            "safe_np_funcs": sorted(SAFE_NP_FUNCS),
            "history": [{"round": r.idx, "proposal": r.proposal, "accepted": r.accepted,
                         "cv_score": r.score, "note": r.note} for r in history],
        }
        try:
            obj = propose(context)
        except Exception as exc:  # agent failure ends guided search; the ratchet champion stands
            history.append(GuidedRound(i, {}, False, None, f"agent error: {type(exc).__name__}"))
            break
        if not obj or obj.get("stop"):
            history.append(GuidedRound(i, obj or {}, False, None, "agent stopped"))
            break
        family, hyperparams, transforms = parse_proposal(obj, best_cand)
        try:
            X_aug, applied, _ = apply_transforms(X[base_features], list(best_transforms) + transforms)
            cand = Candidate(family=family, features=list(X_aug.columns),
                             hyperparams={**hyperparams, "random_state": random_state},
                             description=str(obj.get("rationale") or "agent-guided")[:120])
            cv = cv_score(make_fit_predict(cand, is_classification=is_classification), X_aug, y,
                          metric=metric, is_classification=is_classification,
                          is_timeseries=is_timeseries, folds=folds, random_state=random_state)
        except Exception as exc:
            history.append(GuidedRound(i, obj, False, None, f"eval failed: {type(exc).__name__}"))
            continue
        accepted = is_better(metric, cv.mean, best_cv.mean)
        if accepted:
            best_cand, best_cv = cand, cv
            best_transforms = list(applied)
        history.append(GuidedRound(i, obj, accepted, float(cv.mean),
                                   "kept" if accepted else "discarded (no improvement)"))

    return GuidedResult(
        champion=best_cand, champion_cv=best_cv, transforms=best_transforms,
        rounds=history, improved=is_better(metric, best_cv.mean, champion_cv.mean),
    )
