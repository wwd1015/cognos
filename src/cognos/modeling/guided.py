"""LLM-guided search — the reasoning layer as the mutation function (ADR-0001 stage B).

After the deterministic ratchet establishes a champion, the LLM proposes the *next* experiment given
the ledger so far — typically new feature engineering, sometimes a different family/hyperparameters.
The deterministic engine *disposes*: each proposal is applied target-hidden, scored with the same
leakage-safe CV, and kept only if it beats the incumbent on the frozen metric. The LLM can therefore
drive exploration without being able to hallucinate a result into the record.

Every proposal (prompt + raw response) is logged to the run's reasoning transcript for replay/audit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fit import DEFAULT_FAMILIES, GLM_FAMILIES, LINEAR_FAMILIES, Candidate, make_fit_predict
from .metrics import CVResult, cv_score, is_better
from .transforms import SAFE_NP_FUNCS, TransformSpec, apply_transforms

_KNOWN_FAMILIES = set(LINEAR_FAMILIES) | set(GLM_FAMILIES) | {
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


def build_prompt(profile: dict, champion: Candidate, champion_score: float, metric: str,
                 direction: str, columns: list[str]) -> str:
    return (
        "You are COGNOS's modeling agent. Propose ONE next experiment to improve a model.\n"
        f"Task metric: {metric} ({direction} is better). Current champion: {champion.family} on "
        f"{len(champion.features)} features, CV {metric}={champion_score:.5f}.\n"
        f"Available feature columns: {columns}\n"
        f"Allowed model families: {sorted(_KNOWN_FAMILIES)}\n"
        "You may propose feature-engineering transforms as expressions over the EXISTING columns using "
        f"only np.<fn> with fn in {sorted(SAFE_NP_FUNCS)} and arithmetic. Do NOT reference the target.\n"
        "Respond with ONLY a JSON object: {\"family\": <str>, \"hyperparams\": {..}, "
        "\"transforms\": [{\"name\": <str>, \"expr\": <str>}], \"rationale\": <str>}. "
        "Use [] for no transforms."
    )


def parse_proposal(obj: dict, champion: Candidate) -> tuple[str, dict, list[TransformSpec]]:
    family = obj.get("family") if obj.get("family") in _KNOWN_FAMILIES else champion.family
    hyperparams = obj.get("hyperparams") if isinstance(obj.get("hyperparams"), dict) else {}
    transforms = []
    for t in obj.get("transforms", []) or []:
        if isinstance(t, dict) and t.get("name") and t.get("expr"):
            transforms.append(TransformSpec(name=str(t["name"]), expr=str(t["expr"])))
    return family, hyperparams, transforms


def guided_search(
    brain,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    profile: dict,
    champion: Candidate,
    champion_cv: CVResult,
    metric: str,
    direction: str,
    is_classification: bool,
    is_timeseries: bool = False,
    folds: int = 5,
    random_state: int = 42,
    rounds: int = 6,
    log_fn: Callable[[str, str], None] | None = None,
) -> GuidedResult:
    """Run up to ``rounds`` LLM-proposed experiments; keep any that beat the incumbent on the metric."""
    base_features = list(champion.features)
    best_cand, best_cv, best_transforms = champion, champion_cv, []
    history: list[GuidedRound] = []

    for i in range(rounds):
        prompt = build_prompt(profile, best_cand, best_cv.mean, metric, direction, list(X.columns))
        try:
            raw = brain.generate(prompt, max_tokens=600)
        except Exception as exc:  # brain failure ends guided search, deterministic champion stands
            history.append(GuidedRound(i, {}, False, None, f"brain error: {type(exc).__name__}"))
            break
        if log_fn:
            log_fn(prompt, raw)
        obj = _safe_json(raw)
        if not obj:
            history.append(GuidedRound(i, {}, False, None, "unparseable proposal"))
            continue
        family, hyperparams, transforms = parse_proposal(obj, best_cand)
        try:
            X_aug, applied, _ = apply_transforms(X[base_features], list(best_transforms) + transforms)
            cand = Candidate(family=family, features=list(X_aug.columns),
                             hyperparams={**hyperparams, "random_state": random_state},
                             description=obj.get("rationale", "llm-guided")[:120])
            cv = cv_score(make_fit_predict(cand, is_classification=is_classification), X_aug, y,
                          metric=metric, is_classification=is_classification,
                          is_timeseries=is_timeseries, folds=folds, random_state=random_state)
        except Exception as exc:
            history.append(GuidedRound(i, obj, False, None, f"eval failed: {type(exc).__name__}"))
            continue
        accepted = is_better(metric, cv.mean, best_cv.mean)
        if accepted:
            best_cand, best_cv = cand, cv
            best_transforms = list(best_transforms) + applied
        history.append(GuidedRound(i, obj, accepted, cv.mean,
                                   "kept" if accepted else "discarded (no improvement)"))

    return GuidedResult(
        champion=best_cand, champion_cv=best_cv, transforms=best_transforms,
        rounds=history, improved=is_better(metric, best_cv.mean, champion_cv.mean),
    )


def _safe_json(text: str) -> dict:
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}
