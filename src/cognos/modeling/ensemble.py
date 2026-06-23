"""Caruana greedy ensemble selection over the search survivors' out-of-fold predictions.

Reuses the OOF predictions already computed during the ratchet search, so building the ensemble is
near-free. Starting from an empty ensemble, greedily add (with replacement) the candidate whose
inclusion most improves the OOF metric; repeating a candidate increases its weight (Caruana et al.
2004, as used by auto-sklearn/AutoGluon).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .metrics import is_better, score


@dataclass
class EnsembleResult:
    member_indices: list[int]  # indices into the candidate list (with repeats => weights)
    weights: dict[int, float]
    metric: str
    ensemble_score: float
    best_single_score: float

    @property
    def improved(self) -> bool:
        return is_better(self.metric, self.ensemble_score, self.best_single_score)


def greedy_ensemble(
    oof_preds: list[np.ndarray],
    y: np.ndarray,
    *,
    metric: str,
    is_classification: bool,
    n_rounds: int = 25,
) -> EnsembleResult | None:
    """``oof_preds[i]`` is candidate i's OOF prediction (proba for classification)."""
    if len(oof_preds) < 2:
        return None
    y = np.asarray(y, dtype=float)
    # Restrict to positions every candidate predicted (TimeSeriesSplit leaves early rows as NaN).
    valid = np.ones(len(y), dtype=bool)
    for p in oof_preds:
        valid &= ~np.isnan(p)
    if valid.sum() < 4:
        return None
    yv = y[valid]
    preds = [p[valid] for p in oof_preds]

    def _score(vec: np.ndarray) -> float:
        if is_classification:
            point = (vec >= 0.5).astype(int)
            return score(metric, yv, point, y_proba=vec)
        return score(metric, yv, vec)

    singles = [_score(p) for p in preds]
    best_single = max(singles) if metric_is_max(metric) else min(singles)

    chosen: list[int] = []
    current = np.zeros_like(yv)
    best_score: float | None = None
    for _ in range(n_rounds):
        round_best_idx, round_best_score, round_best_vec = None, None, None
        for i, p in enumerate(preds):
            k = len(chosen)
            blended = (current * k + p) / (k + 1)
            s = _score(blended)
            if round_best_score is None or is_better(metric, s, round_best_score):
                round_best_idx, round_best_score, round_best_vec = i, s, blended
        if round_best_idx is None:
            break
        # Stop if adding the best candidate no longer improves the ensemble.
        if best_score is not None and not is_better(metric, round_best_score, best_score):
            break
        chosen.append(round_best_idx)
        current = round_best_vec
        best_score = round_best_score

    if not chosen:
        return None
    counts = Counter(chosen)
    total = sum(counts.values())
    weights = {i: c / total for i, c in counts.items()}
    return EnsembleResult(
        member_indices=chosen, weights=weights, metric=metric,
        ensemble_score=float(best_score), best_single_score=float(best_single),
    )


def metric_is_max(metric: str) -> bool:
    from .metrics import metric_direction

    return metric_direction(metric) == "maximize"
