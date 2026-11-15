"""The candidate space over model family, feature subset and hyperparameters.

``build_search_space`` lists candidates cheap/simple first (FLAML's cost-frugal spirit); the
selection methods (``selection.py``) evaluate them through one ``Evaluator``. ``ExperimentRecord``
is one row of the experiment ledger (``ledger.tsv``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fit import DEFAULT_FAMILIES, HAZARD_FAMILIES, Candidate


@dataclass
class ExperimentRecord:
    """One row of the experiment ledger (``ledger.tsv``)."""

    idx: int
    label: str
    family: str
    n_features: int
    metric: str
    metric_value: float
    cv_std: float
    adjusted: float
    status: str  # keep | discard | crash
    description: str = ""

    def tsv_row(self) -> str:
        return (
            f"{self.idx}\t{self.label}\t{self.family}\t{self.n_features}\t{self.metric}\t"
            f"{self.metric_value:.6f}\t{self.cv_std:.6f}\t{self.adjusted:.6f}\t{self.status}\t"
            f"{self.description}"
        )


def _rank_numeric_features(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not num:
        return []
    corrs = {}
    yv = np.asarray(y, dtype=float)
    for c in num:
        col = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
        if np.nanstd(col) == 0:
            corrs[c] = 0.0
            continue
        corrs[c] = abs(np.corrcoef(np.nan_to_num(col), yv)[0, 1])
    return sorted(num, key=lambda c: corrs.get(c, 0.0), reverse=True)


def _hp_grid(family: str) -> list[dict]:
    if family in ("ridge", "lasso", "elasticnet"):
        alphas = [0.01, 0.1, 1.0] if family != "elasticnet" else [0.01, 0.1]
        if family == "elasticnet":
            return [{"alpha": a, "l1_ratio": r} for a in alphas for r in (0.2, 0.5, 0.8)]
        return [{"alpha": a} for a in alphas]
    if family in ("ridge_logit", "lasso_logit"):
        return [{"C": c} for c in (0.1, 1.0, 10.0)]
    if family in ("poisson", "gamma"):
        return [{"alpha": a} for a in (0.0, 0.1, 1.0)]
    if family == "tweedie":
        return [{"power": p, "alpha": a} for p in (1.2, 1.5, 1.8) for a in (0.1, 1.0)]
    if family == "random_forest":
        return [{"max_depth": d, "n_estimators": 200} for d in (None, 6, 12)]
    if family == "gradient_boosting":
        return [{"max_depth": d, "learning_rate": lr} for d in (2, 3) for lr in (0.05, 0.1)]
    return [{}]  # ols / logit


def build_search_space(
    task: str,
    columns: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    families: list[str] | None = None,
    max_features: int | None = None,
    random_state: int = 42,
    hazard_meta: dict | None = None,
) -> list[Candidate]:
    families = families or DEFAULT_FAMILIES.get(task, DEFAULT_FAMILIES["regression"])
    ranked = _rank_numeric_features(X[columns], y)
    cat = [c for c in columns if c not in ranked]
    all_feats = columns
    top_k = ranked[: min(5, len(ranked))] + cat
    feature_sets = [("top", top_k)] if top_k else []
    if set(all_feats) != set(top_k):
        feature_sets.append(("all", all_feats))
    if max_features:
        feature_sets = [(n, fs[:max_features]) for n, fs in feature_sets]

    candidates: list[Candidate] = []
    for family in families:
        if family in HAZARD_FAMILIES and not hazard_meta:
            continue  # hazard needs event-time metadata (data.event_time_col) to be fittable
        for hp in _hp_grid(family):
            hp = {**hp, "random_state": random_state}
            if family in HAZARD_FAMILIES:
                hp = {**hp, **hazard_meta}
            for fs_name, feats in feature_sets:
                if not feats:
                    continue
                candidates.append(
                    Candidate(family=family, features=list(feats), hyperparams=hp,
                              description=f"{family}/{fs_name}")
                )
    # Cheap/simple first: linear families and smaller feature sets lead.
    candidates.sort(key=lambda c: (0 if c.is_linear else 1, len(c.features)))
    return candidates
