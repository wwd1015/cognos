"""Budget-aware ratchet search over the CASH space.

Implements the accept-if-better-else-reject hill-climb from autoresearch/autoforge as code: each
candidate (model family + feature subset + hyperparameters) is evaluated with leakage-safe CV; if
its parsimony-adjusted score beats the incumbent champion it is kept, otherwise it is discarded.
Every attempt is logged to an experiment ledger (the ``results.tsv`` analogue). Candidates are
ordered cheap/simple first (FLAML's cost-frugal spirit), and the search respects an explicit
candidate budget and optional wall-clock budget.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .fit import DEFAULT_FAMILIES, Candidate, make_fit_predict
from .metrics import CVResult, cv_score, is_better, metric_direction


@dataclass
class ExperimentRecord:
    """One row of the experiment ledger (autoforge results.tsv analogue)."""

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


@dataclass
class SearchResult:
    champion: Candidate
    champion_cv: CVResult
    metric: str
    ledger: list[ExperimentRecord] = field(default_factory=list)
    evaluated: list[tuple] = field(default_factory=list)  # [(Candidate, CVResult)] survivors for ensemble
    n_tried: int = 0

    def ledger_tsv(self) -> str:
        header = "idx\tlabel\tfamily\tn_features\tmetric\tmetric_value\tcv_std\tadjusted\tstatus\tdescription"
        return "\n".join([header, *(r.tsv_row() for r in self.ledger)]) + "\n"

    def ledger_records(self) -> list[dict]:
        return [asdict(r) for r in self.ledger]


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
) -> list[Candidate]:
    families = families or DEFAULT_FAMILIES.get(task, DEFAULT_FAMILIES["regression"])
    ranked = _rank_numeric_features(X, y)
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
        for hp in _hp_grid(family):
            hp = {**hp, "random_state": random_state}
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


def ratchet_search(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    metric: str,
    is_classification: bool,
    is_timeseries: bool = False,
    families: list[str] | None = None,
    max_candidates: int = 24,
    folds: int = 5,
    random_state: int = 42,
    complexity_penalty: float = 0.0,
    time_budget_s: float | None = None,
    max_features: int | None = None,
) -> SearchResult:
    columns = list(X.columns)
    space = build_search_space(task, columns, X, y, families=families,
                              max_features=max_features, random_state=random_state)[:max_candidates]

    direction = metric_direction(metric)
    n_total = max(1, len(columns))
    champion: Candidate | None = None
    champion_cv: CVResult | None = None
    champion_adj: float | None = None
    ledger: list[ExperimentRecord] = []
    evaluated: list[tuple] = []
    start = time.monotonic()

    for i, cand in enumerate(space):
        if time_budget_s is not None and (time.monotonic() - start) > time_budget_s:
            break
        try:
            cv = cv_score(
                make_fit_predict(cand, is_classification=is_classification),
                X, y, metric=metric, is_classification=is_classification,
                is_timeseries=is_timeseries, folds=folds, random_state=random_state,
            )
            penalty = complexity_penalty * (len(cand.features) / n_total)
            adjusted = cv.mean - penalty if direction == "maximize" else cv.mean + penalty
            keep = is_better(metric, adjusted, champion_adj)
            status = "keep" if keep else "discard"
            if keep:
                champion, champion_cv, champion_adj = cand, cv, adjusted
            evaluated.append((cand, cv))
            ledger.append(ExperimentRecord(
                idx=i, label=cand.label(), family=cand.family, n_features=len(cand.features),
                metric=metric, metric_value=cv.mean, cv_std=cv.std, adjusted=adjusted,
                status=status, description=cand.description,
            ))
        except Exception as exc:  # crash => log 0 and continue (autoforge contract)
            ledger.append(ExperimentRecord(
                idx=i, label=cand.label(), family=cand.family, n_features=len(cand.features),
                metric=metric, metric_value=0.0, cv_std=0.0, adjusted=0.0, status="crash",
                description=f"{cand.description}: {type(exc).__name__}: {exc}",
            ))

    if champion is None or champion_cv is None:
        raise RuntimeError("Search produced no viable candidate (all crashed). Check data/config.")

    return SearchResult(champion=champion, champion_cv=champion_cv, metric=metric,
                        ledger=ledger, evaluated=evaluated, n_tried=len(ledger))
