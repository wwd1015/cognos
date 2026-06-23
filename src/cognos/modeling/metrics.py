"""Frozen metric registry and leakage-safe cross-validation.

The metric definitions and the holdout split live here, in a module the modeling/idea agents do
*not* edit at runtime — the "frozen substrate" guardrail borrowed from Karpathy's autoresearch
(prepare.py). This is what prevents an autonomous search loop from gaming its own yardstick.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

MAXIMIZE = {"roc_auc", "accuracy", "r2", "f1", "average_precision", "direction_accuracy"}
NEEDS_PROBA = {"roc_auc", "log_loss", "average_precision", "brier"}


def metric_direction(name: str) -> str:
    return "maximize" if name in MAXIMIZE else "minimize"


def is_better(name: str, candidate: float, incumbent: float | None) -> bool:
    """Accept-if-better test used by the ratchet (autoresearch/autoforge KEEP/DISCARD)."""
    if incumbent is None:
        return True
    if metric_direction(name) == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def score(name: str, y_true, y_pred, y_proba=None) -> float:
    """Compute a named metric. ``y_proba`` is the positive-class probability (classification)."""
    y_true = np.asarray(y_true)
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    if name == "accuracy":
        return float(accuracy_score(y_true, np.round(np.asarray(y_pred)).astype(int)))
    if name == "f1":
        return float(f1_score(y_true, np.round(np.asarray(y_pred)).astype(int), zero_division=0))
    if name == "roc_auc":
        proba = y_proba if y_proba is not None else y_pred
        return float(roc_auc_score(y_true, proba))
    if name == "average_precision":
        proba = y_proba if y_proba is not None else y_pred
        return float(average_precision_score(y_true, proba))
    if name == "log_loss":
        proba = y_proba if y_proba is not None else y_pred
        proba = np.clip(np.asarray(proba), 1e-9, 1 - 1e-9)
        return float(log_loss(y_true, proba, labels=[0, 1]))
    if name == "brier":
        proba = y_proba if y_proba is not None else y_pred
        return float(np.mean((np.asarray(proba) - y_true) ** 2))
    if name == "direction_accuracy":
        return float(np.mean(np.sign(np.asarray(y_pred)) == np.sign(y_true)))
    raise ValueError(f"Unknown metric '{name}'")


@dataclass
class CVResult:
    mean: float
    std: float
    oof_pred: np.ndarray  # out-of-fold predictions (probabilities for classification)
    fold_scores: list[float]
    n_folds: int


def _splitter(task_is_classification: bool, is_timeseries: bool, folds: int, random_state: int):
    if is_timeseries:
        return TimeSeriesSplit(n_splits=folds)
    if task_is_classification:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def cv_score(
    fit_predict: Callable[[pd.DataFrame, np.ndarray], tuple],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    metric: str,
    is_classification: bool,
    is_timeseries: bool = False,
    folds: int = 5,
    random_state: int = 42,
) -> CVResult:
    """Leakage-safe CV: a *fresh* model (incl. all preprocessing) is fit per training fold only.

    ``fit_predict(X_train, y_train)`` must return a callable ``predict(X) -> (point, proba)`` where
    ``proba`` is None for regression. All data-dependent preprocessing must happen inside it so that
    nothing from the validation fold leaks in (the #1 silent bug in automated modeling).
    """
    y = np.asarray(y)
    n = len(y)
    folds = max(2, min(folds, n // 2)) if n >= 4 else 2
    splitter = _splitter(is_classification, is_timeseries, folds, random_state)
    oof = np.full(n, np.nan, dtype=float)
    fold_scores: list[float] = []

    for train_idx, val_idx in splitter.split(X, y if is_classification and not is_timeseries else None):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        predict = fit_predict(X_tr, y_tr)
        point, proba = predict(X_val)
        stored = proba if (is_classification and proba is not None) else point
        oof[val_idx] = np.asarray(stored, dtype=float)
        if is_classification:
            fold_scores.append(score(metric, y_val, point, y_proba=proba))
        else:
            fold_scores.append(score(metric, y_val, point))

    # TimeSeriesSplit leaves the first block unscored; fold_scores already excludes those folds.
    return CVResult(
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        oof_pred=oof,
        fold_scores=fold_scores,
        n_folds=len(fold_scores),
    )


def holdout_split(
    df: pd.DataFrame,
    *,
    holdout_fraction: float,
    datetime_col: str | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seal a final holdout. Time-ordered when a datetime column is given (no look-ahead)."""
    if holdout_fraction <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    if datetime_col and datetime_col in df.columns:
        ordered = df.sort_values(datetime_col).reset_index(drop=True)
        cut = int(len(ordered) * (1 - holdout_fraction))
        return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - holdout_fraction))
    return shuffled.iloc[:cut].reset_index(drop=True), shuffled.iloc[cut:].reset_index(drop=True)
