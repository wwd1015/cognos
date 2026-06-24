"""Credit-risk model validation metrics (SR 11-7 "outcomes analysis").

The right toolkit for validating a PD/scoring model — discrimination, calibration, and population
stability — as opposed to the trading-strategy metrics (PBO, Deflated Sharpe) that presume a returns
series. Pure functions over (y_true, score) arrays, so they are trivially unit-testable and feed the
backtest stage directly. See ADR-0005.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y_true, scores) -> float:
    """Gini coefficient = 2*AUC - 1 (discrimination). 0 = no skill, 1 = perfect ranking."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(2 * roc_auc_score(y_true, scores) - 1)


def ks_statistic(y_true, scores) -> float:
    """Kolmogorov-Smirnov: max separation between the cumulative score distributions of the classes."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.unique(scores))
    cdf_pos = np.searchsorted(np.sort(pos), grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _quantile_bins(scores, n_bands: int) -> np.ndarray:
    edges = np.quantile(np.asarray(scores, dtype=float), np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def calibration_table(y_true, scores, n_bands: int = 10) -> list[dict]:
    """Per score-band: count, mean predicted probability, observed event rate (expected vs observed)."""
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = _quantile_bins(scores, n_bands)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
    table: list[dict] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        table.append({
            "band": b,
            "n": n,
            "predicted": float(np.mean(scores[mask])),
            "observed": float(np.mean(y_true[mask])),
        })
    return table


def expected_calibration_error(y_true, scores, n_bands: int = 10) -> float:
    """Sample-weighted mean |observed - predicted| across score bands. Lower is better-calibrated."""
    table = calibration_table(y_true, scores, n_bands)
    if not table:
        return float("nan")
    total = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["observed"] - r["predicted"]) for r in table) / total)


def psi(expected_scores, actual_scores, n_bins: int = 10) -> float:
    """Population Stability Index between a baseline (dev) and a new (OOT) score distribution.

    Rule of thumb: <0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift.
    """
    expected = np.asarray(expected_scores, dtype=float)
    actual = np.asarray(actual_scores, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = _quantile_bins(expected, n_bins)
    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    e_pct = np.clip(e_counts / max(1, e_counts.sum()), 1e-6, None)
    a_pct = np.clip(a_counts / max(1, a_counts.sum()), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_label(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def credit_outcomes(y_true, scores, *, dev_scores=None, n_bands: int = 10) -> dict:
    """Aggregate credit-risk outcomes analysis: discrimination + calibration + (optional) stability."""
    table = calibration_table(y_true, scores, n_bands)
    psi_val = psi(dev_scores, scores) if dev_scores is not None else float("nan")
    g = gini(y_true, scores)
    return {
        "gini": g,
        "auc": float((g + 1) / 2) if not np.isnan(g) else float("nan"),
        "ks": ks_statistic(y_true, scores),
        "expected_calibration_error": expected_calibration_error(y_true, scores, n_bands),
        "calibration_table": table,
        "psi": psi_val,
        "psi_label": psi_label(psi_val),
        "n_scored": int(len(np.asarray(scores))),
    }
