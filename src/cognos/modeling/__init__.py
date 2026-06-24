"""Modeling primitives: fitters, frozen metrics, ratchet search, ensembling."""

from __future__ import annotations

from .credit_metrics import (
    calibration_table,
    credit_outcomes,
    expected_calibration_error,
    gini,
    ks_statistic,
    psi,
)
from .ensemble import EnsembleResult, greedy_ensemble
from .fit import Candidate, FittedModel, fit_full, make_fit_predict, make_preprocessor
from .metrics import CVResult, cv_score, holdout_split, is_better, metric_direction, score
from .search import ExperimentRecord, SearchResult, build_search_space, ratchet_search

__all__ = [
    "Candidate",
    "FittedModel",
    "fit_full",
    "make_fit_predict",
    "make_preprocessor",
    "CVResult",
    "cv_score",
    "holdout_split",
    "is_better",
    "metric_direction",
    "score",
    "ExperimentRecord",
    "SearchResult",
    "build_search_space",
    "ratchet_search",
    "EnsembleResult",
    "greedy_ensemble",
    "credit_outcomes",
    "gini",
    "ks_statistic",
    "calibration_table",
    "expected_calibration_error",
    "psi",
]
