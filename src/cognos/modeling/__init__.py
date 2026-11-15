"""Modeling primitives: fitters, frozen metrics, the candidate space, ensembling."""

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
from .metrics import CVResult, cv_score, is_better, metric_direction, score
from .search import ExperimentRecord, build_search_space

__all__ = [
    "Candidate",
    "FittedModel",
    "fit_full",
    "make_fit_predict",
    "make_preprocessor",
    "CVResult",
    "cv_score",
    "is_better",
    "metric_direction",
    "score",
    "ExperimentRecord",
    "build_search_space",
    "EnsembleResult",
    "greedy_ensemble",
    "credit_outcomes",
    "gini",
    "ks_statistic",
    "calibration_table",
    "expected_calibration_error",
    "psi",
]
