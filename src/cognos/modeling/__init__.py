"""Modeling primitives: fitters, frozen metrics, ratchet search, ensembling."""

from __future__ import annotations

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
]
