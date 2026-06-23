"""Deployment-time scoring functions.

A trained model is persisted as a lightweight :class:`ScorerBundle` (sklearn pipeline + metadata,
*without* the statsmodels object so it always pickles cleanly). The IMPACT adapter references
``cognos.runtime.score.score_row`` as a derived-field function so the model can be embedded directly
in an IMPACT EntityConfig; the same bundle also powers the built-in fallback scorer.

This module is also the canonical docs<->code link target: the documentation stage points white-paper
paragraphs at ``{@code:src/cognos/runtime/score.py#score_row}`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ScorerBundle:
    pipeline: Any  # fitted sklearn Pipeline
    raw_features: list[str]
    is_classification: bool
    model_id: str = "model"

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.raw_features]
        if self.is_classification:
            return self.pipeline.predict_proba(X)[:, 1]
        return self.pipeline.predict(X)


def save_scorer(path: str, fitted) -> str:
    """Persist a FittedModel as a picklable ScorerBundle. Returns the path."""
    bundle = ScorerBundle(
        pipeline=fitted.pipeline,
        raw_features=list(fitted.raw_features),
        is_classification=bool(fitted.is_classification),
        model_id=fitted.model_id,
    )
    joblib.dump(bundle, path)
    return path


@lru_cache(maxsize=16)
def load_scorer(model_path: str) -> ScorerBundle:
    return joblib.load(model_path)


def score_frame(df: pd.DataFrame, model_path: str) -> np.ndarray:
    """Vectorized scoring of a whole DataFrame (used by the built-in fallback path)."""
    return load_scorer(model_path).score_frame(df)


def score_row(row: pd.Series, model_path: str) -> float:
    """Row-wise scoring — the IMPACT derived-field contract (pd.Series -> scalar).

    IMPACT embeds the model via ``{function: 'cognos.runtime.score.score_row',
    kwargs: {model_path: '<abs path to .joblib>'}}``.
    """
    bundle = load_scorer(model_path)
    frame = pd.DataFrame([row[bundle.raw_features].to_dict()])
    return float(bundle.score_frame(frame)[0])
