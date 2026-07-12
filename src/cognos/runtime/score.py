"""Deployment-time scoring functions.

A trained model is persisted as a lightweight :class:`ScorerBundle` (sklearn pipeline + metadata,
*without* the statsmodels object so it always pickles cleanly). The IMPACT adapter references
``cognos.runtime.score.score_row`` as a derived-field function so the model can be embedded directly
in an IMPACT EntityConfig; the same bundle also powers the built-in fallback scorer.

This module is also the canonical docs<->code link target: the documentation stage points white-paper
paragraphs at ``{@code:src/cognos/runtime/score.py#score_row}`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ScorerBundle:
    pipeline: Any  # fitted sklearn Pipeline
    raw_features: list[str]  # columns the pipeline consumes (base + any engineered)
    is_classification: bool
    model_id: str = "model"
    transforms: list[dict] = field(default_factory=list)  # [{name, expr}] LLM-authored features
    base_features: list[str] | None = None  # original columns needed to recompute transforms
    structural: dict | None = None  # StructuralSpec dict; Merton DD recomputed at serve time
    migration: dict | None = None  # fitted migration model dict; migration_pd recomputed at serve

    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute engineered features (target-hidden) so serving matches training exactly."""
        frame = df
        if self.structural:
            from ..modeling.structural import StructuralSpec, augment_frame

            spec = StructuralSpec.from_dict(self.structural)
            market = (spec.equity_value_col, spec.equity_vol_col, spec.debt_col)
            if spec.dd_feature not in frame.columns and all(c in frame.columns for c in market):
                frame, _ = augment_frame(frame, spec)
        if self.migration:
            from ..modeling.migration import MigrationSpec
            from ..modeling.migration import augment_frame as migration_augment

            mspec = MigrationSpec.from_dict(self.migration["spec"])
            if mspec.pd_feature not in frame.columns and mspec.rating_col in frame.columns:
                frame, _ = migration_augment(frame, self.migration)
        if not self.transforms:
            return frame
        from ..modeling.transforms import TransformSpec, apply_transforms

        base = self.base_features or self.raw_features
        cols = [c for c in dict.fromkeys([*base, *self.raw_features]) if c in frame.columns]
        aug, _, _ = apply_transforms(frame[cols], [TransformSpec(**t) for t in self.transforms])
        return aug

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = self._augment(df)[self.raw_features]
        if self.is_classification:
            return self.pipeline.predict_proba(X)[:, 1]
        return self.pipeline.predict(X)


def save_scorer(path: str, fitted) -> str:
    """Persist a FittedModel as a picklable ScorerBundle. Returns the path."""
    transforms = [t.to_dict() if hasattr(t, "to_dict") else dict(t) for t in getattr(fitted, "transforms", [])]
    bundle = ScorerBundle(
        pipeline=fitted.pipeline,
        raw_features=list(fitted.raw_features),
        is_classification=bool(fitted.is_classification),
        model_id=fitted.model_id,
        transforms=transforms,
        base_features=list(fitted.base_features) if getattr(fitted, "base_features", None) else None,
        structural=getattr(fitted, "structural", None),
        migration=getattr(fitted, "migration", None),
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
    cols = bundle.base_features or bundle.raw_features  # the row carries base features only
    frame = pd.DataFrame([row[cols].to_dict()])
    return float(bundle.score_frame(frame)[0])
