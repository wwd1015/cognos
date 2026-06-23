"""Shared dataset helpers used across stages (single source of truth for feature selection)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CognosConfig


def select_features(df: pd.DataFrame, config: CognosConfig) -> list[str]:
    """Resolve the model feature columns.

    Protected attributes are EXCLUDED from model features by default (disparate-treatment
    avoidance) — they remain available to the compliance stage for disparate-impact testing.
    The datetime column and any drop_columns are also excluded.
    """
    dc = config.data
    exclude = {dc.target, *dc.drop_columns, *dc.protected_attributes}
    if dc.datetime_col:
        exclude.add(dc.datetime_col)
    if dc.features:
        return [c for c in dc.features if c in df.columns and c not in exclude]
    return [c for c in df.columns if c not in exclude]


def coerce_target(df: pd.DataFrame, config: CognosConfig) -> np.ndarray:
    y = df[config.data.target]
    if config.task.is_classification:
        return y.astype(int).to_numpy()
    return y.astype(float).to_numpy()
