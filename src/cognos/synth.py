"""Synthetic dataset generators for tests and the end-to-end demo.

Deterministic (seeded) so tests and the demo are reproducible. Each generator returns a tidy
pandas DataFrame with a named target column and a known signal so model search has something real
to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_regression_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    region_effect = pd.Series(region).map({"north": 0.5, "south": -0.3, "east": 0.1, "west": 0.0}).to_numpy()
    y = 2.0 * x1 - 1.5 * x2 + 0.5 * x3 + region_effect + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": region, "target": y})


def make_classification_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    logit = 1.2 * x1 - 0.8 * x2 + 0.4 * x3 - 0.2
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def make_timeseries_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(0, 3, n)
    season = np.sin(np.arange(n) * 2 * np.pi / 30)
    lag_feat = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.4, n)
    y = trend + season + 0.7 * lag_feat + noise
    return pd.DataFrame({"date": dates, "trend_idx": trend, "season": season,
                         "lag_feat": lag_feat, "target": y})


def make_credit_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Binary default model with a protected attribute, for fair-lending / compliance demos.

    ``group`` is a protected attribute. The data has a mild correlation between group and the
    outcome so the disparate-impact check has something to detect.
    """
    rng = np.random.default_rng(seed)
    income = rng.gamma(2.0, 30000, n)
    dti = np.clip(rng.normal(0.35, 0.12, n), 0.02, 0.95)  # debt-to-income
    utilization = np.clip(rng.beta(2, 3, n), 0, 1)
    employment_years = np.clip(rng.normal(6, 4, n), 0, 40)
    group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    group_shift = np.where(group == "B", 0.4, 0.0)  # injected disparity
    logit = (
        -2.0
        + 3.0 * dti
        + 2.0 * utilization
        - 0.00001 * income
        - 0.05 * employment_years
        + group_shift
        + rng.normal(0, 0.3, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < p_default).astype(int)
    return pd.DataFrame({
        "income": income,
        "dti": dti,
        "utilization": utilization,
        "employment_years": employment_years,
        "group": group,
        "default": default,
    })


GENERATORS = {
    "regression": make_regression_dataset,
    "classification": make_classification_dataset,
    "timeseries": make_timeseries_dataset,
    "credit": make_credit_dataset,
}
