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


def make_commercial_credit_dataset(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Commercial (business) lending default model with a vintage for out-of-time validation.

    Business credit, so there are NO consumer protected attributes. Loans are originated across
    monthly cohorts spanning ~3 years (``vintage``), and later vintages are slightly riskier (a mild
    time trend) so PSI / calibration on an out-of-time holdout is interesting. Default probability is
    a logistic function of the financial features: higher leverage and lower interest coverage push
    PD up.
    """
    rng = np.random.default_rng(seed)

    # ~3 years of monthly origination cohorts; spread loans uniformly across them.
    cohorts = pd.date_range("2021-01-01", periods=36, freq="MS")
    cohort_idx = rng.integers(0, len(cohorts), n)
    vintage = cohorts[cohort_idx]
    # 0 -> oldest cohort, 1 -> newest cohort; drives the mild time trend in risk.
    time_pos = cohort_idx / (len(cohorts) - 1)

    leverage = rng.gamma(2.0, 1.5, n)  # debt / EBITDA, positive
    interest_coverage = rng.gamma(3.0, 2.0, n) + 0.5  # EBITDA / interest, positive
    current_ratio = np.clip(rng.normal(1.6, 0.5, n), 0.2, 5.0)
    log_assets = rng.normal(16.0, 1.5, n)  # log of total assets
    profit_margin = np.clip(rng.normal(0.08, 0.06, n), -0.30, 0.40)
    sector = rng.choice(["industrials", "tech", "retail", "energy"], size=n)
    sector_effect = pd.Series(sector).map(
        {"industrials": 0.0, "tech": -0.3, "retail": 0.2, "energy": 0.4}
    ).to_numpy()

    logit = (
        -2.3
        + 0.45 * leverage
        - 0.30 * interest_coverage
        - 0.50 * current_ratio
        - 0.20 * (log_assets - 16.0)
        - 4.0 * profit_margin
        + sector_effect
        + 0.8 * time_pos  # later vintages slightly riskier
        + rng.normal(0, 0.3, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < p_default).astype(int)
    return pd.DataFrame({
        "vintage": vintage,
        "leverage": leverage,
        "interest_coverage": interest_coverage,
        "current_ratio": current_ratio,
        "log_assets": log_assets,
        "profit_margin": profit_margin,
        "sector": sector,
        "default": default,
    })


def make_cni_portfolio_dataset(n: int = 2000, seed: int = 7,
                               include_leak: bool = True) -> pd.DataFrame:
    """A realistic C&I (commercial & industrial) loan portfolio for the commercial-risk demo.

    Obligor-level origination sample with quarterly vintages (2019Q1–2023Q4, so the 2020 stress
    period is inside the sample), obligor financial ratios, facility characteristics, sector/region,
    and the macro environment at origination. The target ``default`` is a 12-month default flag
    driven by a logistic model with textbook signs: leverage up-risk; coverage, liquidity, margin,
    and size down-risk; a 2020 vintage stress bump.

    ``obligor_id`` is an identifier (drop it from features). When ``include_leak`` is True the frame
    also carries ``dpd_at_outcome`` — days past due observed at the END of the outcome window. It is
    an *outcome*, not a predictor available at origination, and is near-deterministically related to
    ``default``: the classic post-outcome leakage field explore must flag and the config must drop.
    """
    rng = np.random.default_rng(seed)

    # 20 quarterly origination cohorts spanning the 2020 stress period.
    cohorts = pd.period_range("2019Q1", "2023Q4", freq="Q").to_timestamp()
    cohort_idx = rng.integers(0, len(cohorts), n)
    vintage = cohorts[cohort_idx]
    year = vintage.year
    # Macro at origination: unemployment with a 2020 spike, GDP growth with a 2020 dip.
    unemployment_rate = np.where(year == 2020, rng.normal(9.0, 1.0, n),
                                 rng.normal(4.2, 0.6, n)).clip(2.5, 15.0)
    gdp_growth = np.where(year == 2020, rng.normal(-2.5, 1.0, n),
                          rng.normal(2.3, 0.8, n)).clip(-8.0, 6.0)

    # Obligor financials (ratios in plausible middle-market ranges).
    debt_to_ebitda = np.clip(rng.gamma(2.2, 1.6, n), 0.2, 12.0)
    interest_coverage = np.clip(rng.gamma(2.8, 1.8, n) + 0.3, 0.3, 25.0)
    current_ratio = np.clip(rng.normal(1.7, 0.6, n), 0.2, 5.0)
    operating_margin = np.clip(rng.normal(0.09, 0.07, n), -0.35, 0.45)
    revenue_growth = np.clip(rng.normal(0.05, 0.12, n), -0.60, 0.80)
    log_total_assets = rng.normal(17.0, 1.4, n)  # ~ $10M–$1B obligors

    # Facility characteristics.
    utilization_rate = np.clip(rng.beta(2.0, 2.5, n), 0.0, 1.0)  # revolver drawn/committed
    collateral_coverage = np.clip(rng.gamma(3.0, 0.5, n), 0.0, 6.0)  # collateral value / exposure

    sector = rng.choice(["manufacturing", "services", "retail_trade", "energy", "healthcare"],
                        size=n, p=[0.28, 0.27, 0.20, 0.10, 0.15])
    sector_effect = pd.Series(sector).map({
        "manufacturing": 0.0, "services": -0.10, "retail_trade": 0.35,
        "energy": 0.45, "healthcare": -0.20,
    }).to_numpy()
    region = rng.choice(["northeast", "southeast", "midwest", "west"], size=n)

    logit = (
        -3.4
        + 0.32 * debt_to_ebitda
        - 0.16 * interest_coverage
        - 0.45 * current_ratio
        - 3.0 * operating_margin
        - 0.8 * revenue_growth
        - 0.18 * (log_total_assets - 17.0)
        + 1.2 * utilization_rate
        - 0.25 * collateral_coverage
        + 0.10 * (unemployment_rate - 4.2)
        - 0.06 * gdp_growth
        + sector_effect
        + np.where(year == 2020, 0.5, 0.0)  # vintage stress bump
        + rng.normal(0, 0.35, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < p_default).astype(int)

    df = pd.DataFrame({
        "obligor_id": [f"OBL{100000 + i}" for i in range(n)],
        "vintage": vintage,
        "debt_to_ebitda": debt_to_ebitda,
        "interest_coverage": interest_coverage,
        "current_ratio": current_ratio,
        "operating_margin": operating_margin,
        "revenue_growth": revenue_growth,
        "log_total_assets": log_total_assets,
        "utilization_rate": utilization_rate,
        "collateral_coverage": collateral_coverage,
        "unemployment_rate": unemployment_rate,
        "gdp_growth": gdp_growth,
        "sector": sector,
        "region": region,
        "default": default,
    })
    if include_leak:
        # Observed AFTER origination — belongs to the outcome window, not the information set.
        df["dpd_at_outcome"] = np.where(default == 1, rng.integers(120, 181, n),
                                        rng.integers(0, 6, n))
    return df


GENERATORS = {
    "regression": make_regression_dataset,
    "classification": make_classification_dataset,
    "timeseries": make_timeseries_dataset,
    "credit": make_credit_dataset,
    "commercial": make_commercial_credit_dataset,
    "cni": make_cni_portfolio_dataset,
}
