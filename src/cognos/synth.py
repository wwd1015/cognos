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
                               include_leak: bool = True,
                               include_market: bool = False) -> pd.DataFrame:
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
    if include_market:
        # Traded-market observables for public obligors — the inputs a Merton structural model
        # (modeling/structural.py) needs: equity value, equity volatility, and debt face value,
        # made consistent with the accounting picture so distance-to-default carries real signal.
        total_assets = np.exp(log_total_assets)
        debt_face = np.clip(0.15 + 0.055 * debt_to_ebitda, 0.05, 0.85) * total_assets
        df["equity_value"] = np.maximum(total_assets - debt_face, 0.02 * total_assets)
        df["equity_vol"] = np.clip(0.20 + 0.045 * debt_to_ebitda - 0.5 * operating_margin
                                   + rng.normal(0, 0.05, n), 0.08, 1.2)
        df["debt_face"] = debt_face
    if include_leak:
        # Observed AFTER origination — belongs to the outcome window, not the information set.
        df["dpd_at_outcome"] = np.where(default == 1, rng.integers(120, 181, n),
                                        rng.integers(0, 6, n))
    # Event timing for survival analysis: the quarter (1-4) of the 12-month window in which the
    # default occurred; NaN for censored (non-default) obligors. This is the label's timing —
    # set data.event_time_col so the engine excludes it from features and the hazard families
    # (modeling/hazard.py) can consume it. Riskier quarters skew early (hazard falls with age).
    quarter = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.28, 0.21, 0.16])
    df["default_quarter"] = np.where(default == 1, quarter.astype(float), np.nan)
    return df


# One-year corporate rating-transition rates (%), rows = from-rating, columns =
# AAA AA A BBB BB B CCC D NR. Calibrated to the shape of the published S&P Global long-run
# (1981-2023) averages — sticky diagonals, one-notch-dominant migration, near-zero
# investment-grade default rates, a ~26% CCC default rate, and an NR (withdrawn) share that
# grows as ratings worsen. Synthetic stand-in for licensed CreditPro-style data; rows are
# renormalized to sum to exactly 1 at load.
_AGENCY_TRANSITIONS = {
    "AAA": [87.0, 9.0, 0.5, 0.05, 0.06, 0.03, 0.05, 0.00, 3.31],
    "AA":  [0.5, 87.2, 7.6, 0.5, 0.05, 0.06, 0.02, 0.02, 4.05],
    "A":   [0.03, 1.6, 88.4, 4.9, 0.3, 0.10, 0.02, 0.05, 4.60],
    "BBB": [0.00, 0.09, 3.3, 86.1, 3.3, 0.4, 0.10, 0.15, 6.56],
    "BB":  [0.01, 0.03, 0.1, 4.6, 77.7, 6.3, 0.5, 0.60, 10.16],
    "B":   [0.00, 0.02, 0.06, 0.15, 4.4, 74.1, 4.4, 3.20, 13.67],
    "CCC": [0.00, 0.00, 0.10, 0.20, 0.6, 13.4, 43.1, 26.60, 16.00],
}
_AGENCY_SCALE = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
_RECESSION_YEARS = {1982, 1990, 1991, 2001, 2002, 2008, 2009, 2020}


def make_rating_migration_dataset(n_obligors: int = 450, start_year: int = 1981,
                                  end_year: int = 2023, seed: int = 42,
                                  book: str = "agency") -> pd.DataFrame:
    """S&P-style corporate obligor-year rating panel for migration / loss-forecast modeling.

    One row per obligor-year: the rating at the observation point (``rating``), obligor
    fundamentals consistent with that rating (Compustat-style, so a hybrid model has signal beyond
    the letter grade), the macro environment, exposure (``ead``), and the outcome — the rating one
    year later (``next_rating``, which may be the default state ``D`` or a withdrawal ``NR``) and
    the ``default`` flag. Transitions are drawn from a long-run average matrix shaped like the
    published S&P Global corporate averages, tilted by the macro regime (recession years default
    and downgrade more, upgrade less) and by obligor fundamentals (a levered, low-coverage obligor
    is more likely to migrate down *within* its grade — this is the signal a hybrid champion adds
    over the pure matrix). Defaulted and withdrawn obligors exit; fresh entrants keep the panel
    balanced. Deterministic given the seed.

    ``book="agency"`` mimics the rated public universe (fuller high-grade mix; use a long window
    like 1981-2023). ``book="bank"`` mimics an internal commercial book (speculative-grade-heavy;
    pair with a short window to reproduce the "internal history is too short" problem).
    """
    rng = np.random.default_rng(seed)
    scale = _AGENCY_SCALE
    base = np.array([_AGENCY_TRANSITIONS[s] for s in scale], dtype=float)
    base = base / base.sum(axis=1, keepdims=True)
    n_states = base.shape[1]  # 7 live + D + NR
    entry_mix = (np.array([0.03, 0.09, 0.18, 0.27, 0.22, 0.16, 0.05]) if book == "agency"
                 else np.array([0.005, 0.03, 0.11, 0.26, 0.30, 0.235, 0.06]))
    entry_mix = entry_mix / entry_mix.sum()
    sectors = np.array(["manufacturing", "services", "retail_trade", "energy",
                        "healthcare", "utilities"])

    years = list(range(int(start_year), int(end_year) + 1))
    # Macro series: one unemployment print per year, elevated in recession years.
    unemployment = {y: float(np.clip(rng.normal(9.0 if y in _RECESSION_YEARS else 5.0, 0.7),
                                     3.0, 14.0)) for y in years}

    next_id = 0

    def _new_obligors(k: int) -> dict:
        nonlocal next_id
        ids = [f"OBL{100000 + next_id + i}" for i in range(k)]
        next_id += k
        return {
            "id": ids,
            "rating_idx": rng.choice(len(scale), size=k, p=entry_mix),
            "quality": rng.normal(0, 1, k),  # persistent within-grade credit quality
            "sector": rng.choice(sectors, size=k),
            "size": rng.normal(0, 1, k),  # persistent size factor
        }

    pool = _new_obligors(n_obligors)
    rows: list[dict] = []
    for year in years:
        k = len(pool["id"])
        q = pool["rating_idx"].astype(float)  # 0=AAA ... 6=CCC
        u = pool["quality"]
        eps = rng.normal(0, 1, k)
        # Fundamentals anchored to the grade, blurred by persistent quality + noise.
        debt_to_ebitda = np.clip(1.0 + 0.75 * q + 0.55 * u + rng.normal(0, 0.5, k), 0.1, 12.0)
        interest_coverage = np.clip(13.0 - 1.5 * q - 1.3 * u + rng.normal(0, 1.3, k), 0.2, 30.0)
        operating_margin = np.clip(0.17 - 0.015 * q - 0.020 * u + rng.normal(0, 0.04, k),
                                   -0.30, 0.45)
        log_total_assets = 21.0 - 0.5 * q + 0.9 * pool["size"] + rng.normal(0, 0.4, k)
        ead = np.exp(rng.normal(17.5 - 0.25 * q + 0.7 * pool["size"], 0.6))

        # Idiosyncratic migration tilt: worse-than-grade fundamentals push transitions down.
        z = np.clip(0.75 * u + 0.55 * eps, -3.0, 3.0)
        recession = year in _RECESSION_YEARS

        next_idx = np.empty(k, dtype=int)
        for i_rating in range(len(scale)):
            mask = pool["rating_idx"] == i_rating
            m = int(mask.sum())
            if not m:
                continue
            probs = np.tile(base[i_rating], (m, 1))
            zi = z[mask]
            down = np.arange(n_states - 2) > i_rating  # worse live grades
            up = np.arange(n_states - 2) < i_rating
            mult = np.ones((m, n_states))
            mult[:, :-2][:, down] *= np.exp(0.30 * zi)[:, None]
            mult[:, :-2][:, up] *= np.exp(-0.30 * zi)[:, None]
            mult[:, -2] *= np.exp(0.85 * zi)  # default column
            if recession:
                mult[:, :-2][:, down] *= 1.7
                mult[:, :-2][:, up] *= 0.55
                mult[:, -2] *= 2.3
            probs *= mult
            probs /= probs.sum(axis=1, keepdims=True)
            cum = probs.cumsum(axis=1)
            draw = rng.uniform(size=m)[:, None]
            next_idx[mask] = (draw > cum).sum(axis=1)

        next_state = np.array([*scale, "D", "NR"], dtype=object)[next_idx]
        rows.extend({
            "obligor_id": pool["id"][i],
            "asof": pd.Timestamp(year, 1, 1),
            "rating": scale[pool["rating_idx"][i]],
            "debt_to_ebitda": debt_to_ebitda[i],
            "interest_coverage": interest_coverage[i],
            "operating_margin": operating_margin[i],
            "log_total_assets": log_total_assets[i],
            "sector": pool["sector"][i],
            "unemployment_rate": unemployment[year],
            "regime": "recession" if recession else "expansion",
            "ead": ead[i],
            "next_rating": next_state[i],
            "default": int(next_state[i] == "D"),
        } for i in range(k))

        # Survivors carry their new grade into the next year; exits are replaced by entrants.
        survive = next_idx < len(scale)
        pool = {
            "id": [pool["id"][i] for i in range(k) if survive[i]],
            "rating_idx": next_idx[survive],
            "quality": np.clip(pool["quality"][survive] + rng.normal(0, 0.25, int(survive.sum())),
                               -3.0, 3.0),
            "sector": pool["sector"][survive],
            "size": pool["size"][survive],
        }
        n_exit = k - int(survive.sum())
        if n_exit and year != years[-1]:
            fresh = _new_obligors(n_exit)
            pool = {key: (np.concatenate([pool[key], fresh[key]])
                          if isinstance(pool[key], np.ndarray)
                          else [*pool[key], *fresh[key]])
                    for key in pool}

    return pd.DataFrame(rows)


GENERATORS = {
    "regression": make_regression_dataset,
    "classification": make_classification_dataset,
    "timeseries": make_timeseries_dataset,
    "credit": make_credit_dataset,
    "commercial": make_commercial_credit_dataset,
    "cni": make_cni_portfolio_dataset,
    "migration": make_rating_migration_dataset,
}
