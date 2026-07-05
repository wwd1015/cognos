"""Merton (1974) structural credit model — distance-to-default as a deterministic engine feature.

Equity is a call option on firm assets: given observed equity value E, equity volatility σ_E, and
the face value of debt F, the engine solves the Black–Scholes–Merton system

    E = V·N(d1) − F·e^{−rT}·N(d2)
    σ_E = (V/E)·N(d1)·σ_V

for the unobserved asset value V and asset volatility σ_V (the classic KMV fixed-point iteration),
then computes the **distance to default** DD = (ln(V/F) + (r − σ_V²/2)T) / (σ_V√T) and the
structural PD = N(−DD). Everything is a deterministic solver — no fitting, no randomness — so it
belongs to the engine ("the engine disposes") and recomputes identically at serve time.

Two uses in COGNOS:
- **Hybrid (default)**: DD becomes an engineered feature feeding the reduced-form champion — the
  RiskCalc-style hybrid that industry uses when market observables exist.
- **Pure-structural benchmark**: PD = N(−DD) is scored on the sealed holdout as a labelled
  challenger benchmark, never the deployed model.

Requires traded-market observables; for private middle-market obligors the framework stays
"rejected — no market observables" in the ideate design brief.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

MAX_ITER = 200
TOL = 1e-8


@dataclass
class StructuralSpec:
    """Column mapping + market parameters for the Merton solver (fully serializable)."""

    equity_value_col: str
    equity_vol_col: str
    debt_col: str
    risk_free_rate: float = 0.03
    horizon_years: float = 1.0
    dd_feature: str = "merton_dd"
    pd_feature: str = "merton_pd"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StructuralSpec:
        return cls(**d)


def solve_merton(equity_value, equity_vol, debt_face, *, risk_free_rate: float = 0.03,
                 horizon_years: float = 1.0) -> dict[str, np.ndarray]:
    """Vectorized KMV fixed-point solve of the two-equation Merton system.

    Returns asset value V, asset vol σ_V, distance-to-default DD, and PD = N(−DD). Rows with
    non-positive inputs (or failed convergence) come back NaN — the caller decides the fill policy.
    """
    E = np.asarray(equity_value, dtype=float)
    sE = np.asarray(equity_vol, dtype=float)
    F = np.asarray(debt_face, dtype=float)
    r, T = float(risk_free_rate), float(horizon_years)
    sqrtT = np.sqrt(T)

    ok = np.isfinite(E) & np.isfinite(sE) & np.isfinite(F) & (E > 0) & (sE > 0) & (F > 0)
    V = np.where(ok, E + F, np.nan)
    sV = np.where(ok, sE * E / np.maximum(E + F, 1e-12), np.nan)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(MAX_ITER):
            d1 = (np.log(V / F) + (r + 0.5 * sV**2) * T) / (sV * sqrtT)
            d2 = d1 - sV * sqrtT
            n1 = norm.cdf(d1)
            # Fixed-point updates from the two BSM equations.
            V_new = (E + F * np.exp(-r * T) * norm.cdf(d2)) / np.clip(n1, 1e-12, None)
            sV_new = sE * E / np.clip(V_new * n1, 1e-12, None)
            V_new = np.where(ok, V_new, np.nan)
            sV_new = np.where(ok, np.clip(sV_new, 1e-6, 10.0), np.nan)
            delta = np.nanmax(np.abs(V_new - V) / np.clip(V, 1e-12, None)) if ok.any() else 0.0
            V, sV = V_new, sV_new
            if not np.isfinite(delta) or delta < TOL:
                break

        dd = (np.log(V / F) + (r - 0.5 * sV**2) * T) / (sV * sqrtT)
    pd_ = norm.cdf(-dd)
    bad = ~np.isfinite(dd)
    dd = np.where(bad, np.nan, dd)
    pd_ = np.where(bad, np.nan, pd_)
    return {"asset_value": V, "asset_vol": sV, "dd": dd, "pd": pd_}


def augment_frame(df: pd.DataFrame, spec: StructuralSpec) -> tuple[pd.DataFrame, dict]:
    """Add the DD / structural-PD columns to a frame (target-hidden: market observables only).

    Failed rows are filled with the column median so downstream pipelines never see NaN; the count
    of failures is reported for the run record.
    """
    out = df.copy()
    sol = solve_merton(
        df[spec.equity_value_col], df[spec.equity_vol_col], df[spec.debt_col],
        risk_free_rate=spec.risk_free_rate, horizon_years=spec.horizon_years,
    )
    dd, pd_ = sol["dd"], sol["pd"]
    n_failed = int(np.sum(~np.isfinite(dd)))
    if n_failed and np.isfinite(dd).any():
        dd = np.where(np.isfinite(dd), dd, np.nanmedian(dd))
        pd_ = np.where(np.isfinite(pd_), pd_, np.nanmedian(pd_))
    out[spec.dd_feature] = np.nan_to_num(dd, nan=0.0)
    out[spec.pd_feature] = np.nan_to_num(pd_, nan=0.5)
    info = {"n_rows": int(len(df)), "n_failed": n_failed,
            "mean_dd": float(np.nanmean(dd)) if len(df) else None,
            "mean_structural_pd": float(np.nanmean(pd_)) if len(df) else None}
    return out, info
