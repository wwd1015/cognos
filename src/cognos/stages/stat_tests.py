"""Statistical diagnostic battery (the "statistical testing" half of the modeling stage).

Each test has a known H0 and interpretation, so results are machine-actionable. Residual-based tests
(heteroskedasticity, autocorrelation, normality, specification) require a linear model with a
statsmodels fit; multicollinearity and stationarity tests run more broadly. Every test is wrapped so
an inapplicable or failing-to-converge test is recorded as ``skipped`` rather than crashing the run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ALPHA = 0.05


@dataclass
class TestResult:
    name: str
    category: str
    statistic: float | None
    pvalue: float | None
    passed: bool | None  # None => skipped/not-applicable
    severity: str  # INFO|LOW|MEDIUM|HIGH
    interpretation: str
    h0: str = ""

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return ("__error__", f"{type(exc).__name__}: {exc}")


def run_battery(
    fitted,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    is_classification: bool,
    is_timeseries: bool = False,
) -> dict[str, Any]:
    """Run the diagnostic battery against a fitted model. Returns a structured report."""
    results: list[TestResult] = []
    resid = getattr(fitted, "residuals", None)
    design = getattr(fitted, "design_matrix", None)
    sm_res = getattr(fitted, "sm_result", None)
    has_linear_resid = resid is not None and design is not None and not is_classification

    if has_linear_resid:
        results.extend(_residual_tests(resid, design, sm_res))
    if design is not None:
        results.extend(_collinearity_tests(design))
    if is_timeseries or (np.asarray(y).dtype.kind == "f" and not is_classification):
        results.extend(_stationarity_tests(np.asarray(y, dtype=float)))

    run = [r for r in results if r.passed is not None]
    failed = [r for r in run if r.passed is False]
    severities = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    max_sev = max((severities[r.severity] for r in failed), default=0)
    inv = {v: k for k, v in severities.items()}
    return {
        "tests": [r.dict() for r in results],
        "n_run": len(run),
        "n_passed": sum(1 for r in run if r.passed),
        "n_failed": len(failed),
        "n_skipped": len(results) - len(run),
        "max_failed_severity": inv[max_sev],
        "failed_tests": [r.name for r in failed],
    }


def _residual_tests(resid, design, sm_res) -> list[TestResult]:
    from statsmodels.stats.diagnostic import (
        acorr_breusch_godfrey,
        acorr_ljungbox,
        het_breuschpagan,
        het_white,
        linear_harvey_collier,
        linear_reset,
    )
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    out: list[TestResult] = []
    resid = np.asarray(resid, dtype=float)
    exog = np.asarray(design, dtype=float)

    bp = _safe(lambda: het_breuschpagan(resid, exog))
    if bp and bp[0] != "__error__":
        pv = float(bp[1])
        out.append(TestResult("breusch_pagan", "heteroskedasticity", float(bp[0]), pv,
                              pv > ALPHA, "INFO" if pv > ALPHA else "MEDIUM",
                              "Homoskedastic residuals." if pv > ALPHA else
                              "Heteroskedasticity detected — use robust (HC) standard errors.",
                              h0="residual variance is constant"))
    wh = _safe(lambda: het_white(resid, exog))
    if wh and wh[0] != "__error__":
        pv = float(wh[1])
        out.append(TestResult("white", "heteroskedasticity", float(wh[0]), pv, pv > ALPHA,
                              "INFO" if pv > ALPHA else "MEDIUM",
                              "No heteroskedasticity (White)." if pv > ALPHA else
                              "White test indicates heteroskedasticity.", h0="homoskedastic"))

    dw = _safe(lambda: float(durbin_watson(resid)))
    if isinstance(dw, float):
        ok = 1.5 < dw < 2.5
        out.append(TestResult("durbin_watson", "autocorrelation", dw, None, ok,
                              "INFO" if ok else "MEDIUM",
                              "No first-order autocorrelation." if ok else
                              f"Durbin-Watson={dw:.2f} suggests serial correlation.",
                              h0="no first-order autocorrelation (DW~2)"))
    if sm_res is not None:
        bg = _safe(lambda: acorr_breusch_godfrey(sm_res, nlags=min(4, len(resid) // 5 or 1)))
        if bg and bg[0] != "__error__":
            pv = float(bg[1])
            out.append(TestResult("breusch_godfrey", "autocorrelation", float(bg[0]), pv, pv > ALPHA,
                                  "INFO" if pv > ALPHA else "MEDIUM",
                                  "No higher-order serial correlation." if pv > ALPHA else
                                  "Higher-order serial correlation detected.", h0="no serial correlation"))
    lb = _safe(lambda: acorr_ljungbox(resid, lags=[min(10, len(resid) // 5 or 1)], return_df=True))
    if lb is not None and not isinstance(lb, tuple):
        pv = float(lb["lb_pvalue"].iloc[-1])
        out.append(TestResult("ljung_box", "autocorrelation", float(lb["lb_stat"].iloc[-1]), pv,
                              pv > ALPHA, "INFO" if pv > ALPHA else "LOW",
                              "Residuals are independent." if pv > ALPHA else
                              "Ljung-Box indicates autocorrelation.", h0="residuals independent"))

    jb = _safe(lambda: jarque_bera(resid))
    if jb and jb[0] != "__error__":
        pv = float(jb[1])
        out.append(TestResult("jarque_bera", "normality", float(jb[0]), pv, pv > ALPHA,
                              "INFO" if pv > ALPHA else "LOW",
                              "Residuals ~ normal." if pv > ALPHA else
                              "Non-normal residuals (inference may be affected; OK for large n).",
                              h0="residuals normally distributed"))

    hc = _safe(lambda: linear_harvey_collier(sm_res)) if sm_res is not None else None
    if hc and hc[0] != "__error__":
        pv = float(hc[1])
        out.append(TestResult("harvey_collier", "linearity", float(hc[0]), pv, pv > ALPHA,
                              "INFO" if pv > ALPHA else "MEDIUM",
                              "Linear specification adequate." if pv > ALPHA else
                              "Possible nonlinearity (Harvey-Collier).", h0="relationship is linear"))
    reset = _safe(lambda: linear_reset(sm_res, power=2, use_f=True)) if sm_res is not None else None
    if reset is not None and not isinstance(reset, tuple):
        pv = float(reset.pvalue)
        out.append(TestResult("ramsey_reset", "specification", float(reset.fvalue), pv, pv > ALPHA,
                              "INFO" if pv > ALPHA else "MEDIUM",
                              "No specification error (RESET)." if pv > ALPHA else
                              "RESET suggests omitted nonlinearity / misspecification.",
                              h0="model is correctly specified"))
    return out


def _collinearity_tests(design) -> list[TestResult]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    out: list[TestResult] = []
    df = pd.DataFrame(np.asarray(design, dtype=float),
                      columns=[str(c) for c in getattr(design, "columns", range(np.asarray(design).shape[1]))])
    cols = [c for c in df.columns if str(c).lower() != "const"]
    if len(cols) >= 2:
        sub = df[cols].to_numpy()
        max_vif, worst = 0.0, None
        for i, c in enumerate(cols):
            vif = _safe(lambda i=i: float(variance_inflation_factor(sub, i)))
            if isinstance(vif, float) and np.isfinite(vif) and vif > max_vif:
                max_vif, worst = vif, c
        if worst is not None:
            ok = max_vif < 10
            sev = "INFO" if max_vif < 5 else ("MEDIUM" if max_vif < 10 else "HIGH")
            out.append(TestResult("max_vif", "multicollinearity", max_vif, None, ok, sev,
                                  f"Max VIF={max_vif:.1f} on '{worst}'." +
                                  ("" if ok else " Severe multicollinearity (VIF>10)."),
                                  h0="VIF < 10 (no severe multicollinearity)"))
    cond = _safe(lambda: float(np.linalg.cond(df.to_numpy())))
    if isinstance(cond, float) and np.isfinite(cond):
        ok = cond < 30
        out.append(TestResult("condition_number", "multicollinearity", cond, None, ok,
                              "INFO" if ok else "MEDIUM",
                              f"Design condition number={cond:.1f}." +
                              ("" if ok else " >30 indicates collinearity."),
                              h0="condition number < 30"))
    return out


def _stationarity_tests(y) -> list[TestResult]:
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    out: list[TestResult] = []
    if len(y) < 20 or np.std(y) == 0:
        return out
    # KPSS/ADF emit InterpolationWarning when the statistic is past the lookup table; the p-value is
    # still usable (it is simply reported as a bound), so suppress the noise.
    warnings.simplefilter("ignore")
    adf = _safe(lambda: adfuller(y, autolag="AIC"))
    if adf and adf[0] != "__error__":
        pv = float(adf[1])
        ok = pv < ALPHA  # reject unit root => stationary
        out.append(TestResult("adf", "stationarity", float(adf[0]), pv, ok,
                              "INFO" if ok else "MEDIUM",
                              "ADF: series is stationary." if ok else
                              "ADF: unit root (non-stationary) — consider differencing.",
                              h0="series has a unit root (non-stationary)"))
    kp = _safe(lambda: kpss(y, regression="c", nlags="auto"))
    if kp and kp[0] != "__error__":
        pv = float(kp[1])
        ok = pv > ALPHA  # fail to reject stationarity
        out.append(TestResult("kpss", "stationarity", float(kp[0]), pv, ok,
                              "INFO" if ok else "MEDIUM",
                              "KPSS: stationary." if ok else
                              "KPSS: non-stationary around a constant.",
                              h0="series is stationary"))
    return out
