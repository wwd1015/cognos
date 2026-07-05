"""Discrete-time hazard (survival) modeling for PD term structures — Shumway (2001).

The obligor frame is **panel-expanded inside the estimator**: each obligor contributes one row per
period at risk (1..horizon, stopping at the default period), with the event flag set in the default
period and a categorical period column carrying the baseline hazard. Because the expansion happens
inside ``fit``/``predict``, the CV and holdout machinery keep splitting at the *obligor* level —
panel rows of one obligor can never straddle a fold boundary, which is the classic survival-CV leak.

Links: ``hazard_logit`` (period-indexed logistic hazard) and ``hazard_cloglog`` (the grouped-time
proportional-hazards link). Both are fit by the statsmodels-backed :class:`SMBinaryGLM`, and the
champion refit adds full statsmodels GLM inference on a K-1 panel design, so coefficients and
p-values (including the baseline-hazard period effects) are valid for documentation.

The deliverable beyond a single-horizon PD is the **term structure**: cumulative
PD(t) = 1 − Π_{s≤t} (1 − h_s(x)) for every period t up to the horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fit import Candidate, FittedModel, SMBinaryGLM, _design_columns, make_preprocessor

HAZARD_LINKS = {"hazard_logit": "logit", "hazard_cloglog": "cloglog"}
PERIOD_COL = "_period"  # categorical period index carrying the baseline hazard


def _effective_event_time(event_time, y: np.ndarray, horizon: int) -> np.ndarray:
    """Period (1-based) at which each obligor's event lands, NaN when censored.

    ``event_time`` may be missing/censored-coded; the binary target is the ground truth for
    *whether* an event happened inside the window — a defaulter with no usable timing is placed
    conservatively in the final period.
    """
    et = pd.to_numeric(pd.Series(np.asarray(event_time)), errors="coerce").to_numpy(dtype=float)
    yv = np.asarray(y).astype(int)
    valid = np.isfinite(et) & (et >= 1) & (et <= horizon)
    out = np.where(yv == 1, np.where(valid, et, float(horizon)), np.nan)
    return out


def expand_panel(X: pd.DataFrame, event_time, y: np.ndarray, horizon: int
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    """Obligor frame → obligor-period panel with the event flag in the default period."""
    Xr = X.reset_index(drop=True)
    et = _effective_event_time(event_time, y, horizon)
    periods_at_risk = np.where(np.isfinite(et), et, float(horizon)).astype(int)

    row_idx = np.repeat(np.arange(len(Xr)), periods_at_risk)
    period = np.concatenate([np.arange(1, k + 1) for k in periods_at_risk]) if len(Xr) else np.array([], dtype=int)
    panel = Xr.iloc[row_idx].reset_index(drop=True)
    panel[PERIOD_COL] = np.array([f"p{t:02d}" for t in period], dtype=object)

    y_panel = np.zeros(len(panel), dtype=int)
    if len(panel):
        is_event_row = np.isfinite(et)[row_idx] & (period == periods_at_risk[row_idx])
        y_panel[is_event_row & (np.repeat(np.asarray(y).astype(int), periods_at_risk) == 1)] = 1
    return panel, y_panel


class HazardScorer:
    """Picklable serving object satisfying the ``ScorerBundle.pipeline`` contract.

    ``predict_proba(X)`` returns the cumulative PD over the full horizon per obligor, computed by
    scoring the fitted per-period hazard pipeline at every period and compounding survival.
    """

    def __init__(self, pipeline, features: list[str], horizon: int):
        self.pipeline = pipeline
        self.features = list(features)
        self.horizon = int(horizon)
        self.named_steps = pipeline.named_steps  # duck-type sklearn Pipeline for FittedModel

    def hazard_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """(n_obligors, horizon) per-period hazards h_t(x)."""
        Xf = X[self.features].reset_index(drop=True)
        cols = []
        for t in range(1, self.horizon + 1):
            Xp = Xf.copy()
            Xp[PERIOD_COL] = f"p{t:02d}"
            cols.append(self.pipeline.predict_proba(Xp)[:, 1])
        return np.column_stack(cols)

    def cumulative_pd(self, X: pd.DataFrame, upto: int | None = None) -> np.ndarray:
        h = self.hazard_matrix(X)[:, : (upto or self.horizon)]
        return 1.0 - np.prod(1.0 - h, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.cumulative_pd(X), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _fit_hazard_pipeline(Xf: pd.DataFrame, y, event_time, link: str, horizon: int):
    from sklearn.pipeline import Pipeline

    panel, y_panel = expand_panel(Xf, event_time, y, horizon)
    pre = make_preprocessor(panel, scale=True)  # PERIOD_COL is object → one-hot with the categoricals
    pipe = Pipeline([("pre", pre), ("est", SMBinaryGLM(link=link))])
    pipe.fit(panel, y_panel)
    return pipe, panel, y_panel


def hazard_fit_predict(candidate: Candidate):
    """The leakage-safe CV contract for hazard families (see metrics.cv_score).

    The event-time column travels in ``X`` as metadata (never in ``candidate.features``); folds are
    obligor-level and the panel expansion happens inside each training fold only.
    """
    link = HAZARD_LINKS[candidate.family]
    etc = str(candidate.hyperparams["event_time_col"])
    horizon = int(candidate.hyperparams["horizon"])

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray):
        if etc not in X_train.columns:
            raise ValueError(f"hazard family needs event-time column '{etc}' in the search frame")
        pipe, _, _ = _fit_hazard_pipeline(X_train[candidate.features], y_train,
                                          X_train[etc], link, horizon)
        scorer = HazardScorer(pipe, candidate.features, horizon)

        def predict(X_val: pd.DataFrame):
            proba = np.clip(scorer.cumulative_pd(X_val), 1e-9, 1 - 1e-9)
            return (proba >= 0.5).astype(int), proba

        return predict

    return fit_predict


def fit_full_hazard(candidate: Candidate, X: pd.DataFrame, y: np.ndarray, *, event_time,
                    horizon: int, task: str) -> FittedModel:
    """Refit the hazard champion on the full training set with valid panel GLM inference."""
    import statsmodels.api as sm

    from .fit import build_inference_design

    link = HAZARD_LINKS[candidate.family]
    Xf = X[candidate.features]
    pipe, panel, y_panel = _fit_hazard_pipeline(Xf, y, event_time, link, horizon)
    scorer = HazardScorer(pipe, candidate.features, horizon)

    fitted = FittedModel(
        candidate=candidate, task=task, is_classification=True,
        pipeline=scorer, feature_names=_design_columns(pipe.named_steps["pre"]),
        raw_features=list(candidate.features),
    )
    # Obligor-level residuals (binary outcome vs cumulative PD) keep the diagnostic battery aligned
    # with the obligor frame; the panel design is deliberately NOT exposed as design_matrix.
    fitted.residuals = np.asarray(y).astype(float) - scorer.cumulative_pd(Xf)
    try:
        design = build_inference_design(panel)  # K-1 coding incl. the period baseline hazard
        res = sm.GLM(np.asarray(y_panel, dtype=float), design,
                     family=SMBinaryGLM._family(link)).fit(maxiter=200)
        fitted.sm_result = res
    except Exception:
        fitted.sm_result = None
    return fitted


def term_structure(scorer: HazardScorer, X: pd.DataFrame) -> dict:
    """Portfolio-average PD term structure: cumulative PD and marginal hazard per period."""
    h = scorer.hazard_matrix(X)
    surv = np.cumprod(1.0 - h, axis=1)
    cum_pd = 1.0 - surv
    return {
        "periods": list(range(1, scorer.horizon + 1)),
        "mean_hazard": [float(v) for v in h.mean(axis=0)],
        "mean_cumulative_pd": [float(v) for v in cum_pd.mean(axis=0)],
    }
