"""Model fitters for both traditional statistical models and ML models.

Linear/statistical families are fit twice: an sklearn ``Pipeline`` (preprocessing + estimator) is
used for prediction and leakage-safe CV, while a parallel ``statsmodels`` fit on the full design
matrix yields coefficients, p-values and residuals for the statistical-testing + documentation
stages. ML families use sklearn only and expose feature importances.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# family -> (is_linear, needs_scaling, is_classifier-capable)
LINEAR_FAMILIES = {"ols", "ridge", "lasso", "elasticnet", "logit", "ridge_logit", "lasso_logit"}

DEFAULT_FAMILIES = {
    "regression": ["ols", "ridge", "lasso", "elasticnet"],
    "timeseries": ["ols", "ridge", "lasso"],
    "classification": ["logit", "ridge_logit", "lasso_logit"],
    "ml_regression": ["random_forest", "gradient_boosting"],
    "ml_classification": ["random_forest", "gradient_boosting"],
}


@dataclass
class Candidate:
    """One experiment in the search: a model family + feature subset + hyperparameters."""

    family: str
    features: list[str]
    hyperparams: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def is_linear(self) -> bool:
        return self.family in LINEAR_FAMILIES

    def label(self) -> str:
        hp = ",".join(f"{k}={v}" for k, v in sorted(self.hyperparams.items()))
        return f"{self.family}({len(self.features)}f;{hp})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "features": self.features,
            "hyperparams": self.hyperparams,
            "description": self.description or self.label(),
        }


def _estimator(family: str, is_classification: bool, hp: dict[str, Any]):
    rs = hp.get("random_state", 42)
    if family == "ols":
        return LinearRegression()
    if family == "ridge":
        return Ridge(alpha=hp.get("alpha", 1.0), random_state=rs)
    if family == "lasso":
        return Lasso(alpha=hp.get("alpha", 0.01), max_iter=5000, random_state=rs)
    if family == "elasticnet":
        return ElasticNet(alpha=hp.get("alpha", 0.01), l1_ratio=hp.get("l1_ratio", 0.5),
                          max_iter=5000, random_state=rs)
    if family == "logit":
        return LogisticRegression(C=1e6, max_iter=2000)
    if family == "ridge_logit":
        return LogisticRegression(C=hp.get("C", 1.0), penalty="l2", max_iter=2000)
    if family == "lasso_logit":
        return LogisticRegression(C=hp.get("C", 1.0), penalty="l1", solver="liblinear", max_iter=2000)
    if family == "random_forest":
        cls = RandomForestClassifier if is_classification else RandomForestRegressor
        return cls(n_estimators=hp.get("n_estimators", 200), max_depth=hp.get("max_depth", None),
                   random_state=rs, n_jobs=1)
    if family == "gradient_boosting":
        cls = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
        return cls(n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", 3),
                   learning_rate=hp.get("learning_rate", 0.1), random_state=rs)
    raise ValueError(f"Unknown model family '{family}'")


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    transformers = []
    if num:
        transformers.append(("num", StandardScaler() if scale else "passthrough", num))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(transformers, remainder="drop")


def make_fit_predict(candidate: Candidate, *, is_classification: bool) -> Callable:
    """Return ``fit_predict(X_train, y_train) -> predict`` for leakage-safe CV (see metrics.cv_score)."""

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray):
        Xf = X_train[candidate.features]
        pre = make_preprocessor(Xf, scale=candidate.is_linear)
        est = _estimator(candidate.family, is_classification, candidate.hyperparams)
        pipe = Pipeline([("pre", pre), ("est", est)])
        pipe.fit(Xf, y_train)

        def predict(X_val: pd.DataFrame):
            Xv = X_val[candidate.features]
            if is_classification:
                proba = pipe.predict_proba(Xv)[:, 1]
                point = (proba >= 0.5).astype(int)
                return point, proba
            return pipe.predict(Xv), None

        return predict

    return fit_predict


@dataclass
class FittedModel:
    """A trained champion model with everything downstream stages need."""

    candidate: Candidate
    task: str
    is_classification: bool
    pipeline: Pipeline
    feature_names: list[str]  # post-transform design columns
    raw_features: list[str]
    sm_result: Any = None  # statsmodels result (linear families only)
    design_matrix: pd.DataFrame | None = None  # exog incl. const (for diagnostics)
    residuals: np.ndarray | None = None

    @property
    def model_id(self) -> str:
        return f"{self.candidate.family}"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.raw_features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.is_classification:
            return None
        return self.pipeline.predict_proba(X[self.raw_features])[:, 1]

    def coefficients(self) -> dict[str, float] | None:
        if self.sm_result is not None:
            return {k: float(v) for k, v in self.sm_result.params.items()}
        est = self.pipeline.named_steps["est"]
        if hasattr(est, "coef_"):
            coef = np.ravel(est.coef_)
            return {n: float(c) for n, c in zip(self.feature_names, coef, strict=False)}
        return None

    def pvalues(self) -> dict[str, float] | None:
        if self.sm_result is not None and hasattr(self.sm_result, "pvalues"):
            return {k: float(v) for k, v in self.sm_result.pvalues.items()}
        return None

    def feature_importances(self) -> dict[str, float] | None:
        est = self.pipeline.named_steps["est"]
        if hasattr(est, "feature_importances_"):
            return {n: float(i) for n, i in zip(self.feature_names, est.feature_importances_, strict=False)}
        return None


def _design_columns(pre: ColumnTransformer) -> list[str]:
    try:
        return [str(c) for c in pre.get_feature_names_out()]
    except Exception:  # pragma: no cover
        return [f"x{i}" for i in range(pre.transform_count_)]  # type: ignore


def build_inference_design(Xf: pd.DataFrame) -> pd.DataFrame:
    """Full-rank design for valid statsmodels inference, decoupled from the prediction pipeline.

    Uses **K-1 (drop-first) dummy coding** for categoricals + standardized numerics + an intercept.
    The prediction pipeline deliberately uses all-K one-hot for serving robustness, but all-K dummies
    plus an intercept are perfectly collinear (the dummy-variable trap), which makes coefficients,
    standard errors and p-values ill-defined. K-1 coding here yields a full-rank design so the
    inference COGNOS reports is statistically valid — a hard requirement for SR 11-7 model validation.
    Inference runs on training data only, so `handle_unknown` robustness is not needed.
    """
    import statsmodels.api as sm

    num = Xf.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in Xf.columns if c not in num]
    parts: list[pd.DataFrame] = []
    if num:
        scaler = StandardScaler()
        parts.append(pd.DataFrame(scaler.fit_transform(Xf[num].astype(float)),
                                  columns=num, index=Xf.index))
    if cat:
        parts.append(pd.get_dummies(Xf[cat].astype("object"), drop_first=True, dtype=float))
    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=Xf.index)
    return sm.add_constant(design, has_constant="add")


def fit_full(candidate: Candidate, X: pd.DataFrame, y: np.ndarray, *, task: str,
             is_classification: bool) -> FittedModel:
    """Fit the champion on the full training set; add statsmodels inference for linear families."""
    import statsmodels.api as sm

    Xf = X[candidate.features]
    pre = make_preprocessor(Xf, scale=candidate.is_linear)
    est = _estimator(candidate.family, is_classification, candidate.hyperparams)
    pipe = Pipeline([("pre", pre), ("est", est)])
    pipe.fit(Xf, y)
    feat_names = _design_columns(pipe.named_steps["pre"])

    fitted = FittedModel(
        candidate=candidate, task=task, is_classification=is_classification,
        pipeline=pipe, feature_names=feat_names, raw_features=candidate.features,
    )

    if candidate.is_linear:
        try:
            design_df = build_inference_design(Xf)  # full-rank K-1 design (valid p-values)
            yv = np.asarray(y).astype(float)
            if is_classification:
                res = sm.Logit(yv, design_df).fit(disp=0, maxiter=200)
                fitted.residuals = yv - np.asarray(res.predict(design_df))
            else:
                res = sm.OLS(yv, design_df).fit()
                fitted.residuals = np.asarray(res.resid)
            fitted.sm_result = res
            fitted.design_matrix = design_df
        except Exception:
            # statsmodels can fail to converge (e.g. perfect separation); keep sklearn coefficients.
            fitted.sm_result = None

    return fitted
