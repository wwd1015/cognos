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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# GLM/econometric families: linear (scaled + sklearn-predicted) but with no statsmodels OLS/Logit
# inference here, so they skip build_inference_design and report sklearn coef_ only.
GLM_FAMILIES = {"poisson", "gamma", "tweedie"}

# Binary-response GLM links beyond logit. Fit via a statsmodels-backed estimator (sklearn has no
# probit/cloglog) with full statsmodels inference on the K-1 design. cloglog is the grouped-time
# proportional-hazards link, which is why the discrete-time hazard families build on it.
BINARY_GLM_LINKS = {"probit", "cloglog"}

# Discrete-time hazard families (Shumway 2001): the obligor frame is panel-expanded inside the
# estimator, so CV splits stay at the obligor level (leakage-safe). See modeling/hazard.py.
HAZARD_FAMILIES = {"hazard_logit", "hazard_cloglog"}

# family -> (is_linear, needs_scaling, is_classifier-capable). GLMs are linear for scaling/predict.
LINEAR_FAMILIES = {
    "ols", "ridge", "lasso", "elasticnet", "logit", "ridge_logit", "lasso_logit",
    *GLM_FAMILIES, *BINARY_GLM_LINKS, *HAZARD_FAMILIES,
}

DEFAULT_FAMILIES = {
    "regression": ["ols", "ridge", "lasso", "elasticnet"],
    "glm_regression": ["poisson", "gamma", "tweedie", "ols", "ridge"],
    "timeseries": ["ols", "ridge", "lasso"],
    "classification": ["logit", "probit", "cloglog", "ridge_logit", "lasso_logit"],
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


class SMBinaryGLM(ClassifierMixin, BaseEstimator):
    """sklearn-compatible binary GLM with a statsmodels backend (probit / cloglog links).

    sklearn has no probit or complementary-log-log classifier; statsmodels does, with proper IRLS
    fitting. This thin wrapper exposes fit/predict_proba/predict so the estimator drops into the
    existing leakage-safe CV pipeline (BaseEstimator supplies clone/params/tags plumbing).
    """

    def __init__(self, link: str = "probit", maxiter: int = 200):
        self.link = link
        self.maxiter = maxiter

    @staticmethod
    def _family(link: str):
        import statsmodels.api as sm

        links = {"probit": sm.families.links.Probit(), "cloglog": sm.families.links.CLogLog(),
                 "logit": sm.families.links.Logit()}
        return sm.genmod.families.Binomial(link=links[link])

    def fit(self, X, y) -> SMBinaryGLM:
        import statsmodels.api as sm

        Xd = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
        self.result_ = sm.GLM(np.asarray(y, dtype=float), Xd, family=self._family(self.link)).fit(
            maxiter=self.maxiter)
        self.classes_ = np.array([0, 1])
        self.coef_ = np.asarray(self.result_.params[1:]).reshape(1, -1)  # excl. intercept
        return self

    def predict_proba(self, X) -> np.ndarray:
        import statsmodels.api as sm

        Xd = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
        p = np.clip(np.asarray(self.result_.predict(Xd), dtype=float), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _penalty(kind: str) -> dict[str, Any]:
    """LogisticRegression regularization kwargs: sklearn >= 1.8 deprecates ``penalty`` in favour
    of ``l1_ratio`` (0 = ridge, 1 = lasso); older versions only understand ``penalty``."""
    import sklearn

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return {"l1_ratio": 1.0 if kind == "l1" else 0.0}
    return {"penalty": kind}


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
    if family == "poisson":
        return PoissonRegressor(alpha=hp.get("alpha", 1.0), max_iter=300)
    if family == "gamma":
        return GammaRegressor(alpha=hp.get("alpha", 1.0), max_iter=300)
    if family == "tweedie":
        return TweedieRegressor(power=hp.get("power", 1.5), alpha=hp.get("alpha", 1.0), max_iter=300)
    if family == "logit":
        return LogisticRegression(C=1e6, max_iter=2000)
    if family in BINARY_GLM_LINKS:
        return SMBinaryGLM(link=family, maxiter=hp.get("maxiter", 200))
    if family == "ridge_logit":
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=2000, **_penalty("l2"))
    if family == "lasso_logit":
        return LogisticRegression(C=hp.get("C", 1.0), solver="liblinear", max_iter=2000,
                                  **_penalty("l1"))
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
    if candidate.family in HAZARD_FAMILIES:
        from .hazard import hazard_fit_predict

        return hazard_fit_predict(candidate)

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
    transforms: list = field(default_factory=list)  # list[TransformSpec] LLM-authored, target-hidden
    base_features: list[str] | None = None  # original columns needed to recompute the transforms
    structural: dict | None = None  # StructuralSpec dict; Merton DD recomputed at predict/serve
    migration: dict | None = None  # fitted migration model dict; migration_pd recomputed at serve

    @property
    def model_id(self) -> str:
        return f"{self.candidate.family}"

    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Recompute engineered features (target-hidden) so predict matches how the model was fit."""
        frame = X
        if self.structural:
            from .structural import StructuralSpec, augment_frame

            spec = StructuralSpec.from_dict(self.structural)
            market = (spec.equity_value_col, spec.equity_vol_col, spec.debt_col)
            if spec.dd_feature not in frame.columns and all(c in frame.columns for c in market):
                frame, _ = augment_frame(frame, spec)
        if self.migration:
            from .migration import MigrationSpec
            from .migration import augment_frame as migration_augment

            mspec = MigrationSpec.from_dict(self.migration["spec"])
            if mspec.pd_feature not in frame.columns and mspec.rating_col in frame.columns:
                frame, _ = migration_augment(frame, self.migration)
        if not self.transforms:
            return frame
        from .transforms import apply_transforms

        base = self.base_features or self.raw_features
        cols = [c for c in dict.fromkeys([*base, *self.raw_features]) if c in frame.columns]
        aug, _, _ = apply_transforms(frame[cols], list(self.transforms))
        return aug

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._augment(X)[self.raw_features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.is_classification:
            return None
        return self.pipeline.predict_proba(self._augment(X)[self.raw_features])[:, 1]

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

    # GLMs are linear (scaled + sklearn-predicted) but have no statsmodels OLS/Logit inference here;
    # they report sklearn coef_ via FittedModel.coefficients() instead. Hazard families never reach
    # this function — the model stage fits them via modeling.hazard.fit_full_hazard.
    if candidate.is_linear and candidate.family not in GLM_FAMILIES:
        try:
            design_df = build_inference_design(Xf)  # full-rank K-1 design (valid p-values)
            yv = np.asarray(y).astype(float)
            if is_classification:
                if candidate.family in BINARY_GLM_LINKS:
                    res = sm.GLM(yv, design_df,
                                 family=SMBinaryGLM._family(candidate.family)).fit(maxiter=200)
                else:
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
