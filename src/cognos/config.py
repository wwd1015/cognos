"""COGNOS configuration schema — the per-project profile.

Mirrors deputy's ``projects/<name>.yaml`` idea: the *only* place project specifics live. The
agents/stages are project-agnostic; adding a new modeling problem means writing one YAML file.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    REGRESSION = "regression"  # traditional statistical regression (OLS/GLM/...)
    CLASSIFICATION = "classification"  # traditional statistical classification (logit/...)
    ML_REGRESSION = "ml_regression"  # tree/boosting/NN regressors
    ML_CLASSIFICATION = "ml_classification"  # tree/boosting/NN classifiers
    TIMESERIES = "timeseries"  # ARIMA/forecasting

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.CLASSIFICATION, TaskType.ML_CLASSIFICATION)

    @property
    def is_ml(self) -> bool:
        return self in (TaskType.ML_REGRESSION, TaskType.ML_CLASSIFICATION)


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Mode(str, Enum):
    AUTONOMOUS = "autonomous"  # quick-and-dirty prototype; gates auto-approve
    INTERACTIVE = "interactive"  # human-in-the-loop; pause at configured gates


class BrainKind(str, Enum):
    HEURISTIC = "heuristic"  # deterministic, no LLM (default; powers tests + offline demo)
    LLM = "llm"  # Claude-backed reasoning for idea-gen / judgment / prose


class DataConfig(BaseModel):
    path: str | None = None  # CSV/Parquet path; may be None when a DataFrame is passed in code
    format: str = "csv"  # csv | parquet
    target: str  # target column name
    features: list[str] = Field(default_factory=list)  # empty => use all non-target columns
    datetime_col: str | None = None  # for time-series / walk-forward ordering
    drop_columns: list[str] = Field(default_factory=list)
    protected_attributes: list[str] = Field(default_factory=list)  # for fair-lending checks


class MetricConfig(BaseModel):
    name: str = "auto"  # auto => rmse for regression, roc_auc for classification
    direction: Direction | None = None  # auto-inferred from metric when None


class SearchConfig(BaseModel):
    max_candidates: int = 24  # ratchet experiment budget
    time_budget_s: float | None = None  # wall-clock cap (autoresearch-style); None = unbounded
    cv_folds: int = 5
    holdout_fraction: float = 0.2  # sealed final holdout (frozen substrate)
    random_state: int = 42  # pinned seed (reproducibility / autoforge contract)
    model_families: list[str] = Field(default_factory=list)  # empty => task defaults
    ensemble: bool = False  # off by default (ADR-0007); when on, an ensemble is a labelled
    # challenger benchmark only — the deployed model is always the single interpretable champion.
    max_features_per_candidate: int | None = None
    complexity_penalty: float = 0.0  # parsimony / simplicity bias (>=0)


class ComplianceConfig(BaseModel):
    regimes: list[str] = Field(default_factory=lambda: ["SR11-7", "NIST-AI-RMF"])
    risk_tier: str = "medium"  # low | medium | high — drives validation intensity
    intended_use: str = ""
    out_of_scope_use: str = ""
    fair_lending: bool = False  # run ECOA/Reg B SCAN + disparate-impact checks
    disparate_impact_threshold: float = 0.8  # 4/5ths rule
    jurisdictions: list[str] = Field(default_factory=lambda: ["US"])  # US | EU


class ImpactConfig(BaseModel):
    enabled: bool = True  # use the IMPACT library if importable, else built-in fallback
    entity_name: str = "ScoredEntity"
    primary_key: str | None = None  # defaults to a synthesized row id
    extra_fields: list[dict[str, Any]] = Field(default_factory=list)


class BacktestConfig(BaseModel):
    enabled: bool = True
    scheme: str = "walk_forward"  # walk_forward | cscv | holdout
    n_splits: int = 5
    deflate_sharpe: bool = True  # Deflated Sharpe Ratio (multiple-testing correction)
    pbo: bool = True  # Probability of Backtest Overfitting via CSCV
    returns_column: str | None = None  # if the task is a trading/return signal


class BrainConfig(BaseModel):
    kind: BrainKind = BrainKind.HEURISTIC
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.2


class StagesConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "explore",
            "ideate",
            "model",
            "backtest",
            "validate",
            "comply",
            "document",
            "review",
        ]
    )
    gates: list[str] = Field(default_factory=lambda: ["validate", "review"])
    # Stages that may BLOCK the pipeline. In interactive mode these are the human pause points.
    # Compliance is intentionally NOT a gate (ADR-0006): it is a non-gating readiness report.
    halt_on_block: bool = True  # autonomous mode: stop the run when a gate BLOCKs


class CognosConfig(BaseModel):
    """The complete COGNOS project profile."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    task: TaskType
    mode: Mode = Mode.AUTONOMOUS
    data: DataConfig
    metric: MetricConfig = Field(default_factory=MetricConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    runs_dir: str = "runs"

    @model_validator(mode="after")
    def _fill_defaults(self) -> CognosConfig:
        # Resolve auto metric + direction from the task type.
        if self.metric.name == "auto":
            self.metric.name = "roc_auc" if self.task.is_classification else "rmse"
        if self.metric.direction is None:
            maximize = {"roc_auc", "accuracy", "r2", "f1", "direction_accuracy", "average_precision"}
            self.metric.direction = (
                Direction.MAXIMIZE if self.metric.name in maximize else Direction.MINIMIZE
            )
        return self

    # --- IO ----------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> CognosConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping, got {type(raw)}")
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CognosConfig:
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
