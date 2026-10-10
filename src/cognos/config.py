"""COGNOS configuration schema — the per-project profile.

Mirrors deputy's ``projects/<name>.yaml`` idea: the *only* place project specifics live. The
agents/stages are project-agnostic; adding a new modeling problem means writing one YAML file.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

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


class DataConfig(BaseModel):
    path: str | None = None  # CSV/Parquet path; may be None when a DataFrame is passed in code
    format: str = "csv"  # csv | parquet
    target: str  # target column name
    features: list[str] = Field(default_factory=list)  # empty => use all non-target columns
    datetime_col: str | None = None  # for time-series / walk-forward ordering
    drop_columns: list[str] = Field(default_factory=list)
    protected_attributes: list[str] = Field(default_factory=list)  # for fair-lending checks
    # Discrete-time hazard (survival) support: the 1-based period in which the event occurred
    # (NaN/0 for censored obligors). Outcome data — excluded from features like the target; it is
    # the label's *timing*, consumed only by the hazard families (modeling/hazard.py).
    event_time_col: str | None = None
    horizon_periods: int | None = None  # outcome window in periods; None => max observed event time


class DesignConfig(BaseModel):
    """The modeling **design brief** — what the model sponsor (e.g. an MD in model development)
    has decided about the problem, before any algorithm is chosen.

    Ideate consumes this to ground its econometric-framework assessment; every *unanswered* field
    becomes an explicit open design question in the design brief (MD triangulation) instead of a
    silent assumption. All fields are optional free text so the engine never blocks on them.
    """

    use_case: str = ""  # e.g. origination | surveillance | CECL/IFRS9 | IRB | stress-testing
    horizon: str = ""  # outcome window, e.g. "12-month PD", "lifetime"
    default_definition: str = ""  # e.g. "90+ DPD or nonaccrual, per SR 11-7 documentation"
    segment: str = ""  # e.g. "C&I middle-market", "CRE income-producing", "small business"
    interpretability: str = "required"  # required | preferred | flexible
    notes: str = ""  # free-form sponsor guidance (sign expectations, exclusions, priors)

    def unanswered(self) -> list[str]:
        """Names of the design questions the sponsor has not answered yet."""
        blanks = [name for name in ("use_case", "horizon", "default_definition", "segment")
                  if not getattr(self, name).strip()]
        return blanks


class PriorArtifact(BaseModel):
    """One artifact of the model being updated. ``auto`` infers the role from the file name."""

    path: str
    role: Literal["auto", "whitepaper", "code", "validation", "monitoring", "other"] = "auto"


class EngagementConfig(BaseModel):
    """The development mode and the documents the run starts from (engagement.py).

    ``new`` is a complete new model development; ``update`` changes an existing model and needs
    that model's artifacts plus a change request. The business intent document (and the change
    request) follow one template: ``cognos intent-template [--kind update]``. Paths are copied
    into the run (``runs/<id>/inputs/``) when it is created.
    """

    kind: Literal["new", "update"] = "new"
    intent: str | None = None  # the business intent document / model update request
    supporting: list[str] = Field(default_factory=list)  # background material
    prior_artifacts: list[PriorArtifact] = Field(default_factory=list)  # update: the existing model
    prior_run: str | None = None  # update: an earlier COGNOS run (id or directory) of that model

    @model_validator(mode="before")
    @classmethod
    def _plain_paths(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and raw.get("prior_artifacts"):
            raw = dict(raw)
            raw["prior_artifacts"] = [{"path": a} if isinstance(a, str) else a
                                      for a in raw["prior_artifacts"]]
        return raw

    @model_validator(mode="after")
    def _check_kind(self) -> EngagementConfig:
        has_prior = bool(self.prior_artifacts or self.prior_run)
        if self.kind == "update":
            if not has_prior:
                raise ValueError(
                    "a model update needs the existing model: give engagement.prior_artifacts "
                    "(white paper, code, reports) or engagement.prior_run")
            if not self.intent:
                raise ValueError(
                    "a model update needs the update request: give engagement.intent "
                    "(template: cognos intent-template --kind update)")
        elif has_prior:
            raise ValueError("prior-model artifacts are read only in a model update; set "
                             "engagement.kind: update")
        return self


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
    guided: bool = False  # opt-in agent-guided search (ADR-0001 stage B): the modeler proposes experiments
    guided_rounds: int = 6  # number of LLM-proposed experiments after the deterministic ratchet


class StructuralConfig(BaseModel):
    """Merton structural model support (modeling/structural.py) — needs market observables.

    When enabled and the three columns exist, the engine solves for distance-to-default and adds it
    as an engineered feature (hybrid mode), and scores the pure structural PD on the sealed holdout
    as a labelled challenger benchmark.
    """

    enabled: bool = False
    equity_value_col: str | None = None
    equity_vol_col: str | None = None
    debt_col: str | None = None
    risk_free_rate: float = 0.03
    horizon_years: float = 1.0


class MigrationConfig(BaseModel):
    """Rating-transition (migration) support (modeling/migration.py) — needs a rating history.

    When enabled and the rating/next-rating columns exist, the model stage estimates a cohort
    transition matrix on the training partition (NR-adjusted, Laplace-smoothed), swaps the raw
    rating for the horizon cumulative PD it implies (``migration_pd``, the hybrid feature), scores
    the pure-migration PD on the sealed holdout as a labelled challenger benchmark, and reports a
    by-rating expected-loss forecast. ``next_rating_col`` is outcome data — like the target, it is
    never a feature.
    """

    enabled: bool = False
    rating_col: str | None = None  # rating at the observation point (in the information set)
    next_rating_col: str | None = None  # rating at the end of the outcome window (outcome data)
    rating_scale: list[str] = Field(default_factory=list)  # best -> worst, excluding the default
    # state; empty => inferred from observed default rates (reported in the run record)
    default_state: str = "D"
    withdrawn_states: list[str] = Field(default_factory=lambda: ["NR"])  # denominator-adjusted
    horizon_periods: int = 1  # loss-forecast horizon in rating periods (matrix powers)
    smoothing: float = 0.5  # Laplace prior count per cell
    monotone_pd: bool = True  # PAVA-monotonize the default column (reported when it adjusts)
    condition_col: str | None = None  # e.g. macro-regime column -> conditional matrices (stress)
    lgd: float = 0.45  # loss-given-default for the expected-loss forecast
    ead_column: str | None = None  # exposure-at-default column; None => equal-weighted


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


class PortfolioConfig(BaseModel):
    """Vasicek one-factor portfolio loss simulation + Basel IRB capital (modeling/simulate.py).

    Opt-in, reported by the backtest stage; never feeds champion selection. PDs in are the model's
    scores — read them next to the calibration section of the outcomes analysis.
    """

    enabled: bool = False
    lgd: float = 0.45  # loss given default assumption
    ead_column: str | None = None  # exposure-at-default column; None => equal-weighted
    asset_correlation: float | None = None  # None => Basel IRB corporate formula rho(PD)
    confidence: float = 0.999
    n_sims: int = 20000
    seed: int = 42


class StressConfig(BaseModel):
    """Macro-scenario stress testing: shock covariates, re-score through the deployed scorer.

    Each scenario: {name: str, shocks: {column: {"add": x} | {"mul": x} | {"set": x}}}.
    """

    enabled: bool = False
    scenarios: list[dict] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    """Which backend runs the stage agents (agents/providers.yaml) and the runner's limits.

    ``auto`` resolves COGNOS_PROVIDER, then the first available LLM provider, else ``heuristic``
    (the deterministic offline agents). Tests and offline demos pin ``heuristic``.
    """

    provider: str = "auto"
    model: str | None = None  # override the provider's default model
    max_retries: int = 2  # validation retries per agent call (errors are fed back)
    time_limit_s: float = 300.0  # wall-clock limit per agent call, all attempts
    budget_usd: float = 5.0  # spend cap per run (estimated from token usage); 0 = unlimited
    replay_dir: str | None = None  # provider "replay": directory of recorded <agent>.json outputs


class WorkflowConfig(BaseModel):
    """Human gates and the automatic challenge loop (engine/graph.py)."""

    gates: list[str] = Field(default_factory=lambda: [
        "gate_intent", "gate_data", "gate_design", "gate_champion", "gate_validation",
        "gate_signoff"])
    auto_challenge_loops: int = 2  # validator high findings routed back automatically, at most N times


class StagesConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "intake",
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
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    design: DesignConfig = Field(default_factory=DesignConfig)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    structural: StructuralConfig = Field(default_factory=StructuralConfig)
    migration: MigrationConfig = Field(default_factory=MigrationConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    stress: StressConfig = Field(default_factory=StressConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    runs_dir: str = "runs"

    @model_validator(mode="before")
    @classmethod
    def _migrate_brain(cls, raw: Any) -> Any:
        """v0.x profiles configured an LLM ``brain``; v1 configures ``agents`` (a provider)."""
        if isinstance(raw, dict) and "brain" in raw:
            raw = dict(raw)
            brain = raw.pop("brain") or {}
            agents = dict(raw.get("agents") or {})
            if brain.get("kind") == "llm":
                agents.setdefault("provider", "anthropic")
            elif brain.get("kind") == "heuristic":
                agents.setdefault("provider", "heuristic")
            raw["agents"] = agents
        return raw

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
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping, got {type(raw)}")
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CognosConfig:
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
