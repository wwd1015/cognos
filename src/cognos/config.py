"""COGNOS configuration schema — the per-project profile.

Mirrors deputy's ``projects/<name>.yaml`` idea: the *only* place project specifics live. The
agents/stages are project-agnostic; adding a new modeling problem means writing one YAML file.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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


class SourceConfig(BaseModel):
    """Where the dataset comes from (datasources.py). ``file`` reads a local or uploaded file;
    ``sqlite`` and ``snowflake`` read a table or a read-only query; a plugin may add more kinds.

    ``options`` are the connector's settings (account, warehouse, database, schema, role, …).
    Never a secret: a value written ``env:NAME`` is read from the environment when the data is
    fetched, and passwords are only ever taken from the environment.
    """

    kind: str = "file"
    path: str | None = None  # file: the data file; sqlite: the database file
    table: str | None = None
    query: str | None = None  # a single read-only SELECT (instead of table)
    limit: int | None = None  # row cap applied by the engine
    options: dict[str, str] = Field(default_factory=dict)
    # From the data catalog (catalog.py, ADR-0020): the table's id, what it is for, and what its
    # columns mean, copied when the data is given so the run keeps them if the catalog changes.
    catalog: str | None = None
    description: str = ""
    columns: dict[str, str] = Field(default_factory=dict)
    keys: list[str] = Field(default_factory=list)  # the columns that identify a row
    outcomes: list[str] = Field(default_factory=list)  # outcome columns, never features


class NamedSource(SourceConfig):
    """One of several inputs (``data.sources``). ``name`` is how the join plan, the column
    origins and the white paper refer to it."""

    name: str

    @field_validator("name")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", v or ""):
            raise ValueError(f"a source name is letters, digits and underscores, got {v!r}")
        return v


class JoinStep(BaseModel):
    """One stated join: add ``right`` to what is already joined, matching ``left_on`` (columns
    of ``left``, the base table or one added earlier) to ``right_on``."""

    left: str
    left_on: list[str]
    right: str
    right_on: list[str]
    # What to do when a key matches several rows of ``right``: one row per base row is kept.
    many: Literal["aggregate", "first", "refuse"] = "refuse"
    # A base row with no match: keep it with missing values, or drop it (counted, reported).
    unmatched: Literal["keep", "drop"] = "keep"
    note: str = ""  # why the link exists, in the data owner's words (a documented link)

    @field_validator("left_on", "right_on", mode="before")
    @classmethod
    def _one_or_many(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _same_width(self) -> JoinStep:
        if not self.left_on or len(self.left_on) != len(self.right_on):
            raise ValueError("left_on and right_on must name the same number of columns")
        return self


class DataConfig(BaseModel):
    path: str | None = None  # CSV/Parquet path; may be None when a DataFrame is passed in code
    format: str = "csv"  # csv | parquet
    source: SourceConfig | None = None  # a connector instead of path (file | sqlite | snowflake)
    # Several inputs of any kind (a Snowflake table and a CSV, ...). The engine profiles how
    # they connect, the Data Analyst proposes the join, the developer confirms it at the data
    # gate. ``base`` and ``join`` state the join instead (no proposal); see linking.py.
    sources: list[NamedSource] = Field(default_factory=list)
    base: str | None = None  # the table whose rows the model is built on
    join: list[JoinStep] = Field(default_factory=list)
    # Links the data catalog documents between the sources: always offered at explore and
    # measured like any other, never binding (ADR-0020).
    links: list[JoinStep] = Field(default_factory=list)
    # The catalog tables a template starts with: preselected at gate_source (ADR-0020).
    catalog: list[str] = Field(default_factory=list)
    # The dependent variable. Leave it empty and the Data Analyst proposes it from the business
    # intent; the model developer confirms it at the data gate.
    target: str = ""
    features: list[str] = Field(default_factory=list)  # empty => use all non-target columns
    datetime_col: str | None = None  # for time-series / walk-forward ordering
    drop_columns: list[str] = Field(default_factory=list)
    protected_attributes: list[str] = Field(default_factory=list)  # for fair-lending checks
    # Discrete-time hazard (survival) support: the 1-based period in which the event occurred
    # (NaN/0 for censored obligors). Outcome data — excluded from features like the target; it is
    # the label's *timing*, consumed only by the hazard families (modeling/hazard.py).
    event_time_col: str | None = None
    horizon_periods: int | None = None  # outcome window in periods; None => max observed event time
    # What data this kind of model is usually built on, in a template's words (ADR-0018). Shown
    # to the developer when the data is given; never read by the engine.
    guide: str = ""

    @property
    def given(self) -> bool:
        """Whether the profile says where the data comes from. A run started from a template
        has none until the developer gives it at gate_source."""
        return bool(self.path or self.source is not None or self.sources)

    @model_validator(mode="after")
    def _sources_consistent(self) -> DataConfig:
        names = [s.name for s in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("data.sources: every source needs its own name")
        if self.sources and (self.source is not None or self.path):
            raise ValueError("give data.sources, or one data.source / data.path, not both")
        if (self.base or self.join or self.links) and len(self.sources) < 2:
            raise ValueError("data.base, data.join and data.links need two or more data.sources")
        known = set(names)
        for name in [self.base, *(t for j in [*self.join, *self.links]
                                  for t in (j.left, j.right))]:
            if name and name not in known:
                raise ValueError(f"data.join names an unknown source {name!r}; known: {sorted(known)}")
        return self


class DesignConfig(BaseModel):
    """The modeling **design brief** — what the model sponsor (e.g. an MD in model development)
    has decided about the problem, before any algorithm is chosen.

    Ideate consumes this to ground its econometric-framework assessment; every *unanswered* field
    becomes an explicit open design question in the design brief (MD triangulation) instead of a
    silent assumption. All fields are optional free text so the engine never blocks on them.
    """

    use_case: str = ""  # e.g. origination | surveillance | CECL/IFRS9 | IRB | stress-testing
    horizon: str = ""  # outcome window, e.g. "12-month PD", "lifetime"
    default_definition: str = ""  # e.g. "90+ DPD or nonaccrual, per SR 26-2 documentation"
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


class RankRule(BaseModel):
    """One line of the ranking: a metric from the frozen registry and the sample it is read on."""

    metric: str = "auto"  # auto => metric.name
    sample: Literal["validation", "in_sample", "test"] = "validation"


class EvaluationConfig(BaseModel):
    """The test plan: how the data are divided to build, compare and confirm models
    (modeling/splits.py). A field left at ``auto`` is recommended from the data and the intent,
    proposed by the Design Lead and confirmed at the design gate (ADR-0016).

    Three samples at most: *development* (fit, and re-sampled for ``validation``), *test* (the
    candidate pool is compared on it, so it is used to choose) and a sealed *holdout* (scored
    once, for the champion, after the choice).
    """

    test: Literal["auto", "out_of_time", "random", "none"] = "auto"
    test_fraction: float | None = None  # None => search.holdout_fraction
    holdout_fraction: float | None = None  # sealed final sample; None => recommended, 0 = none
    validation: Literal["auto", "none", "kfold", "stratified_kfold", "group_kfold",
                        "rolling_origin", "sliding_window"] = "auto"
    folds: int | None = None  # None => search.cv_folds
    group_col: str | None = None  # group_kfold: rows of one group stay in one fold
    horizon: int | None = None  # rolling_origin / sliding_window: periods per validation block
    window: int | None = None  # sliding_window: training periods kept
    # How candidates are ranked, most important first. Empty => the run's metric on the
    # validation sample (in sample when there is no validation).
    rank_by: list[RankRule] = Field(default_factory=list)
    pool_size: int = 6  # candidates the engine proposes; the developer may carry any number


class FactorConfig(BaseModel):
    """The factor library and the single-factor screen (modeling/factors.py).

    ``transforms`` widens the explored columns into a library (one row per period: lags,
    differences, growth rates, moving averages; otherwise logs of skewed amounts). Every factor
    is fitted alone, and the rules below, applied in the order written, cut the library to a
    shortlist. A rule set to ``None`` is off.
    """

    transforms: list[str] = Field(default_factory=lambda: ["auto"])
    # auto | none | lag | diff | pct | yoy_diff | yoy_pct | ma | log | sqrt | square
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    windows: list[int] = Field(default_factory=lambda: [4])  # moving-average lengths
    period: int | None = None  # periods per year (yoy); None => read from the date spacing
    # What a factor is ranked on when fitted alone: the ranking metric on the validation sample
    # or in sample (auto => validation when the plan has one).
    criterion: Literal["auto", "validation", "in_sample"] = "auto"
    max_missing: float | None = 0.2
    max_p_value: float | None = 0.10  # single-factor coefficient
    enforce_sign: bool = True  # drop a factor whose sign contradicts the expected one
    expected_signs: dict[str, Literal["+", "-"]] = Field(default_factory=dict)  # by column
    min_strength: float | None = None  # the ranking metric a factor must reach on its own
    best_per_source: int | None = 2  # transformations kept per explored column
    max_correlation: float | None = 0.8  # of a correlated pair the weaker one goes
    top_n: int | None = 12
    force_in: list[str] = Field(default_factory=list)
    force_out: list[str] = Field(default_factory=list)


class SelectionConfig(BaseModel):
    """How single factors become multi-factor models (modeling/selection.py). One method or
    several; every model any of them evaluates lands in one ledger and is ranked together."""

    methods: list[Literal["auto", "grid", "forward", "backward", "stepwise",
                          "best_subset"]] = Field(default_factory=lambda: ["auto"])
    # What a step has to improve: the ranking metric on the validation sample or in sample,
    # or an information criterion (aic / bic / adj_r2, families with statsmodels inference).
    criterion: Literal["auto", "validation", "in_sample", "aic", "bic", "adj_r2"] = "auto"
    max_factors: int = 4
    must_include: list[str] = Field(default_factory=list)  # factors every model carries
    max_p_value: float | None = None  # every coefficient of an accepted model
    enforce_sign: bool = True
    max_correlation: float | None = 0.8  # between two factors of one model
    max_models: int = 400  # budget per method and family


class AnalysisConfig(BaseModel):
    """Exploratory analysis the Data Analyst may request (analysis/). Each round the agent asks
    for tool runs (built in or from a plugin) or writes Python; the engine executes them."""

    rounds: int = 2  # scouting rounds before the analyst's recommendation
    max_requests: int = 8  # analyses per round
    allow_code: bool = True  # let the agent write Python (run restricted, kept as an artifact)
    code_timeout_s: float = 30.0  # wall-clock limit per script
    stage_tools: bool = True  # offer every stage's agent the tools registered for its stage


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


# Regime ids that a later letter replaced, mapped to the one in force.
SUPERSEDED_REGIMES = {"SR11-7": "SR26-2"}


class ComplianceConfig(BaseModel):
    regimes: list[str] = Field(default_factory=lambda: ["SR26-2", "NIST-AI-RMF"])
    risk_tier: str = "medium"  # low | medium | high — drives validation intensity
    intended_use: str = ""
    out_of_scope_use: str = ""
    fair_lending: bool = False  # run ECOA/Reg B SCAN + disparate-impact checks
    disparate_impact_threshold: float = 0.8  # 4/5ths rule
    jurisdictions: list[str] = Field(default_factory=lambda: ["US"])  # US | EU

    @field_validator("regimes")
    @classmethod
    def _current_guidance(cls, v: list[str]) -> list[str]:
        """SR 26-2 (April 2026) supersedes SR 11-7: a profile or run that names the old letter
        is read as the guidance in force (ADR-0019)."""
        out = [SUPERSEDED_REGIMES.get(r, r) for r in v]
        return list(dict.fromkeys(out))


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
    """Human gates and the automatic challenge loop (engine/graph.py). ``gate_source`` is not
    listed: it is always part of a run, and closes by itself when the profile gives the data."""

    gates: list[str] = Field(default_factory=lambda: [
        "gate_intent", "gate_data", "gate_design", "gate_candidates", "gate_champion",
        "gate_validation", "gate_signoff"])
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


class TemplateInfo(BaseModel):
    """What a template is (ADR-0018): a profile with no data, a starting point for one kind of
    model. Built in (``cognos/templates/``) or saved by a team from a finished run
    (``<runs>/_templates/``). A run started from one keeps this section, so it says where it
    started."""

    id: str = ""  # the file name without .yaml
    title: str = ""  # what a person picks, e.g. "PD — logistic scorecard"
    origin: Literal["built-in", "team"] = "built-in"
    saved_by: str = ""
    saved_at: str = ""
    from_run: str = ""  # the run it was saved from


class CognosConfig(BaseModel):
    """The complete COGNOS project profile."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    task: TaskType | None = None  # None => inferred from the confirmed target column
    mode: Mode = Mode.AUTONOMOUS
    data: DataConfig = Field(default_factory=DataConfig)  # empty in a template (gate_source)
    template: TemplateInfo | None = None
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    plugins: list[str] = Field(default_factory=list)  # modules exposing register(registry)
    design: DesignConfig = Field(default_factory=DesignConfig)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    factors: FactorConfig = Field(default_factory=FactorConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
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
        # Resolve auto metric + direction from the task type (once the task is known).
        if self.task is None:
            return self
        if self.metric.name == "auto":
            self.metric.name = "roc_auc" if self.task.is_classification else "rmse"
        if self.metric.direction is None:
            maximize = {"roc_auc", "accuracy", "r2", "f1", "direction_accuracy", "average_precision"}
            self.metric.direction = (
                Direction.MAXIMIZE if self.metric.name in maximize else Direction.MINIMIZE
            )
        return self

    def with_data(self, data: dict[str, Any] | None) -> CognosConfig:
        """The profile with the data given at gate_source (``overrides.data``) in its data
        section: where the data comes from is replaced, everything else is kept."""
        if not data:
            return self
        raw = self.model_dump(mode="json")
        raw["data"] = {**raw["data"], "path": None, "source": None, "sources": [], "base": None,
                       "join": [], "links": [], **data}
        return CognosConfig.model_validate(raw)

    def with_target(self, target: str | None, task: str | None) -> CognosConfig:
        """The config once the dependent variable is decided: target and task filled where the
        profile left them open, and the metric resolved from the task."""
        if (not target or self.data.target) and (not task or self.task is not None):
            return self
        raw = self.model_dump(mode="json")
        if target and not self.data.target:
            raw["data"]["target"] = target
        if task and self.task is None:
            raw["task"] = task
        return CognosConfig.model_validate(raw)

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
