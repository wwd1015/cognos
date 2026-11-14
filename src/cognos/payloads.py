"""The declared shape of the two most-read stage payloads: ``model`` and ``backtest``.

Stages hand each other plain dictionaries (``StageResult.payload``) and keep doing so: nothing
here is used at run time. This module is a contract the tests hold both sides to
(``tests/unit/test_payload_shapes.py``): every key a writer records must be declared here, and
every key a reader asks for must be declared here, so a key that one side stops writing and the
other keeps asking for fails a test instead of reading ``None`` for months.

Every field is optional with a default, so results recorded by older versions still load, and
``extra="allow"`` keeps whatever an older run carried. Deep diagnostics stay ``dict[str, Any]``
on purpose: only the shapes read in many places (a pool member, a backtest candidate row) are
spelled out.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Shape(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- the model stage ------------------------------------------------------------------------
class PoolMember(_Shape):
    """One candidate entry: a member of ``pool`` (finalized: scorer, detail, diagnostics) or of
    ``leaders`` (the ranked shortlist the Modeling agent chooses from)."""

    # who it is
    id: str = ""
    name: str | None = None
    label: str | None = None
    family: str | None = None
    features: list[str] = Field(default_factory=list)
    n_features: int | None = None
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None  # how it entered the pool: engine, agent or human
    # where it stands in the ranking and how it got there
    rank: int | None = None
    methods: list[str] = Field(default_factory=list)
    role: str | None = None
    interpretable: bool | None = None
    engine_pick: bool | None = None
    within_one_se: bool | None = None
    # its scores (``scores`` is {sample: {metric: value}}; ``score`` the first ranking line)
    scores: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    # what the plan's rules found
    coefficient_signs: list[dict[str, Any]] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    max_p_value: float | None = None
    transforms: list[dict[str, Any]] = Field(default_factory=list)  # leaders only
    n_transforms: int | None = None
    # the statistical tests run on a finalized member
    diagnostics_run: int | None = None
    diagnostics_passed: int | None = None
    diagnostics_failed: list[str] = Field(default_factory=list)
    diagnostics_worst: str | None = None
    # on disk, relative to the run (read through pool.scorer_path / pool.detail)
    scorer_path: str | None = None
    detail_path: str | None = None


class ModelPayload(_Shape):
    """What the model stage records: a plan, a factor library, a screen, a ranking and a pool.
    Never a champion (ADR-0016): that is laid over it by ``pool.view`` (see ``ChampionView``)."""

    # the development plan
    plan: dict[str, Any] = Field(default_factory=dict)
    plan_summary: str | None = None
    samples: dict[str, Any] = Field(default_factory=dict)
    # the factor library and the single-factor screen
    library: dict[str, Any] = Field(default_factory=dict)
    transforms: dict[str, dict[str, Any]] = Field(default_factory=dict)  # {column: form}
    single_factor: list[dict[str, Any]] = Field(default_factory=list)
    screen: list[dict[str, Any]] = Field(default_factory=list)
    exceptions: list[dict[str, Any]] = Field(default_factory=list)
    shortlist: list[str] = Field(default_factory=list)
    expected_signs: dict[str, Any] = Field(default_factory=dict)
    raw_features: list[str] = Field(default_factory=list)
    factors: list[dict[str, Any]] = Field(default_factory=list)
    # model rules, ranking and the selection paths
    model_funnel: dict[str, Any] = Field(default_factory=dict)
    rank_mode: str | None = None
    rank_by: list[dict[str, Any]] = Field(default_factory=list)
    rank_metric: str | None = None
    blend: dict[str, Any] = Field(default_factory=dict)
    paths: list[dict[str, Any]] = Field(default_factory=list)
    metric: str | None = None
    direction: str | None = None
    best_id: str | None = None
    best_score: float | None = None
    # the ranked leaders and the pool
    leaders: list[PoolMember] = Field(default_factory=list)
    pool: list[PoolMember] = Field(default_factory=list)
    pool_ids: list[str] = Field(default_factory=list)
    pool_source: str | None = None
    evaluated_ids: list[str] = Field(default_factory=list)
    names: dict[str, str] = Field(default_factory=dict)
    ranking_path: str | None = None
    # the search and its looks at the test sample
    search_fingerprint: str | None = None
    n_candidates_tried: int | None = None
    n_configs_for_deflation: int | None = None
    n_search_strategies: int | None = None
    test_evaluations: int = 0
    oof_perf_path: str | None = None
    # engineered and exploratory parts (present only when the profile asks for them)
    challenger_benchmark: dict[str, Any] | None = None
    structural: dict[str, Any] | None = None
    migration: dict[str, Any] | None = None
    guided: dict[str, Any] | None = None
    # the task and the sample sizes
    is_classification: bool | None = None
    is_timeseries: bool | None = None
    n_train: int | None = None
    n_test: int | None = None
    n_holdout: int | None = None
    # attach_recommendation (stages/base.py)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    tool_runs: list[dict[str, Any]] = Field(default_factory=list)
    tools_unavailable: list[Any] = Field(default_factory=list)


class ChampionView(ModelPayload):
    """``pool.view`` / ``ctx.champion()``: the model payload with the champion in force laid
    over it under the names later stages read. ``transforms`` changes meaning here: the
    champion's own transformations (a list), not the columns' forms (a dict)."""

    # the champion in force and how it was chosen
    champion: dict[str, Any] = Field(default_factory=dict)  # family, features, hyperparams ...
    champion_id: str | None = None
    champion_label: str | None = None
    champion_name: str | None = None
    champion_source: str | None = None
    champion_decided: bool = False
    admissible_set: list[PoolMember] = Field(default_factory=list)
    # its scores during development
    cv_mean: float | None = None
    cv_std: float | None = None
    cv_fold_scores: list[float] = Field(default_factory=list)
    development_score: float | None = None
    scores: dict[str, Any] = Field(default_factory=dict)
    # its fitted record (pool.detail)
    coefficients: dict[str, Any] | None = None
    pvalues: dict[str, Any] | None = None
    feature_importances: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    hazard: dict[str, Any] | None = None
    transforms: list[dict[str, Any]] = Field(default_factory=list)  # type: ignore[assignment]
    base_features: list[str] = Field(default_factory=list)
    scorer_path: str | None = None
    # the samples it was scored on after the choice
    test_metric: float | None = None
    holdout_metric: float | None = None
    holdout_evaluations: int = 0


# --- the backtest stage ---------------------------------------------------------------------
class BacktestCandidate(_Shape):
    """One pool member as outcomes analysis measured it on the comparison sample."""

    # who it is (copied from the pool member)
    id: str = ""
    name: str | None = None
    label: str | None = None
    family: str | None = None
    n_features: int | None = None
    features: list[str] = Field(default_factory=list)
    role: str | None = None
    interpretable: bool | None = None
    methods: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    # development scores, and what it scored on the comparison sample
    development: dict[str, Any] = Field(default_factory=dict)
    development_score: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    outcomes: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    error: str | None = None  # a member that could not be scored
    # stress scenarios, when the profile has them
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    scenario_baseline: float | None = None
    scenarios_error: str | None = None


class BacktestPayload(_Shape):
    """What the backtest stage records: the pool compared on one sample, the champion
    recommended (or chosen at the gate) and its outcomes analysis."""

    # the comparison of the pool
    evaluation_sample: str | None = None
    candidates: list[BacktestCandidate] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)  # metric, direction, sample, n_*
    # the champion in force
    champion_id: str | None = None
    champion_label: str | None = None
    champion_name: str | None = None
    champion_source: str | None = None
    champion_rationale: str | None = None
    recommended_id: str | None = None
    # looks at the samples
    test_evaluations: int = 0
    sealed_holdout: bool | None = None
    n_trials: int | None = None
    # the champion's analysis (each part present only when the task and profile allow it)
    oos_metric: float | None = None
    oos_metric_name: str | None = None
    used_impact: bool | None = None
    impact_note: str | None = None
    scored_rows: int | None = None
    forecasts_path: str | None = None
    outcomes_analysis: dict[str, Any] | None = None
    forecast_analysis: dict[str, Any] | None = None
    portfolio_analysis: dict[str, Any] | None = None
    stress_testing: dict[str, Any] | None = None
    walk_forward: dict[str, Any] | None = None
    trading: dict[str, Any] | None = None
    interpretation: dict[str, str] = Field(default_factory=dict)
    # attach_recommendation (stages/base.py)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    tool_runs: list[dict[str, Any]] = Field(default_factory=list)
    tools_unavailable: list[Any] = Field(default_factory=list)


SHAPES: dict[str, type[BaseModel]] = {"model": ModelPayload, "backtest": BacktestPayload}
# What readers see that no stage records as such: ``ctx.champion()`` / ``pool.view``.
VIEWS: dict[str, type[BaseModel]] = {"champion": ChampionView}


def declared(stage: str) -> frozenset[str]:
    """The top-level keys declared for a stage's payload (or for a view, ``"champion"``)."""
    shape = SHAPES.get(stage) or VIEWS[stage]
    return frozenset(shape.model_fields)
