# ADR-0008 — Econometric core: survival, structural, and portfolio simulation live in the engine

Date: 2026-07-04 (v0.4.0)
Status: accepted

## Context

The sponsor's direction for COGNOS's commercial-risk depth is **traditional regression and
simulation-based methods close to the structural model** — not further ML breadth. Three method
families were added: discrete-time hazard (survival), the Merton structural model, and
portfolio/stress simulation. Each poses a design question about where it sits relative to COGNOS's
non-negotiables (offline-runnable, determinism in the plumbing, frozen substrate, single
interpretable champion).

## Decisions

### 1. Hazard models panel-expand *inside* the estimator; event timing is label data

A discrete-time hazard model (Shumway 2001) is fit on obligor-period panel rows, but naive panel
construction before cross-validation lets one obligor's periods straddle a fold boundary — the
classic survival-CV leak. We therefore:

- keep the **obligor** as the unit the search machinery sees: CV folds, the sealed holdout, and the
  scorer contract all operate on obligor rows;
- perform the panel expansion **inside** `hazard_fit_predict` / `fit_full_hazard`
  (`modeling/hazard.py`), per training fold only;
- treat `data.event_time_col` as the **label's timing**: it is outcome data, excluded from features
  exactly like the target (datautil), and travels through the search frame only as metadata that
  hazard candidates may consume;
- serve through a `HazardScorer` that computes cumulative PD over the horizon from obligor features
  alone, satisfying the existing `ScorerBundle.pipeline` contract (backtest/IMPACT unchanged);
- report the **PD term structure** as a first-class payload (`hazard.term_structure`) because the
  term structure — not the single-horizon PD — is what lifetime (CECL/IFRS 9) use cases need.

Inference runs as a statsmodels GLM (logit or cloglog link — cloglog being the grouped-time
proportional-hazards link) on a full-rank K-1 panel design, so period baseline-hazard effects carry
valid p-values.

### 2. The structural model is a solver, not a fit; hybrid is the default use

Merton distance-to-default is computed by a deterministic KMV fixed-point solver
(`modeling/structural.py`) from market observables. Because it involves no fitting and no
randomness, it belongs to the **engine** ("the engine disposes"):

- **Hybrid mode (default)**: `merton_dd` becomes an engineered feature feeding the reduced-form
  champion — the industry (RiskCalc-style) pattern. It is recomputed **target-hidden at serve
  time** on both `FittedModel` and `ScorerBundle`, the same contract as LLM-authored transforms.
- **Pure structural PD** (= N(−DD)) is scored on the sealed holdout only as a **labelled challenger
  benchmark** (`structural.benchmark`, `deployed: false`) — ADR-0007's single-champion rule is
  unchanged.
- When market observables exist but the `structural:` block is off, ideate marks the framework
  **available** and raises an unlock question; when they don't exist (private obligors), the
  framework stays **rejected with a stated reason** — that recorded rejection is SR 11-7
  "alternatives considered" evidence.

### 3. Simulation is reported, never selected on

Vasicek one-factor portfolio losses (+ closed-form Basel IRB capital) and macro-scenario stress
re-scoring (`modeling/simulate.py`) are **reports the backtest stage emits**, conditioned on the
champion's PDs. They never enter champion selection, so the frozen metric / sealed holdout
guarantees are untouched. All simulation is **seeded** (bit-reproducible) and the stress path is
fully deterministic (covariate shocks re-scored through the deployed scorer). Scenario shocks
against unknown columns are surfaced as findings, never silently dropped. PDs in are model scores;
the reports sit next to the calibration section for exactly that reason.

## Consequences

- The default classification slate widens to logit/probit/cloglog (+ hazard families when event
  timing exists); all are interpretable GLMs, so the interpretability-first ordering is preserved.
- The scorer bundle gains two optional recompute steps (structural spec, transforms) but its
  external contract is unchanged.
- A hazard champion's diagnostics run on obligor-level residuals; panel-level design is deliberately
  not exposed to the generic battery (inapplicable tests skip rather than mislead).
- Portfolio/stress numbers inherit the model's calibration errors — documented as a standing caveat
  in the backtest skill pack and payload notes.
