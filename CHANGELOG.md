# Changelog

All notable changes to COGNOS are documented here. Format loosely follows Keep a Changelog;
versioning is SemVer.

## [0.2.0] — 2026-06-24

Design-review outcomes (see `CONTEXT.md` and `docs/adr/0001-0007`). COGNOS is now explicitly a
two-layer system — an LLM reasoning layer that *proposes* and a deterministic engine that *disposes*
— focused on commercial model development.

### Added
- **Reasoning-driven core** (ADR-0001): LLM-driven ideation that emits engine-validated feature
  transforms, and an opt-in **LLM-guided search** (`search.guided`) where the LLM proposes the next
  experiment from the ledger and the engine keeps it only if it beats the incumbent on the frozen
  metric.
- **Safe, target-hidden transforms** (`modeling/transforms.py`, ADR-0002): AST-whitelisted expression
  executor over a features-only view; transforms cannot reference the target, persist on the champion
  + scorer, re-apply at serving, and round-trip to IMPACT derived fields.
- **Two-tier reproducibility** (ADR-0003): reasoning transcript recorded to
  `runs/<id>/reasoning/transcript.jsonl`; `ScriptedBrain` deterministic test double.
- **Credit-risk outcomes analysis** (`modeling/credit_metrics.py`, ADR-0005): Gini/KS, calibration
  (expected-vs-observed + ECE), PSI, on an out-of-time sample — now the default meaning of "backtest".
- Opt-in **GLM/econometric families** (poisson, gamma, tweedie); a synthetic **commercial** demo
  dataset with a vintage column (`cognos demo --task commercial`).

### Changed
- **Valid statistical inference** (Q6): coefficients/p-values now come from a full-rank K-1 inference
  design, decoupled from the all-K prediction pipeline (fixes the dummy-variable trap).
- **Compliance is a non-gating model-risk readiness report** (ADR-0006), never a verdict and never
  rubber-stamped; the only gates are now `validate` and `review`.
- **Single interpretable champion** deployed (ADR-0007); the ensemble is an opt-in, labelled challenger
  benchmark, never silently shipped.
- **Backtesting**: PBO / Deflated Sharpe demoted to an opt-in trading mode (`backtest.returns_column`).
- **Scope** (ADR-0004): primary domain is commercial model development; consumer fair lending is an
  optional, off-by-default module.

## [0.1.0] — 2026-06-22

Initial release: the full COGNOS system.

### Added
- **Eight-stage pipeline**: `explore`, `ideate`, `model`, `backtest`, `validate`, `comply`,
  `document`, `review`, each a single-responsibility agent with a uniform `run(ctx) -> StageResult`
  interface.
- **Mechanical orchestrator** with two operating modes (autonomous and interactive/stage-by-stage),
  gate handling, per-stage checkpointing, and resume-from-failure.
- **Pluggable brain**: deterministic `HeuristicBrain` (default, offline) and optional Claude
  `LLMBrain` that degrades gracefully.
- **Modeling core**: CASH ratchet search, leakage-safe nested CV, frozen substrate (sealed holdout +
  metrics), statsmodels inference for linear families, ML fitters, Caruana ensembling.
- **Statistical diagnostic battery** (heteroskedasticity, autocorrelation, normality, linearity,
  multicollinearity, stationarity).
- **IMPACT integration**: model embedded as an `EntityPipeline` derived field, with a built-in
  fallback scorer.
- **Backtest analytics**: Probability of Backtest Overfitting (CSCV) and Deflated Sharpe Ratio.
- **Independent validation** (SR 11-7 effective challenge) with a five-axis rubric.
- **Compliance**: SR 11-7 × NIST AI RMF × trustworthy-AI, ECOA/Reg B fair-lending disparate-impact
  scan, reason codes, model inventory, EU AI Act Annex IV flagging.
- **Documentation**: white paper as an OKF v0.1 bundle + Google Model Card + EU Annex IV, with
  docs↔code link anchors.
- **Consistency review**: AST-based docs↔code drift detection over the OKF graph.
- **CLI** (`run`, `run-stage`, `demo`, `init`, `explain`, `report`, `list-runs`, `agents`) and Python
  API (`run_pipeline`, `Orchestrator`).
- **Claude-Code-native agent layer** (`.claude/`), per-project profiles (`projects/`), and an eval
  harness (`evals/`).
- Synthetic data generators, a runnable end-to-end example, and a unit + integration test suite.
