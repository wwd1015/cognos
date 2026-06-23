# Changelog

All notable changes to COGNOS are documented here. Format loosely follows Keep a Changelog;
versioning is SemVer.

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
