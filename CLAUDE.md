# CLAUDE.md — principles for working on COGNOS itself

These are binding design principles when editing COGNOS. They keep the system coherent, testable, and
trustworthy.

## Non-negotiables
1. **Always offline-runnable.** Every stage must work with the default `HeuristicBrain` (no API key).
   LLM usage is additive only: branch `if ctx.brain.available: … else: <deterministic>` and never let
   a missing key or SDK change a verdict or break a run.
2. **Determinism in the plumbing, judgment in the agents.** The orchestrator and hooks must stay
   mechanical (sequence, checkpoint, honor gates, escalate). Don't add a "smart supervisor."
3. **Heavy artifacts by reference.** Datasets, fitted models, tables, and the OKF bundle live on disk
   under the run dir; `StageResult.payload` carries only small structured data. A downstream stage in a
   fresh process must be able to reconstruct everything from disk.
4. **The frozen substrate stays frozen.** The sealed holdout (`data/holdout.parquet`) and metric
   definitions (`modeling/metrics.py`) must not be editable by the search/idea stages. This is the
   core anti-reward-hacking guarantee.
5. **Independent challenge.** `validate`, `comply`, and `review` must not reuse the modeling stage's
   reasoning; they re-derive risk from artifacts. Keep them separate.
6. **Gates BLOCK only on high-confidence harm.** Reserve `BLOCK` for confirmed target leakage
   (`validate`), disparate-impact violations (`comply`), and stale docs↔code references (`review`).
   Noisy signals (e.g. PBO) are WARN/FAIL, not BLOCK.
7. **Tests + lint must pass.** `pytest` green and `ruff check src/ tests/` clean before any commit.

## Stage contract
A stage is a `Stage` subclass with `name`, `requires`, `is_gate`, and `run(ctx) -> StageResult`. It
reads prior outputs via `ctx.require(<stage>).payload`, writes artifacts under
`runs/<id>/stages/<name>/`, sets `verdict`, `summary`, `metrics`, `payload`, `findings`, `artifacts`.
Register it with `@register_stage` and add it to `stages/__init__._STAGE_MODULES`.

## Extending COGNOS
- **New model family** → add an estimator branch in `modeling/fit.py::_estimator`, a hyperparameter
  grid in `modeling/search.py::_hp_grid`, and (if it's a defaulted family) `DEFAULT_FAMILIES`.
- **New metric** → add it to `modeling/metrics.py::score` and, if higher-is-better, to `MAXIMIZE`.
- **New statistical test** → add it to `stages/stat_tests.py` with its H0, severity, and a `_safe`
  wrapper so an inapplicable test is `skipped`, never a crash.
- **New compliance regime / jurisdiction** → extend `stages/comply.py`; emit structured evidence the
  `document` stage can render and the `review` stage can trace.
- **New stage** → keep it single-responsibility; subtract tools rather than add; wire `requires`.

## Integrations
- **IMPACT** is optional. `integrations/impact_adapter.py` must prefer the real `EntityPipeline` and
  fall back to the built-in scorer on any error, recording `used_impact`. Never hard-depend on it.
- **OKF** is the documentation substrate. Keep producers permissive and consumers tolerant (only
  `type` is required; broken links are tolerated but reported).

## The agent layer (`.claude/`)
The declarative agents + orchestrator command mirror the engine. If you change the CLI surface or a
stage's outputs, update the corresponding `.claude/agents/<stage>.md` and `commands/cognos-run.md`.
