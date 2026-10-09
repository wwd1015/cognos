# CLAUDE.md — principles for working on COGNOS itself

These are binding design principles when editing COGNOS. They keep the system coherent, testable, and
trustworthy. v1.0 architecture: **agents recommend, humans decide, the engine disposes** (ADR-0010).

## Non-negotiables
1. **Always offline-runnable.** Every stage runs with the deterministic (heuristic) agents — no key, no
   network. Tests pin `COGNOS_PROVIDER=heuristic` (autouse fixture); a missing key, SDK or CLI makes a
   provider *unavailable*, never a crash (`auto` falls back to heuristic).
2. **Determinism in the plumbing, judgment in the agents, decisions with humans.** The engine
   (`engine/`) stays mechanical: sequence, checkpoint, mark stale, pause at gates, apply decisions,
   route challenges, escalate. Judgment enters only through `ctx.recommend(agent, ...)`. Don't add a
   "smart supervisor."
3. **No LLM math.** Every recorded number comes from the engine. Agents cite `facts` by id; the
   writer uses `{{fact:<id>}}` placeholders the engine renders. Unknown fact ids and bare metric
   numbers are rejected by `agents/checks.py`.
4. **Heavy artifacts by reference.** Datasets, fitted models, tables, and the OKF bundle live on disk
   under the run dir; `StageResult.payload` carries only small structured data. A stage in a fresh
   process must reconstruct everything from disk (`state.json` + `stages/*/result.json`).
5. **The frozen substrate stays frozen.** The sealed holdout (`data/holdout.parquet`) and metric
   definitions (`modeling/metrics.py`) are never agent-editable. The modeler chooses the champion
   from the admissible set *before* the holdout is scored and never sees holdout facts; every holdout
   evaluation is counted (`holdout_evaluations`).
6. **Independent challenge.** `validate`, `comply`, and `review` re-derive risk from artifacts. The
   validator's slice never contains the modeler's rationale (`agents/slices.py` scopes).
7. **Gates BLOCK only on high-confidence harm.** Verdict gates: `validate` (confirmed target
   leakage) and `review` (stale docs↔code references). Agents can raise FAIL/WARN, never BLOCK, and a
   BLOCK can never be *accepted* at a human gate — only sent back or rejected. Compliance is a
   non-gating report (ADR-0006). Noisy signals (e.g. PBO) are WARN/FAIL, not BLOCK.
8. **Tests + lint must pass.** `pytest` green and `ruff check src/ tests/` clean before any commit.
   All file I/O passes `encoding="utf-8"` (Windows); checkpoint/state writes go through
   `fsutil.atomic_write`. For local end-to-end verification with real agents, run the opt-in live
   test through `claude -p`: `COGNOS_LIVE=1 pytest tests/live` (skipped by default).

## The workflow (engine/graph.py)
```
explore → gate_data → ideate → gate_design → model → gate_champion → backtest → validate
        → gate_validation → comply → document → review → gate_signoff
```
`RunState` (`runs/<id>/state.json`) holds step statuses, gate decisions, challenges, gaps (sponsor
questions) and overrides. Human decisions change the *effective* config through overrides (design
answers → `ctx.config.design`; exclusions → `ctx.profile()`; slate → model families; champion →
`overrides.champion`) — the profile YAML is never edited. A changed decision marks the downstream
steps `stale`; a step invalidated while running stays stale when it finishes.

Seats (`engine/process.py`): developer on data, design and champion; reviewer on validation;
approver on sign-off. The engine refuses a seat on another seat's gate. Use, horizon, default
definition and segment cannot be assumed, and `approve` refuses while any of them is open.
`approve` writes `packages/vN.json` once (a digest of recorded facts). A later invalidation
supersedes the live pointer and does not rewrite the file. Autonomous acceptance is seat
`express` — preparation, not a signature, and it does not seal a package. The record
(`process.journal`) is computed on read from stamps the run already carries.

Every `state.invalidate(steps, why)` must say **why** (use `gates.why(...)` for a gate decision): the
reason is stored on each stale step (`StepState.rerun_reason`), kept through the re-run, and shown to
the developer. Before a stage runs again the engine keeps its last result as `result.prev.json`.

Comparisons (`compare.py`: run vs run, re-run vs previous) are mechanical reads of recorded results:
no judgment, nothing stored, and "better" only where the metric has a direction (`_HIGHER` / `_LOWER`,
or the run's own metric direction). Exports (`export.py`) never include `data/`, `models/` or binary
caches — the sealed holdout stays sealed; a new artifact type that should travel must be a text format.

## Stage contract
A stage is a `Stage` subclass with `name`, `requires`, `is_gate`, and `run(ctx) -> StageResult`. It
reads prior outputs via `ctx.require(<stage>).payload` (explore's via `ctx.profile()`), writes
artifacts under `runs/<id>/stages/<name>/`, and obtains judgment only via
`ctx.recommend(agent, data, fresh={stage: provisional_result}, check=...)`, recording it with
`attach_recommendation(payload, ctx, agent, out)`. Questions for the sponsor go in
`payload["questions"]` (the engine syncs them to gaps). Register with `@register_stage` and add to
`stages/__init__._STAGE_MODULES`.

## Agents (agents/)
Each agent = a role prompt (`prompts/<agent>.md` + `_common.md`), an output contract
(`contracts.py`, `extra="forbid"`, strict-grammar-friendly types — no free-form maps), a slice scope
(`slices.FACT_SCOPE`), engine checks (`checks.py`), and a deterministic implementation
(`heuristic.py`, a pure function of the same slice). Adding or changing an agent means touching all
five, and the heuristic must pass the checks (the tests hold it to that). Contracts carry
`responses_to_challenges`; the default check requires one per open challenge.

Providers (`providers.yaml`): `heuristic`, `replay` (recorded outputs, heuristic fallback),
`claude_cli` (`claude -p`, isolated: empty cwd, `--setting-sources ""`, `--tools ""`,
`--strict-mcp-config`, prompt on stdin), `anthropic` (structured outputs, adaptive thinking,
server-side refusal fallbacks), `openai_compat` (`submit_answer` function). The runner audits every
attempt to `runs/<id>/agents/` and enforces time and spend limits.

## Extending COGNOS
- **New model family** → estimator branch in `modeling/fit.py::_estimator`, grid in
  `modeling/search.py::_hp_grid`, `DEFAULT_FAMILIES` if defaulted, and `stages/ideate.py::_engine_families`
  so the design lead may propose it. Survival/structural/simulation extension points: ADR-0008.
- **New metric** → `modeling/metrics.py::score` (+ `MAXIMIZE`); expose it as a fact in `agents/facts.py`
  if agents should cite it.
- **New statistical test** → `stages/stat_tests.py` with H0, severity, and a `_safe` wrapper.
- **New gate action** → `engine/gates.py::ACTIONS` + handler + a UI control in `ui/panels.py`.
- **New compliance regime** → extend `stages/comply.py`; emit evidence `document`/`review` can trace.
- **New stage** → single-responsibility; wire `requires`, a graph node, and (optionally) an agent.
- **New stage metric** → if it has a good direction, add it to `compare._HIGHER` / `_LOWER` and give it a
  label in `compare.METRIC_LABELS`; otherwise comparisons report it as a difference without a verdict.

## Integrations
- **IMPACT** is optional. `integrations/impact_adapter.py` prefers the real `EntityPipeline` and falls
  back to the built-in scorer on any error, recording `used_impact`. Never hard-depend on it.
- **OKF** is the documentation substrate. Producers permissive, consumers tolerant.

## Front ends
`service.py` is the only boundary the CLI (`cli.py`) and the Dash + Mantine workbench (`ui/`) use.
UI panels are pure functions of (result, state, scheme, seat) — keep them testable without a browser
(`tests/ui`). Never write `component or fallback`: Dash components define `__len__`, so a childless
component is falsy — use `graph(fig, placeholder)` / explicit `is None` checks. Charts follow the
reference palette in `ui/theme.py` (single y-axis, legends for ≥ 2 series, status colors only with an
icon + label).
