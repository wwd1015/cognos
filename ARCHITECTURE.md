# COGNOS architecture

## Two faces

COGNOS is both:

1. **An engine** — a tested Python package with a *pluggable brain* per stage. Each stage has solid
   deterministic logic and an optional Claude-backed brain for the parts that benefit from open-ended
   reasoning (idea narratives, judgment, prose). The deterministic path makes the whole system
   runnable and testable offline; the LLM path makes it a richer agent in production. This is the
   single seam (`ctx.brain.available`) that reconciles "comprehensive automated tests" with "a real
   multi-agent AI system."

2. **A Claude-Code-native agent layer** — declarative agent specs in `.claude/agents/*.md`, a
   mechanical orchestrator slash command, PreToolUse safety hooks, and per-project YAML profiles, in
   the spirit of `deputy`. The agent layer drives the engine through the `cognos` CLI.

## Lineage (what we borrowed and from where)

| Source | Pattern adopted |
|--------|-----------------|
| `deputy` | Sequential pipeline of single-responsibility declarative agents; a *mechanical* orchestrator (not a clever supervisor); **load-bearing on-disk artifacts** + greppable **verdict tokens** as the inter-stage contract; per-project YAML; PreToolUse hooks as deterministic backstops; an eval harness; checkpoint/resume. |
| `autoforge` | The `name: value` stdout + `results.tsv` ledger optimization protocol, reimplemented in `integrations/autoforge_loop.py`; accept-if-better-else-discard ratchet. |
| `autoresearch` (Karpathy) | The ratchet hill-climb; the **frozen-substrate / mutable-surface** split (sealed metric + holdout the search can't touch); fixed-budget experiments; TSV experiment ledger. |
| `IMPACT` | In-process integration via `EntityPipeline`: the model is embedded as a **derived field** (`cognos.runtime.score.score_row`) so IMPACT builds a standardized scored feature table. |
| Prior-art research | Orchestrator-worker with a sequential dependent pipeline; interrupt/checkpoint/resume for HITL; independent critic/validator agents (avoid "degeneration of thought"); CASH search + nested CV + Caruana ensembling; statistical diagnostic battery; PBO/Deflated-Sharpe; SR 11-7 × NIST AI RMF; Model Cards + EU Annex IV; OKF docs + AST drift detection. |

## Control flow

The orchestrator (`orchestrator.py`) is deliberately mechanical: it sequences stages in the legal
lifecycle order, persists each `StageResult` to disk, and honors gate verdicts. Judgment lives in the
stage agents; determinism lives in the orchestrator and the safety hooks.

```
for stage in enabled:
    result = stage.run_guarded(ctx)     # times + crash-isolates; missing deps -> ERROR
    ctx.record(result)                  # checkpoint to runs/<id>/stages/<stage>/result.json
    if stage is a gate and verdict not OK:
        autonomous : BLOCK halts (if halt_on_block); FAIL/OPEN_QUESTIONS recorded, run continues
        interactive: gate_handler(result) -> approve | reject (reject halts)
    if verdict == ERROR: halt
```

Because every stage checkpoints, a second run with the same `run_id` reuses completed stages
(resume-from-failure), and any single stage can be re-run in a fresh process via `cognos run-stage`.

## The stage contract

Every agent is a `Stage` with one method, `run(ctx) -> StageResult`. It reads inputs from prior
stages through the `RunContext` and writes its outputs as artifacts under `runs/<id>/stages/<stage>/`.

- `StageResult`: `stage`, `verdict` (PASS/WARN/FAIL/BLOCK/OPEN_QUESTIONS/ERROR/SKIP), `summary`,
  `metrics{}`, `payload{}` (the structured hand-off), `findings[]`, `artifacts[]`.
- **Heavy artifacts are passed by reference** (paths under the run dir), never inlined — datasets,
  fitted models, result tables, and the OKF bundle all live on disk; `payload` carries only small
  structured data. This is what lets a downstream stage in a fresh process reconstruct everything.
- Verdict tokens are greppable (`COGNOS_STAGE: model VERDICT: WARN FINDINGS: 1`) for the agent layer.

## Run directory

```
runs/<run_id>/
  manifest.json            # run metadata + per-stage verdicts
  summary.json summary.txt # machine- and human-readable run summary (final + per-stage verdicts)
  data/   dataset.parquet train.parquet holdout.parquet   # sealed holdout lives here
  models/ champion_scorer.joblib                           # deployable scorer (IMPACT entry point)
  docs/   *.md index.md log.md                             # the OKF white-paper bundle
  stages/<stage>/result.json + artifacts (profile.json, ledger.tsv, diagnostics.json, oof_perf.npz,
                                          scored.parquet, impact_entity.yaml, compliance.json, …)
```

## The pluggable brain

`brains/base.py` defines `Brain` with `available: bool` and `generate()/judge()`. The default
`HeuristicBrain` is `available=False`, so stages take their deterministic path. `LLMBrain` (Claude
via the `anthropic` SDK) activates only when the SDK and `ANTHROPIC_API_KEY` are present, and degrades
gracefully to the heuristic brain otherwise. Stages always branch `if ctx.brain.available: … else: …`
and never *require* the LLM — additive only.

## Key design decisions

- **PBO is computed over the full search library**, not a hand-picked few, and counts overfitting
  strictly (a tied/equally-good library yields ~0, not a false alarm). It is an informational signal;
  the **CV-vs-sealed-holdout gap** is the reliable overfit trigger, and **confirmed leakage** is the
  only thing the validation gate hard-BLOCKs on.
- **Protected attributes are excluded from model features** by default (disparate-treatment
  avoidance) but retained for the compliance stage's disparate-impact test.
- **IMPACT is optional**: the adapter prefers the real `EntityPipeline` and falls back to the
  built-in scorer transparently, recording which path ran (`used_impact`).
- **OKF over a bespoke format**: a permissive, vendor-neutral, agent-readable markdown spec where
  documentation, code, data and results are all cross-linked nodes the `review` gate can traverse.
