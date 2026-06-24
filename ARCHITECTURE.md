# COGNOS architecture

## Two layers

COGNOS is a two-layer system in which an LLM **reasoning layer** and a deterministic **engine** are
both first-class and interdependent ([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)):

1. **The reasoning layer proposes.** An LLM-driven decision layer makes the calls a human modeler
   would make — design, model choice, feature engineering, and the next experiment to try. It only
   ever *proposes* (specs, transforms, hypotheses, prose); it never decides what is good.

2. **The deterministic engine disposes.** The non-LLM core fits models, scores them on a frozen
   metric and sealed holdout, runs the statistical battery, and persists artifacts. It is the **sole
   authority** on what is kept and on any metric or verdict that becomes a recorded fact. This
   determinism is the anti-hallucination mechanism: a proposal cannot enter the record unless the
   engine independently measures that it improves on the auditable yardstick.

Neither layer is optional. Running with no LLM is the deterministic substrate + test double (a
`ScriptedBrain` test double exists), not a degraded product — it is how the deliverable's analysis is
always reproduced and served ([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)).

COGNOS also ships a **Claude-Code-native agent layer** — declarative agent specs in
`.claude/agents/*.md`, a mechanical orchestrator slash command, PreToolUse safety hooks, and
per-project YAML profiles, in the spirit of `deputy`. The agent layer drives the engine through the
`cognos` CLI.

## Reasoning-driven loop

The reasoning layer enters in two staged depths
([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)):

- **(A) LLM-driven ideation** emits *executable* feature-engineering transforms (not just prose).
- **(B) Opt-in LLM-guided search** (`search.guided`) where the LLM is the mutation function proposing
  the next experiment from the experiment ledger. Deterministic grid search is the baseline and the
  test double.

LLM-authored transforms run **target-hidden**
([ADR-0002](docs/adr/0002-llm-authored-transforms-safe-execution.md)): a safe, AST-whitelisted
expression executor (`modeling/transforms.py`) evaluates them over a features-only view (`X`) plus a
fixed set of `np.<fn>` functions. The target `y` is never in scope, so a transform physically cannot
leak the target — even onto the labelled holdout. A kept transform round-trips verbatim to an IMPACT
derived field, is persisted on the champion and the deployed scorer, and is re-applied target-hidden
at serving.

The two-tier reproducibility split makes this auditable
([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)): the **analysis** (every number the engine
computes) is fully bit-reproducible offline with no LLM, while the **reasoning trajectory** is
non-deterministic, recorded to `runs/<id>/reasoning/transcript.jsonl` for replay/audit, and
human-gated. The LLM is required to automate the search, never to reproduce the result.

## Lineage (what we borrowed and from where)

| Source | Pattern adopted |
|--------|-----------------|
| `deputy` | Sequential pipeline of single-responsibility declarative agents; a *mechanical* orchestrator (not a clever supervisor); **load-bearing on-disk artifacts** + greppable **verdict tokens** as the inter-stage contract; per-project YAML; PreToolUse hooks as deterministic backstops; an eval harness; checkpoint/resume. |
| `autoforge` | The `name: value` stdout + `results.tsv` ledger optimization protocol, reimplemented in `integrations/autoforge_loop.py`; accept-if-better-else-discard ratchet. |
| `autoresearch` (Karpathy) | The ratchet hill-climb; the **frozen-substrate / mutable-surface** split (sealed metric + holdout the search can't touch); fixed-budget experiments; TSV experiment ledger; auditability from *provenance* (every change committed) plus a *frozen evaluator*; the LLM as the mutation function proposing the next experiment from the ledger ([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)). |
| `IMPACT` | In-process integration via `EntityPipeline`: the model is embedded as a **derived field** (`cognos.runtime.score.score_row`) so IMPACT builds a standardized scored feature table. A kept target-hidden transform round-trips verbatim to a derived field so train- and serve-time feature logic are identical ([ADR-0002](docs/adr/0002-llm-authored-transforms-safe-execution.md)). |
| Prior-art research | Orchestrator-worker with a sequential dependent pipeline; interrupt/checkpoint/resume for HITL; independent critic/validator agents (avoid "degeneration of thought"); CASH search + leakage-safe cross-validation + the single interpretable champion (Caruana ensembling kept only as an opt-in labelled challenger benchmark, [ADR-0007](docs/adr/0007-single-interpretable-champion-no-silent-ensemble.md)); statistical diagnostic battery; SR 11-7 outcomes analysis (Gini/KS, calibration, PSI) with PBO/Deflated-Sharpe demoted to opt-in trading mode ([ADR-0005](docs/adr/0005-backtesting-is-credit-risk-outcomes-analysis.md)); SR 11-7 × NIST AI RMF; Model Cards + EU Annex IV; OKF docs + AST drift detection. |

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
`HeuristicBrain` is `available=False`, so stages take their deterministic path — the substrate that
reproduces and serves the deliverable's analysis, not a fallback. A `ScriptedBrain` test double
replays canned responses for deterministic testing of the reasoning-driven path. `LLMBrain` (Claude
via the `anthropic` SDK) activates only when the SDK and `ANTHROPIC_API_KEY` are present, and degrades
to the deterministic path otherwise. Stages always branch `if ctx.brain.available: … else: …` and
never *require* the LLM to reproduce a result — the LLM automates the search, never the analysis
([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)).

## Key design decisions

- **Reasoning proposes, the engine disposes** ([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)):
  both layers are first-class and interdependent. Determinism is the anti-hallucination mechanism — no
  proposal becomes a recorded fact until the engine independently verifies it on the frozen substrate.
- **Target-hidden transform execution** ([ADR-0002](docs/adr/0002-llm-authored-transforms-safe-execution.md)):
  LLM-authored feature transforms run through a safe AST-whitelisted expression executor
  (`modeling/transforms.py`) over a features-only view (`X`) + a fixed `np.<fn>` set; `y` is never in
  scope, so transforms cannot leak the target even onto the labelled holdout.
- **Two-tier reproducibility** ([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)): the analysis is
  bit-reproducible offline with no LLM; the reasoning trajectory is recorded to
  `runs/<id>/reasoning/transcript.jsonl` for replay/audit and is human-gated.
- **Primary domain is commercial** model development under SR 11-7
  ([ADR-0004](docs/adr/0004-primary-domain-commercial-fair-lending-optional.md)). Consumer fair-lending
  (ECOA/Reg B, disparate impact, reason codes) does **not** apply and is an optional, off-by-default
  consumer-only module (`compliance.fair_lending: false`) — not a default-pipeline feature.
- **Backtest = SR 11-7 outcomes analysis** ([ADR-0005](docs/adr/0005-backtesting-is-credit-risk-outcomes-analysis.md)):
  discrimination (Gini/KS), calibration (expected-vs-observed by band + ECE), and stability (PSI) on an
  **out-of-time** sample by default (`modeling/credit_metrics.py`). The holdout is time-ordered when a
  datetime column is set. PBO + Deflated Sharpe are demoted to an opt-in trading mode
  (`backtest.returns_column`), not run on credit models.
- **`comply` is non-gating** ([ADR-0006](docs/adr/0006-compliance-is-non-gating-readiness-report.md)): a
  model-risk readiness report that never PASSes or BLOCKs, never marks an unevidenced element compliant
  (ongoing monitoring is always outstanding at dev time), and lists human-only steps (independent
  validation sign-off, monitoring plan, governance). The **only gates are `validate` and `review`**;
  `validate` hard-BLOCKs only on confirmed target leakage, and `review` BLOCKs only on stale docs↔code
  references.
- **Single interpretable champion** ([ADR-0007](docs/adr/0007-single-interpretable-champion-no-silent-ensemble.md)):
  the deployed model is always the single interpretable champion; the Caruana ensemble is no longer a
  silent default and is reframed as an optional, labelled challenger benchmark (`search.ensemble`, off
  by default). Nothing in the docs implies a model that did not ship.
- **Valid inference design** (grilling Q6): statsmodels coefficients/p-values come from a separate
  full-rank K-1 (drop-first) **inference design** (`build_inference_design`), decoupled from the all-K
  prediction pipeline, so reported significances are statistically valid (no dummy-variable trap /
  astronomical condition number). Model selection uses **leakage-safe cross-validation in search + a
  sealed/out-of-time holdout** (not nested cross-validation).
- **IMPACT is optional**: the adapter prefers the real `EntityPipeline` and falls back to the
  built-in scorer transparently, recording which path ran (`used_impact`).
- **OKF over a bespoke format**: a permissive, vendor-neutral, agent-readable markdown spec where
  documentation, code, data and results are all cross-linked nodes the `review` gate can traverse.
