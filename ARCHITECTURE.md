# COGNOS architecture

> v1.0: **agents recommend, humans decide, the engine disposes**
> ([ADR-0010](docs/adr/0010-agents-recommend-humans-decide.md)).

```
      Dash + Mantine workbench (cognos ui)            CLI (cognos …)
                        └──────────────┬──────────────┘
                              cognos.service  (the single boundary)
                                       │
          engine/  graph · RunState · gates · challenges · gaps · events · staleness
                                       │
          stages/  prepare ─▶ ctx.recommend(agent) ─▶ finalize (engine disposes)
             │                           │
          modeling/ okf/ runtime/     agents/  prompts · contracts · slices · facts · checks ·
          (numerical core,            heuristic · runner · providers: heuristic | replay |
           frozen substrate)          claude_cli | anthropic | openai_compat
```

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

Neither layer is optional. Running with the deterministic (heuristic) agents is the offline and demo
path and the test substrate (`replay` re-runs recorded agent outputs), not a degraded product — it is
how the deliverable's analysis is always reproduced ([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)).
In production the LLM agents recommend and a model developer decides at five review gates.

The agents run **inside the engine** (v1.0): each stage consults its agent through
`ctx.recommend()`, and the runner calls the configured backend — `claude -p` in an isolated
temp directory, the Anthropic API, any OpenAI-compatible API, recorded outputs (`replay`), or the
deterministic `heuristic` agents. The v0.x `.claude/agents` wrappers are retired; their domain
playbooks are the agents' role prompts (`src/cognos/agents/prompts/`).

## Reasoning-driven loop

Before any reasoning enters, `ideate` does the *design* work deterministically: it assesses the
data structure (cross-sectional vs vintage/panel, event support), judges which econometric
frameworks apply (structural vs reduced-form vs hazard vs ML challenger — rejected ones recorded
with reasons as SR 11-7 "alternatives considered"), and triangulates the config's `design:` brief
with the model sponsor — every unanswered design point becomes an open question in the run's
`design_brief.md`, never a silent assumption.

The reasoning layer enters in two staged depths
([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)):

- **(A) LLM-driven ideation** emits *executable* feature-engineering transforms (not just prose)
  and a design review that may add sponsor questions and engine-fittable candidate specs.
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
computes) is fully bit-reproducible offline with no LLM, while the **recommendations** are
non-deterministic, recorded to `runs/<id>/agents/` (every attempt's prompt, context slice and raw
output) for audit and `replay`, and human-gated at the review gates. The LLM is required to automate
the judgment, never to reproduce the result.

## Lineage (what we borrowed and from where)

| Source | Pattern adopted |
|--------|-----------------|
| `deputy` | Sequential pipeline of single-responsibility declarative agents; a *mechanical* orchestrator (not a clever supervisor); **load-bearing on-disk artifacts** + greppable **verdict tokens** as the inter-stage contract; per-project YAML; PreToolUse hooks as deterministic backstops; an eval harness; checkpoint/resume. |
| `autoforge` | The `name: value` stdout + `results.tsv` ledger optimization protocol, reimplemented in `integrations/autoforge_loop.py`; accept-if-better-else-discard ratchet. |
| `autoresearch` (Karpathy) | The ratchet hill-climb; the **frozen-substrate / mutable-surface** split (sealed metric + holdout the search can't touch); fixed-budget experiments; TSV experiment ledger; auditability from *provenance* (every change committed) plus a *frozen evaluator*; the LLM as the mutation function proposing the next experiment from the ledger ([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)). |
| `IMPACT` | In-process integration via `EntityPipeline`: the model is embedded as a **derived field** (`cognos.runtime.score.score_row`) so IMPACT builds a standardized scored feature table. A kept target-hidden transform round-trips verbatim to a derived field so train- and serve-time feature logic are identical ([ADR-0002](docs/adr/0002-llm-authored-transforms-safe-execution.md)). |
| Cyber Credit Officer | Engine-run agents with role prompts + Pydantic output contracts + context slices; validate-and-retry against deterministic tools; human decision gates (accept / edit / override); a challenger whose high-severity findings loop back to the responsible agent (bounded); information gaps with targeted re-entry; per-call audit (prompt hash, context hash, raw I/O, cost); `claude -p` / Anthropic / OpenAI-compatible backends; practice (mock) mode ([ADR-0010](docs/adr/0010-agents-recommend-humans-decide.md)). |
| Prior-art research | Orchestrator-worker with a sequential dependent pipeline; interrupt/checkpoint/resume for HITL; independent critic/validator agents (avoid "degeneration of thought"); CASH search + leakage-safe cross-validation + the single interpretable champion (Caruana ensembling kept only as an opt-in labelled challenger benchmark, [ADR-0007](docs/adr/0007-single-interpretable-champion-no-silent-ensemble.md)); statistical diagnostic battery; SR 11-7 outcomes analysis (Gini/KS, calibration, PSI) with PBO/Deflated-Sharpe demoted to opt-in trading mode ([ADR-0005](docs/adr/0005-backtesting-is-credit-risk-outcomes-analysis.md)); SR 11-7 × NIST AI RMF; Model Cards + EU Annex IV; OKF docs + AST drift detection. |

## Control flow

The engine (`engine/engine.py`) is deliberately mechanical. It walks the step graph
(`engine/graph.py`), runs every ready step, checkpoints, pauses at human gates, applies decisions,
routes challenges, syncs gaps, and marks stale work — it never judges.

```
explore → gate_data → ideate → gate_design → model → gate_champion → backtest → validate
        → gate_validation → comply → document → review → gate_signoff

loop:
  ready = steps whose dependencies are done/skipped and that are pending or stale
  stage step : result = stage.run_guarded(ctx); record; sync gaps + challenge responses
               ERROR -> failed (retry later) · BLOCK at validate/review -> blocked (halts)
               validate: high validator findings -> challenges -> target stage stale (≤ N loops)
  gate step  : interactive -> awaiting (the human decides) · autonomous -> auto-accept
  decision   : handler updates overrides / challenges / gaps -> changed steps + descendants stale
```

Two drivers share one step executor: `run_until_idle()` (synchronous — CLI, tests, autonomous runs)
and `start()`/`advance()` (background threads — the UI). Every read-modify-write of `state.json`
happens under the run's lock after re-loading from disk; a step invalidated while it was running
stays stale when it finishes. Because stages checkpoint and state is on disk, any process (a CLI
command, the UI after a restart) resumes a run exactly; `cognos run-stage` still re-runs one stage.

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
  config.yaml              # the profile the run was created from (rehydrates the engine)
  state.json               # RunState: steps, gate decisions, challenges, gaps, overrides, loops, spend,
                           # the live package pointer
  packages/ vN.json        # sealed decision packages: written once by approve, never rewritten
  events.jsonl             # the activity feed (UI, CLI)
  manifest.json            # run metadata + per-stage verdicts
  summary.json summary.txt # machine- and human-readable run summary
  agents/  audit.jsonl <call>.input.json <call>.output.json <call>.transcript.jsonl
  data/    dataset.parquet train.parquet holdout.parquet   # sealed holdout lives here
  models/  champion_scorer.joblib                          # deployable scorer (IMPACT entry point)
  docs/    *.md index.md log.md narrative.md decisions.md  # the OKF white-paper bundle
  stages/<stage>/result.json + artifacts (profile.json, hypotheses.json, design_brief.md,
                                          ledger.tsv/json, search_cache.joblib, diagnostics.json, …)
  stages/<stage>/result.prev.json          # the result a re-run replaced (kept for comparison)
```

Three mechanical readers sit beside the engine. `engine/process.py` maps each gate to a seat
(developer, reviewer, approver), refuses a seat on another seat's gate, and seals `packages/vN.json`
on `approve`; a later invalidation supersedes the pointer in `state.json`. `compare.py` restates
two records and subtracts (run vs run, re-run vs `result.prev.json`). `export.py` zips the text
record of a run with a SHA-256 listing; `data/`, `models/` and binary caches never travel.

## The agent layer

Each agent is five pieces (`src/cognos/agents/`):

| Piece | Role |
|---|---|
| `prompts/<agent>.md` + `_common.md` | the role and its domain playbook; shared rules (numbers from facts, answer every challenge, never assume the design) |
| `contracts.py` | the typed recommendation (`extra="forbid"`, strict-grammar-friendly JSON types) |
| `slices.py` | what the agent may see: project brief + stage evidence + `facts` scoped to its independence (the modeler sees no model/backtest facts; the validator never sees the modeler's rationale) + its open challenges |
| `checks.py` | what the engine re-checks (cited facts exist, every challenge answered, champion ∈ admissible set, fittable families, target never excluded, no typed metrics in prose, …) |
| `heuristic.py` | the deterministic implementation of the same contract — the offline path |

`runner.py` builds the prompt (task + context JSON + challenges + the previous attempt's errors),
calls the backend, validates, retries up to `agents.max_retries`, and audits every attempt; it
enforces a per-call time limit and a per-run spend budget. Provider resolution:
`COGNOS_PROVIDER` → `agents.provider` → `auto` (first available LLM backend, else `heuristic`).

## Key design decisions

- **Reasoning proposes, the engine disposes** ([ADR-0001](docs/adr/0001-reasoning-proposes-engine-disposes.md)):
  both layers are first-class and interdependent. Determinism is the anti-hallucination mechanism — no
  proposal becomes a recorded fact until the engine independently verifies it on the frozen substrate.
- **Target-hidden transform execution** ([ADR-0002](docs/adr/0002-llm-authored-transforms-safe-execution.md)):
  LLM-authored feature transforms run through a safe AST-whitelisted expression executor
  (`modeling/transforms.py`) over a features-only view (`X`) + a fixed `np.<fn>` set; `y` is never in
  scope, so transforms cannot leak the target even onto the labelled holdout.
- **Two-tier reproducibility** ([ADR-0003](docs/adr/0003-two-tier-reproducibility.md)): the analysis is
  bit-reproducible offline with no LLM; the recommendations are recorded to `runs/<id>/agents/`
  (every attempt's prompt, context slice and raw output) with the human decisions in `state.json`
  and the decision log, and are replayable with the `replay` provider.
- **Agents recommend, humans decide** ([ADR-0010](docs/adr/0010-agents-recommend-humans-decide.md)):
  five review gates; one pushback mechanism (challenges) for human send-backs and validator findings;
  sponsor questions as tracked gaps; the modeler chooses from the one-standard-error admissible set
  before the holdout is scored; only the engine BLOCKs, and a BLOCK is never acceptable.
- **Primary domain is commercial** model development under SR 11-7
  ([ADR-0004](docs/adr/0004-primary-domain-commercial-fair-lending-optional.md)). Fair-lending scans
  are an optional, off-by-default module (`compliance.fair_lending: false`) — out of scope for a
  commercial profile by choice, not a default-pipeline feature.
- **Backtest = SR 11-7 outcomes analysis** ([ADR-0005](docs/adr/0005-backtesting-is-credit-risk-outcomes-analysis.md)):
  discrimination (Gini/KS), calibration (expected-vs-observed by band + ECE), and stability (PSI) on an
  **out-of-time** sample by default (`modeling/credit_metrics.py`). The holdout is time-ordered when a
  datetime column is set. PBO + Deflated Sharpe are demoted to an opt-in trading mode
  (`backtest.returns_column`), not run on credit models.
- **`comply` is non-gating** ([ADR-0006](docs/adr/0006-compliance-is-non-gating-readiness-report.md)): a
  model-risk readiness report that never PASSes or BLOCKs, never marks an unevidenced element compliant
  (ongoing monitoring is always outstanding at dev time), and lists human-only steps (independent
  validation sign-off, monitoring plan, governance). The **only verdict gates are `validate` and
  `review`** (the five human review gates decide, they never BLOCK);
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
- **Econometric core: survival, structural, and simulation live in the engine**
  ([ADR-0008](docs/adr/0008-econometric-core-survival-structural-simulation.md)): discrete-time
  hazard families panel-expand obligor-periods **inside the estimator** so CV/holdout stay
  obligor-level (the survival-CV leak is impossible by construction) and report a **PD term
  structure**; the Merton structural model is a deterministic KMV **solver, not a fit** — DD feeds
  the champion as a serve-time-recomputed feature (hybrid), with the pure structural PD kept as a
  labelled challenger benchmark; Vasicek portfolio losses + Basel IRB capital and macro-scenario
  stress are seeded **reports** the backtest stage emits, never selection criteria.
- **Rating migration is a fitted engine layer**
  ([ADR-0009](docs/adr/0009-rating-migration-external-agency-data.md)): the cohort transition
  matrix is estimated on the **training partition only** (NR-adjusted, Laplace-smoothed,
  PAVA rank-ordered with adjustments reported); the rating-implied horizon PD replaces the raw
  rating as the champion's serve-time-recomputed hybrid feature, the pure-migration PD is a
  sealed-holdout challenger benchmark, and the matrix-power term structure + by-rating
  expected-loss forecast (pooled and regime-conditioned) are **reports** — the CECL/stress
  deliverable, never selection criteria.
- **IMPACT is optional**: the adapter prefers the real `EntityPipeline` and falls back to the
  built-in scorer transparently, recording which path ran (`used_impact`).
- **OKF over a bespoke format**: a permissive, vendor-neutral, agent-readable markdown spec where
  documentation, code, data and results are all cross-linked nodes the `review` gate can traverse.
