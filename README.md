# COGNOS

**An autonomous multi-agent system for end-to-end statistical & ML model development.**

COGNOS automates the model-development lifecycle for **commercial** modeling (e.g. commercial
credit risk, validated under SR 11-7): data exploration, idea generation, model search + statistical
testing, standardized backtesting, independent validation, a model-risk readiness report,
documentation, and a docs↔code consistency check.

It is a **two-layer** system (see [`docs/adr/`](docs/adr/)): an LLM **reasoning layer** that
*proposes* the decisions a human modeler would make — design, model choice, feature engineering, the
next experiment to try — and a deterministic **engine** that *disposes*: it fits, scores on a frozen
metric and a sealed holdout, runs the statistical battery, and is the sole authority on what is kept
and on any number that becomes a recorded fact. The reasoning automates judgment; the determinism is
the anti-hallucination mechanism — **the LLM proposes, the engine disposes.**

It draws on [`deputy`](../deputy) (sequential declarative agents + mechanical orchestrator + on-disk
artifact hand-offs), reimplements the [`autoforge`](../autoforge) / Karpathy-`autoresearch` ratchet,
integrates the [`IMPACT`](../IMPACT) feature-table engine, and emits its white paper in Google's
**Open Knowledge Format (OKF)**.

---

## The pipeline

```
   dataset ─▶ explore ─▶ ideate ─▶ model ─▶ backtest ─▶ validate ─▶ comply ─▶ document ─▶ review ─▶ verdict
              profile   ranked     ratchet   outcomes    SR 11-7     readiness   OKF white   docs↔code
              +leakage  hypotheses search +   analysis    challenge   report      paper       drift gate ⛔
                        (+LLM)     stat tests  (OOT)       ⛔          (non-gate)
```

`⛔` = a **gate**: it can BLOCK the pipeline (autonomous) or pause for approval (interactive). The
gates are **`validate`** and **`review`** only. Compliance is a non-gating report (ADR-0006).

| Stage | What it does | Key outputs |
|------|------------|-------------|
| `explore`  | Profiles data; flags target-leakage suspects, missingness, imbalance | data profile |
| `ideate`   | Ranks candidate model specs; with an LLM, proposes engine-validated feature transforms | hypotheses |
| `model`    | Ratchet search over the CASH space, leakage-safe CV, full-rank statsmodels inference, statistical diagnostic battery, single interpretable champion; opt-in **LLM-guided search** | champion, valid coefficients, diagnostics, experiment ledger |
| `backtest` | Scores via **IMPACT**; **outcomes analysis** on an out-of-time sample: Gini/KS, calibration, PSI (PBO/Deflated-Sharpe only in opt-in trading mode) | scored table, outcomes report |
| `validate` ⛔ | **Independent** SR 11-7 challenge: leakage (BLOCKs), overfitting, stability, diagnostics, significance | rubric + verdict |
| `comply`   | Non-gating model-risk **readiness report**: SR 11-7 evidence + outstanding human steps | readiness report |
| `document` | White paper as an **OKF bundle** + Google **Model Card** + EU **Annex IV**, with `{@code:…}` docs↔code links | OKF docs bundle |
| `review` ⛔ | AST-verifies every docs↔code link; BLOCKs on stale references | consistency report |

## Install

```bash
pip install -e ".[dev]"      # core + tests (uv: uv pip install -e ".[dev]")
pip install -e ".[llm]"      # optional: Claude reasoning layer (else deterministic engine only)
pip install -e ../IMPACT     # optional: real IMPACT feature-table engine (else built-in scorer)
```

Python ≥ 3.11. Core deps: numpy, pandas, scikit-learn, statsmodels, scipy, pydantic, PyYAML, joblib.

## Quickstart

```bash
# End-to-end on synthetic data (no config needed):
cognos demo --task commercial      # commercial credit, out-of-time outcomes analysis
cognos demo --task regression

# Your own data:
cognos init -o cognos.yaml         # write + edit a config template
cognos explain --config cognos.yaml
cognos run --config cognos.yaml             # autonomous
cognos run --config cognos.yaml --interactive   # pause at gates (validate, review)

# Stage-by-stage (human-in-the-loop): every agent is individually invocable
cognos run-stage model --config cognos.yaml --run <run_id>

cognos report --run <run_id>
cognos agents
```

Python API:

```python
from cognos import CognosConfig, run_pipeline
cfg = CognosConfig.from_yaml("cognos.yaml")
ctx, summary = run_pipeline(cfg)
print(summary.token_block(), "white paper:", ctx.docs_dir)
```

## The reasoning layer (propose / dispose)

The LLM enters in two staged depths (off unless an LLM brain is configured):

- **Ideation** — proposes feature-engineering transforms grounded in the data profile; the engine
  validates each (target-hidden) before it counts.
- **Guided search** (`search.guided: true`) — after the deterministic ratchet, the LLM proposes the
  *next* experiment from the experiment ledger; the engine applies it target-hidden, scores it with
  leakage-safe CV, and **keeps it only if it beats the incumbent on the frozen metric**.

A proposal can never enter the record unless the engine independently verifies it. LLM-authored
feature transforms run on a **features-only view** — the target is never in scope, so a transform
physically cannot leak the answer. Every prompt+response is recorded to
`runs/<id>/reasoning/transcript.jsonl` for replay/audit.

## Two operating modes, one control flow

- **Autonomous** — the full pipeline runs unattended; a gate BLOCK halts. For quick prototypes.
- **Stage-by-stage / interactive** — every stage is independently invocable (`cognos run-stage …`),
  and `--interactive` pauses at each gate for approve/reject. For human-in-the-loop work.

Each stage checkpoints its result, so runs **resume from failure** and any stage re-runs in a fresh
process.

## What makes it trustworthy

- **Propose / dispose** — the LLM never asserts a metric or verdict; only the deterministic engine
  does. A wrong proposal simply fails to beat the holdout and is discarded.
- **Two-tier reproducibility** — the *analysis* is bit-reproducible offline with no LLM; the
  *reasoning trajectory* is recorded/replayable and human-gated (ADR-0003).
- **Frozen substrate** — metric definitions + the sealed holdout are not editable by the search.
- **Leakage-safe CV + target-hidden transforms** — preprocessing is fit inside each fold; feature
  code can't see the target.
- **Valid inference** — coefficients/p-values come from a full-rank K-1 design (no dummy-variable
  trap), as SR 11-7 validation requires.
- **Independent challenge** — validation runs separately from modeling and BLOCKs on confirmed
  leakage.
- **Honest backtesting** — Gini/KS + calibration + PSI on an out-of-time sample; no rubber-stamped
  compliance (compliance is a readiness report, not a verdict).
- **Docs that can't silently drift** — the `review` gate AST-verifies docs↔code links every run.

## Layout

```
src/cognos/
  config.py context.py artifacts.py orchestrator.py cli.py okf.py synth.py datautil.py
  brains/        heuristic (default) + optional Claude LLMBrain + ScriptedBrain (test double)
  stages/        the 8 agents + stat_tests battery
  modeling/      metrics, fitters (+ GLMs), ratchet search, credit_metrics, backtest_stats,
                 transforms (target-hidden), guided (LLM-guided search), ensemble
  integrations/  impact_adapter, autoforge_loop
  runtime/       score (deployment scorer; re-applies transforms; the IMPACT derived-field entry point)
.claude/         declarative agent specs + orchestrator command + safety hooks (deputy-style)
projects/  examples/end_to_end/  evals/  tests/
docs/adr/        architecture decision records (0001-0007)
CONTEXT.md       ubiquitous-language glossary
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FEATURES.md`](FEATURES.md), [`CONTEXT.md`](CONTEXT.md),
the decision records in [`docs/adr/`](docs/adr/), and [`CLAUDE.md`](CLAUDE.md) (principles for
changing COGNOS itself).

## License

MIT.
