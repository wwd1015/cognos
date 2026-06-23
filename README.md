# COGNOS

**An autonomous multi-agent system for end-to-end statistical & ML model development.**

COGNOS takes a dataset and a one-file project profile and runs the whole model-development
lifecycle — data exploration, idea generation, model search + statistical testing, standardized
backtesting, independent validation, regulatory compliance, documentation, and a docs↔code
consistency check — producing a validated, documented, deployment-ready model and an auditable
trail of every decision.

It is built in the spirit of [`deputy`](../deputy) (a sequential pipeline of declarative agents with
on-disk artifact hand-offs and a mechanical orchestrator), reimplements the
[`autoforge`](../autoforge) / Karpathy-`autoresearch` *ratchet* loop in code, integrates the
[`IMPACT`](../IMPACT) feature-table engine, and emits its white paper in Google's **Open Knowledge
Format (OKF)**.

---

## The pipeline

```
                ┌─────────┐   ┌────────┐   ┌───────┐   ┌──────────┐
   dataset ───▶ │ explore │─▶ │ ideate │─▶ │ model │─▶ │ backtest │─▶ ...
                └─────────┘   └────────┘   └───────┘   └──────────┘
                  profile     ranked        ratchet      IMPACT scoring
                  + leakage   hypotheses    search +     + PBO / Deflated
                  scan                      stat tests   Sharpe
       ...  ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌────────┐
       ───▶ │ validate │─▶ │ comply  │─▶ │ document │─▶ │ review │─▶ verdict
            └──────────┘   └─────────┘   └──────────┘   └────────┘
             SR 11-7         SR 11-7 ×     OKF white      docs↔code
             effective       NIST × fair-  paper + model  drift gate
             challenge ⛔     lending ⛔     card + AnnexⅣ  ⛔
```

`⛔` = a **gate**: it can BLOCK the pipeline (autonomous mode) or pause for human approval
(interactive mode).

| Stage | Agent does | Key outputs |
|------|------------|-------------|
| `explore`  | Profiles data; flags target-leakage suspects, missingness, imbalance | data profile |
| `ideate`   | Proposes & ranks candidate model specs (family + feature strategy) | hypotheses |
| `model`    | Ratchet search over the CASH space, leakage-safe nested CV, statsmodels inference, Caruana ensemble, **statistical diagnostic battery** | champion model, coefficients, diagnostics, experiment ledger |
| `backtest` | Embeds the model as an **IMPACT derived field**, runs `EntityPipeline`; computes **PBO** (CSCV) & **Deflated Sharpe** | scored table, overfitting analytics |
| `validate` | **Independent** SR 11-7 effective challenge: leakage, overfitting, stability, diagnostics, significance | rubric + verdict |
| `comply`   | SR 11-7 × NIST AI RMF × 7 trustworthy-AI traits; ECOA/Reg B **fair-lending** disparate-impact scan + reason codes | compliance report |
| `document` | White paper as an **OKF bundle** + Google **Model Card** (9 sections) + EU AI Act **Annex IV** pack, with `{@code:…}` links to deployment code | OKF docs bundle |
| `review`   | AST-verifies every docs↔code link; flags drift between the white paper and the actual code/model | consistency report |

## Install

```bash
pip install -e ".[dev]"        # core + test tooling (uv works too: uv pip install -e ".[dev]")
pip install -e ".[llm]"        # optional: Claude-backed agent reasoning (else fully deterministic)
pip install -e ../IMPACT        # optional: the real IMPACT feature-table engine (else built-in scorer)
```

Python ≥ 3.11. Core deps: numpy, pandas, scikit-learn, statsmodels, scipy, pydantic, PyYAML, joblib.

## Quickstart

```bash
# See it run end-to-end on synthetic data (no config needed):
cognos demo --task regression
cognos demo --task credit          # demonstrates the fair-lending BLOCK

# Your own data:
cognos init -o cognos.yaml         # write a config template, then edit it
cognos explain --config cognos.yaml
cognos run --config cognos.yaml             # autonomous
cognos run --config cognos.yaml --interactive   # pause at gates for approve/reject

# Stage-by-stage (human-in-the-loop): each agent is individually invocable
cognos run-stage explore --config cognos.yaml --run <run_id>
cognos run-stage model   --config cognos.yaml --run <run_id>

cognos report --run <run_id>       # print the run summary
cognos agents                      # list the 8 stage agents
```

Python API:

```python
from cognos import CognosConfig, run_pipeline

cfg = CognosConfig.from_yaml("cognos.yaml")
ctx, summary = run_pipeline(cfg)                  # autonomous
print(summary.token_block())
print("white paper:", ctx.docs_dir)
```

## Two operating modes, one control flow

- **Autonomous** — the full pipeline runs unattended; gates auto-approve and a `BLOCK` halts the run.
  For quick, end-to-end prototypes.
- **Stage-by-stage / interactive** — every stage is an independently invocable agent
  (`cognos run-stage …`), and `--interactive` pauses at each gate for approve / reject. For work
  where human judgment is heavily involved.

Both modes share the same orchestrator: each stage checkpoints its `StageResult` to disk, so runs
**resume from failure** and any stage can be re-run in a fresh process.

## What makes it trustworthy

- **Frozen substrate** — the metric definitions and the sealed holdout live in a module the search
  agents cannot edit, so an autonomous loop cannot game its own yardstick (from Karpathy's
  `autoresearch`).
- **Leakage-safe nested CV** — all preprocessing is fit inside each training fold.
- **Independent challenge** — validation / compliance / consistency agents run separately from the
  modeling agent (SR 11-7's "effective challenge"; avoids self-grading rationalization).
- **Multiple-testing honesty** — Probability of Backtest Overfitting and Deflated Sharpe account for
  the number of configurations tried.
- **Docs that can't silently drift** — the white paper links paragraphs to source symbols, and the
  `review` gate AST-verifies those links every run.

## Layout

```
src/cognos/
  config.py context.py artifacts.py orchestrator.py cli.py okf.py synth.py datautil.py
  brains/        heuristic (default, offline) + optional Claude LLM brain
  stages/        the 8 agents + stat_tests battery
  modeling/      metrics, fitters, ratchet search, Caruana ensemble, backtest stats (PBO/DSR)
  integrations/  impact_adapter, autoforge_loop
  runtime/       score (deployment scorer; the IMPACT derived-field entry point)
.claude/         declarative agent specs + orchestrator command + safety hooks (deputy-style)
projects/        example project profiles
examples/end_to_end/   runnable synthetic-data demo
evals/           autonomous-run eval harness
tests/           unit + integration (run: pytest)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design rationale, [`FEATURES.md`](FEATURES.md) for
the comprehensive feature list, and [`CLAUDE.md`](CLAUDE.md) for the principles that govern changes
to COGNOS itself.

## License

MIT.
