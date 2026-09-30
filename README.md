# COGNOS

**A multi-agent workbench for regulated model development: agents recommend, you decide, the
engine disposes.**

COGNOS takes a dataset and a design brief through the model-development lifecycle for **commercial**
modeling (e.g. commercial credit risk, validated under SR 11-7): data exploration, design, model
search and statistical testing, outcomes analysis, independent validation, a model-risk readiness
report, the white paper, and a docs↔code consistency check.

At every judgment step an **LLM agent recommends** — which columns may be inputs, which econometric
framework fits, which champion to ship, what a validator would challenge. A **model developer
decides** at five review gates: accept, edit, override, or challenge the agent and send the work
back. A **deterministic engine disposes**: it computes every number, keeps the sealed holdout sealed,
checks every agent answer (and makes the agent retry when it fails), and is the only thing that can
BLOCK. The white paper records who recommended what, and who decided.

v1.0 adopts the design proven in Cyber Credit Officer — engine-run agents with contracts, human
gates, a challenger loop, and an audit trail — on top of COGNOS's econometric engine
([ADR-0010](docs/adr/0010-agents-recommend-humans-decide.md)).

---

## The workflow

```
 explore ─▶ ideate ─▶ model ─▶ backtest ─▶ validate ─▶ comply ─▶ document ─▶ review
    │          │         │                    │ ⛔                              │ ⛔
 gate_data  gate_design  gate_champion   gate_validation                 gate_signoff
 (you)       (you)        (you)            (you)                           (you)
```

| Stage | The engine computes | The agent recommends | You decide at the gate |
|---|---|---|---|
| `explore` | profile, missingness, leakage suspects | **Data Analyst** — keep/exclude each suspect, data-quality issues, sponsor questions | which columns are inputs |
| `ideate` | data structure, framework applicability, fittable families | **Design Lead** — framework roles, ranked slate, feature transforms (engine-validated), sponsor questions | the slate; answers to design questions |
| `model` | budgeted leakage-safe search, the **admissible set** (one-SE rule), refit, inference, battery, sealed-holdout score | **Modeler** — the champion from the admissible set (blind to the holdout), economic sign checks; opt-in guided experiments | accept / override the champion |
| `backtest` | IMPACT scoring, Gini/KS, calibration, PSI, portfolio sim, stress | **Outcomes Analyst** — reads discrimination, calibration, stability | — |
| `validate` ⛔ | the effective-challenge rubric; **BLOCKs on confirmed leakage** | **Independent Validator** — findings routed back to the stage that must change (never sees the modeler's reasoning) | accept the risk / send back / reject |
| `comply` | SR 11-7 / NIST AI RMF readiness report (non-gating) | **Model-Risk Analyst** — readiness and priority human actions | — |
| `document` | OKF bundle, Model Card, EU Annex IV, decision log | **Technical Writer** — narrative with `{{fact:…}}` placeholders the engine renders | — |
| `review` ⛔ | AST-verified docs↔code anchors; **BLOCKs on stale references** | — | sign off / reject |

Any decision can be revised later; the engine marks everything downstream stale and re-runs it.
Send-backs and high-severity validator findings reach the stage's agent as **challenges** it must
answer. Unanswered design points (use case, horizon, default definition, segment) are **tracked
questions**, never silent assumptions.

## Install

```bash
pip install -e ".[dev,ui,llm]"   # engine + tests + the workbench + LLM backends
pip install -e ../IMPACT         # optional: the real IMPACT feature-table engine
```

Python ≥ 3.11. Agents run on any of: the local **Claude Code CLI** (`claude -p`, your login), the
**Anthropic API** (`ANTHROPIC_API_KEY`), any **OpenAI-compatible** API (OpenAI, xAI, OpenRouter,
Ollama), or the **deterministic heuristic agents** (offline, no key). `cognos providers` shows what
is available.

## Quickstart

```bash
cognos ui                                    # the workbench at http://127.0.0.1:8050
cognos demo --task commercial --interactive  # review each gate in the terminal (heuristic agents)
cognos demo --task cni --provider claude_cli # live Claude agents, autonomous
```

Your own data:

```bash
cognos init -o cognos.yaml                   # profile template (design brief, agents, gates)
cognos explain --config cognos.yaml
cognos run --config cognos.yaml --interactive
cognos status --run <run_id>                 # steps, questions, challenges
cognos gate gate_champion --run <run_id> --action override --payload '{"champion": "c3"}' --reason "..."
cognos gate gate_validation --run <run_id> --action send_back --target model --message "..."
cognos answer --run <run_id> --gap design-use_case --text "origination underwriting"
```

Python:

```python
from cognos import service
cfg = service.demo_config("commercial")
run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
state = service.run_until_idle(run_id)                 # pauses at gate_data
service.submit_gate(run_id, "gate_data", "accept", background=False)
```

`Orchestrator` / `run_pipeline` still work as a compatibility wrapper (review gates auto-accepted).

## The workbench

`cognos ui` opens a Dash + Mantine app built for model developers: a runs list with a new-run drawer
(demo presets or `projects/*.yaml`, interactive or autonomous, agent backend), and a run workspace
with the stage rail, each stage's engine evidence (profile, framework assessment, slate, experiment
ledger, admissible set, coefficient and calibration charts, rubric, findings, the rendered white
paper) beside its agent's recommendation, the gate form, live activity, questions & challenges, and
an agent audit where every call's prompt, context slice and raw output can be inspected. Light and
dark themes; all state is on disk, so a refresh or restart loses nothing.

## What makes it trustworthy

- **Agents recommend; the engine disposes.** Every agent answer is validated against a contract and
  engine checks (unknown fact ids, a champion outside the admissible set, a non-fittable family, an
  excluded target, a typed metric value…) and retried with the errors; failures are visible, never
  silent.
- **No LLM math.** Every recorded number comes from the engine; agents cite facts by id and the
  writer's prose is rendered from placeholders.
- **Frozen substrate.** Metric definitions and the sealed holdout are not agent-editable; the
  modeler chooses before the holdout is scored, and every holdout evaluation is counted.
- **Independent challenge.** The validator's context never contains the modeler's rationale; its
  high findings loop back to the responsible stage (bounded) and the rest reach you.
- **BLOCK is load-bearing.** Only the engine BLOCKs (confirmed leakage, stale docs↔code references);
  a BLOCK can be sent back or rejected, never accepted.
- **Leakage-safe CV + target-hidden transforms; valid inference** (full-rank K−1 design).
- **Honest backtesting** — Gini/KS + calibration + PSI on an out-of-time sample; compliance is a
  readiness report, not a verdict.
- **Two-tier reproducibility** — the analysis re-derives offline with no LLM; the recommendations and
  decisions are recorded (agent audit + decision log) and replayable (ADR-0003).
- **Docs that can't silently drift** — `review` AST-verifies docs↔code links every run.

## Layout

```
src/cognos/
  engine/        workflow graph, RunState, gates, events, the Engine (DAG + staleness + gates)
  agents/        contracts, prompts/, slices (independence), facts, checks, heuristic agents,
                 providers.yaml, runner (retry/audit/limits), backends/ (claude_cli, anthropic, openai)
  stages/        the 8 stages + stat_tests battery (engine work; judgment via ctx.recommend)
  modeling/      metrics, fitters (GLM links incl. probit/cloglog), ratchet search, hazard,
                 structural (Merton), migration, simulate (Vasicek + stress), credit_metrics,
                 transforms (target-hidden), guided (agent-guided search), ensemble
  ui/            the Dash + Mantine workbench
  service.py     the boundary the CLI and UI use
  cli.py  config.py  context.py  artifacts.py  orchestrator.py (compat)  okf.py  synth.py
  integrations/  impact_adapter, autoforge_loop
  runtime/       deployment scorer (IMPACT derived-field entry point)
projects/  examples/  evals/  tests/  docs/adr/ (0001–0010)  CONTEXT.md (glossary)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FEATURES.md`](FEATURES.md), [`CONTEXT.md`](CONTEXT.md),
[`CHANGELOG.md`](CHANGELOG.md), the decision records in [`docs/adr/`](docs/adr/), and
[`CLAUDE.md`](CLAUDE.md) (principles for changing COGNOS itself).

## License

MIT.
