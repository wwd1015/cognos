# COGNOS

**A multi-agent workbench for regulated model development: agents recommend, you decide, the
engine disposes.**

COGNOS takes a business intent and a dataset through the model-development lifecycle for
**commercial** modeling (e.g. commercial credit risk, validated under SR 11-7): intake of the
sponsor's intent, data exploration, design, model search and statistical testing, outcomes
analysis, independent validation, a model-risk readiness report, the white paper, and a docs↔code
consistency check.

It runs in two **development modes**:

- **New model development** starts from the sponsor's **business intent document** (plus any
  background material). There is a template to fill in beforehand: `cognos intent-template`.
- **Model update** starts from the **existing model's artifacts** (white paper, development or
  deployment code, validation and monitoring reports, or an earlier COGNOS run) and an **update
  request** written on the same template: `cognos intent-template --kind update`.

In both, an **Intake Analyst** agent reads the documents, restates the goal, and **interviews**
you wherever the intent is not clear, before any data is touched.

At every judgment step an **LLM agent recommends** — which columns may be inputs, which econometric
framework fits, which champion to ship, what a validator would challenge. **People decide in seats**:
the model developer at intent, data, design and champion; an independent reviewer at validation; an approver
at sign-off. A seat cannot act on another seat's gate. A **deterministic engine disposes**: it
computes every number, keeps the sealed holdout sealed, checks every agent answer (and makes the
agent retry when it fails), and is the only thing that can BLOCK.

Finishing the analysis is not a signature. Autonomous mode is express preparation: the gates are
accepted and marked as such, and no package is sealed. An approver seals one only after use,
horizon, default definition and segment are answered. The package file is written once; a later
edit supersedes it and does not rewrite it. Approval is not deployment. The white paper records
who recommended what, and which seat decided.

v1.0 adopts the design proven in Cyber Credit Officer — engine-run agents with contracts, human
gates, a challenger loop, and an audit trail — on top of COGNOS's econometric engine
([ADR-0010](docs/adr/0010-agents-recommend-humans-decide.md)).

---

## The workflow

```
 intake ─▶ explore ─▶ ideate ─▶ model ─▶ backtest ─▶ validate ─▶ comply ─▶ document ─▶ review
    │         │          │         │                    │ ⛔                              │ ⛔
 gate_intent gate_data gate_design gate_champion  gate_validation                 gate_signoff
 (you)       (you)      (you)       (you)           (you)                           (you)
```

| Stage | The engine computes | The agent recommends | You decide at the gate |
|---|---|---|---|
| `intake` | reads the intent document and the prior model's artifacts, parses the template, inventories and hashes every document | **Intake Analyst** — the engagement brief (every "stated" position quoted from the documents), whether the goal is clear, interview questions; for an update, the change items and scope | answer the interview; confirm the intent |
| `explore` | profile, missingness, leakage suspects | **Data Analyst** — keep/exclude each suspect, data-quality issues, sponsor questions | which columns are inputs |
| `ideate` | data structure, framework applicability, fittable families | **Design Lead** — framework roles, ranked slate, feature transforms (engine-validated), sponsor questions | the slate; answers to design questions |
| `model` | budgeted leakage-safe search, the **admissible set** (one-SE rule), refit, inference, battery, sealed-holdout score | **Modeler** — the champion from the admissible set (blind to the holdout), economic sign checks; opt-in guided experiments | accept / override the champion |
| `backtest` | IMPACT scoring, Gini/KS, calibration, PSI, portfolio sim, stress | **Outcomes Analyst** — reads discrimination, calibration, stability | — |
| `validate` ⛔ | the effective-challenge rubric; **BLOCKs on confirmed leakage** | **Independent Validator** — findings routed back to the stage that must change (never sees the modeler's reasoning) | accept the risk / send back / reject |
| `comply` | SR 11-7 / NIST AI RMF readiness report (non-gating) | **Model-Risk Analyst** — readiness and priority human actions | — |
| `document` | OKF bundle, Model Card, EU Annex IV, decision log | **Technical Writer** — narrative with `{{fact:…}}` placeholders the engine renders | — |
| `review` ⛔ | AST-verified docs↔code anchors; **BLOCKs on stale references** | — | sign off / reject |

### The interview

Intake does not guess. Each field of the brief is *stated* (the engine finds the agent's quote in
your documents, or rejects the answer), *inferred* (to be confirmed) or *missing*. Every required
field that is not stated becomes a **blocking question**. You answer at the intent gate (or with
`cognos answer`), intake re-reads the brief with the answer, and the gate re-opens until nothing
blocks. You may confirm the intent with questions still open, with a reason; they stay open, and
nobody can sign a package until use, horizon, default definition and segment are answered. What
the sponsor's document states becomes part of the effective config when you confirm it.

### A model update

The engine reads the existing model mechanically: the family named in its artifacts, the dataset
columns its code refers to and, when it is an earlier COGNOS run, the champion and scores that run
recorded. The Intake Analyst turns the request into change items and a scope (recalibrate,
re-estimate, redevelop). The design keeps the existing model's family at the front of the slate as
the benchmark, the validator checks that every requested change was delivered, and the white paper
gains a **model change record** with the existing model beside the update. The existing model's
scores reach only the agents that read results; the modeler still chooses blind.

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
cognos init -o cognos.yaml                   # profile template (data, agents, gates)
cognos intent-template -o intent.md          # the business intent document: fill it in first
cognos explain --config cognos.yaml
cognos run --config cognos.yaml --intent intent.md --interactive      # a new model development

cognos intent-template --kind update -o request.md                    # a model update
cognos run --config cognos.yaml --kind update --intent request.md \
           --prior whitepaper.docx --prior score.py --prior-run <run_id> --interactive
cognos status --run <run_id>                 # steps, questions, challenges
cognos compare <run_a> <run_b>               # what was decided differently, and what it changed
cognos export <run_id>                       # one zip: documents, results, decisions, audit (never the data)
cognos gate gate_champion --run <run_id> --action override --payload '{"champion": "c3"}' --reason "..."
cognos gate gate_validation --run <run_id> --action send_back --target model --message "..."
cognos answer --run <run_id> --gap design-use_case --text "origination underwriting"
```

The documents can also be named in the profile (`engagement:` block). They are copied into the run
(`runs/<id>/inputs/`), read as text (`.md`, `.txt`, `.docx`, source code, notebooks; `.pdf` when
`pypdf` is installed), and an unreadable one is a finding, not a crash.

Python:

```python
from cognos import service
cfg = service.demo_config("commercial")
run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
state = service.run_until_idle(run_id)                 # pauses at gate_intent
service.submit_gate(run_id, "gate_intent", "edit", {"answers": {"design-horizon": "12-month"}},
                    background=False)                  # the interview: intake re-reads the brief
service.submit_gate(run_id, "gate_intent", "accept", reason="the rest is with the sponsor",
                    background=False)

update = service.create_run(cfg, engagement={          # a model update of that run
    "kind": "update", "intent": "request.md", "prior_artifacts": ["whitepaper.docx", "score.py"],
    "prior_run": run_id})
```

`Orchestrator` / `run_pipeline` still work as a compatibility wrapper (review gates auto-accepted).

## The workbench

`cognos ui` opens a Dash + Mantine app built for model developers: a runs list with a new-run drawer
(new model or model update, demo presets or `projects/*.yaml`, upload of the intent document and
the existing model's artifacts, a button to download the template, interactive or autonomous, agent
backend), and a run workspace with the stage rail, each stage's engine evidence (the engagement
brief and the interview form, profile, framework assessment, slate, experiment
ledger, admissible set, coefficient and calibration charts, rubric, findings, the rendered white
paper) beside its agent's recommendation, the gate form, live activity, questions & challenges, and
an agent audit where every call's prompt, context slice and raw output can be inspected. Light and
dark themes; all state is on disk, so a refresh or restart loses nothing.

## What makes it trustworthy

- **Agents recommend; the engine disposes.** Every agent answer is validated against a contract and
  engine checks (unknown fact ids, a champion outside the admissible set, a non-fittable family, an
  excluded target, a typed metric value…) and retried with the errors; failures are visible, never
  silent.
- **No assumed intent.** The Intake Analyst may only call a sponsor position "stated" with a quote
  the engine finds in the documents; everything else is asked, and the answers are on the record.
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
  engagement.py  development modes, the intent template, document reading, ingest into the run
  stages/        the 9 stages + stat_tests battery (engine work; judgment via ctx.recommend)
  modeling/      metrics, fitters (GLM links incl. probit/cloglog), ratchet search, hazard,
                 structural (Merton), migration, simulate (Vasicek + stress), credit_metrics,
                 transforms (target-hidden), guided (agent-guided search), ensemble
  ui/            the Dash + Mantine workbench
  service.py     the boundary the CLI and UI use
  cli.py  config.py  context.py  artifacts.py  orchestrator.py (compat)  okf.py  synth.py
  integrations/  impact_adapter, autoforge_loop
  runtime/       deployment scorer (IMPACT derived-field entry point)
projects/  examples/  evals/  tests/  docs/adr/ (0001–0011)  docs/templates/  CONTEXT.md (glossary)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FEATURES.md`](FEATURES.md), [`CONTEXT.md`](CONTEXT.md),
[`CHANGELOG.md`](CHANGELOG.md), the decision records in [`docs/adr/`](docs/adr/), and
[`CLAUDE.md`](CLAUDE.md) (principles for changing COGNOS itself).

## License

MIT.
