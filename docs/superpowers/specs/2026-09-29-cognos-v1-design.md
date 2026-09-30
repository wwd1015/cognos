# COGNOS v1.0 — agents recommend, humans decide, the engine disposes

Date: 2026-09-29 · Status: approved for build (owner delegated all phase approvals) · Branch: `revamp`

## 1. Intent

**What the owner asked for.** Revamp COGNOS to adopt the design choices proven in Cyber Credit Officer
(CCO): LLM agents make the *recommendation* at every stage; a model developer *reviews, challenges or
sends it back* at human gates; the deterministic path exists only for offline/demo use. Include a
UI (Dash + Mantine, for model developers). Run agents the way CCO does (in-engine runner:
`claude -p` / Anthropic API / OpenAI-compatible), retiring the `.claude/agents` wrappers. Ship as v1.0.

**Assumptions (made explicit).**
- COGNOS's anti-reward-hacking core is untouched: the engine computes every number; the sealed
  holdout and metric definitions are not agent-editable; `validate` (leakage) and `review` (stale
  code refs) remain the only BLOCKs, and a BLOCK is never overridable by accept.
- One mechanism for pushback: a human send-back and an automated validator finding are the same
  object (a **Challenge**) routed to a stage's agent, which must answer it.
- The `cognos-*` skill playbooks become the agents' role prompts (no knowledge is lost).
- CLAUDE.md non-negotiable #1 is reworded: *every stage runs offline with the deterministic
  (heuristic) agents; tests never need a key or network.*

**Success.** A model developer opens the UI, creates a run on a dataset + design brief, watches agents
work live, reviews each recommendation (accept / edit / override / send back), and ends with a
champion, an independent validation, and an OKF white paper that records every agent recommendation
and human decision. The same run completes offline with heuristic agents; `pytest` + `ruff` stay
green with no key.

## 2. Architecture

```
            ┌──────────── Dash + Mantine UI (cognos ui) ────────────┐      CLI (cognos …)
            │ runs list · run workspace · gate panels · audit view  │          │
            └───────────────────────────┬───────────────────────────┘          │
                                        ▼                                      ▼
                         cognos.service  (the single API boundary: create run, advance,
                                          submit gate, answer gap, retry, read state/events)
                                        │
             ┌──────────────────────────┴───────────────────────────┐
             ▼                                                      ▼
   engine/  deterministic workflow                        agents/  recommendation layer
   graph (STEPS/DEPS) · RunState · gates ·                prompts · contracts · slices ·
   staleness · challenges · gaps · events                 runner (validate+retry+audit) ·
             │                                            providers: heuristic | replay |
             ▼                                            claude_cli | anthropic | openai_compat
   stages/  engine work per stage: prepare → ctx.recommend(agent) → finalize (engine disposes)
   modeling/ · okf · runtime  (unchanged numerical core, frozen substrate)
```

Layers do not blur: the engine never calls an LLM directly; stages obtain judgment only through
`ctx.recommend(...)`; agents never compute a recorded number; the UI and CLI only talk to `service`.

## 3. The workflow graph

Steps are stages (engine + agent) and gates (human decision points). Linear by default, but the
engine is a generic DAG runner (`DEPS`) with staleness:

```
explore → gate_data → ideate → gate_design → model → gate_champion → backtest → validate
        → gate_validation → comply → document → review → gate_signoff
```

Step status: `pending | running | done | awaiting | stale | failed | blocked | skipped`.
`skipped` (a disabled stage or gate) satisfies dependencies. Editing a completed gate or sending
work back marks the target step and all descendants `stale`; stale steps re-run. A step that is
marked stale while running stays stale when it finishes (and re-runs).

**Modes.** `interactive`: gates pause (`awaiting`) for a human decision. `autonomous`: gates
auto-accept the agent's recommendation (recorded as `auto`). In both modes a `BLOCK` from `validate`
or `review` sets the step `blocked` and descendants do not run (`halt_on_block`); in interactive
mode the human's only options on a BLOCK are *send back* or *reject*.

## 4. Human gates

| Gate | Shows | Actions (all recorded as `GateDecision` with reason) | Effect |
|---|---|---|---|
| `gate_data` | profile, leakage suspects, data analyst's column decisions | accept · edit exclusions · send back | `overrides.exclude_columns`; downstream uses effective features |
| `gate_design` | frameworks, slate, transforms, sponsor questions (gaps) | accept · edit slate · answer questions · send back | answers → `overrides.design`, re-enter at ideate; slate edits → `overrides.slate` |
| `gate_champion` | admissible set, recommended champion + rationale, sign checks, coefficients, diagnostics, holdout of the chosen | accept · override champion (admissible set) · send back | `overrides.champion`; re-finalize model; holdout evaluations counted |
| `gate_validation` | verdict, rubric, findings, validator recommendation | accept (reason required on FAIL; disabled on BLOCK) · send back to explore/ideate/model | Challenge routed to the stage agent |
| `gate_signoff` | review verdict, white paper | approve · reject | run status `approved`/`rejected` |

Gates can be disabled per project (`workflow.gates`).

## 5. Agents

Each agent: a role prompt (`agents/prompts/<agent>.md` + shared `_common.md`), a Pydantic output
contract (`extra="forbid"`), a context **slice** (what it may see), an engine **validator** (what
the engine re-checks), and a deterministic **heuristic** implementation (the offline path, a pure
function of the same slice).

| Agent | Stage | Recommends | Engine re-checks (reject → retry with errors) |
|---|---|---|---|
| `data_analyst` | explore | keep/exclude per leakage suspect & quality issue; data-quality claims | every suspect decided; columns exist; target never excluded |
| `design_lead` | ideate | framework roles, ranked slate, feature transforms, sponsor questions | families engine-fittable; frameworks consistent with applicability; transforms validated target-hidden |
| `modeler` | model | champion from the engine's **admissible set**; economic sign checks; (opt-in) guided experiments | pick ∈ admissible set; features exist; guided proposals kept only if CV beats incumbent |
| `outcomes_analyst` | backtest | interpretation of Gini/KS/calibration/PSI; findings | cited facts exist |
| `validator` | validate | independent findings (severity, target stage, remedy); approve / conditions / send back | cited facts exist; target stage valid; cannot BLOCK |
| `risk_analyst` | comply | readiness narrative, priority human actions | cited facts exist |
| `writer` | document | narrative sections using `{{fact:<id>}}` placeholders | unknown fact ids rejected; bare metric-like numbers rejected (no LLM math) |

`review` stays purely deterministic (AST docs↔code check) — no agent.

**Evidence.** Every slice carries a flat `facts` map (`"model.cv_mean": 0.7812`, …). Claims and
findings cite fact ids; unknown ids are rejected. Numbers in recorded prose come only from facts.

**Independence.** The validator's slice contains artifacts and engine metrics, never the modeler's
rationale. The modeler never sees holdout metrics when choosing (holdout is evaluated after the
choice, in `finalize`).

**Admissible set (model).** Evaluated candidates whose CV mean is within one CV standard deviation
of the best (one-standard-error rule), capped at 6, plus their coefficient signs from a train-only
fit. The heuristic modeler picks the ratchet champion (identical to v0.5 behaviour).

**Challenges.** `Challenge{id, source: human|validator, target_stage, severity, message, evidence,
status: open|answered|closed, response}`. Open challenges for a stage appear in its agent's context;
the contract requires one `responses_to_challenges` entry per challenge id. Validator findings with
severity `high` and a target stage auto-route back up to `workflow.auto_challenge_loops` times
(default 2); what remains goes to `gate_validation`. BLOCKs never auto-route.

**Gaps.** Ideate's open questions become `Gap{id, question, category: design|data, status:
open|answered|assumed, answer, reentry}`. Answering a design gap writes `overrides.design.<field>`
(when the gap maps to a design field) and re-enters at ideate.

## 6. Providers and the runner

`AgentRunner.recommend(agent, stage, context, contract, validate)`: builds the prompt (task +
context JSON + challenges + previous validation errors), calls the backend, validates with Pydantic
and the engine validator, retries up to `agents.max_retries` (default 2) with the errors, writes raw
input/output to `runs/<id>/agents/<call_id>.{input,output}.json`, appends an `AuditEntry` (agent,
stage, attempt, provider, model, prompt hash, context hash, status, duration, cost, turns) to
`runs/<id>/agents/audit.jsonl`, and emits events. A per-call wall-clock limit and a per-run spend
budget apply. On exhausted retries or backend failure the step fails (UI shows it; `retry` re-runs).

| Provider | Kind | Notes |
|---|---|---|
| `heuristic` | deterministic | default for tests and offline demo; no key |
| `replay` | recorded | replays a directory of recorded outputs (`<agent>.json`) — tests the plumbing |
| `claude_cli` | `claude -p` | empty temp cwd, `--setting-sources ""`, `--tools ""`, `--json-schema`, `--strict-mcp-config` |
| `anthropic` | Messages API | structured outputs from the contract schema; Pydantic re-validates |
| `openai` / `xai` / `openrouter` / `ollama` … | `openai_compat` | final answer via a `submit_answer` function call |

Resolution: `COGNOS_PROVIDER` env → `agents.provider` in the profile → `auto` (first available of
claude_cli, anthropic, openai_compat entries; else heuristic). Tests pin `heuristic`.
v1.0 agents are single-shot (no tool use): the slice is the complete context. Guided search is an
iterative loop of single-shot `modeler` experiment proposals, each disposed by the engine.

## 7. Run state and the service API

`runs/<id>/state.json` (`RunState`): steps, gate decisions, challenges, gaps, overrides, provider,
spend, loop counters, status (`running|awaiting|blocked|failed|completed|approved|rejected`).
Read-modify-write happens under a per-run lock, re-loading from disk. `events.jsonl` is the live
activity feed. Heavy artifacts stay by reference under `runs/<id>/` (unchanged).

`RunContext.config` is the **effective** config: profile + overrides (exclusions append to
`data.drop_columns`, design answers fill `design`). `ctx.features()` = explore features minus
exclusions.

`cognos.service`: `create_run(profile|config, mode, provider)`, `start(run_id)` (background
advance), `run_until_idle(run_id)` (sync), `submit_gate(run_id, gate, action, payload, reason)`,
`answer_gap(...)`, `retry(run_id, step)`, `state(run_id)`, `events(run_id, since)`, `list_runs()`.
`Orchestrator` / `run_pipeline` remain as thin compatibility wrappers over the engine.

## 8. UI (Dash + Mantine)

`cognos ui` launches the app. Pages: **Runs** (list + new-run drawer: project profile or synthetic
demo preset, mode, provider with availability) and **Run workspace**: step rail with status badges,
header (project, provider, spend, verdict), main panel per step, live activity feed, and an **Audit**
tab (agent calls with raw I/O, gate decisions, challenges). Stage panels show the engine's evidence
(profile table, framework/slate tables, ledger chart, coefficient chart, calibration chart, rubric,
findings, rendered white paper) beside the agent's recommendation; the gate form sits below. The
browser polls state every second; a refresh loses nothing (all state is on disk).

## 9. Error handling

- Agent failure after retries → step `failed` with message; `retry` re-runs; audit keeps every attempt.
- A stage crash is a verdict (`ERROR`), as today.
- Missing SDK/key/CLI → provider unavailable, never a crash; `auto` falls back to heuristic.
- Spend budget exceeded → agent call refused with a clear message (step fails, retry after raising).

## 10. Testing

- Unit: graph/staleness, state persistence, gate handlers, challenge routing, gap re-entry, contracts,
  validators, heuristic agents, runner retry/validation with a fake backend, provider resolution,
  fact placeholders.
- Integration: full autonomous heuristic run; interactive run driven through `service` (accept,
  override champion, send back, answer gap); leakage BLOCK → send back → completes; replay provider
  run; CLI commands; all existing suites (numerical core unchanged).
- UI: every page and step panel builds for a completed run; gate callbacks drive the service.
- Live check (manual, not CI): one real `claude_cli` run recorded as an example.

## 11. Out of scope for v1.0

Agent tool use (MCP), multi-user auth, remote deployment, a REST adapter (the service layer is the
seam for one), React front end.
