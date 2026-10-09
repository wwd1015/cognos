# Agents recommend, humans decide, the engine disposes (v1.0)

ADR-0001 made COGNOS a two-layer system — reasoning proposes, the engine disposes — but in v0.x the
reasoning layer was thin: stages called an optional `brain.generate()` in a few places, parsed JSON
loosely, and the `.claude/agents` wrappers only ran one CLI command each. Humans could approve or
reject only at the two verdict gates. v1.0 adopts the design proven in Cyber Credit Officer and makes
the agents, the human decisions, and the engine's checks first-class.

## Decision

1. **Every judgment stage has an agent that recommends.** Seven agents (data analyst, design lead,
   modeler, outcomes analyst, independent validator, model-risk analyst, technical writer) each have a
   role prompt (the former `cognos-*` playbooks), a strict output contract, an independence-scoped
   context slice, and engine checks. A rejected answer is retried with the errors fed back; every
   attempt is audited with its raw input and output. `review` stays purely deterministic.
2. **Humans decide at five review gates** — data, design, champion, validation, sign-off — with
   accept / edit / override / send back / approve / reject. Decisions shape the *effective* config
   through overrides; the profile YAML is never edited. Autonomous mode auto-accepts (recorded as
   `auto`) for prototypes.
3. **One pushback mechanism.** A human send-back and a high-severity validator finding are the same
   object — a Challenge routed to a stage's agent, which must answer it. Validator findings loop back
   automatically at most `workflow.auto_challenge_loops` times; the rest reach the human.
4. **Open questions are tracked gaps.** Design questions the sponsor has not answered are records the
   human answers (filling the design brief and re-running the stage) or accepts as assumptions.
5. **The engine is a DAG runner with staleness.** A changed decision marks everything downstream
   stale; the engine re-runs it. State lives on disk (`state.json`), so any process, the CLI, or a
   browser refresh resumes exactly.
6. **Agents run in the engine**, not as Claude Code subagents: `claude -p` (isolated), the Anthropic
   API, or any OpenAI-compatible API — plus the deterministic `heuristic` agents and `replay`.
7. **No LLM math.** Agents cite fact ids; the writer's prose carries `{{fact:<id>}}` placeholders the
   engine renders; unknown ids and typed metric values are rejected.

## Invariants carried forward

- The frozen substrate: the modeler chooses from the admissible set (one-standard-error rule) before
  the sealed holdout is scored and never sees holdout evidence; holdout evaluations are counted and
  re-selection is flagged by the validator.
- Only the engine BLOCKs (confirmed leakage at `validate`, stale code references at `review`); agents
  can raise FAIL/WARN. A BLOCK can never be accepted at a gate — only sent back or rejected.
- Offline: every stage runs with the heuristic agents; tests never need a key (ADR-0003's two-tier
  reproducibility now reads: the analysis is reproducible offline; the recommendations and decisions
  are recorded — agent audit + decision log — and replayable via the `replay` provider).

## Consequences

- The `brains/` package and the `.claude/agents` + `/cognos-run` orchestrator are removed; the
  playbooks live on as `src/cognos/agents/prompts/`. `Orchestrator` / `run_pipeline` remain as a
  compatibility wrapper (review gates auto-accepted; a legacy gate handler can no longer approve a
  BLOCK).
- A Dash + Mantine workbench (`cognos ui`) is the primary interface for model developers; the CLI
  exposes the same decisions (`cognos gate`, `cognos answer`, `cognos status`).
- The white paper gains a decision log: every agent recommendation (with its backend), every human
  decision, and every challenge with its response.
