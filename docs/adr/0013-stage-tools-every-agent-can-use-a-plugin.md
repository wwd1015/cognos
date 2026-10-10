# Stage tools: every stage's agent can use a tool it was not built with (v1.3)

ADR-0012 gave the Data Analyst tools: it requests an analysis, the engine runs it. Plugins could
add tools, but only to data exploration. The other stages had whatever the engine computed and
nothing else. That is the wrong shape for where COGNOS sits: it is one part of a larger system,
and the tests an institution cares about most (its own validation suite, its segment tests, its
policy checks) are developed outside COGNOS. Validation in particular must be able to run custom
tests through tools such as IMPACT.

## Decision

1. **Every stage agent gets the request step.** Before the Intake Analyst, Design Lead, Modeler,
   Outcomes Analyst, Independent Validator, Model-Risk Analyst or Technical Writer recommends,
   the engine offers it the tools registered for its stage (`ctx.consult_tools`). The agent
   returns requests (`ToolRequestOutput`: tool, parameters, the question each run answers); the
   engine checks them, runs them, and hands the results back, for up to `analysis.rounds`
   rounds. The Data Analyst keeps its own step (`data_scout`), which may also write code.
   Agents still execute nothing.
2. **Tooling is pluggable; guidance is not.** A plugin adds what an agent can ask for. How an
   agent is told to think stays inside COGNOS: the request step runs under the agent's own role
   prompt plus one shared skill (`agents/prompts/_tools.md`). A plugin cannot ship a prompt, and
   a tool's description is data in the agent's context, not an instruction to it.
3. **A tool declares its stages and its inputs.** `Tool(stages=(...), needs=(...))`. The engine
   hands a tool exactly what it declared: `data` (the snapshot), `documents`, `results` (upstream
   stage payloads), `train`, `holdout`, `model`. A declaration the stage cannot satisfy is
   refused when the plugin loads.
4. **The sealed holdout stays sealed until the choice is made.** `holdout` and `model` exist
   only from `backtest` on. A tool registered for intake, explore, ideate or model cannot
   declare them, so the Modeler's tools see the development sample and nothing else. Tools at
   ideate and model are also not given the scores an earlier run recorded for the model being
   updated. Tool results are facts scoped like every other fact (`tools.<stage>.<id>.<name>`):
   an agent sees its own stage's and those of the stages it may already read.
5. **A tool can fail a test; it cannot block.** A result may carry `checks` (name, passed,
   severity, detail). A failed check is a finding of the stage. At validation a `high` one fails
   the validation, as a rubric finding of that severity does. Severity is capped at `high`:
   BLOCK stays reserved for confirmed leakage and stale documentation references.
6. **No tool, no step.** A stage with no available tool makes no extra agent call and behaves
   exactly as before. `analysis.stage_tools: false` switches the step off everywhere.
7. **An unavailable tool is on the record.** A tool may say it is unavailable and why
   (`available()`). It is not offered to the agent, and the stage records it under
   `tools_unavailable`; the workbench and the white paper list it as registered but not run. A
   reviewer sees that a test exists and was not performed.
8. **IMPACT is a registered placeholder.** IMPACT's test interface does not exist yet. COGNOS
   registers `impact_test_suite` for validation as unavailable, so every validation says the
   suite was not run. A later registration under the same name replaces an earlier one: the day
   IMPACT ships its tests, a plugin (or `integrations/impact_tools.py`) supplies the function
   and the Independent Validator can request it with no other change.

## What is kept

Every run is recorded in the stage payload (`tool_runs`) with the tool's origin and version, the
inputs it was given, its parameters and its checks; the full result (table, chart spec) is a
JSON artifact under `stages/<stage>/analyses/`, which the export includes. The white paper lists
tool runs by stage. Each request step is an audited agent call like any other.

## Consequences

- `AnalysisTool` gains `stages`, `needs`, `available` and `version`; `Tool` is the general name.
  A tool written for ADR-0012 is an explore tool over the dataset and needs no change.
- The deterministic agents run every tool offered to their stage once, with default parameters,
  so a plugin's tests run offline and in CI.
- With a live provider, each stage that has a tool costs at least one more agent call.

## Limits, stated plainly

- A tool is code the institution reviewed and installed; it runs in the engine's process with
  the inputs it declared. COGNOS limits what it is handed, not what it does. Do not install a
  plugin you have not read.
- Only the Data Analyst writes code. Elsewhere an agent can use a tool or go without.
- Tool runs are not re-executed at validation as scripts are: a tool is reviewed code, not
  agent-written code. Its version is recorded so a later reader knows what ran.
- A validation tool reads the sealed holdout after the champion is chosen. If its finding sends
  the run back to modeling, the next selection is no longer blind to it; the existing
  `holdout_evaluations` count and the `holdout-reuse` finding report that.
