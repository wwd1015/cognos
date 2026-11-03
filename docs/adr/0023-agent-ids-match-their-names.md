# Agent ids match their names

ADR-0021 renamed the agents for their work (Data agent, Validation agent, ...) but kept the
old ids (`data_analyst`, `validator`, ...), because recorded runs and replays were keyed on
them. The ids still showed wherever a page or a file printed one raw: the run record listed
`intake_analyst` beside "Intake agent", audit file names said `writer`, a prompt file was
`modeler.md`. Two names for one agent is the confusion ADR-0021 set out to remove.

## Decision

1. **An id is the display name in snake case**: `intake_agent`, `data_agent`,
   `data_agent_requests`, `data_agent_joins`, `design_agent`, `modeling_agent`,
   `modeling_agent_search`, `backtest_agent`, `validation_agent`, `compliance_agent`,
   `writing_agent`. Steps keep their suffixes (`_tools`, `_ask`). Contracts, checks,
   heuristics, prompts and slice scopes use the new ids; so do the payload key the Validation
   agent writes (`validation_agent`), a challenge's `source`, and the loop count.
   `tests/unit/test_agent_ids.py` holds every id to its name.

2. **Old records are read, never rewritten.** `agents/ids.py` maps each old id to the new one.
   `canonical` is used where an id is read from disk (a challenge's source, the loop count, a
   recorded recommendation); `FRIENDLY` names the old ids as well, so an old run's pages show
   the current names; a replay looks for the new file name, then the old one.

## Consequences

- Runs, audit files and examples recorded before this keep their old ids on disk.
- A new agent's id is its name: there is no separate list of display names to keep in step.
