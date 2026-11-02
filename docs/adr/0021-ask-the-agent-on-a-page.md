# Ask the agent on a page, and agents named for their work

A developer reading a stage page often needs to understand it before deciding anything: why a
column is excluded, why a specification ranks first, where a number comes from. The only way
to put a question to an agent was a send-back, which is a decision: it records a challenge,
marks work stale and re-runs the stage. There was nothing in between for "explain this to me".

The agents also carried job titles (Data Analyst, Design Lead, Independent Validator,
Model-Risk Analyst). In a bank those are people, and a reader could not tell from a sentence
whether a person or an agent had done something.

## Decision

1. **Every stage page has an Ask window.** A floating button opens a small window, bottom
   right, where a person asks the agent that owns the page (`contracts.STAGE_AGENT`). A page
   without an agent has no button. The window floats rather than taking a column: the
   workspace already has three, and a fourth does not fit a small display.

2. **A question is a step of the page's agent** (`<agent>_ask`, contract `AskAnswer`, prompt
   `_ask.md` after the role prompt). It has its agent's slice scope (`slices.fact_scope`), so
   the walls hold: the Validation agent never sees the Modeling agent's reasoning, and no agent
   that chooses sees a holdout fact. The page hands it only that agent's own recorded
   recommendation (`ask._page`), never another agent's.

3. **No LLM math here either.** The answer writes every number as a `{{fact:<id>}}`
   placeholder and lists what it relied on in `cites`; `checks.ask` rejects typed numbers and
   unknown ids, and the engine renders the answer. Offline, the deterministic agent answers from
   the record (the facts whose names or values share the question's words, and its recorded
   summary) and says that it did.

4. **Asking decides nothing.** It writes no state, no result and no gap. When the question
   shows something the stage should reconsider, the answer may carry a `suggested_challenge`;
   at an open gate the window copies it into the gate's send-back form, and a person still
   presses Send back. The interview with the sponsor stays the gaps (ADR-0011): the Ask window
   is not a second question store.

5. **Kept on the record.** Each turn (who asked, when, the question, the rendered answer, the
   facts cited, the call id) is kept in `runs/<id>/asks/<stage>.json`, written under the run's
   lock, never across the agent call. The call itself is audited like any other
   (`runs/<id>/agents/`, time and spend limits apply). Threads travel in the export.

6. **Agents are named for their work.** Intake, Data, Design, Modeling, Backtest, Validation,
   Compliance and Writing agent. Job titles belong to the people's seats (developer, reviewer,
   approver). The ids (`data_analyst`, `validator`, ...) do not change, since recorded runs
   and replays are keyed on them; `contracts.FRIENDLY` is the one place an id becomes a name.

## Consequences

- Questions cost agent calls. They count against the run's budget like any other call.
- A thread is advice a developer read, not a decision; the run record does not list it. A
  reviewer who wants to know what the developer was told reads the thread or the agent calls.
- Recorded runs and older documents keep the old titles where they were written.
