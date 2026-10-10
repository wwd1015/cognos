# Development modes and intake: a run starts from the sponsor's intent (v1.1)

Through v1.0 a run started from a dataset and a profile. What the model was *for* lived in six
free-text fields of the `design:` block, and the first time anyone was asked about a blank one was
the design gate, after the data had been profiled. There was one way to run: build a model from
nothing. Real model development has at least two, and both begin with a document, not a dataset: a
sponsor's statement of intent for a new model, or a change request against a model that already
exists and has a white paper, code and a validation history.

## Decision

1. **Two development modes**, `engagement.kind`: `new` (a complete new model development) and
   `update` (a change to an existing model). The name is deliberately not "mode": that word
   already means autonomous / interactive. The development mode changes what a run starts from and
   what its agents are told; it never changes how the engine measures.
2. **One template for both.** The business intent document is a Markdown template the sponsor fills
   in before the run (`cognos intent-template`, the workbench's download button,
   `docs/templates/`). The model update request is the same document with five more sections (the
   existing model, the reason, the requested changes, what must not change, known findings). The
   engine parses the template mechanically; a free-form document is read by the agent.
3. **A new first stage, `intake`, with its own agent and gate.** The **Intake Analyst** restates
   the goal, fills an engagement brief, and writes interview questions. `gate_intent` belongs to
   the model developer, who answers for the sponsor or carries the questions to them. Nothing
   downstream runs before it: no data is profiled against an intent nobody has confirmed.
4. **The interview is the gap machinery, not a chat.** A question is a gap; an answer re-runs
   intake; the gate re-opens with whatever is still open. Every turn is a recorded stage run with
   an audited agent call, so the interview is on the record and replayable like any other
   recommendation. A required field that is not stated is a *blocking* question.
5. **The agent cannot invent the sponsor's position.** A brief entry is `stated`, `inferred` or
   `missing`. `stated` must carry a quote the engine finds in the documents the agent was shown;
   otherwise the answer is rejected and retried. This is the "no LLM math" rule applied to intent.
   An inferred position is never applied; it is asked about.
6. **A confirmed document becomes configuration by the existing route.** On accept, what the
   document states about use, horizon, default definition, segment, interpretability and intended
   use is written to the run's overrides, exactly as a gate answer is. An answer already on the
   run wins over the document; the document wins over the profile (and a disagreement is a
   finding). The profile YAML is still never edited.
7. **A blocking question does not deadlock a run.** The developer may confirm the intent with
   questions open by giving a reason, the same rule as accepting a FAIL at validation. The
   questions stay open, and the four sponsor decisions still cannot be assumed or signed past.
   Autonomous (express) preparation accepts the gate as it accepts the others.
8. **The prior model is read mechanically; its scores are scoped.** For an update the engine
   inventories and hashes the artifacts, counts the model families they name, finds the dataset
   columns they refer to, and, when the prior model is an earlier COGNOS run, reads that run's
   recorded champion and metrics. The family and inputs go to the design lead (the incumbent is
   always on the slate, ranked first). The scores are `prior.*` facts for the agents that read
   results only: the modeler still chooses blind.
9. **The engine re-estimates in every update.** `recalibrate` / `re_estimate` / `redevelop` is the
   Intake Analyst's classification of how deep the request goes. It steers the design (stay close
   to the incumbent, or reopen it) and the validator (how much to re-examine). The engine has no
   intercept-only recalibration, and the scope does not pretend otherwise.

## Invariants carried forward

- Offline: the heuristic Intake Analyst reads the template as written (a filled section is stated
  and quoted, an empty required one is a blocking question) and passes the same checks. A run
  with no intent document still works: the brief is assembled from the profile and the interview
  covers the rest.
- Heavy artifacts by reference: documents are copied into `runs/<id>/inputs/` when the run is
  created (a missing one refuses the run), and the text the engine read is kept in
  `stages/intake/corpus.json`. The stage payload carries the brief and an inventory, never the text.
- The sealed holdout stays sealed: intake never loads the dataset. An earlier run's holdout figure
  is that run's record, reported beside this run's and labelled as measured on a different sample.
- Runs recorded before v1.1 have no intake step; on load it reads as skipped.

## Consequences

- Nine stages and six review gates. An interactive run now pauses first at `gate_intent`.
- The four core design questions are raised once, by intake, under the ids `ideate` used
  (`design-<field>`). After the intent is confirmed, answering one re-enters at `ideate`, as
  before; an answer to any other intake question re-runs intake and everything after it.
- Every agent's context gains `project.engagement` (the confirmed brief). The validator's checklist
  gains "fit to the intent"; the white paper gains a business-intent section and, for an update, a
  model change record.
- Uploaded originals are not exported (they may be binary, and the prior model's code is not ours
  to redistribute by default); the brief and the text as read are.
- Not done here, on purpose: materiality rules that route an update to a lighter validation, a
  true recalibration fitter, and re-scoring an external prior model on the new holdout. Each needs
  an engine capability, not an agent opinion.
