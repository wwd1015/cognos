# Reasoning proposes, the deterministic engine disposes

COGNOS is a two-layer system in which an LLM reasoning layer and a deterministic engine are both
first-class and interdependent. The reasoning layer **proposes** decisions a human modeler would make
(design, model choice, feature engineering, and the next experiment to try); the deterministic engine
**disposes** — it fits, scores on a frozen metric and sealed holdout, runs the statistical battery,
and is the sole authority on what is kept and on any metric or verdict that becomes a recorded fact.
This is the invariant that bounds hallucination: an LLM proposal cannot enter the record unless the
engine independently verifies it measurably improves on the auditable yardstick.

The LLM enters the loop in two staged depths: **(A)** LLM-driven ideation that emits *executable*
candidate specs (not just prose), then **(B)** an opt-in LLM-guided search where the LLM is the
mutation function proposing the next experiment from the ledger. The deterministic grid search remains
the baseline and the test double.

## Considered options

- **Pure-LLM agents that reason their way through modeling** — rejected: not reproducible or
  auditable, and hallucination-prone (the core criticism we exist to avoid).
- **Pure deterministic AutoML** — rejected: cannot automate the human judgment (design, model choice,
  path exploration) that is the point; reasoning is not a nice-to-have.
- **Hybrid: reasoning drives, determinism grounds (chosen)** — gets automation of judgment *and*
  reproducibility/auditability, because every proposal is empirically verified before it counts.

## Consequences

- Requires safe execution of LLM-proposed feature transforms (see ADR-0002).
- Requires redefining "reproducible" for the LLM-in-the-loop path as *replayable from a recorded
  transcript* (pinned prompts + temperature 0 + saved responses), not bit-identical re-derivation
  (see ADR-0003).
- Running with no LLM is a deterministic **test double + degraded offline mode**, not a co-equal
  product mode.
