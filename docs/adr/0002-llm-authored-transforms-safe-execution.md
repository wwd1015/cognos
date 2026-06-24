# Safe execution of LLM-authored feature transforms

The LLM may author arbitrary Python feature-engineering code (not a restricted DSL), following the
autoresearch model where auditability comes from *provenance* (every change is a commit) plus a
*frozen evaluator*, rather than from restricting the language. Arbitrary code is made safe by four
guards: (1) every LLM-authored change is committed to an experiment history; (2) the sealed holdout
and metric live in code the LLM cannot edit (no metric gaming); (3) **feature-construction code runs
against a target-hidden view of the data (`X` only; `y` is never in scope), so it physically cannot
leak the target**; (4) execution is process/worktree-isolated.

Guard (3) is the COGNOS-specific addition autoresearch does not need: autoresearch trains a language
model on frozen data with the metric computed entirely inside the un-editable evaluator, so target
leakage is impossible. COGNOS does supervised tabular learning where the label sits in the same table
as the features, and — because the holdout must keep its labels to be scored — a transform that reads
the target would leak *even on the holdout*, defeating the frozen-holdout backstop. Removing `y` from
the transform's scope is what closes that hole.

## Considered options

- **Restricted expression DSL (AST-whitelisted)** — safe and auditable by *restriction*; kept as an
  optional stricter mode for regulated/multi-tenant deployments where arbitrary execution is
  unacceptable. Not the default: it caps the feature-engineering creativity that is the point.
- **Arbitrary Python, audited by commit history + frozen substrate + target-hidden exec + isolation
  (chosen)** — the autoresearch model adapted for tabular supervised learning.

## Consequences

- A kept transform is a committed, importable function → it round-trips verbatim to an IMPACT
  derived field (`{function: 'dotted.path'}`) so train-time and serve-time feature logic are identical.
- Requires building the target-hidden execution scope and process isolation (new engine capability).
