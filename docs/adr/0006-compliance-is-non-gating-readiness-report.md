# Compliance is an optional, non-gating model-risk readiness report

The `comply` stage is demoted from a verdict-issuing gate to an **optional, non-gating model-risk
readiness report**. It neither PASSes nor BLOCKs.

An agent cannot *adjudicate* regulatory compliance — SR 11-7 requires sign-off by a human validator
independent of development — so a PASS/BLOCK on "compliance" is overreach, and a verdict tempts
rubber-stamping, which is the worst possible failure for a tool whose entire purpose is trustworthy,
auditable output. The substance compliance cares about (defensible coefficients, calibration,
stability, reproducible documentation) is essential and is already produced by the `model`,
`backtest`, `validate`, and `document` stages; `comply` only *organizes* it.

## What the report does

- Maps the evidence the other stages produced into the SR 11-7 structure (conceptual soundness /
  ongoing monitoring / outcomes analysis).
- **Never marks an element compliant without concrete, checkable evidence** — unevidenced items are
  listed as outstanding gaps, never silently passed. (This subsumes the earlier "must not overclaim"
  concern.)
- Explicitly lists the human-only steps that remain: independent validation sign-off, a monitoring
  plan with thresholds, override/governance policy.
- Is opt-in (on when prepping a regulated commercial model for MRM submission; off for prototypes).

## Consequences

- The only gates are now **`validate`** (technical soundness) and **`review`** (docs↔code
  consistency). This supersedes the original design where `comply` was a gate.
- COGNOS's value concentrates in the substantive stages; compliance is a deliverable, not a judge.
