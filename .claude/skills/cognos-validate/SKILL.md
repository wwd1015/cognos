---
name: cognos-validate
description: SR 11-7 effective-challenge playbook for the validate agent — independence discipline, the leakage confirmation protocol, benchmarking, and BLOCK materiality.
---

# Validate playbook — effective challenge (SR 11-7)

Validate is the first gate and the pipeline's **independent** second line. The engine re-derives
risk from artifacts; use this playbook to judge like a validator, not like the developer.

## Independence discipline

- Never accept the modeling stage's reasoning as evidence — the rubric re-derives every risk from
  the artifacts (data, ledger, diagnostics, holdout). If validate's evidence merely quotes model's
  summary, that is a defect in the run worth surfacing.
- The champion's CV score is the developer's claim; the sealed holdout and OOT outcomes are the
  validator's facts.

## Leakage confirmation protocol (the only hard BLOCK)

Explore *suspects* leakage; validate *confirms* it. Confirmation means the evidence chain holds:
a near-deterministic feature↔target relationship AND the champion actually uses the feature AND
no sponsor attestation that the field exists at prediction time. Confirmed target leakage on a
commercial default model (post-outcome DPD, charge-off, workout fields) invalidates every metric
downstream — that is why it is the hard BLOCK. The remedy is always the same: drop the column and
re-run from explore.

## The challenge checklist

1. **Conceptual soundness** — does the framework match the data structure (see the ideate design
   brief)? Were alternatives considered and rejected for stated reasons?
2. **Developmental evidence** — EPV adequate; signs match economic priors; significance not
   over-claimed for regularized families.
3. **Overfitting** — CV→holdout gap; ledger breadth (was the "best" model one of dozens tried on
   the same folds?).
4. **Benchmarking** — the champion must beat naive baselines (single-ratio leverage model,
   base-rate predictor) by a margin that justifies its complexity; a challenger far above the
   champion means the interpretability price should be stated, not hidden.
5. **Stability** — coefficients and discrimination should hold across vintages/segments, not just
   in aggregate.

## Materiality — when to BLOCK vs WARN

- **BLOCK**: confirmed target leakage, or evidence the frozen substrate was violated (holdout
  touched during search). High-confidence, model-invalidating harm only.
- **FAIL/WARN**: overfit gaps, failed diagnostics, thin events, sign flips — real findings the
  human weighs, but noisy signals never BLOCK (CLAUDE.md non-negotiable #6).
- In interactive mode the human decides at the gate; your job is that they decide on verbatim
  findings, ranked by severity, with the evidence chain stated.
