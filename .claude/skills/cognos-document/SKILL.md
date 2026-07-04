---
name: cognos-document
description: Whitepaper playbook for the document agent — SR 11-7 documentation structure, how the design brief and framework assessment feed "alternatives considered", and traceability standards.
---

# Document playbook — the model whitepaper (commercial risk)

The engine renders the OKF bundle, Model Card, and EU Annex IV pack. Use this playbook to judge
whether the documentation would survive a validator or examiner reading it cold.

## The SR 11-7 whitepaper skeleton

A commercial credit model document must let an independent party **reproduce and challenge**:

1. **Purpose & use** — the design brief's use case/horizon verbatim; out-of-scope uses stated.
2. **Portfolio & segment** — what population the sample represents, sampling window, vintages.
3. **Data** — sources, default definition, exclusions, treatment of leakage-flagged fields
   (dropped columns are documented, not silently vanished).
4. **Methodology & alternatives considered** — this is where ideate's `framework_assessment`
   lands: the chosen framework AND the rejected ones with reasons (structural rejected for missing
   market observables, etc.). An examiner reads absence of alternatives as absence of thought.
5. **Estimation results** — coefficients with signs interpreted economically, inference caveats
   for regularized families.
6. **Developmental evidence & outcomes analysis** — CV, sealed holdout, OOT Gini/KS, calibration,
   PSI, challenger-benchmark gap.
7. **Limitations & assumptions** — EPV, right-censoring, regime coverage (was a stress period in
   sample?).
8. **Monitoring plan** — metrics, thresholds, frequency, and the recalibration trigger.

## Traceability standard

- Every number in the paper traces to a run artifact; every code reference uses `{@code:…}` links
  the review gate can AST-verify. A pretty paper with untraceable numbers is worse than an ugly
  traceable one.
- The Model Card and Annex IV pack are *views* of the same facts — inconsistency between views is
  a finding.

## Voice

Write for the validator, not the developer: claims are qualified ("on the OOT sample", "for the
2019–2023 vintages"), the challenger gap is stated as the price of interpretability, and
open design questions from ideate appear as documented limitations if still unanswered.
