---
name: cognos-comply
description: Model-risk readiness playbook for the comply agent — SR 11-7 evidence mapping, NIST AI RMF, the human-only steps, and what ECOA/Reg B means for commercial credit.
---

# Comply playbook — model-risk readiness report (non-gating)

Comply organizes evidence; it never gates (ADR-0006). Use this playbook to judge whether the
readiness report is complete and honest about what remains **human-only**.

## SR 11-7 evidence map

A commercial credit model's readiness evidence groups into:

1. **Development** — design brief (use case, horizon, default definition, segment), data lineage,
   alternatives considered (ideate's framework assessment), estimation results with valid
   inference.
2. **Validation** — the effective-challenge rubric, outcomes analysis (discrimination,
   calibration, stability), benchmarking.
3. **Governance** — intended use / out-of-scope use, monitoring plan with thresholds (e.g. PSI
   0.10/0.25, Gini floor), issue-remediation path.

The report's job is traceability: every claim points at a run artifact.

## NIST AI RMF mapping

Govern (policies, roles) / Map (context, intended use) / Measure (the statistical evidence) /
Manage (monitoring, decommission triggers). The engine can fill Map and Measure from artifacts;
Govern and Manage are mostly organizational — they belong on the human-steps list, not claimed as
done.

## The human-only steps (never claim these)

- Model **risk-tier assignment** and inventory registration.
- **MRM committee approval** and effective-challenge sign-off by an independent human validator.
- **Monitoring thresholds adopted** by the model owner (the report proposes; owners adopt).
- Annual review scheduling; use-restriction attestations by the business.

A readiness report that reads as "compliant" is defective; the honest output is "evidence
organized, N steps outstanding, here is the list."

## Fair lending in *commercial* credit — the common misconception

ECOA / Regulation B **does apply to business-purpose credit**, not just consumer: adverse-action
notice duties (simplified for larger businesses), and the small-business data-collection rule
(Section 1071 / Reg B subpart B) for covered lenders. What differs from consumer lending is the
absence of protected attributes in typical obligor data and no FCRA score-disclosure regime.
COGNOS treats fair-lending scans as **optional** for commercial profiles (ADR-0004,
`compliance.fair_lending`); when it is off, the report should say "not in scope for this profile",
not "not applicable to commercial lending" — the sponsor decides scope.

## Verdict discipline

Comply always PASSes ("report produced"). If you feel an urge to fail the run from here, the
finding belongs in validate's or review's domain — say so instead of gating.
