---
type: engagement
title: Business Intent & Model Change Record
description: What the sponsor asked for, what stayed open, and what an update changed.
resource: stages/intake/brief.json
tags:
- intent
- governance
timestamp: '2026-10-10T02:10:23.517195+00:00'
---

# Business Intent

- **Development mode:** Model update
- **Objective:** Rank middle-market commercial borrowers by default risk so that underwriters price and size new facilities consistently.
- **Intent at intake:** clear

| Field | Sponsor position | Source |
| --- | --- | --- |
| Business objective | Rank middle-market commercial borrowers by default risk so that underwriters price and size new facilities consistently. | intent document |
| Decision the model supports | Origination underwriting | intent document |
| Portfolio segment | C&I middle-market | intent document |
| Outcome definition | 90+ days past due or nonaccrual | intent document |
| Outcome horizon | 12 months from origination | intent document |
| Intended use and users | not stated |  |
| Out-of-scope uses | not stated |  |
| Interpretability requirement | Required: a credit officer must be able to explain a decline. | intent document |
| Success criteria | Rank-ordering no worse than the current expert scorecard on an out-of-time sample. | intent document |
| Constraints | not stated |  |
| Data sources | not stated |  |
| Stakeholders | not stated |  |
| Existing model | Commercial PD v1 (the first run of this demo), in use for one year. | intent document |
| Reason for the update | The annual review found the model under-predicting default in the two latest vintages. | intent document |
| Requested changes | - Refresh the development data through the latest vintage - Re-estimate the coefficients on the refreshed sample - Close validation finding V-12 on calibration | intent document |
| What must not change | The segment, the default definition and the 12-month horizon. | intent document |
| Known issues and findings | not stated |  |

## Open at intake

Nothing.

## Documents received

| Document | Role | SHA-256 |
| --- | --- | --- |
| update_request.md | intent | 1167bc94e733a593 |
| score_v1.py | code | ac2300c4e1f0ef0e |
| prior_run_whitepaper.md | whitepaper | cf8364de03da94cd |

## Model change record

- **Scope:** re estimate — The deepest of the 3 requested change(s) is data refresh.
- **Existing model family:** ridge_logit

| Requested change | Type | Enters at |
| --- | --- | --- |
| Refresh the development data through the latest vintage | data refresh | explore |
| Re-estimate the coefficients on the refreshed sample | re estimation | ideate |
| Close validation finding V-12 on calibration | remediation | model |

## Existing model against this update

The existing model is COGNOS run `20261010T021022Z-6aae664`. Both columns are what each run recorded; the two runs keep separate sealed holdouts, so the holdout figures are not measured on the same sample.

|  | Existing model | This update |
| --- | --- | --- |
| Champion family | ridge_logit | ridge_logit |
| Features | 6 | 6 |
| Metric | roc_auc | roc_auc |
| CV roc_auc | 0.8182 | 0.8182 |
| Holdout roc_auc | 0.8531 | 0.8531 |
| Validation verdict | PASS | see the validation section |
