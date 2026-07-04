---
name: cognos-explore
description: Commercial credit data-quality playbook for the explore agent — information-set discipline, post-outcome leakage patterns, default-flag conventions, ratio sanity ranges, and event support.
---

# Explore playbook — commercial credit data quality

Domain knowledge for judging an `explore` result on a commercial-risk dataset. The engine computes
the profile; you use this playbook to decide which findings matter and what to say to the human.

## Information-set discipline (the #1 kill criterion)

Every feature must be knowable **at the moment the score would be used** (origination or the
surveillance snapshot). Classic post-outcome leaks in commercial credit data pulls:

- **Days-past-due measured at/after the outcome window** (`dpd_at_outcome`, `max_dpd_12m`) — near
  deterministic of the default flag itself.
- **Recovery / charge-off / workout fields** (`chargeoff_amount`, `recovery_rate`, `workout_flag`,
  `restructure_date`) — only exist because the default happened.
- **Post-event rating actions** — an internal grade refreshed after the borrower defaulted.
- **Statement data spread after origination** — financials with an as-of date inside the outcome
  window.

A |corr| ≥ 0.98 leakage suspect on a commercial default flag is almost never a "great predictor";
treat it as leakage until the sponsor proves the field is in the origination information set. The
remedy is `data.drop_columns`, not a modeling trick.

## Default-flag conventions

- Standard commercial default definitions: **90+ DPD, nonaccrual, bankruptcy, or charge-off**
  within a fixed window (usually 12 months from the observation point; Basel uses 90 DPD/unlikely
  to pay).
- **Right-censoring:** the newest vintages have not lived the full outcome window — their observed
  default rate is biased low. If the sample includes very recent cohorts, flag it.
- Typical annual commercial default rates are **1–5%** (middle market higher than large corporate).
  A rate far above ~10% suggests a mislabeled flag or a stressed/workout subpopulation; far below
  ~0.5% means event counts will starve the model.

## Financial-ratio sanity ranges (middle-market obligors)

| Ratio | Plausible range | Outside it usually means |
|---|---|---|
| Debt/EBITDA | 0–12x | negative-EBITDA obligors coded as huge positives; spreading errors |
| Interest coverage (EBITDA/interest) | 0–25x | zero-interest denominators |
| Current ratio | 0.2–5 | consolidation or unit errors |
| Operating margin | −35%…+45% | percent-vs-decimal confusion |
| Revolver utilization | 0–100% | committed-line data errors |

Missingness in commercial financials is usually **not at random** — smaller/private borrowers file
less; heavy missingness on statement fields correlates with risk. A ≥30% missing statement field is
a modeling decision (segment or drop), not an imputation detail.

## Event support

Note the event count, not just the row count: a 50k-row portfolio with 200 defaults supports a
handful of coefficients. Downstream (ideate) applies the events-per-variable rule; your job is to
make the raw counts visible.

## What to escalate loudly

1. Any leakage suspect (poisons every downstream stage — cheapest catch in the pipeline).
2. Default rate incompatible with the stated definition/window.
3. Right-censored recent vintages when a `datetime_col` is present.
4. Ratio columns outside sanity ranges (spreading/unit errors).
