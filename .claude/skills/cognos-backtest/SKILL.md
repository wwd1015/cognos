---
name: cognos-backtest
description: SR 11-7 outcomes-analysis playbook for the backtest agent — out-of-time design, Gini/KS benchmarks, calibration reading, and PSI thresholds for commercial credit.
---

# Backtest playbook — SR 11-7 outcomes analysis (commercial credit)

The engine scores the champion via IMPACT and computes the outcomes analysis; use this playbook to
read the numbers like a model-risk analyst.

## Out-of-time (OOT) design

- The evaluation sample should be the **latest vintages held out in time**, not a random split —
  a PD model's job is predicting the future, and OOT is the honest test.
- **Right-censoring caveat:** the newest cohorts may not have lived the full outcome window; their
  realized default rate is biased low, which depresses measured calibration, not discrimination.

## Discrimination — Gini / KS

- Gini = 2·AUC − 1. Wholesale/commercial obligor models typically land at **Gini 40–70%**
  (AUC 0.70–0.85). KS typically 25–50 on the same portfolios.
- Gini **> 85–90%** on an OOT sample is a leakage alarm, not a triumph — cross-check explore's
  leakage suspects before endorsing.
- Discrimination degrading a few points from CV to OOT is normal; a collapse (> 15 Gini points)
  suggests regime change or an unstable feature.

## Calibration

- Compare predicted PD vs realized default rate overall and by score decile. Systematic
  over-prediction on post-stress vintages (or under-prediction going into stress) is expected for
  point-in-time models with macro covariates — describe the direction, don't just pass/fail.
- With few events per decile, calibration tests are noisy — a failed Hosmer–Lemeshow style test on
  < 20 events/bucket is weak evidence.

## Stability — PSI

Industry-standard thresholds for the population stability index:

| PSI | Reading |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | monitor / investigate drivers |
| > 0.25 | significant shift — recalibration candidate |

Score-level PSI answers "did the scored population move?"; feature-level PSI tells you *why*. A
high PSI driven by the macro covariates around a stress period is explainable; a high PSI on a
financial ratio hints at a data-pull change.

## Portfolio simulation & stress (opt-in sections)

When the config enables them, the payload carries two simulation sections — read them as *reports
conditioned on the model's PDs*, not as new model verdicts:

- **`portfolio_analysis`** (Vasicek one-factor, seeded MC + closed-form IRB): EL should ≈ mean
  PD × LGD; VaR/ES at the configured confidence are the tail; `irb_capital_mean` is the analytic
  Basel-formula cross-check. All losses are fractions of total EAD. **Caveat to always state:**
  PDs in are the model's scores — if the calibration section shows bias, the loss distribution
  inherits it. The asset correlation defaults to the Basel formula; a portfolio-calibrated ρ is a
  sponsor decision (ideate raises it as an open question).
- **`stress_testing`**: per-scenario mean PD and EL deltas from deterministically re-scored
  shocked covariates. Sanity: adverse macro shocks must move PD in the economically right
  direction; a scenario shocking unknown columns is reported (`missing_columns`), never silently
  dropped. Stress deltas only reflect covariates the champion actually uses.

## What does NOT apply here

PBO and the Deflated Sharpe Ratio are **trading-strategy** overfitting statistics (opt-in trading
mode). They are not part of a commercial PD outcomes analysis — if they appear, they are labelled
extras, and a noisy PBO is WARN-grade evidence at most, never a BLOCK (per ADR-0005/0006 the gates
don't act on it).
