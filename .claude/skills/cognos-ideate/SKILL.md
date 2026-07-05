---
name: cognos-ideate
description: Framework-selection and MD-triangulation playbook for the ideate agent — structural vs reduced-form vs hazard vs ML challenger, algorithm trade-offs in regulated commercial risk, and the design questions that must never be assumed.
---

# Ideate playbook — framework selection & MD triangulation (commercial risk)

The ideate stage now works like a senior modeler opening an engagement: data structure first, then
the **econometric framework**, then the algorithm — and it triangulates the design brief with the
model sponsor (the MD) instead of assuming. Use this playbook to judge whether the stage's
`framework_assessment`, `open_questions`, and hypothesis slate are sound.

## The framework decision comes before the algorithm

| Framework | When it applies | Engine support | Canonical reference |
|---|---|---|---|
| **Structural (Merton distance-to-default)** | Obligor has traded-market observables: equity value/volatility, debt face value. Public corporates. | **Real support via the `structural:` config block**: deterministic KMV solver → DD as a hybrid feature feeding the champion, plus a pure-structural PD challenger benchmark on the sealed holdout | Merton (1974); KMV / Moody's EDF |
| **Reduced-form PD (scorecard / GLM)** | Private/middle-market obligors: financial ratios + facility + macro covariates, binary default flag | logit, **probit, cloglog** (statsmodels inference), ridge_logit, lasso_logit | Altman (1968) Z-score; Ohlson (1980); Basel IRB |
| **Discrete-time hazard (survival)** | Vintage/panel data with **event timing** (`data.event_time_col`) | **Full support**: `hazard_logit` / `hazard_cloglog` — obligor-period panel expansion inside the estimator (obligor-level CV stays leakage-safe), valid panel GLM inference, **PD term structure** | Shumway (2001) |
| **Rating-transition matrix** | Internal rating history at successive snapshots | not in engine — flag as human follow-up | CreditMetrics (1997) |
| **ML challenger (RF/GBM)** | Always available as a predictive-ceiling benchmark; champion only if the sponsor relaxes interpretability | random_forest, gradient_boosting | SR 11-7 benchmarking; champion–challenger practice |

Downstream of the model, two **simulation capabilities** connect PDs to portfolio decisions (opt-in,
reported by backtest, never used for champion selection): **Vasicek one-factor loss simulation**
(`portfolio:` block — EL/UL/VaR/ES + closed-form Basel IRB capital; asset correlation ρ defaults to
the Basel formula) and **macro-scenario stress testing** (`stress:` block — CCAR-flavoured shocked
covariates re-scored deterministically). Unanswered ρ/LGD provenance and missing scenario sets
surface as open design questions.

Judge the assessment against the data structure the stage reports: a rejected framework must have a
concrete reason (e.g. "no market observables — private obligors"), because the rejected
alternatives become the **"alternatives considered"** section SR 11-7 documentation requires.

## Algorithm trade-offs in regulated commercial credit

- **Logit/probit/cloglog GLM**: coefficients are the model — signs must match economic priors
  (leverage ↑ risk; coverage, liquidity, margin, size ↓ risk; utilization ↑ risk). Valid p-values,
  cheap validation, standard for deployment. cloglog is the grouped-time proportional-hazards
  link — prefer it when the hazard reading matters.
- **Discrete-time hazard**: same GLM machinery on the obligor-period panel; buys a PD *term
  structure* (PD(1)…PD(K)) — what CECL/IFRS 9 lifetime use cases actually need — at the price of
  more parameters (period dummies) against the same event count.
- **Regularized logit**: use when collinearity (ratio families share numerators/denominators) or
  low events-per-variable destabilizes plain logit; the price is biased coefficients — inference is
  qualified.
- **RF/GBM**: typically +0.02–0.05 AUC over a good scorecard on tabular credit data. Under
  `interpretability: required` they are **challenger benchmarks only** — the gap they reveal is the
  price of interpretability, and reporting that gap honestly is the point.
- **Events-per-variable**: below ~10 events per candidate parameter (Peduzzi 1996), prefer short
  feature lists; the stage up-weights parsimonious strategies — check that it did.

## MD triangulation — never assume the design

Four design points change the whole build; unanswered ones must appear in `open_questions`:

1. **Use case** (origination / surveillance / CECL–IFRS 9 / IRB / stress testing) — fixes horizon,
   target construction, calibration philosophy (PIT vs TTC), and documentation depth.
2. **Horizon** — the outcome window must match how the target column was labelled.
3. **Default definition** — 90+ DPD? nonaccrual? bankruptcy? Validation re-derives risk from this.
4. **Segment** — pooling C&I with CRE or small business biases coefficients; segmentation is a
   sponsor decision, not a statistical afterthought.

A slate that silently assumes these is a defective slate even if the AUC would be fine. Conversely,
once the `design:` block answers them, the questions must disappear and the brief should echo the
answers.

## Judging the slate

- Leakage suspects from explore must **not** appear in `clean_top_features`.
- Under `interpretability: required`, interpretable families are searched first and tree
  hypotheses carry `role: challenger`.
- LLM-added hypotheses (`source: llm`) must name engine-fittable families; the ratchet — not the
  LLM — decides if they survive.
- The slate should fit `search.max_candidates`; a slate wider than the budget wastes the search.
