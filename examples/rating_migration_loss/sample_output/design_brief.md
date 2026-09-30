# Design brief — corporate_migration_loss

## Sponsor design brief (MD triangulation)

- **Use case:** portfolio loss forecasting for CECL provisioning and enterprise stress testing
- **Horizon:** 1-year default, extended to a 3-year cumulative term structure via matrix powers
- **Default definition:** agency default state D (payment default, distressed exchange, or bankruptcy)
- **Segment:** large corporate (agency-rated universe, mapped to the internal master scale)
- **Interpretability:** required
- **Sponsor notes:** Internal history (2019+) is too short for a through-the-cycle matrix; licensed agency data carries the long-run default experience. Expected signs: leverage up-risk; coverage, margin, size down-risk.

## Data structure

- shape: panel
- datetime_col: asof
- n_rows: 19350
- n_features: 7
- n_numeric: 5
- n_categorical: 2
- leakage_suspects: []
- n_events: 269
- event_rate: 0.013901808785529716
- events_per_variable: 38.4

## Framework assessment (alternatives considered)

| Framework | Applicable | Role | Reason | Reference |
|---|---|---|---|---|
| Reduced-form PD (obligor scorecard / GLM) | True | primary | Binary outcome with obligor-level financial and facility covariates — the standard framework for commercial PD at origination or surveillance. | Altman (1968); Ohlson (1980); Basel IRB PD; SR 11-7 |
| Discrete-time hazard (survival) | partial | candidate | Time-indexed cohorts present — a period-indexed logit reads as a discrete-time hazard; set data.event_time_col (default period per obligor) to unlock the full hazard families and PD term structure. | Shumway (2001) hazard bankruptcy model |
| Structural (Merton distance-to-default) | False | rejected | Requires traded-market observables (equity value/volatility, liability structure) to compute distance-to-default; none present — typical for private middle-market obligors. | Merton (1974); KMV/Moody's EDF |
| Rating-transition matrix (migration) | True | candidate | Rating columns detected (rating, next_rating); the migration: config block is enabled — the engine estimates a cohort transition matrix on the training partition (NR-adjusted), feeds the rating-implied horizon PD to the champion (hybrid), scores the pure-migration PD as a challenger benchmark, and reports a by-rating expected-loss forecast. | CreditMetrics (1997); S&P CreditPro rating-migration practice |
| Machine-learning challenger (trees/boosting) | True | challenger | Interpretability is required for the deployed model, so nonlinear learners serve as challenger benchmarks quantifying the predictive ceiling. | champion–challenger practice; SR 11-7 benchmarking |

## Design Lead decisions

| Framework | Decision | Reason |
|---|---|---|
| reduced_form_pd | primary | Binary outcome with obligor-level financial and facility covariates — the standard framework for commercial PD at origination or surveillance. |
| discrete_time_hazard | candidate | Time-indexed cohorts present — a period-indexed logit reads as a discrete-time hazard; set data.event_time_col (default period per obligor) to unlock the full hazard families and PD term structure. |
| structural_merton | rejected | Requires traded-market observables (equity value/volatility, liability structure) to compute distance-to-default; none present — typical for private middle-market obligors. |
| transition_matrix | candidate | Rating columns detected (rating, next_rating); the migration: config block is enabled — the engine estimates a cohort transition matrix on the training partition (NR-adjusted), feeds the rating-implied horizon PD to the champion (hybrid), scores the pure-migration PD as a challenger benchmark, and reports a by-rating expected-loss forecast. |
| ml_challenger | challenger | Interpretability is required for the deployed model, so nonlinear learners serve as challenger benchmarks quantifying the predictive ceiling. |

## Open questions for the sponsor

- **[design-asset-correlation]** Portfolio simulation uses the Basel IRB asset-correlation formula and LGD=0.4 — confirm both against portfolio evidence or supply calibrated values.

## Ranked hypothesis slate (top 10)

| # | Family | Features | Framework | Role | Priority | Rationale |
|---|---|---|---|---|---|---|
| h1 | logit | top | reduced_form_pd | candidate | 1.0 | Baseline interpretable LOGIT using the strongest clean predictors; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h2 | probit | top | reduced_form_pd | candidate | 0.98 | Binary GLM with the probit link using the strongest clean predictors; statsmodels inference with valid p-values. |
| h3 | cloglog | top | reduced_form_pd | candidate | 0.97 | Binary GLM with the cloglog link using the strongest clean predictors; statsmodels inference with valid p-values. cloglog is the grouped-time proportional-hazards link. |
| h4 | logit | all | reduced_form_pd | candidate | 0.95 | Baseline interpretable LOGIT using all features; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h5 | probit | all | reduced_form_pd | candidate | 0.93 | Binary GLM with the probit link using all features; statsmodels inference with valid p-values. |
| h6 | lasso_logit | top | reduced_form_pd | candidate | 0.93 | Regularized linear (lasso_logit) using the strongest clean predictors to control variance/collinearity. |
| h7 | cloglog | all | reduced_form_pd | candidate | 0.92 | Binary GLM with the cloglog link using all features; statsmodels inference with valid p-values. cloglog is the grouped-time proportional-hazards link. |
| h8 | ridge_logit | top | reduced_form_pd | candidate | 0.9 | Regularized linear (ridge_logit) using the strongest clean predictors to control variance/collinearity. |
| h9 | lasso_logit | all | reduced_form_pd | candidate | 0.88 | Regularized linear (lasso_logit) using all features to control variance/collinearity. |
| h10 | ridge_logit | all | reduced_form_pd | candidate | 0.85 | Regularized linear (ridge_logit) using all features to control variance/collinearity. |

_Notes: 5 model families x feature strategies = 10 hypotheses. Strongest clean signals: debt_to_ebitda, interest_coverage, operating_margin, log_total_assets, unemployment_rate. Design Lead: Primary framework: reduced_form_pd. Ranked 10 engine-fittable specifications, interpretable families first when interpretability is required._
