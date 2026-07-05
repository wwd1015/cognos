# Design brief — cni_middle_market_pd

## Sponsor design brief (MD triangulation)

- **Use case:** origination underwriting
- **Horizon:** 12-month PD
- **Default definition:** 90+ DPD or nonaccrual within 12 months of origination
- **Segment:** C&I middle-market
- **Interpretability:** required
- **Sponsor notes:** Signs must match credit intuition: leverage up-risk; coverage, liquidity, margin, size down-risk. Trees are benchmarks only.

## Data structure

- shape: panel
- datetime_col: vintage
- n_rows: 2000
- n_features: 12
- n_numeric: 10
- n_categorical: 2
- leakage_suspects: []
- n_events: 93
- event_rate: 0.0465
- events_per_variable: 7.8

## Framework assessment (alternatives considered)

| Framework | Applicable | Role | Reason | Reference |
|---|---|---|---|---|
| Reduced-form PD (obligor scorecard / GLM) | True | primary | Binary outcome with obligor-level financial and facility covariates — the standard framework for commercial PD at origination or surveillance. | Altman (1968); Ohlson (1980); Basel IRB PD; SR 11-7 |
| Discrete-time hazard (survival) | True | candidate | Event timing present (data.event_time_col) — the engine panel-expands obligor-periods and fits a full discrete-time hazard (logit or cloglog grouped-time PH link) with a PD term structure. | Shumway (2001) hazard bankruptcy model |
| Structural (Merton distance-to-default) | False | rejected | Requires traded-market observables (equity value/volatility, liability structure) to compute distance-to-default; none present — typical for private middle-market obligors. | Merton (1974); KMV/Moody's EDF |
| Rating-transition matrix (migration) | False | rejected | Requires an internal rating history (grade at successive snapshots); none present. | CreditMetrics (1997); rating-migration practice |
| Machine-learning challenger (trees/boosting) | True | challenger | Interpretability is required for the deployed model, so nonlinear learners serve as challenger benchmarks quantifying the predictive ceiling. | champion–challenger practice; SR 11-7 benchmarking |

## Open questions for the sponsor

- **[data-epv]** Only 93 events for 12 candidate features (≈7.8 events per variable, below the ~10 rule of thumb) — confirm appetite for a short feature list, a coarser segmentation, or a longer sampling window.
- **[design-asset-correlation]** Portfolio simulation uses the Basel IRB asset-correlation formula and LGD=0.45 — confirm both against portfolio evidence or supply calibrated values.

## Ranked hypothesis slate (top 10)

| # | Family | Features | Framework | Role | Priority | Rationale |
|---|---|---|---|---|---|---|
| h1 | logit | top | reduced_form_pd | candidate | 1.0 | Baseline interpretable LOGIT using the strongest clean predictors; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h3 | probit | top | reduced_form_pd | candidate | 1.0 | Binary GLM with the probit link using the strongest clean predictors; statsmodels inference with valid p-values. |
| h5 | cloglog | top | reduced_form_pd | candidate | 1.0 | Binary GLM with the cloglog link using the strongest clean predictors; statsmodels inference with valid p-values. cloglog is the grouped-time proportional-hazards link. |
| h7 | hazard_logit | top | discrete_time_hazard | candidate | 1.0 | Discrete-time hazard (logit link) on the obligor-period panel using the strongest clean predictors — PD term structure over the outcome window (Shumway 2001). |
| h9 | hazard_cloglog | top | discrete_time_hazard | candidate | 1.0 | Discrete-time hazard (cloglog link) on the obligor-period panel using the strongest clean predictors — PD term structure over the outcome window (Shumway 2001). |
| h11 | lasso_logit | top | reduced_form_pd | candidate | 1.0 | Regularized linear (lasso_logit) using the strongest clean predictors to control variance/collinearity. |
| h13 | ridge_logit | top | reduced_form_pd | candidate | 0.97 | Regularized linear (ridge_logit) using the strongest clean predictors to control variance/collinearity. |
| h2 | logit | all | reduced_form_pd | candidate | 0.95 | Baseline interpretable LOGIT using all features; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h4 | probit | all | reduced_form_pd | candidate | 0.93 | Binary GLM with the probit link using all features; statsmodels inference with valid p-values. |
| h6 | cloglog | all | reduced_form_pd | candidate | 0.92 | Binary GLM with the cloglog link using all features; statsmodels inference with valid p-values. cloglog is the grouped-time proportional-hazards link. |

_Notes: 7 model families x feature strategies = 14 hypotheses. Strongest clean signals: debt_to_ebitda, gdp_growth, unemployment_rate, utilization_rate, log_total_assets. Low event support (EPV≈7.8) — parsimonious feature sets up-weighted._
