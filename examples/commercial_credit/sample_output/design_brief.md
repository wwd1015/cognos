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
| Discrete-time hazard (survival) | partial | candidate | Time-indexed cohorts present — a period-indexed logit reads as a discrete-time hazard; full survival machinery (Cox, time-varying covariates) is outside the engine. | Shumway (2001) hazard bankruptcy model |
| Structural (Merton distance-to-default) | False | rejected | Requires traded-market observables (equity value/volatility, liability structure) to compute distance-to-default; none present — typical for private middle-market obligors. | Merton (1974); KMV/Moody's EDF |
| Rating-transition matrix (migration) | False | rejected | Requires an internal rating history (grade at successive snapshots); none present. | CreditMetrics (1997); rating-migration practice |
| Machine-learning challenger (trees/boosting) | True | challenger | Interpretability is required for the deployed model, so nonlinear learners serve as challenger benchmarks quantifying the predictive ceiling. | champion–challenger practice; SR 11-7 benchmarking |

## Open questions for the sponsor

- **[data-epv]** Only 93 events for 12 candidate features (≈7.8 events per variable, below the ~10 rule of thumb) — confirm appetite for a short feature list, a coarser segmentation, or a longer sampling window.

## Ranked hypothesis slate (top 10)

| # | Family | Features | Framework | Role | Priority | Rationale |
|---|---|---|---|---|---|---|
| h1 | logit | top | discrete_time_hazard | candidate | 1.0 | Baseline interpretable LOGIT using the strongest clean predictors; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h3 | lasso_logit | top | discrete_time_hazard | candidate | 1.0 | Regularized linear (lasso_logit) using the strongest clean predictors to control variance/collinearity. |
| h5 | ridge_logit | top | discrete_time_hazard | candidate | 0.97 | Regularized linear (ridge_logit) using the strongest clean predictors to control variance/collinearity. |
| h2 | logit | all | discrete_time_hazard | candidate | 0.95 | Baseline interpretable LOGIT using all features; defensible and easy to validate. On vintage-indexed data this reads as a discrete-time hazard (Shumway 2001). |
| h4 | lasso_logit | all | discrete_time_hazard | candidate | 0.88 | Regularized linear (lasso_logit) using all features to control variance/collinearity. |
| h6 | ridge_logit | all | discrete_time_hazard | candidate | 0.85 | Regularized linear (ridge_logit) using all features to control variance/collinearity. |

_Notes: 3 model families x feature strategies = 6 hypotheses. Strongest clean signals: debt_to_ebitda, gdp_growth, unemployment_rate, utilization_rate, log_total_assets. Low event support (EPV≈7.8) — parsimonious feature sets up-weighted._
