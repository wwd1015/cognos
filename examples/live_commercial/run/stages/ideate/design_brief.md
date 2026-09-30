# Design brief — demo_commercial

## Sponsor design brief (MD triangulation)

- **Use case:** _unanswered — see open questions_
- **Horizon:** _unanswered — see open questions_
- **Default definition:** _unanswered — see open questions_
- **Segment:** _unanswered — see open questions_
- **Interpretability:** required
- **Sponsor notes:** _unanswered — see open questions_

## Data structure

- shape: panel
- datetime_col: vintage
- n_rows: 1200
- n_features: 6
- n_numeric: 5
- n_categorical: 1
- leakage_suspects: []
- n_events: 81
- event_rate: 0.0675
- events_per_variable: 13.5

## Framework assessment (alternatives considered)

| Framework | Applicable | Role | Reason | Reference |
|---|---|---|---|---|
| Reduced-form PD (obligor scorecard / GLM) | True | primary | Binary outcome with obligor-level financial and facility covariates — the standard framework for commercial PD at origination or surveillance. | Altman (1968); Ohlson (1980); Basel IRB PD; SR 11-7 |
| Discrete-time hazard (survival) | partial | candidate | Time-indexed cohorts present — a period-indexed logit reads as a discrete-time hazard; set data.event_time_col (default period per obligor) to unlock the full hazard families and PD term structure. | Shumway (2001) hazard bankruptcy model |
| Structural (Merton distance-to-default) | False | rejected | Requires traded-market observables (equity value/volatility, liability structure) to compute distance-to-default; none present — typical for private middle-market obligors. | Merton (1974); KMV/Moody's EDF |
| Rating-transition matrix (migration) | False | rejected | Requires a rating history (grade at successive snapshots, e.g. an internal grade or agency rating panel); none present. | CreditMetrics (1997); S&P CreditPro rating-migration practice |
| Machine-learning challenger (trees/boosting) | True | challenger | Interpretability is required for the deployed model, so nonlinear learners serve as challenger benchmarks quantifying the predictive ceiling. | champion–challenger practice; SR 11-7 benchmarking |

## Design Lead decisions

| Framework | Decision | Reason |
|---|---|---|
| reduced_form_pd | primary | The obligors are private and the outcome is a binary default flag, with financial ratios and sector as covariates. A GLM scorecard is the standard, interpretable framework for this, and it meets the interpretability requirement. |
| discrete_time_hazard | candidate | The data is indexed by vintage, so a logit or cloglog on this panel reads as a discrete-time hazard (Shumway 2001). The full hazard families and a PD term structure need data.event_time_col to be set, which matters if the use case turns out to be CECL or IFRS 9. |
| structural_merton | rejected | There are no traded-market observables, such as equity value, equity volatility or liability structure, so distance-to-default cannot be computed for these private obligors. |
| transition_matrix | rejected | The data holds no internal or agency rating history across successive snapshots, so migration matrices cannot be estimated. |
| ml_challenger | challenger | Interpretability is required, so random forest and gradient boosting serve only as benchmarks. They measure the predictive ceiling and the price paid for interpretability. |

## Open questions for the sponsor

- **[design-use_case]** What decision will the model support (origination underwriting, portfolio surveillance, CECL/IFRS 9, IRB, stress testing)? The use case fixes the target definition, horizon, and documentation depth.
- **[design-horizon]** What outcome window defines the target (e.g. 12-month default)? The window must match how the target column was labelled.
- **[design-default_definition]** What event definition labels 'default' (e.g. 90+ DPD, nonaccrual, bankruptcy)? Validation re-derives risk from this.
- **[design-segment]** What portfolio segment does this sample represent (C&I, CRE, small business…)? Pooling heterogeneous segments biases coefficients.
- **[agent-056fd5]** What decision will the model support: origination underwriting, portfolio surveillance, CECL or IFRS 9, IRB, or stress testing?
- **[agent-dd7c11]** What outcome window was used to label the default flag (for example, default within 12 months of the vintage date)?
- **[agent-4ad9ed]** What event defines default: 90+ days past due, nonaccrual, charge-off, or bankruptcy?
- **[agent-26bc9c]** Which portfolio segment does this sample represent (C&I, CRE, small business), and should sectors be pooled or modeled separately?
- **[agent-35ca8e]** Can an event-time or observation-period column per obligor be supplied?

## Ranked hypothesis slate (top 10)

| # | Family | Features | Framework | Role | Priority | Rationale |
|---|---|---|---|---|---|---|
| h1 | logit | top | reduced_form_pd | candidate | 1.0 | This is the most defensible baseline: a parsimonious logit on the clean top predictors, with leverage the strongest univariate signal (explore.corr.leverage). With 81 events (ideate.n_events), a parsimonious set keeps the events per parameter comfortable. |
| h2 | cloglog | top | reduced_form_pd | candidate | 0.97 | The cloglog link is the grouped-time proportional-hazards link, so it fits the vintage panel structure and the hazard interpretation. |
| h3 | probit | top | reduced_form_pd | candidate | 0.95 | This checks whether the choice of link changes the model; the coefficients and ranking should closely match the logit. |
| h4 | logit | all | reduced_form_pd | candidate | 0.92 | This adds sector to the numeric ratios to test for segment effects. The sector dummies use up degrees of freedom, and events per variable is 13.5 before expansion (ideate.events_per_variable). |
| h5 | cloglog | all | reduced_form_pd | candidate | 0.88 | This is the hazard-link version of the full-feature specification, including sector. |
| h6 | probit | all | reduced_form_pd | candidate | 0.86 | This is the link-sensitivity check for the full-feature specification. |
| h7 | lasso_logit | all | reduced_form_pd | challenger | 0.84 | The lasso does data-driven selection across all features, including the sector levels, and so guards against overfitting at a modest event count. Its coefficients are shrunk and biased, so it serves as a challenger. |
| h8 | ridge_logit | all | reduced_form_pd | challenger | 0.8 | The ridge penalty stabilizes coefficients if the liquidity, coverage and margin ratios are collinear. The shrinkage bias is why it is a challenger rather than the champion. |
| h9 | lasso_logit | top | reduced_form_pd | challenger | 0.76 | This checks whether the weak predictors, such as profit_margin (explore.corr.profit_margin), are dropped under a penalty. |
| h10 | ridge_logit | top | reduced_form_pd | challenger | 0.72 | This is a shrinkage-stability check on the parsimonious feature set. |

_Notes: 7 model families x feature strategies = 14 hypotheses. Strongest clean signals: leverage, interest_coverage, log_assets, current_ratio, profit_margin. Design Lead: I recommend a reduced-form PD scorecard as the primary framework, estimated with binary GLMs (logit, probit, cloglog), because the obligors are private and the data has a binary default flag. Because the data is a vintage panel, the logit and cloglog specifications can also be read as discrete-time hazard models; the full hazard framework needs an event-time column before it can be used. Regularized logits and tree ensembles are included as challengers: the tree models show the predictive ceiling, and interpretability is required for the deployed model. Use case, horizon, default definition and segment are all blank, so the sponsor must settle them before the design can be finalized._
