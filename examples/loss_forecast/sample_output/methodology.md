---
type: methodology
title: Methodology
description: Search procedure and statistical battery.
tags:
- method
timestamp: '2026-10-12T23:43:38.519999+00:00'
---

# Methodology

## From factors to a champion

Development follows a plan confirmed at the design gate. The data are cut into a development sample, a test sample and, when the data can spare the rows, a sealed holdout ({@code:src/cognos/modeling/splits.py#split_samples}). Explored columns are widened into a factor library, every factor is fitted alone, and the plan's rules cut the library to a shortlist ({@code:src/cognos/modeling/factors.py#screen}). Multi-factor models are built from the shortlist by the plan's selection methods ({@code:src/cognos/modeling/selection.py#run_methods}); every model is scored with all preprocessing fitted inside each validation block, and all of them are ranked together. The top of the ranking forms a candidate pool, the pool is compared on the test sample, and the champion is chosen on that comparison. The deployed model is one model. The stage is orchestrated at {@code:src/cognos/stages/model.py}.

## The development plan

| choice | value | set by | reason |
| --- | --- | --- | --- |
| Test sample | out of time, 0.20 of the rows | engine | One row per period: a forecast is judged on periods after the ones it was fitted on, so the latest periods are held back. |
| Validation | rolling origin, 12 block(s) | engine | A series cannot be shuffled. Each block is forecast from the history before it, and the origin rolls forward. |
| Sealed holdout | none | engine | 120 periods: too few to hold back a third sample. The champion's test result is then also what it was chosen on, and validation says so. |
| Ranking | rmse (validation), then r2 (in sample) | human | rmse on the validation sample: every model is scored on rows it was not fitted on, and the test sample stays unused until the pool is compared. |
| Selection methods | forward | engine | Few periods and many transformations: start from the best single factor and add one at a time while the model improves, so every factor's contribution is on record. One method is recommended. Choose another, or several: every model any of them evaluates is ranked in the same list, so the leaders are the best across the methods you ran. Families on the slate without coefficient tests cannot be stepped; each is fitted on the shortlist so it still has a candidate. |
| Step criterion | validation | engine | A step is kept when the ranking metric improves on rows the model was not fitted on. |

Rows: 90 development, 22 test, 0 sealed holdout; 8 leading period(s) dropped for lack of history.

## Single-factor analysis

The library holds 200 factor(s) from 8 explored column(s) (transformations: lag, diff, yoy_diff, yoy_pct, ma). Each was fitted alone on the development sample; the rules below were applied in this order.

| rule | setting | factors in | factors out |
| --- | --- | --- | --- |
| not fitted |  | 200 | 200 |
| max missing | 0.2 | 200 | 200 |
| max p value | 0.05 | 200 | 79 |
| enforce sign | True | 79 | 71 |
| best per source | 2 | 71 | 10 |
| max correlation | 0.8 | 10 | 6 |
| top n | 12 | 6 | 6 |

Shortlist (6): bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, gdp_growth__ma4__lag2, hpi__yoy_diff, cre_price_index__yoy_pct__lag1, cre_price_index__yoy_pct__lag2. The full table, with the rule and the reason for every dropped factor, is `stages/model/single_factor.json`.

## Multi-factor selection

**Forward** (ols; criterion validation): stopped at the size limit

| step | action | factor | factors in the model | criterion | change |
| --- | --- | --- | --- | --- | --- |
| 1 | add | bbb_spread__ma4 | bbb_spread__ma4 | 0.1846 |  |
| 2 | add | unemployment_rate__yoy_pct__lag2 | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2 | 0.1000 | -0.0846 |
| 3 | add | hpi__yoy_diff | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, hpi__yoy_diff | 0.0898 | -0.0102 |

**Grid** (lasso, ridge; criterion validation): these families have no coefficient tests, so they cannot be stepped: each is fitted on the strongest shortlisted factors and on all of them (12 specification(s))

## Candidate pool

221 model(s) were evaluated and ranked. The pool carried to outcomes analysis was set by the human.

| candidate | specification | factors | development | rmse | mae | r2 | bias | champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c30b46b | ols(3f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, cre_price_index__yoy_pct__lag1 | 0.1001 | 0.1046 | 0.0794 | 0.9449 | -0.0112 | yes |
| c8914d2 | lasso(6f;alpha=0.01,random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, gdp_growth__ma4__lag2, hpi__yoy_diff, cre_price_index__yoy_pct__lag1, cre_price_index__yoy_pct__lag2 | 0.0866 | 0.1049 | 0.0868 | 0.9446 | -0.0252 |  |
| cf27eed | lasso(5f;alpha=0.01,random_state=42) | unemployment_rate__yoy_pct__lag2, bbb_spread__ma4, hpi__yoy_diff, gdp_growth__ma4__lag2, cre_price_index__yoy_pct__lag2 | 0.0866 | 0.1049 | 0.0868 | 0.9446 | -0.0252 |  |
| c229cb4 | ols(3f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, hpi__yoy_diff | 0.0898 | 0.1050 | 0.0872 | 0.9445 | -0.0227 |  |
| cf05663 | ridge(6f;alpha=1.0,random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, gdp_growth__ma4__lag2, hpi__yoy_diff, cre_price_index__yoy_pct__lag1, cre_price_index__yoy_pct__lag2 | 0.0891 | 0.1066 | 0.0882 | 0.9428 | -0.0226 |  |
| c79c8af | ols(3f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, cre_price_index__yoy_pct__lag2 | 0.0952 | 0.1085 | 0.0802 | 0.9407 | -0.0111 |  |
| c41e6dd | ols(2f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2 | 0.1000 | 0.1103 | 0.0851 | 0.9388 | -0.0088 |  |

Compared on the out of time sample (rmse first). Champion chosen by the engine: OLS 3F-C has the best rmse on the out of time sample among the 7 pool member(s); Lasso 6F-A is next.

- **Candidates tried:** 221
- **Hypothesis families considered:** ols, lasso, ridge

## Alternatives tried

This model was developed in 2 runs. A developer may copy a run at a decision and take it differently in a *trial run* ({@code:src/cognos/trials.py#branch}); a trial run stops before validation, so it never opens the sealed holdout, and it stays on record when its files are deleted. Across all runs the test sample was used 235 time(s), 7 of them in the run documented here.

- <run_id> (main run); the run validated here; champion OLS 3F-C, rmse 0.1046 on the out_of_time sample; test sample used 7 time(s).
- <trial run> (trial run, from “Review the design”); to try: rank the search on the test sample; changed the plan; trial finished at the champion; champion OLS 3F-C, rmse 0.1046 on the out_of_time sample; test sample used 228 time(s); files deleted 2026-10-12 by the developer.

## Statistical battery

A battery of 12 statistical tests is run on the champion (12 passed, 0 failed). See [diagnostics](./diagnostics.md) for the full table.

## Deployment scoring

The trained champion is persisted as a picklable scorer bundle. The deployment scoring entry point is {@code:src/cognos/runtime/score.py#score_row}, which evaluates a single row as the IMPACT derived-field contract. See [model](./model.md).
