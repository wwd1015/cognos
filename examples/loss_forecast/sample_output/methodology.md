---
type: methodology
title: Methodology
description: Search procedure and statistical battery.
tags:
- method
timestamp: '2026-10-11T15:49:42.823059+00:00'
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
| Ranking | rmse (validation), then r2 (in sample) | profile | Stated in the profile. |
| Selection methods | forward + best subset | profile | Stated in the profile. |
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

**Best subset** (ols; criterion validation): best model of each size; the path ends on the best of them

| step | action | factor | factors in the model | criterion | change |
| --- | --- | --- | --- | --- | --- |
| 1 | size |  | bbb_spread__ma4 | 0.1846 |  |
| 2 | size |  | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2 | 0.1000 | -0.0846 |
| 3 | size |  | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, hpi__yoy_diff | 0.0898 | -0.0102 |

**Grid** (lasso, ridge; criterion validation): 12 specification(s) from the slate

## Candidate pool

247 model(s) were evaluated and ranked. The pool carried to outcomes analysis was set by the human.

| candidate | specification | factors | development | rmse | mae | r2 | bias | champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c8914d2 | lasso(6f;alpha=0.01,random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, gdp_growth__ma4__lag2, hpi__yoy_diff, cre_price_index__yoy_pct__lag1, cre_price_index__yoy_pct__lag2 | 0.0866 | 0.1049 | 0.0868 | 0.9446 | -0.0252 |  |
| c229cb4 | ols(3f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, hpi__yoy_diff | 0.0898 | 0.1050 | 0.0872 | 0.9445 | -0.0227 | yes |
| c79c8af | ols(3f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2, cre_price_index__yoy_pct__lag2 | 0.0952 | 0.1085 | 0.0802 | 0.9407 | -0.0111 |  |
| c41e6dd | ols(2f;random_state=42) | bbb_spread__ma4, unemployment_rate__yoy_pct__lag2 | 0.1000 | 0.1103 | 0.0851 | 0.9388 | -0.0088 |  |

Compared on the out of time sample (rmse first). Champion chosen by the engine: c229cb4 is within 2% of the best rmse on the out of time sample (c8914d2) with fewer factors; c8914d2 is next.

- **Candidates tried:** 247
- **Hypothesis families considered:** ols, lasso, ridge

## Statistical battery

A battery of 12 statistical tests is run on the champion (12 passed, 0 failed). See [diagnostics](./diagnostics.md) for the full table.

## Deployment scoring

The trained champion is persisted as a picklable scorer bundle. The deployment scoring entry point is {@code:src/cognos/runtime/score.py#score_row}, which evaluates a single row as the IMPACT derived-field contract. See [model](./model.md).
