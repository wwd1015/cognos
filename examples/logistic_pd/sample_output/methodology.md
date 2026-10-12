---
type: methodology
title: Methodology
description: Search procedure and statistical battery.
tags:
- method
timestamp: '2026-10-12T23:43:15.499002+00:00'
---

# Methodology

## From factors to a champion

Development follows a plan confirmed at the design gate. The data are cut into a development sample, a test sample and, when the data can spare the rows, a sealed holdout ({@code:src/cognos/modeling/splits.py#split_samples}). Explored columns are widened into a factor library, every factor is fitted alone, and the plan's rules cut the library to a shortlist ({@code:src/cognos/modeling/factors.py#screen}). Multi-factor models are built from the shortlist by the plan's selection methods ({@code:src/cognos/modeling/selection.py#run_methods}); every model is scored with all preprocessing fitted inside each validation block, and all of them are ranked together. The top of the ranking forms a candidate pool, the pool is compared on the test sample, and the champion is chosen on that comparison. The deployed model is one model. The stage is orchestrated at {@code:src/cognos/stages/model.py}.

## The development plan

| choice | value | set by | reason |
| --- | --- | --- | --- |
| Test sample | out of time, 0.20 of the rows | engine | The rows carry dates: the model will be used on periods after the ones it was built on, so the latest ones are held back. |
| Validation | stratified kfold, 5 block(s) | engine | Folds keep the event rate. Rows of the same entity can sit on both sides of a fold: name evaluation.group_col to prevent it. |
| Sealed holdout | 0.20 of the rows | engine | 2,400 rows are enough to seal a final sample. The candidates are compared on the test sample, so only a sample nobody chose on gives an unbiased last number. |
| Ranking | roc_auc (validation), then log_loss (validation) | human | roc_auc on the validation sample: every model is scored on rows it was not fitted on, and the test sample stays unused until the pool is compared. |
| Selection methods | stepwise | engine | Add the factor that helps most, then drop any that no longer earns its place. One method is recommended. Choose another, or several: every model any of them evaluates is ranked in the same list, so the leaders are the best across the methods you ran. |
| Step criterion | validation | engine | A step is kept when the ranking metric improves on rows the model was not fitted on. |

Rows: 1440 development, 480 test, 480 sealed holdout.

## Single-factor analysis

The library holds 16 factor(s) from 12 explored column(s) (transformations: log). Each was fitted alone on the development sample; the rules below were applied in this order.

| rule | setting | factors in | factors out |
| --- | --- | --- | --- |
| not fitted |  | 16 | 16 |
| max missing | 0.2 | 16 | 16 |
| max p value | 0.1 | 16 | 9 |
| enforce sign | True | 9 | 9 |
| best per source | 2 | 9 | 9 |
| max correlation | 0.8 | 9 | 5 |
| top n | 12 | 5 | 5 |

Shortlist (5): debt_to_ebitda, gdp_growth, interest_coverage, current_ratio, operating_margin. The full table, with the rule and the reason for every dropped factor, is `stages/model/single_factor.json`.

## Multi-factor selection

**Stepwise** (logit; criterion validation): no step improves the criterion within the constraints

| step | action | factor | factors in the model | criterion | change |
| --- | --- | --- | --- | --- | --- |
| 1 | add | debt_to_ebitda | debt_to_ebitda | 0.7032 |  |
| 2 | add | gdp_growth | debt_to_ebitda, gdp_growth | 0.7445 | 0.0413 |
| 3 | add | interest_coverage | debt_to_ebitda, gdp_growth, interest_coverage | 0.7780 | 0.0335 |
| 4 | add | current_ratio | debt_to_ebitda, gdp_growth, interest_coverage, current_ratio | 0.7895 | 0.0115 |

**Same factors** (probit; criterion validation): the factor sets of the logit path(s), fitted as probit

## Candidate pool

33 model(s) were evaluated and ranked. The pool carried to outcomes analysis was set by the human.

| candidate | specification | factors | development | gini | ks | expected_calibration_error | psi | champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cc59646 | logit(3f;random_state=42) | debt_to_ebitda, interest_coverage, current_ratio | 0.7497 | 0.5423 | 0.4119 | 0.0332 | 0.0219 |  |
| c5482c3 | probit(3f;random_state=42) | debt_to_ebitda, gdp_growth, interest_coverage | 0.7781 | 0.4815 | 0.4268 | 0.0136 | 0.1811 |  |
| c46d7e4 | probit(4f;random_state=42) | debt_to_ebitda, gdp_growth, interest_coverage, current_ratio | 0.7899 | 0.4428 | 0.3949 | 0.0175 | 0.1390 | yes |
| cfc227e | logit(4f;random_state=42) | debt_to_ebitda, gdp_growth, interest_coverage, current_ratio | 0.7895 | 0.4338 | 0.3999 | 0.0173 | 0.1441 |  |
| c0cd73f | probit(2f;random_state=42) | debt_to_ebitda, gdp_growth | 0.7450 | 0.3687 | 0.4579 | 0.0175 | 0.3028 |  |
| cfde806 | logit(2f;random_state=42) | debt_to_ebitda, gdp_growth | 0.7445 | 0.3654 | 0.4579 | 0.0172 | 0.3206 |  |
| ccb8810 | logit(3f;random_state=42) | debt_to_ebitda, gdp_growth, current_ratio | 0.7647 | 0.3607 | 0.4289 | 0.0169 | 0.2067 |  |

Compared on the out of time sample (roc_auc first). Champion chosen by the human: Logit 3F-C has the best roc_auc on the out of time sample among the 7 pool member(s); Probit 3F-A is next.

- **Candidates tried:** 33
- **Hypothesis families considered:** logit, probit

## Alternatives tried

This model was developed in 2 runs. A developer may copy a run at a decision and take it differently in a *trial run* ({@code:src/cognos/trials.py#branch}); a trial run stops before validation, so it never opens the sealed holdout, and it stays on record when its files are deleted. Across all runs the test sample was used 14 time(s), 7 of them in the run documented here.

- <first run> (trial run, the main run until 2026-10-12); waiting at “Choose the champion”; champion Logit 3F-C, roc_auc 0.7712 on the out_of_time sample; test sample used 7 time(s).
- <run_id> (main run, from “Choose the champion”); to try: carry Probit 4F-A, the development leader; changed champion; the run validated here; champion Probit 4F-A, roc_auc 0.7214 on the out_of_time sample; test sample used 7 time(s).

## Statistical battery

A battery of 2 statistical tests is run on the champion (2 passed, 0 failed). See [diagnostics](./diagnostics.md) for the full table.

## Deployment scoring

The trained champion is persisted as a picklable scorer bundle. The deployment scoring entry point is {@code:src/cognos/runtime/score.py#score_row}, which evaluates a single row as the IMPACT derived-field contract. See [model](./model.md).
