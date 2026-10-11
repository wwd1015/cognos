# Logistic-regression PD — three samples, a scorecard pool, a champion chosen out of time

A 12-month probability-of-default model for 2,400 C&I obligors observed by vintage, built as a
logistic regression. It walks the development plan
([ADR-0016](../../docs/adr/0016-development-plan-candidate-pool-champion-on-outcomes.md)) gate
by gate, and is the companion of [`loss_forecast`](../loss_forecast): the same method, and a
different answer at every step because the data are different.

```bash
python examples/logistic_pd/run_demo.py                 # the walkthrough, offline, under a minute
python examples/logistic_pd/run_demo.py --save-sample   # also refresh sample_output/
```

| Step | Here | In the loss-forecast example |
|---|---|---|
| Test sample | the later vintages (out of time) | the latest quarters (out of time) |
| Validation | stratified 5-fold | rolling forecast origin, 12 origins |
| Sealed holdout | kept: 2,400 rows can spare a third sample | none: 120 quarters cannot |
| Factor library | each column, plus the logarithm of skewed ones (16 factors) | lags, changes, year-on-year moves, moving averages (about 200) |
| Selection method | stepwise, logit; probit on the same factor sets | forward, OLS |
| The developer's own criteria | AUC then log loss; every coefficient significant at 5% | RMSE then in-sample R²; single factors significant at 5% |
| Compared on | Gini, KS, calibration, PSI, PD under scenarios | RMSE, bias, R², loss rate under scenarios |

What the captured run in [`sample_output/walkthrough.txt`](sample_output/walkthrough.txt) shows:

- The screen drops `unemployment_rate` for being 0.89 correlated with the stronger `gdp_growth`,
  and four ratios that are not significant alone. Each keeps its rule and reason.
- Stepwise builds a four-factor scorecard. A fifth factor would raise validation AUC, and is
  refused because its coefficient is not significant at 5%.
- Every model has a name (`Logit 4F-A`, `Probit 3F-A`); the developer widens the pool with a
  two-factor scorecard and names it.
- The four-factor models lead on the development sample. **Out of time, a three-factor probit
  is better**, and it is the one recommended. That is the reason the pool exists.
- The sealed holdout, opened once at validation, comes in well below both (AUC 0.65 against
  0.78 in validation and 0.74 on test), and validation **fails** the model for it. The synthetic
  portfolio drifts across vintages; the point of the example is that the test sample, having
  been used to choose, did not show how much.

`sample_output/methodology.md` is the white paper's methodology page for the run (the plan, the
screen, each path, the pool comparison). `sample_output/single_factor.json` is the full
single-factor table.
