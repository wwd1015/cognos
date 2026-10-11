# A development plan, a candidate pool, and a champion chosen on outcomes (v1.5)

Until now the model stage did four things under one fixed recipe. It split the data one way (a
sealed fifth), scored candidates one way (five shuffled folds unless the task was `timeseries`),
picked factors one way (the five most correlated, or all of them) and ended on one model: the
Modeler chose a champion from the candidates within one standard error of the best, before any
out-of-sample number existed.

That is not how a model is developed. How the data are divided depends on what the data are and
what the model is for. Factors are widened into transformations, tested one at a time and cut
down by rules the developer owns. Models grow from one factor to several. And several models
usually fit about equally well, while their forecasts and their behaviour under a scenario
differ; that difference, not the last decimal of a fit statistic, is what decides.

## Decision

1. **A development plan, recommended from the data and confirmed by a person.**
   `modeling/plan.py` reads the shape of the sample (one row per period, dated rows of many
   entities, or no dates) and the intent, and returns a complete plan with a reason for every
   choice and the choices the data allow. The Design Lead may change it among those options;
   the developer confirms or edits it at the design gate. Precedence: the developer's edit
   (`overrides.plan`) > the profile (`evaluation`, `factors`, `selection`) > the Design Lead >
   the engine. A plan that cannot be run on the data is refused at the gate with the reason.

2. **Up to three samples.** *Development* is what models are fitted on; `validation`
   re-samples it (k-fold, stratified, grouped by entity, or a rolling forecast origin for a
   series). *Test* is out of sample for every fit, and the candidate pool is compared on it.
   A sealed *holdout* is cut first and scored once, for the champion, after the choice. The
   recommendation keeps a holdout when the data can spare the rows and says why not when they
   cannot.

3. **A factor library and a single-factor screen.** Explored columns are widened into lags,
   changes, growth rates and moving averages (one row per period) or logarithms (otherwise).
   Every factor is fitted alone. The developer's rules, applied in the order written (missing,
   significance, expected sign, strength, best per column, correlation, shortlist size), cut
   the library; every dropped factor keeps the rule and the reason.

4. **Selection methods, one or several.** Forward, backward, stepwise and best subset build
   models from the shortlist with a family that has coefficient tests; the grid covers the
   families that do not. A step is kept when the criterion improves (the ranking metric on the
   validation sample or in sample, AIC, BIC or adjusted R²) and the model passes the
   constraints (every coefficient significant, signs as expected, no two factors too alike, a
   size limit). Each path is recorded step by step. Every model any method evaluates lands in
   one ledger and is ranked by the plan's ranking: metrics from the frozen registry, each read
   on the sample the developer names.

5. **The model stage ends on a pool, not a champion.** The Modeler recommends which leaders go
   forward; the developer adds or removes members at a new gate (`gate_candidates`), by id,
   from anything the search evaluated. Every member is fitted, tested and saved.

6. **The champion is chosen on outcomes.** `backtest` scores every pool member on the test
   sample the same way (forecast error and bias, or discrimination, calibration and
   stability) and under every scenario. The Outcomes Analyst recommends the champion;
   `gate_champion` now follows `backtest`. Later stages read the champion in force through
   `ctx.champion()` (`pool.view`).

```
… → ideate → gate_design → model → gate_candidates → backtest → gate_champion → validate → …
```

## What stays frozen

Principle 5 holds, with the sample named exactly. The **sealed holdout** is never read by the
model stage or by outcomes analysis, no agent that chooses can cite it, and validation scores
it once per champion; a second champion scored on it is counted and reported
(`holdout_scored`). The **test sample** is, by design, used to choose: every evaluation on it
is counted (`test_evaluations`), and when a plan seals nothing, validation says that the
champion's test result is also what it was chosen on. Ranking on the test sample during the
search is allowed because developers do it; it is counted the same way.

Metric definitions stay in `modeling/metrics.py`. A developer chooses which metric and which
sample, never the formula. Information criteria are selection criteria, not recorded metrics.

[ADR-0007](0007-single-interpretable-champion-no-silent-ensemble.md) stands: one model is
deployed. It is now chosen later, on more evidence.

## Consequences

- Seven review gates. A send-back at the champion gate goes to the search, since outcomes
  analysis only measures.
- Candidate ids follow the specification (a hash of family, factors and settings), so a pool
  or a champion chosen at a gate cannot silently come to mean another model after a re-search.
- A model's scorer carries the library transformations it uses and recomputes them from raw
  columns. A lag has no value on the first rows of a frame: those rows score as missing.
  Scenarios recompute the transformations of a shocked column, with the development sample as
  history.
- Lagged values of the *target* are not in the library: a model with autoregressive terms needs
  a recursive forecast, which the scorer does not do.
- Time operations need one row per period. Dated rows of many entities get logarithms only;
  lags by entity would need the entity column and are not built.
- Standard errors in the single-factor screen are autocorrelation-robust (HAC) for a series.
  The multi-factor constraints use ordinary standard errors.
- The engine's default pool prefers leaders with different factors that are within one
  standard error of the best; near-copies are added only to fill the pool.
- A rolling-origin validation score and an out-of-time test score are not the same exercise
  (one block ahead from a refitted model, against one fixed origin), so a gap between them is
  a caution at validation, not a failure.
- Runs recorded before this change still open: `pool.view` returns their model stage as it was.
  Recorded agent answers from before it do not replay (the Modeler's contract changed).
