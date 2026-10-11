# Loss forecasting — from a factor library to a champion chosen on outcomes

A quarterly net charge-off rate on macro drivers: one row per quarter, 30 years. The drivers act
through lags and changes, and the series is short, so nothing about the build can be taken for
granted. This example walks the development plan
([ADR-0016](../../docs/adr/0016-development-plan-candidate-pool-champion-on-outcomes.md)) gate
by gate.

```bash
python examples/loss_forecast/run_demo.py     # the walkthrough, offline, about 30 seconds
cognos demo --task loss_forecast --interactive # the same data in the workbench: cognos ui
```

| Step | What the engine does | What the developer decides |
|---|---|---|
| The plan | reads "one row per period" and recommends an out-of-time test sample, a rolling forecast origin, no sealed third sample (and says why) | the ranking, the screen rules, the selection methods, the size limit |
| Single-factor analysis | widens 8 columns into about 200 factors and fits each alone | the rules that cut the library: significance, expected sign, best per column, correlation |
| Multi-factor selection | forward selection one factor at a time, checked against best subset; every step on record | one method or several, the step criterion, the constraints |
| Candidate pool | ranks every model evaluated; proposes a pool of different models | adds or removes members, from anything in the ledger |
| Champion | scores every pool member on the held-back quarters and under each scenario | accepts the recommendation or chooses another member |

The synthetic series (`cognos.synth.make_loss_forecast_dataset`) is driven by the change in
unemployment over a year (two quarters back), the credit spread (one quarter back) and
house-price growth over a year (one quarter back). `vix`, `cpi_inflation` and `treasury_10y`
carry no signal; `gdp_growth` and `cre_price_index` move with the true drivers. A sound build
finds a transformation of each true driver and leaves the rest out.

The profile is [`projects/example-loss-forecast.yaml`](../../projects/example-loss-forecast.yaml).
Lagged values of the loss rate itself are not offered as factors: an autoregressive model needs
a recursive forecast, which the scorer does not do.
