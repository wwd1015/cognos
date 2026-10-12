# Loss forecasting — from a factor library to a champion chosen on outcomes

A quarterly net charge-off rate on macro drivers: one row per quarter, 30 years. The drivers act
through lags and changes, and the series is short, so nothing about the build can be taken for
granted. This example walks the development plan
([ADR-0016](../../docs/adr/0016-development-plan-candidate-pool-champion-on-outcomes.md)) gate
by gate.

```bash
python examples/loss_forecast/run_demo.py     # the walkthrough, offline, about 30 seconds
python examples/loss_forecast/run_demo.py --save-sample   # also refresh sample_output/
cognos demo --task loss_forecast --interactive # the same data in the workbench: cognos ui
```

| Step | What the engine does | What the developer decides |
|---|---|---|
| The plan | reads "one row per period" and recommends an out-of-time test sample, a rolling forecast origin, no sealed third sample (and says why) | their own criteria: the primary metric and its sample, the screen rules, the selection method, the size limit |
| Single-factor analysis | widens 8 columns into about 200 factors and fits each alone | the rules that cut the library: significance, expected sign, best per column, correlation |
| Multi-factor selection | forward selection, one factor at a time, every step on record | the method (one, or several ranked together), the step criterion, the constraints |
| Candidate pool | ranks and names every model evaluated; proposes a pool of different models | carries as many as they want to compare, from anything in the ranking; renames |
| Champion | scores every pool member on the held-back quarters and under each scenario | accepts the recommendation or chooses another member |
| A trial run | copies the run at the design gate, re-runs only what the changed plan changes, counts its looks at the test sample, and stops before validation | what to try ("rank the search on the test sample"), and whether to keep the main run or make the trial run the main run |

Step 6 is a trial run ([ADR-0017](../../docs/adr/0017-trial-runs-a-decision-taken-differently.md)).
The developer asks what the search would find if it ranked on the test sample. Re-opening the
design gate would replace steps 2 to 5, so they make a trial run instead. In the captured run
it finds nothing better and costs 228 looks at the test sample against the main run's 7. The
developer deletes it. Its line stays in the model's record, validation reports 235 looks
across the two runs, and the white paper lists it under "Alternatives tried".

The synthetic series (`cognos.synth.make_loss_forecast_dataset`) is driven by the change in
unemployment over a year (two quarters back), the credit spread (one quarter back) and
house-price growth over a year (one quarter back). `vix`, `cpi_inflation` and `treasury_10y`
carry no signal; `gdp_growth` and `cre_price_index` move with the true drivers. A sound build
finds a transformation of each true driver and leaves the rest out.

The profile is [`projects/example-loss-forecast.yaml`](../../projects/example-loss-forecast.yaml).
Lagged values of the loss rate itself are not offered as factors: an autoregressive model needs
a recursive forecast, which the scorer does not do.

The captured run is in [`sample_output/`](sample_output): `walkthrough.txt` (this script's
output), `methodology.md` (the white paper's methodology page for the run) and
`single_factor.json` (all 200 factors, each with its fate). The companion example on
cross-sectional data is [`logistic_pd`](../logistic_pd).
