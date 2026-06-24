# Backtesting means credit-risk outcomes analysis, not trading-strategy overfitting metrics

For the primary (commercial credit-risk) domain, the backtest stage produces SR 11-7 **outcomes
analysis**, computed on an **out-of-time (OOT)** sample by default:

- **Discrimination** — Gini / AUC and the KS statistic.
- **Calibration** — predicted vs actual default rate by score band (expected-vs-observed).
- **Stability** — PSI / CSI between the development and OOT populations.
- **Rank-ordering / monotonicity** across score bands.

The **Probability of Backtest Overfitting (CSCV)** and **Deflated Sharpe Ratio** are trading-signal
metrics (they presume a returns series) and are **demoted to an opt-in "trading/returns" mode**
(`backtest.returns_column` set), not run on credit-risk models.

## Why

PBO/DSR answer "is this strategy's Sharpe an artifact of multiple testing?" — meaningless for a
PD/LGD model, which has no P&L. SR 11-7 outcomes analysis asks "do predicted risks match realized
outcomes, and is the population stable?" — which is exactly Gini/KS + calibration + PSI on an
out-of-time sample.

## Consequences

- Credit backtesting requires an out-of-time split (train on older vintages, evaluate on newer), not
  a random holdout. The dataset must carry a date/vintage column to split on.
- The IMPACT scored table (predicted score, actual outcome, segment, time) is the natural input to
  these metrics.
- The `credit` synthetic demo needs a vintage/date so it can show OOT calibration/PSI, not just a
  random holdout.
