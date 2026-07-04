---
name: cognos-model
description: Champion-selection playbook for the model agent — economic sign checks, inference quality, interpretability constraints, and plausibility bands for commercial PD metrics.
---

# Model playbook — champion selection in regulated commercial modeling

The engine runs the ratchet and the statistical battery; use this playbook to judge whether the
champion it reports is *economically* credible, not just numerically best.

## Economic sign checks (commercial PD priors)

For a default model, coefficient signs should match how credit officers reason:

| Driver | Expected sign on PD |
|---|---|
| Leverage (Debt/EBITDA) | + |
| Interest coverage | − |
| Liquidity (current/quick ratio) | − |
| Profitability (margins) | − |
| Size (log assets) | − |
| Revolver utilization | + |
| Collateral coverage | − |
| Unemployment at origination | + |
| GDP growth at origination | − |

A "wrong sign" on a significant coefficient is a red flag for collinearity, segment mixing, or
leakage — surface it even when the verdict is PASS. (Regularized families shrink coefficients;
signs still shouldn't flip.)

## Inference quality

- Coefficients/p-values come from the **full-rank (K−1 coded) inference design** — they are valid
  for documentation only for plain OLS/logit champions. Regularized and GLM families report
  point estimates without honest p-values; say so rather than over-claiming.
- Insignificant-everything with a decent AUC suggests collinearity (ratio families share
  numerators/denominators) or too many parameters for the event count.

## Plausibility bands (obligor-level commercial PD)

- CV AUC **0.70–0.85** is the normal range for a 12-month obligor default model; **> 0.90** is
  suspicious — check the leakage suspects and the CV→holdout gap before celebrating.
- The CV→holdout degradation finding (> 15% relative) is an overfitting signal the validate gate
  will re-derive; don't smooth it over.

## Interpretability constraint

- The deliverable is a **single interpretable champion** (ADR-0007). An ensemble or GBM appearing
  in the payload is a labelled **challenger benchmark**: report the gap to the champion as "the
  price of interpretability", never as a reason to swap the deployed model.
- If the design brief says `interpretability: flexible` and a tree family won, the validation and
  documentation burden goes up — note it for the downstream stages.

## Diagnostics that matter most here

Residual/specification failures (linearity, heteroskedasticity) on a scorecard usually mean a
missing transform (log of a skewed ratio) or a pooled segment — connect the failed test to that
hypothesis when reporting, don't just enumerate test names.
