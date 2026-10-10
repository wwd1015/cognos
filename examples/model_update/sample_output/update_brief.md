# Engagement brief — commercial_pd

- **Development mode:** Model update
- **Intent:** clear
- **Objective (restated):** Rank middle-market commercial borrowers by default risk so that underwriters price and size new facilities consistently.

## What the sponsor decided

- **Business objective** (stated in the intent document): Rank middle-market commercial borrowers by default risk so that underwriters price and size new facilities consistently.
- **Decision the model supports** (stated in the intent document): Origination underwriting
- **Portfolio segment** (stated in the intent document): C&I middle-market
- **Outcome definition** (stated in the intent document): 90+ days past due or nonaccrual
- **Outcome horizon** (stated in the intent document): 12 months from origination
- **Intended use and users:** _not stated_
- **Out-of-scope uses:** _not stated_
- **Interpretability requirement** (stated in the intent document): Required: a credit officer must be able to explain a decline.
- **Success criteria** (stated in the intent document): Rank-ordering no worse than the current expert scorecard on an out-of-time sample.
- **Constraints:** _not stated_
- **Data sources:** _not stated_
- **Stakeholders:** _not stated_
- **Existing model** (stated in the intent document): Commercial PD v1 (the first run of this demo), in use for one year.
- **Reason for the update** (stated in the intent document): The annual review found the model under-predicting default in the two latest vintages.
- **Requested changes** (stated in the intent document): - Refresh the development data through the latest vintage - Re-estimate the coefficients on the refreshed sample - Close validation finding V-12 on calibration
- **What must not change** (stated in the intent document): The segment, the default definition and the 12-month horizon.
- **Known issues and findings:** _not stated_

## Update request

- **Scope:** re estimate — The deepest of the 3 requested change(s) is data refresh.
- **Existing model family:** ridge_logit

- Refresh the development data through the latest vintage _(data refresh; from explore)_
- Re-estimate the coefficients on the refreshed sample _(re estimation; from ideate)_
- Close validation finding V-12 on calibration _(remediation; from model)_

## Interview

_Nothing open._

## Documents received

- `update_request.md` — intent
- `score_v1.py` — code
- `prior_run_whitepaper.md` — whitepaper
