# Primary domain is commercial model development; fair lending is an optional consumer-only module

COGNOS's primary use case is **commercial** model development (e.g. commercial credit-risk models —
facility / obligor / collateral), validated under **SR 11-7 model risk management**. Consumer
fair-lending law (ECOA / Regulation B: disparate impact, the four-fifths rule, adverse-action reason
codes) **does not apply** to commercial lending, so the fair-lending machinery is an **optional,
consumer-only module that is off by default** (`compliance.fair_lending: false`).

## Consequences

- The compliance stage's load-bearing content for the primary domain is SR 11-7 (conceptual
  soundness / ongoing monitoring / outcomes analysis) and model-validation rigor — *not* disparate
  impact or reason codes.
- The flagship example should not center fair lending. The `credit` demo demonstrates a feature the
  primary user never enables; a commercial model-validation example is the better showcase.
- If the fair-lending module is kept for other (consumer) users, the disparate-impact direction bug
  (it measures the adverse, not favorable, outcome — see grilling Q5) must still be fixed; but it is
  not on the primary path.
- Protected-attribute handling and disparate-impact tests are not part of the default commercial
  pipeline.
