# A single interpretable champion by default; no silent ensemble

The deliverable is a **single, interpretable champion model** chosen with an interpretability/parsimony
preference (not raw metric alone), because commercial credit-risk models under SR 11-7 must be
validable and explainable — a transparent model (often a logistic scorecard) is strongly preferred
over a blended ensemble, which is hard to validate and frequently disallowed for regulatory use.

The previous behavior — computing a Caruana ensemble of search survivors and then **silently
deploying only the single champion** — is removed. A "computed but discarded" feature that still
appears in the white paper misrepresents what shipped.

## Decision

- Default: one interpretable champion; selection respects an interpretability/parsimony preference.
- Ensembling is either removed or kept only as an **explicit, labeled "challenger / predictive-ceiling
  benchmark"** (answering "how much accuracy do we trade for interpretability?"), reported as a
  benchmark and never silently treated as the deliverable.

## Consequences

- Nothing in the documentation implies a model that was not actually shipped.
- A future "performance-over-interpretability" mode (e.g. ML challenger models) is an explicit opt-in,
  not the default.
