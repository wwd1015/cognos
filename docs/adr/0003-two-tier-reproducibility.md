# Reproducibility is two-tier: deterministic analysis, human-gated reasoning

Reproducibility and auditability are core, but an LLM-in-the-loop search is not bit-reproducible. We
split the guarantee in two:

1. **The analysis is fully reproducible.** Every number the deterministic engine produces — fits,
   metrics, statistical tests, backtest results, compliance computations, verdicts — re-derives
   bit-identically given the same decisions and pinned seeds, with **no LLM in the path**. The shipped
   model and its entire analysis can be reproduced and validated offline by an auditor.

2. **The reasoning is non-deterministic and human-gated.** Which candidates/transforms the LLM
   proposes may differ by model, version, or round. This is *tolerated*, for two reasons: every
   proposal is still verified by the deterministic engine before it can affect the record (it cannot
   hallucinate a result into existence), and the reasoning steps are exactly the human-in-the-loop
   review points, so a human mitigates the variability. The full reasoning trajectory (prompts +
   responses + model id) is recorded for audit and is replayable from that transcript, but a live
   re-run of the search may legitimately arrive at a different — also engine-verified — model.

## Consequences

- The reasoning-heavy steps (ideation and, later, LLM-guided search) are the natural gate / HITL
  pause points, not just validate/comply/review.
- "Running with no LLM" is therefore not a degraded *product*; it is how the deliverable's analysis is
  always reproduced and served. The LLM is required to *automate the search*, never to *reproduce the
  result*.
- A regulator can reproduce and challenge the final model deterministically; the LLM never sits inside
  the deliverable's trust boundary.
