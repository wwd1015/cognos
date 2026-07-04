---
name: cognos-review
description: Docs↔code drift playbook for the review agent — the drift patterns that recur in model-development runs and the BLOCK bar for stale references.
---

# Review playbook — docs↔code consistency (the final gate)

Review AST-verifies every `{@code:…}` link in the OKF bundle against the code and artifacts it
claims. Use this playbook for the drift patterns that actually happen in model-development runs.

## Recurring drift patterns

- **Champion swap drift** — the search was re-run, a different family won, but the whitepaper
  still narrates the previous champion's coefficients or family name.
- **Feature rename/drop drift** — a leakage-flagged column was dropped (e.g. `dpd_at_outcome`),
  yet the docs still list it among model inputs, or a transform-created feature (`x1_sq`) is
  documented under a stale name.
- **Metric drift** — docs quote a rounder/better metric than `result.json` holds, or state the
  wrong direction (rmse "higher is better").
- **Design-brief drift** — the config's `design:` answers changed (new horizon, new segment) but
  the whitepaper's purpose section still carries the old ones.
- **Path drift** — `{@code:…}` references to functions/files that were renamed in the engine.

## The BLOCK bar

BLOCK is for **confirmed stale references** — a link that provably points at nothing or at
content contradicting the doc's claim. Style issues, tone, or incomplete sections are WARN/FAIL
material for the human. The gate exists because a model document that misstates its own code is
worse than no document: it manufactures false assurance.

## Judgment guide

When a link fails, state the triple: what the doc claims, what the code/artifact actually says,
and which of the two is current. The human's next step differs (fix docs and re-run document, vs.
re-run the pipeline) and your report should make that obvious.
