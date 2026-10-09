---
type: caveats
title: Limitations & Assumptions
description: Known limitations, assumptions, and recommendations.
tags:
- caveats
timestamp: '2026-09-30T02:35:19.205266+00:00'
---

# Limitations & Assumptions

- The model is valid only within the distribution of its training data; monitor for drift before relying on out-of-distribution scores.
- Out-of-scope use: see the model card.
- No target-leakage suspects were flagged, but feature availability at prediction time should still be confirmed.
- The frozen-holdout metric is a single-shot estimate; real-world performance may differ.

See the [model card](./model_card.md) and [methodology](./methodology.md).
