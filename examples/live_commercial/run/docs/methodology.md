---
type: methodology
title: Methodology
description: Search procedure and statistical battery.
tags:
- method
timestamp: '2026-09-30T02:35:19.199176+00:00'
---

# Methodology

## Champion search

Candidates are explored with a *ratchet* search: each accepted experiment must beat the incumbent on leakage-safe cross-validation (all preprocessing fit inside each training fold) before it becomes the new incumbent. The final score is read once on a sealed *frozen holdout* never touched during search. The deployed model is the single interpretable champion. The ratchet loop is implemented in {@code:src/cognos/modeling/search.py#ratchet_search}, orchestrated by the modeling stage at {@code:src/cognos/stages/model.py}.

- **Candidates tried:** 16
- **Hypothesis families considered:** logit, probit, cloglog, lasso_logit, ridge_logit, gradient_boosting, random_forest

## Statistical battery

A battery of 2 statistical tests is run on the champion (2 passed, 0 failed). See [diagnostics](./diagnostics.md) for the full table.

## Deployment scoring

The trained champion is persisted as a picklable scorer bundle. The deployment scoring entry point is {@code:src/cognos/runtime/score.py#score_row}, which evaluates a single row as the IMPACT derived-field contract. See [model](./model.md).
