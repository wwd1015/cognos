---
type: model
title: Champion Model
description: Family, hyperparameters, and headline metrics.
resource: models/champion_scorer.joblib
tags:
- model
timestamp: '2026-09-30T02:35:19.199741+00:00'
---

# Champion Model

- **Family:** ridge_logit
- **Description:** ridge_logit/top
- **CV roc_auc:** 0.8153 ± 0.0806
- **Frozen-holdout roc_auc:** 0.8774
- **Train / holdout rows:** 960 / 240

## Hyperparameters

| parameter | value |
| --- | --- |
| C | 0.1 |
| random_state | 42 |

## Scoring

Deployment-time scoring is served by {@code:src/cognos/runtime/score.py#score_row}. See [coefficients](./coefficients.md), [diagnostics](./diagnostics.md), and [methodology](./methodology.md).
