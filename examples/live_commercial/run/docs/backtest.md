---
type: backtest
title: Backtest
description: Out-of-sample performance, PBO, and DSR.
resource: stages\backtest\scored.parquet
tags:
- backtest
timestamp: '2026-09-30T02:35:19.202770+00:00'
---

# Out-of-Sample Backtest

- **Scheme:** n/a
- **OOS roc_auc:** 0.8774
- **Scored rows:** 240
- **PBO (prob. of backtest overfitting):** n/a
- **Deflated Sharpe ratio:** n/a
- **IMPACT used:** False — IMPACT not installed; used built-in fallback scorer.



The backtest is implemented at {@code:src/cognos/stages/backtest.py}. See [model](./model.md).
