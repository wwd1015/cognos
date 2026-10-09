---
type: model_card
title: Model Card
description: "Google Model Card \u2014 9 standard sections."
tags:
- model_card
- governance
timestamp: '2026-09-30T02:35:19.204671+00:00'
---

# Model Card

## 1 Model Details

- **Name:** demo_commercial (v0.1.0)
- **Type:** ridge_logit (classification)
- **Developed with:** COGNOS automated model-development pipeline
- **Description:** COGNOS synthetic commercial demo — Commercial PD (vintage panel, out-of-time)

## 2 Intended Use

Intended use not specified.

**Out-of-scope use:** Not specified.

## 3 Factors

Relevant factors include the input feature distribution and, for fairness, the protected attributes: none declared.

## 4 Metrics

- **Primary metric:** roc_auc (maximize)
- **CV roc_auc:** 0.8153 ± 0.0806
- **Frozen-holdout roc_auc:** 0.8774

## 5 Evaluation Data

A sealed frozen holdout of 240 rows, never touched during model search. See [dataset](./dataset.md).

## 6 Training Data

960 training rows over 6 features drawn from the source dataset (1200 total rows).

## 7 Quantitative Analyses

2 of 2 statistical tests passed; see [diagnostics](./diagnostics.md). Out-of-sample backtest roc_auc=0.8774; see [backtest](./backtest.md).

## 8 Ethical Considerations

Fair-lending checks (SR11-7, NIST-AI-RMF) were performed; see [compliance](./compliance.md). Verdict: see compliance stage.

## 9 Caveats & Recommendations

Validity is bounded by the training distribution; monitor for drift and re-validate before high-stakes use. See [limitations](./limitations.md).
