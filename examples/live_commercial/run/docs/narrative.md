---
type: narrative
title: Narrative
description: Agent-drafted narrative with engine-rendered numbers.
tags:
- narrative
timestamp: '2026-09-30T02:35:55.571384+00:00'
---

# Narrative

_Drafted by the Technical Writer agent; every number is rendered from a recorded fact._

## Executive summary

This paper documents a reduced-form probability-of-default scorecard for private commercial obligors. The target is a binary default flag on a vintage-indexed panel of 1200 observations with 6 candidate features and 81 defaults (an event rate of 0.0675). The champion is ridge_logit using all 6 features, chosen from 16 candidates by walk-forward cross-validated ROC AUC (mean 0.8153, standard deviation 0.0806).

On the out-of-time sample of 240 observations, which was evaluated 1 time, the model reached an AUC of 0.8774, a Gini of 0.7549 and a KS of 0.6265. Its expected calibration error was 0.0559, and a PSI of 0.0238 (stable) showed the score distribution holding steady.

The validation verdict is WARN, with 6 findings, and 3 compliance items remain outstanding. The intended use, outcome horizon, default definition and portfolio segment have not been specified by the sponsor. Until they are, the model is suitable only as a development benchmark and not for credit decisions.

## Methodology rationale

**Purpose and use.** The sponsor has not set the use case, horizon, default definition or segment; these remain open questions (9 recorded). No production use is in scope until they are answered. Uses outside this scope include:
- pricing
- regulatory capital
- stress testing
- lifetime-loss accounting
- any portfolio other than the one sampled

**Data.** The panel has 1200 rows and a default rate of 0.0675. No columns were excluded. The leakage screen found none suspects. How the target was labelled (the event definition and outcome window) is undocumented and must be confirmed by the sponsor.

**Framework.** A GLM scorecard is the standard interpretable framework for private obligors with financial-ratio and sector covariates, and interpretability is a stated requirement. A ridge penalty stabilises the estimates given the modest event count. The data were split into 960 training rows and 240 out-of-time rows.

**Coefficients.** The drivers carry economically sensible signs and are significant:
- leverage: 0.9781, p = 0
- interest coverage: -0.9808, p = 0
- log assets: -0.3963, p = 0.0122
- current ratio: -0.377, p = 0.0188

Profit margin is not significant (p = 0.5914), and neither are the sector indicators (p = 0.2358, 0.9761 and 0.1668). The model passed 2 of 2 diagnostics, covering VIF and the condition number.

## Alternatives considered

**Discrete-time hazard: candidate, not yet developed.** Because the data are indexed by vintage, a logit or cloglog on this panel can be read as a discrete-time hazard. Probit, cloglog, lasso and unpenalised logit were all among the families tried (logit, probit, cloglog, lasso_logit, ridge_logit, gradient_boosting, random_forest). A full hazard model and a PD term structure need an event-time column. That column matters if the use case turns out to be lifetime-loss accounting.

**Structural (Merton): rejected.** These private obligors have no traded equity value, equity volatility or liability structure, so distance-to-default cannot be computed.

**Rating-transition matrices: rejected.** The data hold no rating history across successive snapshots.

**Machine-learning challengers: benchmarks only.** Random forest and gradient boosting were run solely to measure the predictive ceiling, because interpretability is required. This context does not record their scores. The challenger gap, which is the price of interpretability, therefore cannot be quantified here and should be added from the model engine's challenger comparison table.

## Limitations and assumptions

**Events per variable.** With 81 defaults, the model has 13.5 events per variable. That is adequate but thin, and it limits how precisely the weaker coefficients can be estimated.

**Performance stability.** Walk-forward AUC varies noticeably across folds (standard deviation 0.0806). The out-of-time AUC (0.8774) exceeds the cross-validated mean (0.8153). Because it rests on a single sample of 240 rows, it should not be read as the expected level of performance.

**Censoring and regime coverage.** Right-censoring cannot be assessed because there is no event-time column. The sample's coverage of economic regimes is undocumented, so behaviour in a downturn is untested.

**Weak variables.** Profit margin has an insignificant, counter-intuitive positive coefficient (0.0893). The sector effects are not significant, and whether sectors should be pooled is unresolved.

**Open sponsor questions.** The use case, horizon, default definition, segment and event-time availability all remain open. Any of these answers could change the target and the model.

## Use and monitoring

**Monitoring.** Once a use is approved, the following should be tracked:
- discrimination: AUC, Gini and KS against the out-of-time baseline of 0.8774, 0.7549 and 0.6265
- calibration: expected calibration error against the baseline of 0.0559
- population stability: PSI against the baseline of 0.0238

Monitoring should run on each new vintage. A material PSI shift, a fall in discrimination, or calibration drift should trigger recalibration or redevelopment. Specific thresholds and cadence depend on the use case and must be set by the sponsor and model risk management. The compliance review rated ongoing monitoring warn, so a formal monitoring plan is an outstanding item. Conceptual soundness (pass) and outcomes analysis (pass) were both rated as passing.

**Governance record.** All four gates (data, design, champion and validation) were accepted automatically in autonomous mode. No human developer reviewed them, and none of the model's challenges were logged. A validator should treat these gates as not yet reviewed by a human: each needs human sign-off before the model is used.
