# COGNOS

COGNOS is an autonomous, governed model-development system. An LLM **reasoning layer** drives the
decisions a human modeler would make — design, model choice, and the search for paths that improve
performance — while a **deterministic engine** grounds, verifies, and records every result so the
output is reproducible and auditable. Neither layer is optional: the reasoning automates judgment and
removes humans from the loop; the determinism is what keeps the reasoning honest (it is the
anti-hallucination mechanism).

**Primary domain:** commercial model development (e.g. commercial credit-risk: facility / obligor /
collateral), governed by **SR 11-7 model risk management**. Fair-lending scans are an **optional,
off-by-default module** for commercial profiles (report them as out of scope for the profile, not as
inapplicable law — ECOA/Reg B does reach business credit);
see [ADR-0004](docs/adr/0004-primary-domain-commercial-fair-lending-optional.md).

## Language

**Reasoning Layer** (a.k.a. the agents):
The LLM-driven decision layer. It proposes design choices, model specifications, and improvement
hypotheses, and explores paths toward better performance. Its purpose is automation of judgment —
reducing dependence on a human modeler.
_Avoid_: "the LLM garnish", "optional AI layer".

**Deterministic Engine**:
The non-LLM core that fits models, scores them on a frozen metric and sealed holdout, runs the
statistical battery, and persists artifacts. Its purpose is reproducibility, auditability, and
**grounding the reasoning layer against hallucination**.
_Avoid_: "the fallback", "the heuristic path" (it is not a fallback; it is the substrate).

**Propose / dispose** (the core loop):
The reasoning layer may only **propose** (specifications, transforms, hypotheses, prose). The
deterministic engine **disposes** — it fits, scores, tests, and is the sole authority on what is kept
and on any metric or verdict that becomes a fact. Every state-changing action passes through
deterministic evaluation. This is the invariant that bounds hallucination.

**Gate**:
A stage that can BLOCK the pipeline (autonomous) or pause for human approval (interactive). The gates
are **`validate`** (technical soundness) and **`review`** (docs↔code consistency) — and *only* those.
_Avoid_: calling `comply` a gate; it is a non-gating report.

**Model-risk readiness report** (the `comply` deliverable):
An optional, non-gating report that organizes the substantive evidence into the SR 11-7 structure and
lists the human-only steps that remain (independent validation sign-off, monitoring plan, governance).
It never adjudicates compliance and never marks an element compliant without concrete evidence.
See [ADR-0006](docs/adr/0006-compliance-is-non-gating-readiness-report.md).

**Frozen Substrate**:
The sealed holdout plus the metric definitions that the reasoning layer is not permitted to edit, so
an autonomous loop cannot game its own yardstick.

**Experiment** (a.k.a. candidate):
One unit of LLM-authored change — a model spec and/or feature transforms — committed per iteration.
The ratchet keeps it (advances the history) only if the deterministic engine measures an improvement
on the frozen metric; otherwise it is reverted. The committed history is the audit trail.

**Target-hidden execution**:
The rule that feature-construction code runs against a features-only view (`X`); the target `y` is
never in scope. It is what makes arbitrary LLM-authored transforms safe — they physically cannot read
the answer, so they cannot leak the target (even onto the labelled holdout).

**Prediction pipeline**:
The fitted transform+estimator used to *score* (serving). For categoricals it uses all-K one-hot with
`handle_unknown='ignore'` so unseen categories never crash at serve time.

**Inference design**:
A *separate*, full-rank (K-1 / drop-first dummy coding + intercept) design used only for statsmodels
**coefficient/p-value estimation** on training data. Decoupled from the prediction pipeline so the
reported coefficients and significances are statistically valid (no dummy-variable trap), as SR 11-7
validation of a regression model requires.

**Outcomes analysis** (SR 11-7 backtesting):
Predicted-vs-actual validation of a credit-risk model on an out-of-time sample — discrimination
(Gini/KS), calibration (expected vs observed default rate by band), and stability (PSI/CSI). This is
what "backtest" means for the primary domain; trading metrics (PBO, Deflated Sharpe) are an opt-in
mode. See [ADR-0005](docs/adr/0005-backtesting-is-credit-risk-outcomes-analysis.md).

**Out-of-time (OOT) sample**:
A holdout split by vintage/date (train on older, evaluate on newer) rather than at random, so
backtesting reflects real temporal generalization and population drift.
_Avoid_: conflating with the random sealed holdout used for in-period model selection.

**Reproducibility (two-tier)**:
The *analysis* (everything the deterministic engine computes) is fully, bit-reproducible offline with
no LLM. The *reasoning trajectory* (what the LLM proposed) is non-deterministic, recorded for audit,
and mitigated by human review. The LLM is required to automate the search, never to reproduce the
result. See [ADR-0003](docs/adr/0003-two-tier-reproducibility.md).

**Design brief (MD triangulation)**:
The sponsor's answers to the design questions that precede any algorithm — use case, horizon,
default definition, segment, interpretability (`design:` config block). Every *unanswered* point
becomes an explicit **open question** in ideate's `design_brief.md`, never a silent assumption.
_Avoid_: answering a sponsor question on the sponsor's behalf.

**Framework assessment (alternatives considered)**:
Ideate's deterministic judgment of which econometric frameworks fit the data structure — structural
(Merton) vs reduced-form PD vs discrete-time hazard vs rating migration vs ML challenger — each
applicable / partial / **rejected with a stated reason**. The rejections are the SR 11-7
"alternatives considered" evidence.

**Discrete-time hazard family** (`hazard_logit` / `hazard_cloglog`):
A survival model (Shumway 2001) fit by **panel-expanding obligor-periods inside the estimator**, so
CV and the sealed holdout keep splitting at the obligor level (the survival-CV leak is impossible by
construction). Unlocked by `data.event_time_col` — the label's *timing*, which is outcome data and
therefore never a feature. Deliverable beyond a single-horizon PD: the **PD term structure**
(cumulative PD per period). See [ADR-0008](docs/adr/0008-econometric-core-survival-structural-simulation.md).

**Structural (Merton) engine**:
A deterministic KMV fixed-point solver (equity value/vol + debt face → asset value/vol,
**distance-to-default**, structural PD). **Hybrid mode** (the default use) feeds DD to the champion
as an engineered feature, recomputed target-hidden at serve time; the pure structural PD is scored
on the sealed holdout only as a **labelled challenger benchmark**, never the deployed model.
_Avoid_: calling the structural PD a fitted model — it is a solver output.

**Rating-migration engine** (`migration:` block):
A cohort-method **transition matrix** estimated on the training partition only (it is a *fitted*
model, unlike the Merton solver, so it must never see the sealed holdout), with the standard
agency-data statistics: **NR adjustment** (withdrawn ratings out of the row denominator), Laplace
smoothing, and **weighted-PAVA rank-ordering** of the default column (adjustments reported, never
silent). **Hybrid mode** swaps the raw rating for the horizon cumulative PD it implies
(`migration_pd`, recomputed target-hidden at serve time — keeping both would be perfectly
collinear); the pure-migration PD is a sealed-holdout **labelled challenger benchmark**. Reports:
matrix-power **cumulative PD term structure**, regime-**conditional matrices**
(`condition_col`), and the by-rating **expected-loss forecast** (EAD × LGD × cumPD). The typical
estimation sample is a long **external agency history** (S&P CreditPro-style) when the internal
rating history is too short — a decision ideate itself evidences (EPV finding + unlock question).
See [ADR-0009](docs/adr/0009-rating-migration-external-agency-data.md).
_Avoid_: estimating the matrix on the full frame — that leaks the holdout into the benchmark.

**Portfolio simulation (Vasicek / ASRF)**:
Seeded one-factor Monte Carlo of the portfolio loss distribution (EL/UL/VaR/ES) plus closed-form
Basel IRB capital, conditioned on the champion's PDs. A *report* the backtest stage emits — it never
feeds champion selection, so the frozen substrate is untouched.

**Stress scenario**:
A named set of covariate shocks (`add`/`mul`/`set`) re-scored **deterministically** through the
deployed scorer; reported as mean-PD / expected-loss deltas vs baseline. Shocks against unknown
columns are surfaced, never silently ignored.

## Resolved

- **"Deterministic mode" status** — resolved: running with no LLM is not a degraded product; it is how
  the deliverable's analysis is always reproduced and served. The LLM automates the *search*; it never
  sits inside the deliverable's reproducibility/trust boundary.

## Example dialogue

> **Modeler:** Can COGNOS just pick the model on its own?
> **COGNOS:** The reasoning layer *proposes* a model spec from the data profile. It never decides on
> its own that the model is good — the deterministic engine fits it, scores it on the sealed holdout,
> and runs the statistical battery. Only a measured improvement on the frozen metric is kept.
> **Modeler:** So if the LLM is confidently wrong?
> **COGNOS:** It can't enter the record. A wrong proposal simply fails to beat the incumbent on the
> holdout and is discarded. The LLM proposes; the engine disposes.
