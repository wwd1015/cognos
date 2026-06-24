# COGNOS — comprehensive feature list

## Pipeline & orchestration
- Eight-stage model-development lifecycle: `explore → ideate → model → backtest → validate → comply → document → review`.
- Mechanical orchestrator: deterministic sequencing, gate handling, escalation (judgment lives in agents).
- **Two operating modes**, one control flow:
  - Autonomous end-to-end run (`cognos run`) for quick prototypes.
  - Stage-by-stage / human-in-the-loop: every stage independently invocable (`cognos run-stage`), plus `--interactive` gate pauses (approve / reject).
- **Gates** (`validate`, `review` — and *only* those) that can BLOCK a run or pause for human approval; `comply` is a non-gating report (ADR-0006).
- **Reasoning steps are HITL pause points** too: the LLM-driven `ideate` step and opt-in LLM-guided search are natural human-in-the-loop review points, not just the gates (ADR-0003).
- **Checkpoint + resume-from-failure**: each stage persists its result; re-running a `run_id` reuses completed stages.
- Per-run directory with full provenance (manifest, summary, data, models, docs, per-stage artifacts).
- Greppable verdict tokens + machine-readable run summary for tooling/CI.
- Per-project YAML profile (`CognosConfig`) — the only place project specifics live; agents are project-agnostic.

## Reasoning layer (propose / dispose)
- **Two-layer system** (ADR-0001): an LLM **reasoning layer** *proposes* (design, model choice, feature engineering, the next experiment) and a deterministic **engine** *disposes* (fits, scores on the frozen metric + sealed holdout, runs the statistical battery; sole authority on what is kept). Both are first-class; the determinism is the anti-hallucination mechanism.
- **Two staged depths of LLM involvement**: (A) LLM-driven **ideation** emitting *executable* feature-engineering transforms (not just prose); (B) opt-in **LLM-guided search** (`search.guided`) where the LLM is the mutation function proposing the next experiment from the ledger. Deterministic grid search is the baseline + the test double.
- **Target-hidden transform execution** (ADR-0002): LLM-authored transforms run against a features-only view (`X`); the target `y` is never in scope, so they cannot leak the target — even onto the labelled holdout. Implemented as a safe **AST-whitelisted expression executor** (`modeling/transforms.py`) over feature columns + a fixed set of `np.<fn>` functions.
- **Round-trips to serving**: a kept transform persists verbatim to an IMPACT derived field, is stored on the champion + the deployed scorer, and is re-applied target-hidden at serve time (train- and serve-time feature logic are identical).
- **Recorded reasoning trajectory** (ADR-0003): every proposal (prompts + responses + model id) is logged to `runs/<id>/reasoning/transcript.jsonl` for replay/audit; the analysis re-derives bit-identically offline with no LLM, while the trajectory is non-deterministic and human-gated. The LLM is required to automate the search, never to reproduce the result.
- **`ScriptedBrain` test double**: replays canned responses so the reasoning-driven path is deterministically testable; running with no LLM is the deterministic substrate + test double, not a degraded product.

## Data exploration (`explore`)
- Schema/dtype/missingness profiling; numeric distribution summaries.
- Target-relationship correlations; **target-leakage detection** (near-perfect correlation suspects).
- Class-imbalance and constant-feature detection.
- Findings with severity; data profile artifact.

## Idea generation (`ideate`)
- Enumerates task-appropriate model families × feature strategies.
- Ranks hypotheses by an interpretability/parsimony heuristic informed by the data profile.
- Per-hypothesis rationale (Reg-friendly: prefers interpretable specs).
- **LLM-driven ideation emits executable feature-engineering transforms** (not just prose), authored and run target-hidden via the safe AST-whitelisted executor (ADR-0001, ADR-0002).

## Modeling & statistical testing (`model`)
- **CASH search** (combined algorithm + hyperparameter selection) as one conditional space.
- **Ratchet search** (accept-if-better-else-discard) with an experiment ledger (`results.tsv`-style), cheap/simple candidates first, candidate + optional wall-clock budgets.
- **Leakage-safe cross-validation in search + a sealed/out-of-time holdout** — all preprocessing fit inside training folds only; selection is validated against a holdout the search cannot touch.
- **Frozen substrate**: sealed holdout + metric definitions the search cannot edit (anti-reward-hacking).
- Traditional statistical models (OLS, Ridge, Lasso, ElasticNet, Logit, regularized logit) **with statsmodels inference** (coefficients, p-values, residuals) — *and* ML models (Random Forest, Gradient Boosting) with feature importances. Configurable for either.
- **Opt-in GLM / econometric regressors**: Poisson, Gamma, Tweedie families.
- **Valid coefficient inference** (ADR/grilling Q6): statsmodels coefficients/p-values come from a separate **full-rank K-1 (drop-first) inference design** (`build_inference_design`), decoupled from the all-K prediction pipeline, so reported significances are statistically valid (no dummy-variable trap / astronomical condition number).
- **Single interpretable champion by default** (ADR-0007): the deployed model is always one interpretable champion (interpretability/parsimony preference). The Caruana ensemble is **not** a silent default — it is an optional, labelled **challenger / predictive-ceiling benchmark** (`search.ensemble`, off by default), never silently treated as the deliverable.
- Parsimony/complexity penalty (simplicity bias).
- **Statistical diagnostic battery** with known H0 per test, machine-actionable pass/warn/fail:
  - Heteroskedasticity: Breusch-Pagan, White, Goldfeld-Quandt.
  - Autocorrelation: Durbin-Watson, Breusch-Godfrey, Ljung-Box.
  - Normality: Jarque-Bera (+ omnibus).
  - Linearity / specification: Harvey-Collier, Ramsey RESET.
  - Multicollinearity: max VIF, design condition number.
  - Stationarity (time series): ADF + KPSS (complementary pair).
- Pinned seeds; deployable scorer persisted for serving/IMPACT.

## Backtesting (SR 11-7 outcomes analysis) & IMPACT integration (`backtest`)
- **Backtest = SR 11-7 outcomes analysis by default** (ADR-0005), computed on an **out-of-time (OOT)** sample via `modeling/credit_metrics.py`:
  - **Discrimination** — Gini / AUC and the KS statistic.
  - **Calibration** — expected-vs-observed default rate by score band, plus Expected Calibration Error (ECE).
  - **Stability** — PSI (and CSI) between the development and OOT populations.
- **Out-of-time holdout**: the holdout is time-ordered (train on older vintages, evaluate on newer) when a datetime/vintage column is configured, not a random split.
- **Real IMPACT integration**: emits an `EntityConfig` YAML embedding the model as a derived field (`cognos.runtime.score.score_row`) and runs `EntityPipeline` to produce a standardized scored feature table (predicted score, actual outcome, segment, time) — the natural input to the outcomes metrics.
- Transparent **built-in fallback** when IMPACT is not installed (records `used_impact`).
- **Opt-in trading/returns mode** (`backtest.returns_column`): **Probability of Backtest Overfitting** (PBO via Combinatorially-Symmetric CV) and **Deflated Sharpe Ratio** + Probabilistic Sharpe Ratio. These presume a returns series and are **not** run on credit-risk models.
- Walk-forward stability of the champion.

## Independent validation (`validate`, gate)
- SR 11-7 **effective challenge**: runs independently of the modeling stage (separate context/agent).
- Five-axis rubric (0–1): leakage, overfitting, stability, diagnostics, statistical significance.
- Hard **BLOCK on confirmed target leakage**; FAIL on a large CV-vs-sealed-holdout gap (the reliable overfit trigger).
- Detects pathological coefficients/p-values (collinearity, separation, numerical blow-up) — soundness checks made meaningful by the decoupled full-rank inference design.
- One of the **two gates** (with `review`); `validate` is the technical-soundness gate.
- Optional LLM "does this model make sense" narrative (additive; never changes the verdict).

## Compliance & model risk (`comply`, non-gating report)
- **Non-gating model-risk readiness report** (ADR-0006): never PASS/BLOCK on compliance. It *organizes* the SR 11-7 evidence the substantive stages already produced; it never adjudicates compliance.
- **Never auto-passes an unevidenced element**: items without concrete, checkable evidence are listed as outstanding gaps (ongoing monitoring is always outstanding at dev time), never silently marked compliant.
- **Lists the human-only outstanding steps**: independent validation sign-off, a monitoring plan with thresholds, override/governance policy.
- **Model inventory** entry (id, version, owner, risk tier, purpose, intended / out-of-scope use).
- **SR 11-7** three core elements: conceptual soundness, ongoing monitoring, outcomes analysis.
- **NIST AI RMF** four functions (Govern / Map / Measure / Manage) mapped to artifacts.
- Seven **trustworthy-AI** characteristics checklist.
- EU jurisdiction → flags required **EU AI Act Annex IV** components.
- Risk-tier-proportional assessment.
- **Optional consumer-only fair-lending module** (`compliance.fair_lending`, **off by default**; ADR-0004): four-fifths disparate-impact ratio per protected attribute, group selection rates, and plain-English adverse-action reason codes. Does **not** apply to the primary commercial domain (ECOA/Reg B is consumer law) and is never part of the default pipeline.

## Documentation (`document`)
- White paper as a **Google Open Knowledge Format (OKF v0.1) bundle**: one markdown concept per artifact (overview, dataset, methodology, model, coefficients, diagnostics, backtest, validation, compliance, model card, limitations, Annex IV) with YAML frontmatter, `index.md`, and `log.md`.
- **Documents only what shipped** (ADR-0007): the single interpretable champion (with its persisted target-hidden transforms); any ensemble appears only as an explicit, labelled challenger benchmark, never as the deliverable.
- **Coefficients reported from the full-rank inference design** so the documented significances are statistically valid.
- **Outcomes analysis** (Gini/KS, calibration/ECE, PSI on the OOT sample) is the backtest section for credit models; trading metrics appear only in opt-in returns mode.
- **Two-tier reproducibility note** (ADR-0003): the analysis is reproducible offline with no LLM; the reasoning trajectory (`runs/<id>/reasoning/transcript.jsonl`) is recorded for replay/audit.
- **Google Model Card** (all 9 sections).
- **EU AI Act Annex IV** technical-documentation pack (when EU is in scope).
- **Docs↔code links**: `{@code:path#symbol}` anchors trace white-paper paragraphs to deployment code, plus cross-linked OKF concepts forming a knowledge graph.
- Single-file human-readable white paper export.

## Consistency / drift review (`review`, gate)
- Walks the OKF graph; **AST-verifies every docs↔code anchor** (missing file or symbol → drift).
- Validates internal markdown links resolve; checks declared resources exist.
- OKF conformance check; quantitative `drift_score`; severity + confidence on findings.
- **BLOCK** when documentation claims code that does not exist (stale references) — keeps docs and code in sync.

## Engine, integrations, and developer experience
- **Pluggable brain**: deterministic by default (offline, no API key — the substrate, not a fallback); a `ScriptedBrain` test double for deterministic testing of the reasoning path; optional Claude (`anthropic`) backend that degrades gracefully.
- **autoforge protocol** reimplemented: `name: value` stdout parsing, experiment ledger, generic ratchet loop.
- Typed config (Pydantic v2) with auto metric/direction resolution and YAML round-trip.
- CLI: `run`, `run-stage`, `demo`, `init`, `explain`, `report`, `list-runs`, `agents`.
- **Demo tasks** including the new **`commercial`** task — a commercial credit-risk model-validation showcase (SR 11-7 + OOT outcomes analysis), the flagship example (ADR-0004). The consumer fair-lending `credit` demo exercises an off-by-default module, not the primary path.
- Python API: `run_pipeline`, `Orchestrator`, `RunContext`, `CognosConfig`.
- Synthetic data generators (regression, classification, time series, commercial credit-risk with a vintage/date column for OOT calibration/PSI, and consumer credit-with-protected-attribute for the optional fair-lending module) for tests and demos.
- **Claude-Code-native agent layer**: declarative `.claude/agents/*.md`, an orchestrator slash command, PreToolUse safety hooks, per-project profiles (deputy-style).
- **Eval harness** for autonomous runs (assert verdicts/gates per case).
- Comprehensive test suite (unit + integration) and a runnable end-to-end example.
- MIT licensed; `uv`/`pip` installable; Python ≥ 3.11.
