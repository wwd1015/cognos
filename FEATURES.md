# COGNOS — comprehensive feature list

## Pipeline & orchestration
- Eight-stage model-development lifecycle: `explore → ideate → model → backtest → validate → comply → document → review`.
- Mechanical orchestrator: deterministic sequencing, gate handling, escalation (judgment lives in agents).
- **Two operating modes**, one control flow:
  - Autonomous end-to-end run (`cognos run`) for quick prototypes.
  - Stage-by-stage / human-in-the-loop: every stage independently invocable (`cognos run-stage`), plus `--interactive` gate pauses (approve / reject).
- **Gates** (`validate`, `comply`, `review`) that can BLOCK a run or pause for human approval.
- **Checkpoint + resume-from-failure**: each stage persists its result; re-running a `run_id` reuses completed stages.
- Per-run directory with full provenance (manifest, summary, data, models, docs, per-stage artifacts).
- Greppable verdict tokens + machine-readable run summary for tooling/CI.
- Per-project YAML profile (`CognosConfig`) — the only place project specifics live; agents are project-agnostic.

## Data exploration (`explore`)
- Schema/dtype/missingness profiling; numeric distribution summaries.
- Target-relationship correlations; **target-leakage detection** (near-perfect correlation suspects).
- Class-imbalance and constant-feature detection.
- Findings with severity; data profile artifact.

## Idea generation (`ideate`)
- Enumerates task-appropriate model families × feature strategies.
- Ranks hypotheses by an interpretability/parsimony heuristic informed by the data profile.
- Per-hypothesis rationale (Reg-friendly: prefers interpretable specs); optional LLM-drafted ideas.

## Modeling & statistical testing (`model`)
- **CASH search** (combined algorithm + hyperparameter selection) as one conditional space.
- **Ratchet search** (accept-if-better-else-discard) with an experiment ledger (`results.tsv`-style), cheap/simple candidates first, candidate + optional wall-clock budgets.
- **Leakage-safe nested cross-validation** — all preprocessing fit inside training folds only.
- **Frozen substrate**: sealed holdout + metric definitions the search cannot edit (anti-reward-hacking).
- Traditional statistical models (OLS, Ridge, Lasso, ElasticNet, Logit, regularized logit) **with statsmodels inference** (coefficients, p-values, residuals) — *and* ML models (Random Forest, Gradient Boosting) with feature importances. Configurable for either.
- **Caruana greedy ensemble** of search survivors (reuses cached OOF predictions; near-free).
- Parsimony/complexity penalty (simplicity bias).
- **Statistical diagnostic battery** with known H0 per test, machine-actionable pass/warn/fail:
  - Heteroskedasticity: Breusch-Pagan, White, Goldfeld-Quandt.
  - Autocorrelation: Durbin-Watson, Breusch-Godfrey, Ljung-Box.
  - Normality: Jarque-Bera (+ omnibus).
  - Linearity / specification: Harvey-Collier, Ramsey RESET.
  - Multicollinearity: max VIF, design condition number.
  - Stationarity (time series): ADF + KPSS (complementary pair).
- Pinned seeds; deployable scorer persisted for serving/IMPACT.

## Standardized backtesting & IMPACT integration (`backtest`)
- **Real IMPACT integration**: emits an `EntityConfig` YAML embedding the model as a derived field (`cognos.runtime.score.score_row`) and runs `EntityPipeline` to produce a standardized scored feature table.
- Transparent **built-in fallback** when IMPACT is not installed (records `used_impact`).
- Out-of-sample metric on the sealed holdout.
- **Probability of Backtest Overfitting** (PBO via Combinatorially-Symmetric CV) over the full search library.
- **Deflated Sharpe Ratio** + Probabilistic Sharpe Ratio (multiple-testing/selection-bias corrected) when a returns column is configured.
- Walk-forward stability of the champion.

## Independent validation (`validate`, gate)
- SR 11-7 **effective challenge**: runs independently of the modeling stage (separate context/agent).
- Five-axis rubric (0–1): leakage, overfitting, stability, diagnostics, statistical significance.
- Hard **BLOCK on confirmed target leakage**; FAIL on a large CV-vs-holdout gap; PBO is an informational signal.
- Detects pathological coefficients/p-values (collinearity, separation, numerical blow-up).
- Optional LLM "does this model make sense" narrative (additive; never changes the verdict).

## Compliance & model risk (`comply`, gate)
- **Model inventory** entry (id, version, owner, risk tier, purpose, intended / out-of-scope use).
- **SR 11-7** three core elements: conceptual soundness, ongoing monitoring, outcomes analysis.
- **NIST AI RMF** four functions (Govern / Map / Measure / Manage) mapped to artifacts.
- Seven **trustworthy-AI** characteristics checklist.
- **Fair-lending (ECOA / Reg B)**: four-fifths **disparate-impact** ratio per protected attribute, group selection rates, **BLOCK** on violation.
- Plain-English **adverse-action reason codes** (top signed drivers, intercept excluded, design prefixes stripped).
- EU jurisdiction → flags required **EU AI Act Annex IV** components.
- Risk-tier-proportional assessment.

## Documentation (`document`)
- White paper as a **Google Open Knowledge Format (OKF v0.1) bundle**: one markdown concept per artifact (overview, dataset, methodology, model, coefficients, diagnostics, backtest, validation, compliance, model card, limitations, Annex IV) with YAML frontmatter, `index.md`, and `log.md`.
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
- **Pluggable brain**: deterministic by default (offline, no API key); optional Claude (`anthropic`) backend that degrades gracefully.
- **autoforge protocol** reimplemented: `name: value` stdout parsing, experiment ledger, generic ratchet loop.
- Typed config (Pydantic v2) with auto metric/direction resolution and YAML round-trip.
- CLI: `run`, `run-stage`, `demo`, `init`, `explain`, `report`, `list-runs`, `agents`.
- Python API: `run_pipeline`, `Orchestrator`, `RunContext`, `CognosConfig`.
- Synthetic data generators (regression, classification, time series, credit-with-protected-attribute) for tests and demos.
- **Claude-Code-native agent layer**: declarative `.claude/agents/*.md`, an orchestrator slash command, PreToolUse safety hooks, per-project profiles (deputy-style).
- **Eval harness** for autonomous runs (assert verdicts/gates per case).
- Comprehensive test suite (unit + integration) and a runnable end-to-end example.
- MIT licensed; `uv`/`pip` installable; Python ≥ 3.11.
