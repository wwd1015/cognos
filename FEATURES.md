# COGNOS — comprehensive feature list

## Development modes and intake (v1.1)
- **Two development modes** (`engagement.kind`, ADR-0011): **new model development**, from the
  sponsor's business intent document and background material; and **model update**, from the
  existing model's artifacts (white paper, development / deployment code, validation and
  monitoring reports, or an earlier COGNOS run) and an update request.
- **One template** for the intent document and the update request (`cognos intent-template
  [--kind update]`, the workbench's download button, `docs/templates/`), prepared before the run.
  The engine parses its sections; headings are matched by alias, hints and `TBD` count as empty.
- **Documents by upload or by path**: `.md`, `.txt`, `.docx` (headings kept), source code,
  notebooks, `.pdf` with `pypdf`. Copied into `runs/<id>/inputs/`, hashed, inventoried; an
  unreadable file is a finding and a missing one refuses the run.
- **Intake stage + Intake Analyst**: restates the objective, fills a 12-field engagement brief (17
  for an update), judges whether the intent is clear / needs clarification / unclear.
- **The interview**: every required field the documents do not state becomes a blocking question;
  the agent adds its own (a vague goal, a contradiction, an unmeasurable success criterion).
  Answers re-run intake until nothing blocks. Answer in the workbench form, with `cognos answer`,
  or in the terminal prompt of `run --interactive`.
- **Grounded brief**: a position is *stated* only with a quote the engine finds in the documents;
  *inferred* positions are asked about, never applied.
- **`gate_intent`**: answer, confirm, or send back. Confirming with blocking questions open needs
  a reason. What the document states then enters the effective config (design brief, intended and
  out-of-scope use); a document that disagrees with the profile is flagged.
- **Model update**: change items typed and routed to the stage they alter; an update scope
  (recalibrate / re-estimate / redevelop); the existing model's family and inputs read from its
  artifacts; the incumbent family always on the slate and ranked first; the validator checks the
  request was delivered; a **model change record** in the white paper with the existing model's
  recorded figures beside the update's when it is an earlier COGNOS run.
- **Scoped prior scores**: `prior.*` facts reach the outcomes analyst, validator, model-risk
  analyst and writer; the design lead and the modeler see the incumbent's family, not its scores.
- **Every agent sees the confirmed brief** (`project.engagement`).

## Workflow engine (v1.0)
- Nine-stage lifecycle `intake → explore → ideate → model → backtest → validate → comply →
  document → review` interleaved with **six human review gates**: `gate_intent`, `gate_data`,
  `gate_design`, `gate_champion`, `gate_validation`, `gate_signoff` (ADR-0010, ADR-0011).
- **Mechanical engine** (`engine/`): a step graph with dependencies, statuses
  (pending / running / done / awaiting / stale / failed / blocked / skipped), and **stale
  propagation** — a revised decision, an answered question or a challenge marks everything
  downstream stale and the engine re-runs it; a step invalidated mid-run stays stale.
- **Two modes, one engine**: interactive (pause at every enabled review gate) and autonomous (gates
  auto-accept the agent's recommendation, recorded as actor `auto` and seat `express` — preparation,
  not a signature); gates can be disabled per project.
- **Seats**: intent, data, design and champion belong to the model developer; validation to the independent
  reviewer; sign-off to the approver. The engine refuses a seat on another seat's gate. `approve`
  seals `packages/vN.json` once (a digest of the recorded design, exclusions, slate, champion and
  verdicts). A later invalidation or a re-opened gate supersedes that package; the file stays.
  Approval is not deployment. Use, horizon, default definition and segment must be answered before
  anyone can sign.
- **Verdict gates** `validate` and `review` — and only those — can BLOCK; a BLOCK halts an
  autonomous run and can only be sent back or rejected in an interactive one. `comply` is a
  non-gating report (ADR-0006).
- **Durable run state** (`state.json`): steps, gate decisions, challenges, gaps, overrides, loop
  counters, spend. Any process — CLI, UI after a restart — resumes exactly; `cognos run-stage` still
  re-runs a single stage; failed steps retry; decided gates can be re-opened and revised.
- Activity feed (`events.jsonl`), per-run directory with full provenance, greppable verdict tokens,
  machine-readable run summary; per-project YAML profile (`CognosConfig`).

## Human decisions
- **Gate actions**: accept, edit (interview answers, exclusions, slate, design answers), override
  (champion, from the admissible set, reason required), send back (a challenge to intake / explore /
  ideate / model), approve and
  reject (sign-off). Accepting a FAIL requires a reason (recorded risk acceptance).
- **Overrides** shape the effective config; the profile YAML is never edited.
- **Tracked questions (gaps)**: every unanswered design point and agent question for the sponsor;
  answer (re-runs the raising stage) or, for a data question, accept as an assumption. Use,
  horizon, default definition and segment are sponsor decisions and cannot be assumed.
- **Decision log**: every agent recommendation (with its backend), human decision and challenge, in
  the OKF bundle (`docs/decisions.md`).

## Agents (propose / dispose)
- **Eight stage agents** — Intake Analyst, Data Analyst, Design Lead, Modeler (+ guided-search role), Outcomes
  Analyst, Independent Validator, Model-Risk Analyst, Technical Writer — each with a role prompt
  (commercial-risk playbook), a Pydantic output contract, an independence-scoped context slice,
  engine checks, and a deterministic heuristic implementation.
- **Validate-and-retry**: a contract violation or failed engine check (unknown fact id, champion
  outside the admissible set, unfittable family, excluded target, unanswered challenge, typed metric
  in prose…) is fed back and retried; exhausted retries fail the step visibly.
- **Challenges**: human send-backs and high-severity validator findings reach the responsible agent,
  which must answer each; validator findings loop back automatically (bounded,
  `workflow.auto_challenge_loops`).
- **Independence**: the modeler's slice has no model/backtest/holdout facts (it chooses before the
  holdout is scored); the validator's slice never contains the modeler's rationale.
- **No LLM math**: facts by id; `{{fact:<id>}}` placeholders rendered by the engine.
- **Providers**: `heuristic` (offline), `replay` (recorded outputs), `claude_cli` (`claude -p`,
  isolated), `anthropic` (structured outputs, adaptive thinking, refusal fallbacks), OpenAI-compatible
  (OpenAI, xAI, OpenRouter, Ollama); `auto` resolution; per-call time limit; per-run spend budget.
- **Agent audit**: every attempt's prompt, context slice and raw output + `audit.jsonl` (status,
  backend, model, prompt/context hashes, duration, cost).
- **Agent-guided search** (opt-in `search.guided`): the modeler proposes experiments; the engine
  applies them target-hidden, scores them with leakage-safe CV, and keeps only winners, which enter
  the admissible set.
- **Target-hidden transform execution** (ADR-0002): agent-authored transforms run on a features-only
  view through an AST-whitelisted executor; kept transforms round-trip to IMPACT and the deployed
  scorer.
- **Two-tier reproducibility** (ADR-0003): the analysis re-derives offline with no LLM; the
  recommendations and decisions are recorded and replayable (`replay`).

## Workbench (`cognos ui`)
- Dash + Mantine app for model developers; light and dark themes.
- **Runs**: every run with status, what it is waiting on, backend, champion, spend; start a run from
  a synthetic demo preset or a project profile, as a new model or a model update (upload the
  intent document, background material and the existing model's artifacts; download the template),
  interactive or autonomous, with a chosen backend.
- **Run workspace**: one stage rail (status, verdict, and the seat a waiting gate belongs to), each
  stage's engine evidence beside its agent's recommendation (uncertainties, responses to
  challenges), the gate form for the seat that owns it, live activity; a seat switch in the page
  (model developer, independent reviewer, approver). Tabs for questions and challenges (the model
  developer answers; the four sponsor facts cannot be assumed), the decision log, a record computed
  on read of who did what, and the agent audit (inspect any call's output, context slice, prompt
  and system prompt).
- **Charts** on the validated reference palette: experiment ledger (kept vs discarded, champion
  ringed), coefficients by significance, calibration by score band, validation rubric, PD term
  structure; admissible set, sign checks, statistical battery, framework assessment, slate, SR 11-7 /
  NIST tables; rendered narrative, decision log, model card and white paper.
- **Out-of-date steps say why**: the decision, answer or validator loop that invalidated a stage is
  shown on the rail and the panel, and once it has re-run, what the re-run changed (champion,
  verdict, metrics, findings) against the result it replaced.
- **Compare runs**: pick two runs (or "Compare with previous run") to see what was decided
  differently and what it did to the results; `cognos compare A B` prints the same.
- **Export**: one zip of a run's documents, results, decisions and audit log with a hashed file
  manifest — never the data or the sealed holdout; `cognos export RUN`.
- Background execution with a 1-second poll; all state on disk, so refreshes and restarts lose
  nothing.

## CLI
- `cognos ui | run [--interactive] [--provider] [--kind --intent --support --prior --prior-run] |
  intent-template | demo | status | gate | answer | retry | run-stage | providers | agents | init |
  explain | report | list-runs | compare | export`.
- Terminal gate review for `run --interactive` (the intake interview, accept / send back),
  auto-accept on non-tty stdin (never past a BLOCK).

## Data exploration (`explore`)
- Schema/dtype/missingness profiling; numeric distribution summaries.
- Target-relationship correlations; **target-leakage detection** (near-perfect correlation suspects).
- Class-imbalance and constant-feature detection.
- Findings with severity; data profile artifact.

## Idea generation / design (`ideate`)
- **Data-structure assessment**: cross-sectional vs. vintage/panel vs. timeseries; event counts and
  **events-per-variable** (Peduzzi ~10 rule) for PD-style tasks, with a finding when support is thin.
- **Econometric framework assessment** (deterministic rules over the profile): structural Merton vs.
  reduced-form PD scorecard vs. discrete-time hazard vs. rating migration vs. ML challenger — each
  applicable/partial/**rejected with a stated reason**, which becomes the SR 11-7 "alternatives
  considered" evidence.
- **MD triangulation via the `design:` config block** (use case, horizon, default definition,
  segment, interpretability): every unanswered design point becomes an explicit **open question to
  the sponsor** (plus data-driven questions: leakage confirmation, panel-column hints, low EPV) —
  never a silent assumption. Answers collapse the questions on re-run.
- **Capability-unlock questions**: when the data supports a framework the config hasn't switched on,
  ideate says so — event-timing columns detected → "set `data.event_time_col` for the hazard
  families"; market observables detected → "enable the `structural:` block"; rating columns
  detected → "enable the `migration:` block for a transition matrix + loss forecast"; portfolio
  simulation on Basel-default ρ/LGD → "confirm provenance"; stress enabled without scenarios →
  "supply the set".
- Enumerates task-appropriate model families × feature strategies; **leakage suspects are excluded
  from the parsimonious feature strategy**; low EPV up-weights parsimonious specs.
- Ranks hypotheses by an interpretability/parsimony heuristic; under `interpretability: required`
  tree families carry `role: challenger` and interpretable families are searched first.
- Human-readable **`design_brief.md` artifact** per run (sponsor brief, data structure, framework
  table, open questions, ranked slate).
- **LLM-driven ideation emits executable feature-engineering transforms** (not just prose), authored and run target-hidden via the safe AST-whitelisted executor (ADR-0001, ADR-0002).
- **LLM design review** (additive): may add sponsor questions and extra candidate specs — only
  engine-fittable families are kept, and the ratchet decides on evidence.

## Modeling & statistical testing (`model`)
- **CASH search** (combined algorithm + hyperparameter selection) as one conditional space.
- **Ratchet search** (accept-if-better-else-discard) with an experiment ledger (`results.tsv`-style), cheap/simple candidates first, candidate + optional wall-clock budgets.
- **Leakage-safe cross-validation in search + a sealed/out-of-time holdout** — all preprocessing fit inside training folds only; selection is validated against a holdout the search cannot touch.
- **Frozen substrate**: sealed holdout + metric definitions the search cannot edit (anti-reward-hacking).
- Traditional statistical models (OLS, Ridge, Lasso, ElasticNet, Logit, regularized logit,
  **probit and cloglog binary GLM links** via a statsmodels-backed estimator) **with statsmodels
  inference** (coefficients, p-values, residuals) — *and* ML models (Random Forest, Gradient
  Boosting) with feature importances. Configurable for either.
- **Discrete-time hazard (survival) families** (`hazard_logit`, `hazard_cloglog`; Shumway 2001):
  obligor-period **panel expansion inside the estimator** (CV and holdout stay obligor-level —
  leakage-safe by construction), valid panel GLM inference including the baseline-hazard period
  effects, a **PD term structure** artifact, and a picklable `HazardScorer` that serves cumulative
  PD through the standard scorer bundle. Unlocked by `data.event_time_col`.
- **Merton structural engine** (`modeling/structural.py`, opt-in `structural:` block): a
  deterministic KMV fixed-point solver maps equity value/vol + debt face value → asset value/vol,
  **distance-to-default**, and structural PD. Hybrid mode feeds DD to the champion as an
  engineered feature (recomputed target-hidden at serve time, like LLM transforms); the pure
  structural PD is scored on the sealed holdout as a labelled challenger benchmark.
- **Rating-migration engine** (`modeling/migration.py`, opt-in `migration:` block; ADR-0009): a
  cohort-method transition matrix **fitted on the training partition only** (never the sealed
  holdout) with the standard agency-data statistics — **NR (withdrawn-rating) denominator
  adjustment**, Laplace smoothing, and **weighted-PAVA rank-ordering** of the default column
  (raw-vs-adjusted reported as a finding). Hybrid mode swaps the raw rating for the horizon
  cumulative PD it implies (`migration_pd`, recomputed target-hidden at serve time); the pure
  migration PD is a labelled sealed-holdout challenger benchmark; the payload carries the
  **matrix-power cumulative PD term structure**, regime-**conditional matrices**
  (`condition_col`), and a by-rating **expected-loss forecast** (EAD × LGD × cumPD) on the
  out-of-time book — the CECL/stress loss-forecast deliverable.
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
- **Opt-in Vasicek portfolio simulation** (`portfolio:` block, `modeling/simulate.py`): seeded
  one-factor Monte Carlo of the portfolio loss distribution — EL, UL, VaR/ES at the configured
  confidence, loss quantiles, MC standard error — plus the **closed-form Basel IRB capital** K as
  the analytic cross-check. Asset correlation defaults to the Basel IRB corporate formula ρ(PD);
  reported next to the calibration section (PDs in are model scores), never used for selection.
- **Opt-in macro-scenario stress testing** (`stress:` block): CCAR-flavoured scenarios shock the
  champion's covariates (`add`/`mul`/`set`), re-score **deterministically** through the deployed
  scorer, and report per-scenario mean-PD and expected-loss deltas; unknown shocked columns are
  reported, never silently ignored.
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
- **Two-tier reproducibility note** (ADR-0003): the analysis is reproducible offline with no LLM; the recommendations (`runs/<id>/agents/`) and decisions (decision log) are recorded for audit and replay.
- **Narrative + decision log** (v1.0): the Technical Writer's prose with engine-rendered `{{fact:…}}` numbers, and every agent recommendation, human decision and challenge.
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
- **Pluggable agent backends**: deterministic heuristic agents (offline, no API key — the substrate, not a fallback), `replay` for deterministic testing of the LLM path, `claude_cli`, `anthropic`, and OpenAI-compatible APIs; an unavailable backend is never a crash.
- **autoforge protocol** reimplemented: `name: value` stdout parsing, experiment ledger, generic ratchet loop.
- Typed config (Pydantic v2) with auto metric/direction resolution and YAML round-trip.
- CLI: see *CLI* above.
- **Demo tasks** including **`commercial`** (SR 11-7 + OOT outcomes analysis), **`cni`** — the flagship C&I showcase with an answered design brief, event timing (hazard families), Vasicek portfolio simulation, and two macro stress scenarios (ADR-0004, ADR-0008) — and **`migration`**: rating-migration loss forecasting on an S&P-style agency panel (transition matrix, term structure, EL forecast, regime conditioning; ADR-0009). The consumer fair-lending `credit` demo exercises an off-by-default module, not the primary path.
- Python API: `cognos.service` (runs, gates, answers, state), `cognos.engine.Engine`, `RunContext`, `CognosConfig`; `run_pipeline` / `Orchestrator` as a compatibility wrapper.
- Synthetic data generators (regression, classification, time series, commercial credit-risk with a vintage/date column for OOT calibration/PSI, a **C&I portfolio** with event timing + optional market observables + a deliberate post-outcome leak, and consumer credit-with-protected-attribute for the optional fair-lending module) for tests and demos.
- **Worked examples**: `examples/commercial_credit/` (private middle-market arc: leakage catch → MD triangulation → full pipeline), `examples/public_obligor_pd/` (public-obligor book: every econometric + structural + simulation capability in one run), and `examples/rating_migration_loss/` (the corporate loss-forecast engagement: internal-data insufficiency → external S&P-style agency data → migration matrix, term structure, EL forecast through all eight stages), all with captured real artifacts under `sample_output/`.
- **Recorded live example** `examples/live_commercial/`: a full run by live Claude agents (every call's input and raw output), replayable offline.
- **Developer safety hooks** (`.claude/hooks`): PreToolUse backstops against destroying or publishing run artifacts and leaking secrets while working on COGNOS itself.
- **Eval harness** for autonomous runs (assert verdicts/gates per case).
- Comprehensive test suite (unit + integration) and a runnable end-to-end example.
- MIT licensed; `uv`/`pip` installable; Python ≥ 3.11.
