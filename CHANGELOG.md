# Changelog

All notable changes to COGNOS are documented here. Format loosely follows Keep a Changelog;
versioning is SemVer.

## [Unreleased]

### Changed
- **Workbench redesign.** The workbench adopts the "Ledger" design system used by IRIS-D: paper
  and warm-black themes, serif display type, mono numerals, one oxblood accent, ruled panels
  instead of boxed cards, a masthead, figure strips, a numbered stage index and a plain activity
  feed. Still pure Dash + Mantine; no behaviour changed.

### Added
- **Tools at every stage** ([ADR-0013](docs/adr/0013-stage-tools-every-agent-can-use-a-plugin.md)).
  A plugin tool declares the stages it serves and the inputs it needs; every stage's agent can
  request the tools registered for its stage before it recommends. The engine runs them, keeps
  the results (`tool_runs`, `stages/<stage>/analyses/`), exposes them as facts
  (`tools.<stage>.<id>.<name>`) and turns failed checks into findings. Guidance stays in COGNOS
  (`prompts/_tools.md`); a plugin adds tooling only. Example:
  `examples/plugins/validation_tools.py`.
- **Custom tests at validation.** A `high` failed check from a validation tool fails the
  validation. A tool can never block a run.
- **IMPACT placeholder.** `impact_test_suite` is registered for validation and reported as not
  run until IMPACT's test interface is available; a plugin replaces it by registering the same
  name.
- Every stage page shows the tools its agent ran and the ones registered but not run; the white
  paper lists both; `cognos plugins` shows each tool's stages and availability.
- **Data sources** ([ADR-0012](docs/adr/0012-data-sources-plugins-and-agent-written-analysis.md)).
  `data.source` reads an uploaded or local file (CSV, Parquet, Excel, JSON lines), SQLite, or
  Snowflake (`pip install 'cognos[snowflake]'`; credentials from `SNOWFLAKE_*` environment
  variables, never stored). The run keeps a snapshot with its provenance and hash.
- **The Data Analyst proposes the dependent variable.** `data.target` and `task` may be left
  empty; the analyst chooses the target against the business intent and lists the features worth
  considering; the developer confirms or changes the target at the data gate.
- **Plugins.** `register(registry)` modules add analysis tools and data sources (entry-point
  group `cognos.plugins`, the profile's `plugins:` list, `COGNOS_PLUGINS`). `cognos plugins`
  lists what is installed. Example: `examples/plugins/credit_tools.py`.
- **Analysis on request.** During explore the analyst requests tools (six built in) or writes
  Python; the engine runs them in rounds and keeps each result with its chart.
- **Agent-written code as a reviewed artifact.** Scripts run in a restricted subprocess, are
  saved with a provenance header and hash, shown at the data gate (the decision records them),
  re-run at validation, given to the Independent Validator, printed in the white paper and
  exported.
- **Visuals.** An analysis gallery, the dependent-variable card, feature candidates and
  data-source provenance in the explore panel; data upload and Snowflake inputs in the new-run
  drawer; `cognos run --data FILE` starts from a data file alone.
- **Development modes** ([ADR-0011](docs/adr/0011-development-modes-and-intake.md)). A run is a
  **new model development** or a **model update** (`engagement.kind`; `cognos run --kind`, the
  switch at the top of the workbench's new-run drawer). A new development starts from the sponsor's
  business intent document and background material. An update starts from the existing model's
  artifacts (white paper, code, validation and monitoring reports, or an earlier COGNOS run via
  `--prior-run`) and an update request.
- **The intent template.** `cognos intent-template [--kind update] [-o FILE]`, the "Download the
  template" button, and `docs/templates/`. One document for both modes; the update request adds
  the change-request sections. Fill it in before the run.
- **Intake stage, Intake Analyst, `gate_intent`.** A new first stage reads the documents (`.md`,
  `.txt`, `.docx`, code, notebooks, `.pdf` with `pypdf`), fills an engagement brief and interviews
  the sponsor where the intent is unclear: a question is a gap, an answer re-runs intake, and the
  gate re-opens until nothing blocks. A "stated" position must quote the documents or the engine
  rejects it. Confirming the brief writes what the document states into the effective config.
- **Model update support downstream.** Typed change items and an update scope; the existing
  model's family read from its artifacts and kept first on the slate; `prior.*` facts from an
  earlier COGNOS run, scoped away from the design lead and the modeler; a validator check that the
  request was delivered; a model change record in the white paper.
- **Uploads.** The new-run drawer takes the intent document, background documents and the
  existing model's artifacts; `service.save_upload` and `service.create_run(engagement=...)` do the
  same from code. Documents are copied into `runs/<id>/inputs/`.
- Every agent's context carries the confirmed brief (`project.engagement`).

### Changed
- Nine stages and six review gates. An interactive run pauses first at `gate_intent`; scripts that
  expected `gate_data` first need to decide it (`accept` with a reason while questions are open).
- The four core design questions are raised by intake (same ids, `design-<field>`); after the
  intent is confirmed their answers re-enter at `ideate` as before.
- Synthetic demos ship an intent document generated from the preset's design brief.
- Runs recorded before this change load with `intake` and `gate_intent` skipped.
- **Why a step is out of date.** Every invalidation names its cause (the gate decision, answered
  question or validator loop) in `StepState.rerun_reason`; the reason stays on the step through
  the re-run. The rail and the stage panel show it.
- **What a re-run changed.** Before a stage runs again the engine keeps its last result as
  `stages/<stage>/result.prev.json`; `service.step_changes` and the stage panel compare the two.
- **Compare runs** (`cognos compare A B`, `/compare/<a>/<b>` in the workbench, "Compare with
  previous run" on a run): what was decided differently (gate actions, overrides, config) and what
  it did to the results (metrics with the difference, verdicts, findings). "Improved" / "worse" is
  claimed only where the metric has a direction (`compare.py`).
- **Export a run** (`cognos export RUN`, the Export button): one zip of the documents, stage
  results and small evidence tables, state (decisions, challenges, questions), config and the agent
  audit log, with `EXPORT.json` listing every file and its SHA-256. The data directory (the sealed
  holdout), fitted models and binary caches are never included; agent prompts and raw outputs only
  with `--with-agent-io`. Sealed packages travel with the export.
- **Seats and a sealed package.** Each gate belongs to one seat: the model developer (data, design,
  champion), the independent reviewer (validation), or the approver (sign-off). The workbench shows
  one map and switches seat; the engine refuses the wrong seat. Use, horizon, default definition
  and segment cannot be waived. `approve` writes `packages/vN.json` once, bound to a digest of
  recorded facts. A later edit supersedes that package and does not rewrite the file. Autonomous
  acceptance is express preparation (`seat=express`), not a signature, and does not seal a package.
  Approval is not deployment. Who did what — person, agent, or the engine — is computed on read.

## [1.0.0] — 2026-09-29

**Agents recommend, humans decide, the engine disposes.** v1.0 adopts the design proven in Cyber
Credit Officer — engine-run agents with contracts, human review gates, a challenger loop, tracked
questions, and a full audit trail — on top of the unchanged econometric engine, and adds a Dash +
Mantine workbench for model developers. Design in ADR-0010.

### Added
- **Workflow engine** (`engine/`): a step graph of the eight stages and five human review gates
  (`gate_data`, `gate_design`, `gate_champion`, `gate_validation`, `gate_signoff`) with stale
  propagation, `RunState` on disk (`state.json`: steps, gate decisions, challenges, gaps, overrides,
  loops, spend), an activity feed (`events.jsonl`), synchronous and background drivers, retry, and
  re-opening a decided gate. Interactive runs pause at gates; autonomous runs auto-accept (recorded).
- **Gate decisions**: accept / edit / override / send back / approve / reject. Exclusions, slate
  edits, design answers and champion overrides become *overrides* on the effective config — the
  profile YAML is never edited. A BLOCK can be sent back or rejected, never accepted.
- **Seven stage agents** (`agents/`): Data Analyst, Design Lead, Modeler (+ guided-search role),
  Outcomes Analyst, Independent Validator, Model-Risk Analyst, Technical Writer — each with a role
  prompt (the former `cognos-*` playbooks), a Pydantic contract, an independence-scoped context slice,
  engine checks with retry-on-error, and a deterministic heuristic implementation.
- **Challenges**: human send-backs and high-severity validator findings are routed to the stage's
  agent, which must answer each one; validator findings loop back automatically at most
  `workflow.auto_challenge_loops` (default 2) times.
- **Gaps**: open design/data questions are tracked; answering one fills the design brief and
  re-runs the stage that raised it, or it can be accepted as an assumption.
- **Admissible set + blind champion choice**: the modeler chooses among candidates within one CV
  standard error of the best, before the sealed holdout is scored; holdout evaluations are counted
  and re-selection is flagged by the validator. The search is cached by an input fingerprint, so an
  override or a send-back re-finalizes without re-searching.
- **Facts and no LLM math**: agents cite fact ids; the writer's prose uses `{{fact:<id>}}`
  placeholders the engine renders; unknown ids and typed metric values are rejected.
- **Providers** (`agents/providers.yaml`): `heuristic`, `replay` (recorded outputs, heuristic
  fallback), `claude_cli` (`claude -p`, isolated: empty cwd, no settings/tools/MCP, prompt on stdin),
  `anthropic` (structured outputs, adaptive thinking, server-side refusal fallbacks), and
  OpenAI-compatible APIs (OpenAI, xAI, OpenRouter, Ollama). `auto` picks the first available and
  falls back to heuristic. Per-call time limit and per-run spend budget.
- **Agent audit**: every attempt's prompt, context slice and raw output under `runs/<id>/agents/`,
  with `audit.jsonl` (status, backend, model, hashes, duration, cost).
- **Decision log** in the OKF bundle (`docs/decisions.md`) and an agent-drafted **narrative**
  (`docs/narrative.md`).
- **Service layer** (`service.py`) — the single boundary for the CLI and UI.
- **Workbench** (`cognos ui`, `[ui]` extra): runs list + new-run drawer; run workspace with stage
  rail, evidence beside recommendation, gate forms, live activity, questions & challenges, agent
  audit with raw I/O; theme-aware charts (experiment ledger, coefficients by significance,
  calibration, rubric, PD term structure); light and dark.
- **CLI**: `ui`, `status`, `gate`, `answer`, `retry`, `providers`; `run --interactive` reviews
  gates in the terminal; `--provider` on `run`, `demo`, `run-stage`.
- `agents:` and `workflow:` profile sections (legacy `brain:` blocks migrate automatically).

### Changed
- Stages obtain judgment only through `ctx.recommend()`; `RunContext.config` is the effective
  config and `ctx.profile()` the explore profile net of human exclusions.
- `Orchestrator` / `run_pipeline` are a compatibility wrapper over the engine; a legacy
  `gate_handler` can no longer approve a BLOCK.
- Guided search routes proposals through the agent runner (validated, audited).
- sklearn ≥ 1.8: `l1_ratio` replaces the deprecated `penalty` argument.

### Removed
- `brains/` (`HeuristicBrain`, `LLMBrain`, `ScriptedBrain`) — replaced by providers and `replay`.
- `.claude/agents`, `.claude/commands/cognos-run.md`, `.claude/skills/cognos-*` — the agents run
  in the engine; the playbooks are their prompts. The `.claude/hooks` safety backstops remain.

### Fixed
- Windows: every file read/write is explicitly UTF-8; `pyarrow` is a declared dependency.

## [0.5.0] — 2026-07-12

The rating-migration release: the loss-forecasting framework corporate banks actually use when the
internal default history is too short — estimate a transition matrix on a long external agency
(S&P CreditPro-style) history and let it carry the long-run default experience. Design decisions in
ADR-0009; the full engagement is demonstrated end-to-end in `examples/rating_migration_loss/`.

### Added
- **Rating-migration engine** (`modeling/migration.py`, opt-in `migration:` config block): a
  cohort-method one-period transition matrix **fitted on the training partition only** (the matrix
  is a fitted model, unlike the deterministic Merton solver, so it must never see the sealed
  holdout), with the standard agency-data statistics — **NR (withdrawn-rating) denominator
  adjustment**, Laplace smoothing, and **weighted-PAVA rank-ordering** of the default column with
  raw-vs-adjusted reported as a finding. Hybrid mode swaps the raw rating for the horizon
  cumulative PD it implies (`migration_pd`), recomputed target-hidden at serve time on both
  `FittedModel` and `ScorerBundle`; the pure-migration PD is scored on the sealed out-of-time
  holdout as a labelled challenger benchmark (`migration.benchmark`, `deployed: false`).
- **Cumulative PD term structure via matrix powers** (absorbing default) and a by-rating
  **expected-loss forecast** (EAD × LGD × cumPD at the horizon) on the out-of-time book, under the
  pooled (through-the-cycle) matrix and per-regime **conditional matrices**
  (`migration.condition_col`) — the baseline/downturn split CECL and stress reviewers ask for.
  Both are reports; neither feeds champion selection.
- **`next_rating_col` is outcome data**: excluded from model features exactly like the target and
  `event_time_col` (datautil).
- **Ideate is migration-aware**: the transition-matrix framework entry is config-aware
  (rejected → available → candidate with stated reasons) and a `data-migration` unlock question
  fires when rating columns exist unconfigured.
- **Synthetic S&P-style agency panel** (`synth.make_rating_migration_dataset`): obligor-year
  rating histories calibrated to the shape of the published S&P long-run (1981–2023) averages,
  with regime-tilted recessions, within-grade fundamental signal, NR withdrawals, and EAD; a
  `book="bank"` short-history variant reproduces the "internal data is too short" problem.
- **`cognos demo --task migration`** preset and the `migration:` block in the `cognos init`
  template.
- **Comprehensive worked example** `examples/rating_migration_loss/`: internal-data insufficiency
  (machine-generated EPV/short-history evidence) → framework unlock on the agency panel → full
  eight-stage run with the matrix, term structure, champion-vs-matrix benchmark, EL forecast,
  Vasicek portfolio, macro stress, and both gates — with captured artifacts in `sample_output/`.
- **ADR-0009** — rating migration is a fitted engine layer; external agency data is a designed
  decision.

## [0.4.0] — 2026-07-04

The econometric & structural release: traditional regression depth (GLM links, discrete-time
hazard with PD term structures) and simulation-based methods near the structural model (Merton
distance-to-default, Vasicek portfolio losses, macro stress). Direction chosen from the
community-survey research (`docs/research/2026-07-04-community-survey.md`) with the sponsor
prioritizing traditional/structural methods over ML-flavored items.

### Added
- **Binary GLM links `probit` and `cloglog`** via a statsmodels-backed sklearn estimator
  (`SMBinaryGLM`) with full statsmodels inference on the K-1 design; both join the default
  classification slate. cloglog is the grouped-time proportional-hazards link.
- **Discrete-time hazard families** `hazard_logit` / `hazard_cloglog` (`modeling/hazard.py`,
  Shumway 2001): obligor-period **panel expansion inside the estimator** so CV/holdout splits stay
  obligor-level (leakage-safe by construction); valid panel GLM inference including baseline-hazard
  period effects; **PD term structure** in the model payload; picklable `HazardScorer` compatible
  with the existing scorer-bundle/IMPACT serving contract. Unlocked by the new
  `data.event_time_col` (+ `data.horizon_periods`); ideate widens the default slate and flips the
  survival framework to fully applicable.
- **Merton structural engine** (`modeling/structural.py`, opt-in `structural:` config block): a
  deterministic KMV fixed-point solver (equity value/vol + debt face → asset value/vol,
  **distance-to-default**, structural PD). Hybrid mode feeds `merton_dd` to the champion as an
  engineered feature — recomputed target-hidden at serve time on both `FittedModel` and
  `ScorerBundle` — and the pure structural PD is scored on the sealed holdout as a labelled
  challenger benchmark. Ideate marks the framework "available" with an unlock question when market
  observables exist but the block is off.
- **Vasicek one-factor portfolio simulation** (`modeling/simulate.py`, opt-in `portfolio:` block):
  seeded Monte Carlo loss distribution (EL/UL/VaR/ES, quantiles, MC standard error) plus
  closed-form **Basel IRB capital** with the ρ(PD) corporate correlation formula; reported by the
  backtest stage, never used for champion selection.
- **Macro-scenario stress testing** (opt-in `stress:` block): scenarios shock covariates
  (`add`/`mul`/`set`), re-score deterministically through the deployed scorer, and report mean-PD /
  expected-loss deltas; unknown shocked columns are surfaced as findings.
- `cognos demo --task cni` now exercises event timing, portfolio simulation, and two stress
  scenarios; the worked example (`examples/commercial_credit/`) shows the full econometric +
  simulation arc with refreshed captured artifacts; `synth.make_cni_portfolio_dataset` gains
  `default_quarter` event timing and opt-in market observables (`include_market=True`).
- New design questions from ideate: hazard/structural capability unlocks, asset-correlation & LGD
  provenance, missing stress scenario sets.
- **ADR-0008** documenting the three design decisions (hazard panel-expansion inside the estimator;
  structural model as a solver with hybrid-default use; simulation reported, never selected on);
  CONTEXT.md glossary entries for the design brief, framework assessment, hazard family, structural
  engine, portfolio simulation, and stress scenarios.
- **Comprehensive showcase** `examples/public_obligor_pd/`: every econometric + structural +
  simulation capability in one run on a public-obligor book (framework unlocks → probit/cloglog
  inference → hazard term structure → Merton hybrid vs pure-structural benchmark → full pipeline
  with Vasicek portfolio and macro stress), with captured real artifacts under `sample_output/`.

## [0.3.0] — 2026-07-04

Commercial-risk domain capabilities: ideate becomes a true *design* stage, every agent gains a
domain playbook, and a worked C&I example ships with the repo.

### Added
- **Ideate as the design stage**: deterministic **data-structure assessment** (cross-sectional vs
  vintage/panel, event counts, events-per-variable with the Peduzzi ~10 rule), an **econometric
  framework assessment** — structural Merton vs reduced-form PD vs discrete-time hazard vs rating
  migration vs ML challenger, each applicable/partial/rejected **with stated reasons** (the SR 11-7
  "alternatives considered") — and a per-run human-readable **`design_brief.md`** artifact.
- **MD triangulation via the new `design:` config block** (`use_case`, `horizon`,
  `default_definition`, `segment`, `interpretability`, `notes`): unanswered design points become
  explicit **open questions to the model sponsor** (plus data-driven ones: leakage confirmation,
  panel-column hints, low EPV) instead of silent assumptions; answers collapse the questions.
- **Interpretability policy**: under `interpretability: required` tree families carry
  `role: challenger` and interpretable families are searched first; `flexible` lets them compete.
  Leakage suspects are excluded from the parsimonious feature strategy (`clean_top_features`).
- **LLM design review** (additive, ADR-0001): the brain may add sponsor questions and extra
  candidate specs; only engine-fittable families are kept and the ratchet decides on evidence.
- **Per-agent commercial-risk skill packs** (`.claude/skills/cognos-<stage>/SKILL.md`): domain
  playbooks for all eight agents (post-outcome leakage patterns, framework selection, economic
  sign checks, Gini/KS/PSI norms, effective-challenge checklist, readiness mapping, whitepaper
  skeleton, drift traps); each stage agent loads its playbook before judging engine output.
- **Realistic C&I portfolio generator** (`synth.make_cni_portfolio_dataset`): quarterly vintages
  spanning the 2020 stress period, obligor ratios, facility terms, macro at origination, and a
  deliberate post-outcome leak (`dpd_at_outcome`); `cognos demo --task cni` preset.
- **Worked example** `examples/commercial_credit/`: the leakage-catch → MD-triangulation →
  answered-design → full-pipeline arc, with **real captured artifacts** committed under
  `sample_output/` (design brief, open questions, run summary).

### Changed
- `cognos init` template now includes the `design:` block.
- Agent specs (`.claude/agents/*.md`) load their skill packs; `comply` wording aligned with
  ADR-0004 (fair-lending scans are out-of-scope-by-choice for commercial profiles, not
  inapplicable law).

## [0.2.0] — 2026-06-24

Design-review outcomes (see `CONTEXT.md` and `docs/adr/0001-0007`). COGNOS is now explicitly a
two-layer system — an LLM reasoning layer that *proposes* and a deterministic engine that *disposes*
— focused on commercial model development.

### Added
- **Reasoning-driven core** (ADR-0001): LLM-driven ideation that emits engine-validated feature
  transforms, and an opt-in **LLM-guided search** (`search.guided`) where the LLM proposes the next
  experiment from the ledger and the engine keeps it only if it beats the incumbent on the frozen
  metric.
- **Safe, target-hidden transforms** (`modeling/transforms.py`, ADR-0002): AST-whitelisted expression
  executor over a features-only view; transforms cannot reference the target, persist on the champion
  + scorer, re-apply at serving, and round-trip to IMPACT derived fields.
- **Two-tier reproducibility** (ADR-0003): reasoning transcript recorded to
  `runs/<id>/reasoning/transcript.jsonl`; `ScriptedBrain` deterministic test double.
- **Credit-risk outcomes analysis** (`modeling/credit_metrics.py`, ADR-0005): Gini/KS, calibration
  (expected-vs-observed + ECE), PSI, on an out-of-time sample — now the default meaning of "backtest".
- Opt-in **GLM/econometric families** (poisson, gamma, tweedie); a synthetic **commercial** demo
  dataset with a vintage column (`cognos demo --task commercial`).

### Changed
- **Valid statistical inference** (Q6): coefficients/p-values now come from a full-rank K-1 inference
  design, decoupled from the all-K prediction pipeline (fixes the dummy-variable trap).
- **Compliance is a non-gating model-risk readiness report** (ADR-0006), never a verdict and never
  rubber-stamped; the only gates are now `validate` and `review`.
- **Single interpretable champion** deployed (ADR-0007); the ensemble is an opt-in, labelled challenger
  benchmark, never silently shipped.
- **Backtesting**: PBO / Deflated Sharpe demoted to an opt-in trading mode (`backtest.returns_column`).
- **Scope** (ADR-0004): primary domain is commercial model development; consumer fair lending is an
  optional, off-by-default module.

## [0.1.0] — 2026-06-22

Initial release: the full COGNOS system.

### Added
- **Eight-stage pipeline**: `explore`, `ideate`, `model`, `backtest`, `validate`, `comply`,
  `document`, `review`, each a single-responsibility agent with a uniform `run(ctx) -> StageResult`
  interface.
- **Mechanical orchestrator** with two operating modes (autonomous and interactive/stage-by-stage),
  gate handling, per-stage checkpointing, and resume-from-failure.
- **Pluggable brain**: deterministic `HeuristicBrain` (default, offline) and optional Claude
  `LLMBrain` that degrades gracefully.
- **Modeling core**: CASH ratchet search, leakage-safe nested CV, frozen substrate (sealed holdout +
  metrics), statsmodels inference for linear families, ML fitters, Caruana ensembling.
- **Statistical diagnostic battery** (heteroskedasticity, autocorrelation, normality, linearity,
  multicollinearity, stationarity).
- **IMPACT integration**: model embedded as an `EntityPipeline` derived field, with a built-in
  fallback scorer.
- **Backtest analytics**: Probability of Backtest Overfitting (CSCV) and Deflated Sharpe Ratio.
- **Independent validation** (SR 11-7 effective challenge) with a five-axis rubric.
- **Compliance**: SR 11-7 × NIST AI RMF × trustworthy-AI, ECOA/Reg B fair-lending disparate-impact
  scan, reason codes, model inventory, EU AI Act Annex IV flagging.
- **Documentation**: white paper as an OKF v0.1 bundle + Google Model Card + EU Annex IV, with
  docs↔code link anchors.
- **Consistency review**: AST-based docs↔code drift detection over the OKF graph.
- **CLI** (`run`, `run-stage`, `demo`, `init`, `explain`, `report`, `list-runs`, `agents`) and Python
  API (`run_pipeline`, `Orchestrator`).
- **Claude-Code-native agent layer** (`.claude/`), per-project profiles (`projects/`), and an eval
  harness (`evals/`).
- Synthetic data generators, a runnable end-to-end example, and a unit + integration test suite.
