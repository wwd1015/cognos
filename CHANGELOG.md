# Changelog

All notable changes to COGNOS are documented here. Format loosely follows Keep a Changelog;
versioning is SemVer.

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
