# Community survey — similar work & borrowable ideas (2026-07-04)

Research-only deliverable: a survey of open-source/community work similar or adjacent to COGNOS,
and a ranked list of improvement suggestions. **Nothing here is implemented yet.**

Method: multi-agent web research (5 parallel search angles → source fetch → claim extraction →
partial adversarial verification; verification was cut short by usage limits, so remaining claims
were vetted against known project facts). Claims that could not be independently re-verified are
marked ⚠︎.

---

## 1. LLM-agent systems for data science / ML engineering

| Project | What it is | Key mechanism |
|---|---|---|
| **AIDE** ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138), [WecoAI/aideml](https://github.com/WecoAI/aideml)) | ML engineering as *tree search over solution scripts* | Nodes = candidate solutions; a **deterministic 3-operator policy** (Draft / Debug / Improve) decides what the LLM does next; a **summarization operator** distills the experiment tree (metrics, hyperparameters, debug hints) into LLM context; best MLE-bench scaffold (16.9% medal rate with o1-preview, ~4× next agent) |
| **MLE-bench** ([arXiv:2410.07095](https://arxiv.org/abs/2410.07095)) | OpenAI benchmark: 75 Kaggle competitions for end-to-end ML agents | Grades against human leaderboards; showed *scaffold choice* drives results as much as the base model |
| **MLE-STAR** ([arXiv:2506.15692](https://arxiv.org/abs/2506.15692)) | Google agent; beats AIDE on MLE-bench Lite (43.9% vs 25.8% same base model) ⚠︎ | Web-retrieval-grounded initial proposals (counters LLM familiarity bias); **ablation-guided targeted refinement** (find the highest-impact pipeline block, refine only it); an explicit **data-leakage checker** module — removing it raised validation accuracy while *test* accuracy collapsed (0.803→0.734) ⚠︎ |
| **AIRA** ([arXiv:2507.02554](https://arxiv.org/abs/2507.02554)) | Formalizes research agents as search policy × operator set ⚠︎ | Findings: **operator quality, not search-policy sophistication, is the bottleneck**; agents systematically overfit the validation proxy — final-node selection on non-privileged signal is worth 9–11.6 points ⚠︎ |
| **DS-Agent** ([arXiv:2402.17453](https://arxiv.org/abs/2402.17453), ICML 2024) | Case-based-reasoning agent | Retrieves + adapts expert Kaggle solutions instead of free-form proposing; persisted case store makes cheap deployment runs possible |
| **Agent Laboratory** ([arXiv:2501.04227](https://arxiv.org/abs/2501.04227)) | Staged research pipeline (lit review → experiment → report) | Human-in-the-loop co-pilot mode at each stage measurably improves output — external validation of COGNOS's gate/HITL design ⚠︎ |

**COGNOS take-aways.** COGNOS's propose/dispose split, frozen metric, and sealed holdout are
*independently validated* by this literature (AIDE uses the same sealed-holdout discipline; AIRA
quantifies the validation-overfitting failure mode the sealed holdout exists to stop). The gaps:
(a) COGNOS's guided search is a linear loop conditioned on the champion only — AIDE's
tree-with-summarized-ledger conditioning is strictly richer; (b) ideate's LLM review is un-grounded
— DS-Agent/MLE-STAR ground proposals in retrieved cases; (c) MLE-STAR's *code-level* leakage
checker complements COGNOS's *statistical* leakage detection.

## 2. AutoML frameworks

| Project | Borrowable mechanism |
|---|---|
| **AutoGluon-Tabular** ([arXiv:2003.06505](https://arxiv.org/pdf/2003.06505)) | Skips CASH search entirely: diverse defaulted families + multi-layer **stacking on out-of-fold predictions only** (anti-leakage discipline); trains cheap/reliable families first; estimates per-model train time and skips models that don't fit the remaining budget; checkpoints every model immediately |
| **FLAML** ([arXiv:1911.04706](https://arxiv.org/pdf/1911.04706), MLSys 2021) | **Estimated-cost-for-improvement (ECI)** budget allocation across learners (sample ∝ 1/ECI: cheap learners first, expensive ones only when justified); low-cost-init local search (CFO/FLOW2, provable cost bounds); explicit rule for holdout-vs-CV choice by dataset size. Notably: FLAML *deliberately* avoids ensembling for deployability of a single model — independent justification for COGNOS's ADR-0007 |
| **auto-sklearn 2.0** ([arXiv:2007.04074](https://arxiv.org/pdf/2007.04074)) | Meta-feature-free **portfolio warm-starting** (greedy, submodular guarantee) + successive-halving budget allocation; also a cautionary tale — its meta-learning is benchmark-contaminated, the same leakage class COGNOS's frozen substrate prevents |
| **AMLB benchmark** (Gijsbers et al., JMLR 2024) | Failed candidate fits are imputed with the **constant-predictor score** rather than dropped — crash-prone configurations can't look artificially good. 1h→4h budget quadrupling yields marginal gains: search-space design binds, not compute |

## 3. Credit-risk scorecard & interpretable-ML tooling

| Project | Borrowable capability |
|---|---|
| **OptBinning** ([repo](https://github.com/guillermo-navas-palencia/optbinning), Apache-2.0, active — v0.21.0 Oct 2025 ⚠︎; [arXiv:2001.08025](https://arxiv.org/abs/2001.08025)) | **Optimal WOE binning as exact CP/MIP optimization** (monotonicity constraints, binary/continuous/multiclass), full Scorecard class (binning + estimator + points scaling), counterfactuals. Deterministic solver output fits "engine disposes"; benchmarked 17× faster than scorecardpy with +12% IV ⚠︎ |
| **scorecardpy** ([repo](https://github.com/ShichenXie/scorecardpy), MIT) | Canonical scorecard workflow decomposition (IV filter → WOE binning → scaling → perf_eva/perf_psi); human-in-the-loop bin adjustment (`woebin_adj`) — a natural propose/dispose surface |
| **PiML** ([SelfExplainML/PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox), banking-focused; transitioning to successor "MoDeVa" ⚠︎) | Menu of 9 inherently interpretable families (GLM, GAM, tree, FIGS, **XGB1/XGB2 depth-limited boosting**, EBM, GAMI-Net, ReLU-DNN); model-agnostic diagnostics: **WeakSpot (residual slicing), Overfit-by-region, Reliability (split conformal), Robustness (noise perturbation), Resilience (OOD degradation)** |
| **InterpretML / EBM** ([interpretml](https://github.com/interpretml/interpret), MIT) | Glassbox GA²M model with accuracy ≈ XGBoost on tabular finance data; auto-detected, inspectable pairwise interactions |
| **Monotone-constrained GBM** (price-of-monotonicity study ⚠︎; XGBoost/LightGBM/CatBoost native) | Monotonicity on credit PD costs ~0–3% AUC (near zero at portfolio scale); a **four-principle protocol for constraint direction** (economic reasoning, literature, train-only evidence, encoding verification) — tailor-made for "LLM proposes signs, engine verifies train-only" |
| **creditR / R ecosystem practice** | Per-grade calibration tests (binomial, Hosmer–Lemeshow, chi-square) and Herfindahl–Hirschman concentration check for rating systems; PSI 0.1/0.25 thresholds (COGNOS already has PSI) |

## 4. Model-risk-management / validation / governance tooling

| Project | Borrowable pattern |
|---|---|
| **ValidMind Library** ([repo](https://github.com/validmind/validmind-library), AGPL/commercial) | **Documentation generated from structured test results** (tests populate doc templates — never free-form); findings with remediation tracking; three-lines-of-defense role separation; ⚠︎ their docs claim SR 11-7 is being superseded by joint interagency guidance ("SR 26-2") — worth verifying independently before touching COGNOS's `comply` regime list |
| **deepchecks** ([repo](https://github.com/deepchecks/deepchecks), **AGPL** — borrow ideas, don't vendor) | Named mechanical leakage checks: *Date Train-Test Leakage Overlap, Train Test Samples Mix, Index Leakage, Feature-Label Correlation (PPS > 0.7), FeatureLabelCorrelationChange (ΔPPS > 0.2)*; Checks→Suites→Conditions architecture with tri-state verdicts (mirrors COGNOS gates); their canonical motivating example is literally a credit "late payments" leak |
| **Evidently** ([repo](https://github.com/evidentlyai/evidently), Apache-2.0) | 100+ metrics incl. PSI; **auto-generates test thresholds from a reference dataset** (derive gates from the development sample, apply to OOT) |
| **Great Expectations** (Apache-2.0) | Auto-profiled expectation suites rendered as human-readable HTML "Data Docs" — an explore-stage evidence pattern |
| **Kapoor & Narayanan leakage taxonomy** (Patterns, 2023) | **8 leakage types in 3 classes** (no clean separation / illegitimate features / test-set distribution mismatch) — implementable as a mechanical checklist; **"model info sheets"**: a 21-question leakage attestation the `document` stage could embed |

## 5. Leakage & backtest-overfitting statistics

| Source | Borrowable statistic |
|---|---|
| **Bailey, Borwein, López de Prado & Zhu — PBO/CSCV** (J. Comp. Finance 2015) | COGNOS already computes PBO over the OOF library. The deeper point: **a sealed holdout alone cannot measure selection-induced overfitting — the trial count N must flow to validation.** COGNOS records `n_configs_for_deflation`; validate should *use* it |
| **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) | Closed-form multiplicity correction given N trials — the cheap parametric complement to CSCV; the same "expected max under null grows with N" logic applies to AUC/Gini champion selection |
| **mlfinlab CPCV** ([repo](https://github.com/hudson-and-thames/mlfinlab); ⚠︎ the mlfinlab.com domain is squatted — cite GitHub) | Combinatorial purged cross-validation: a *distribution* of backtest paths instead of one OOT split, with purging/embargo for temporal leakage |
| **pypbo** (AGPL, dormant) | Confirms the CSCV/PBO logic is ~200 lines — reimplement, never vendor |

---

## Ranked suggestions (research only — not implemented)

Ordered by impact-per-effort for a regulated commercial-PD system. "Stage" = where it slots.

| # | Suggestion | Stage / module | Impact | Effort |
|---|---|---|---|---|
| 1 | **Scorecard family via optimal WOE binning** — add a `scorecard` model family: monotonic optimal binning → logistic on WOE → points scaling (adopt OptBinning, Apache-2.0, or reimplement CP-based binning). This is *the* standard deliverable in commercial credit and COGNOS can't produce one today. Ideate's `reduced_form_pd` framework maps to it directly; `woebin_adj`-style bin edits become a propose/dispose surface. | `modeling/fit.py`, `modeling/transforms.py`, `stages/ideate.py` | High | Medium |
| 2 | **Monotone-constrained GBM + EBM as interpretable challengers** — add `gradient_boosting_monotone` (native XGBoost/LightGBM constraints) and `ebm` families; implement the four-principle constraint-direction protocol: **LLM proposes signs from economic priors, engine verifies against train-only evidence** (extends the existing sign-check idea in the model skill pack into enforced code). Cost is ~0–3% AUC for full monotonicity. | `modeling/fit.py::_estimator`, `modeling/search.py::_hp_grid`, ideate + model skill packs | High | Medium |
| 3 | **Mechanical leakage battery in validate** — deterministic re-derivations per the deepchecks taxonomy + Kapoor–Narayanan checklist: date train/holdout overlap, duplicate rows across the split, index/ID leakage, single-feature PPS threshold, train-vs-holdout PPS change. Complements the current correlation heuristic; BLOCK stays reserved for confirmed leakage. | `stages/validate.py` (+ `stages/stat_tests.py`) | High | Low–Med |
| 4 | **Multiplicity-aware champion verdict** — validate should consume the trial count the model stage already records (`n_configs_for_deflation`): expected-max-metric-under-null given N trials (DSR-style logic applied to AUC/Gini), flag when the champion's holdout edge is within the selection-noise band. Closes the documented "sealed holdout ignores N" gap. | `stages/validate.py`, `modeling/backtest_stats.py` | High | Low |
| 5 | **Per-grade calibration + concentration tests** — binomial / Hosmer–Lemeshow per score band and an HHI rating-concentration check in the outcomes analysis (current calibration is aggregate). Standard IRB/SR 11-7 validation expectations. | `modeling/credit_metrics.py`, `stages/backtest.py` | Med–High | Low |
| 6 | **AIDE-style ledger conditioning for guided search** — feed the LLM a *summarized experiment tree* (per-candidate metric, config, failure hints — the ledger already exists as TSV) instead of only champion+profile; keep the deterministic draft/debug/improve policy in the engine. AIRA caveat: invest in operator quality (better proposal prompts/actions) before fancier search policies. | `modeling/guided.py` | Medium | Medium |
| 7 | **Auto-derived gate thresholds** (Evidently pattern) — derive validate/backtest thresholds from the development sample (reference-based), rather than global constants, with the frozen substrate rule unchanged. | `stages/validate.py`, `stages/backtest.py` | Medium | Low |
| 8 | **Honest failure accounting in the ratchet** (AMLB pattern) — a candidate whose fit crashes is scored as the constant predictor in the ledger, not silently dropped, so fragile configs can't win by being skipped. | `modeling/search.py` | Medium | Trivial |
| 9 | **CPCV as an opt-in backtest scheme** — combinatorial purged CV for panel data: a distribution of OOT paths (with purging/embargo) instead of a single split; reuses the existing PBO plumbing. | `stages/backtest.py`, `modeling/backtest_stats.py` | Medium | Medium |
| 10 | **Leakage attestation in the white paper** — a model-info-sheet-style section (per the 8-type taxonomy) where each leakage class gets an explicit argument/evidence pointer; `review` traces it like any other claim. | `stages/document.py`, OKF bundle | Medium | Low |
| 11 | **Budget-aware family ordering** — ECI/reliability-order ideas: cheapest families first with estimated-time skip rule under `time_budget_s` (extends the current interpretability-first ordering). | `modeling/search.py` | Low–Med | Low |
| 12 | **Case-bank grounded ideation** (DS-Agent/MLE-STAR pattern) — persist design briefs + outcomes across runs as a retrieval corpus for ideate's LLM review; grounds proposals, counters familiarity bias. Needs run history to be useful — defer. | `stages/ideate.py`, new store | Medium | High |

**Follow-up worth checking (not a code change):** the ⚠︎ claim that SR 11-7 has successor
interagency guidance ("SR 26-2", per ValidMind's docs). If confirmed, `comply`'s regime list and
skill pack should reference both.

**License discipline:** OptBinning, Evidently, InterpretML, Great Expectations are Apache/MIT
(usable as dependencies). deepchecks, pypbo, ValidMind are AGPL — borrow *ideas*, never vendor code.

**What the research validates about COGNOS as-is:** sealed holdout + frozen metric (AIDE, AIRA),
single-champion-no-silent-ensemble (FLAML's deployability argument), stage-level human gates
(Agent Laboratory), PBO-over-the-real-search-library (Bailey et al.), PSI thresholds and
interpretable-first families (industry scorecard practice).
