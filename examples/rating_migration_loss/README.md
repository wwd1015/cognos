# Comprehensive showcase — rating-migration loss forecasting on external (S&P-style) agency data

The engagement every commercial/corporate bank recognizes: the sponsor needs a **loss-forecasting
model for the large-corporate book** (CECL provisioning + enterprise stress testing) built on the
industry-standard **rating-migration** framework — and the internal rating history is a few years
old with a couple dozen defaults, nowhere near enough to calibrate a through-the-cycle transition
matrix. The industry answer is to license a long external agency history (S&P CreditPro-style
obligor-year rating data) and let the matrix carry the long-run default experience, with obligor
fundamentals adding discrimination within grade.

One script walks that engagement through **all eight COGNOS stages** and shows the system reaching
each decision for auditable reasons — including the decision to reject the internal-only plan.

```bash
pip install -e .                                        # from the repo root
python examples/rating_migration_loss/run_demo.py       # add --save-sample to refresh sample_output/
```

Fully offline: matrix estimation, PAVA rank-ordering, matrix-power term structures, the loss
forecast, and the seeded simulations all live in the deterministic engine; no API key needed. The
"S&P data" is a synthetic stand-in — an obligor-year panel whose transition rates are calibrated to
the *shape* of the published S&P Global long-run (1981–2023) averages (sticky diagonals, near-zero
investment-grade default rates, a ~26% CCC default rate, an NR share that grows as ratings worsen)
— because the real CreditPro extract is licensed data that cannot ship in a repo.

## The three steps (with real captured numbers)

### 1. Internal data assessment — the engine itself rejects the internal-only plan

Explore + ideate run on the internal book alone (1,300 obligor-years, 2019–2023, **29 defaults**).
No human has to eyeball this; the design stage surfaces the problem and the fix as structured
findings and open questions:

> `[MEDIUM]` Events-per-variable ≈ **4.1** (events=29, features=7) is below the ~10 rule of thumb —
> coefficient estimates may be unstable.
> `[data-migration]` Rating column(s) detected (rating, next_rating) — enable the `migration:`
> config block to estimate a rating-transition matrix…

Five annual cohorts also cannot pin down a through-the-cycle matrix (no full credit cycle
observed). The sponsor's decision — *license the S&P long-run history and develop on the agency
universe mapped to the internal master scale* — is recorded in the design brief, which is exactly
the "alternatives considered / data sufficiency" evidence SR 11-7 documentation wants.

### 2. Framework unlock — ideate names the capability; the human flips the switch

The same two stages on the agency panel (19,350 obligor-years, 43 annual cohorts 1981–2023, 269
defaults) with nothing switched on. The framework assessment resolves all five frameworks — and
marks migration **available** rather than silently assuming it:

| Framework | Applicable | Role |
|---|---|---|
| Reduced-form PD (obligor scorecard / GLM) | true | primary |
| Discrete-time hazard (survival) | partial | candidate |
| Structural (Merton distance-to-default) | false | rejected — no market observables |
| **Rating-transition matrix (migration)** | **true** | **available** → candidate once enabled |
| ML challenger (trees/boosting) | true | challenger (interpretability: required) |

with the unlock question naming the exact switch: *enable the `migration:` config block
(`rating_col` + `next_rating_col`)*. The engine names the capability; switching it on is the
sponsor's call.

### 3. Full pipeline — matrix, term structure, benchmark, loss forecast, and the gates

All eight stages run with the `migration:` block enabled. What the model stage produces:

**The estimated matrix** (cohort method, estimated on the training partition only — the matrix is
*fitted*, so it never sees the sealed holdout):

- **NR adjustment**: 7.3% of transitions end in a withdrawn rating and are removed from the row
  denominator (the standard S&P/CreditPro treatment), reported rather than silently dropped;
- **PAVA rank-ordering**: raw AAA/AA default cells are zero-event small-sample artifacts; the
  weighted pool-adjacent-violators step monotonizes the default column and the run record names
  the adjusted grades (AAA, AA) — a LOW finding, not a silent fix;
- diagonally dominant everywhere (ratings are sticky), one-notch migration dominant.

**The cumulative PD term structure** (matrix powers, absorbing default — the CECL lifetime input):

| Rating | Y1 | Y2 | Y3 |
|---|---|---|---|
| AAA | 0.05% | 0.19% | 0.38% |
| BBB | 0.26% | 0.61% | 1.05% |
| BB | 0.97% | 2.46% | 4.37% |
| B | 5.46% | 12.19% | 18.84% |
| CCC | 38.16% | 56.03% | 65.08% |

**Champion vs. the pure-migration challenger benchmark** (sealed out-of-time holdout, 2015+):

- Pure migration PD (the matrix alone, `deployed: false`): holdout AUC **0.885** (Gini 0.770);
- Hybrid probit champion (`migration_pd` + leverage, coverage, margin, size, sector, macro):
  holdout AUC **0.893**, CV 0.938 — fundamentals add discrimination *within* grade, and the raw
  rating is swapped out of the champion's features (the PD is an exact function of the rating;
  keeping both would make the K-1 inference design perfectly collinear).

**The expected-loss forecast** (the purpose of the model — by rating band on the out-of-time book,
LGD 40%, 3-year horizon, EAD-weighted):

| | EAD share | cum PD | EL |
|---|---|---|---|
| BBB | 34.0% | 1.05% | 179m |
| BB | 11.3% | 4.37% | 247m |
| B | 5.7% | 18.84% | 533m |
| **Total** | | | **1.012% of EAD** through-the-cycle |

with the regime-conditioned matrices as the stress axis: **expansion 0.84% → recession 1.85%** of
EAD — the baseline/downturn split a CECL or CCAR reviewer asks for first.

**Then the rest of the pipeline does its independent jobs:**

- **backtest**: SR 11-7 outcomes analysis on the out-of-time sample — Gini 0.785, KS 0.702, PSI
  0.005 (stable); Vasicek portfolio loss distribution (EL 0.28%, VaR₀.₉₉₉ 2.25%, IRB K 2.10%);
  macro stress re-scored through the deployed scorer (baseline mean PD 0.70% → severely adverse
  1.39%);
- **validate** (gate): independent effective challenge passes with rubric 1.00 — no leakage (the
  declared outcome column `next_rating` never enters the features), CV↔holdout gap within bounds,
  significant coefficients;
- **comply**: the non-gating readiness report lists the human-only steps (sponsor sign-off,
  monitoring plan, agency-data licensing note belongs in the data-provenance section);
- **document**: the white paper renders the matrix, benchmark, and loss forecast into the OKF
  bundle + model card;
- **review** (gate): docs↔code drift 0.000 — the bundle's claims trace to real artifacts.

Final verdict: **WARN** — driven by explore's honest class-imbalance flag (1.4% default rate is
simply what corporate credit looks like), not by any gate. The run is approvable with that finding
documented, which is the correct posture for a credit model.

## Why this demonstrates "a reasonable decision"

- The **rejection of the internal-only plan is machine-generated evidence** (EPV finding + short
  history), not analyst folklore — and it is the documented justification for buying external data.
- The **matrix is treated as a fitted model**: train-only estimation, sealed-holdout benchmark,
  rank-order diagnostics with a transparent PAVA adjustment, NR treatment reported. Nothing about
  the agency data is taken on faith.
- The **champion is the regulated-industry answer**: a single interpretable GLM that *contains* the
  migration signal, benchmarked against the pure matrix so the incremental value of fundamentals
  is quantified — champion/challenger evidence, not vibes.
- The **loss forecast is the deliverable**, produced under through-the-cycle, expansion, and
  recession matrices, and cross-checked by a separately-derived Vasicek loss distribution and
  covariate-shock stress run.

## sample_output/

Real artifacts captured from an actual run (`--save-sample`): `design_brief.md` (the framework
assessment + open questions), `migration.json` (matrix, diagnostics, term structure, pd map, loss
forecast), `stress.json` (scenario deltas), `run_summary.txt` (the canonical verdict block).

## Where each capability lives

| Capability | Engine module | Config switch |
|---|---|---|
| Cohort matrix + NR adjustment + PAVA | `modeling/migration.py` | `migration:` block |
| Hybrid `migration_pd` feature + serve-time recompute | `modeling/migration.py::augment_frame` | `migration.rating_col` |
| Term structure (matrix powers) | `modeling/migration.py::cumulative_default_curve` | `migration.horizon_periods` |
| Regime-conditional matrices | `modeling/migration.py::fit_migration` | `migration.condition_col` |
| Expected-loss forecast | `modeling/migration.py::expected_loss_forecast` | `migration.lgd` / `ead_column` |
| Vasicek portfolio + IRB capital | `modeling/simulate.py` | `portfolio:` block |
| Macro-scenario stress | `modeling/simulate.py` | `stress:` block |

Also runnable as a one-liner on the same synthetic panel: `cognos demo --task migration`.

Design rationale: [ADR-0009](../../docs/adr/0009-rating-migration-external-agency-data.md).
