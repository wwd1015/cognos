# ADR-0009 — Rating migration is a fitted engine layer; external agency data is a designed decision

Date: 2026-07-12 (v0.5.0)
Status: accepted

## Context

The canonical corporate-bank loss-forecasting engagement is rating-migration based: estimate a
rating-transition matrix, power it to the horizon, and read cumulative PDs / expected loss off the
default column (CreditMetrics 1997; CECL and stress-testing practice). Two facts shape the design:

1. **The matrix is a fitted model, not a solver.** Unlike Merton DD (ADR-0008 §2), transition
   probabilities are estimated from data, so they can leak into the sealed holdout if computed on
   the full frame, and they carry small-sample pathologies (zero-default investment-grade rows,
   withdrawn ratings) that need explicit statistical treatment.
2. **The interesting decision is usually about data, not algorithms.** Internal rating histories
   are typically too short (few cohorts, few defaults) to calibrate a through-the-cycle matrix;
   the industry answer is licensed external agency data (S&P CreditPro-style obligor-year panels).
   That decision should be *produced by the system as evidence*, not assumed.

## Decisions

### 1. The migration layer mirrors the structural layer, but fits on the training partition only

`modeling/migration.py` + the opt-in `migration:` config block follow the ADR-0008 hybrid pattern —
with one deliberate difference: the model stage estimates the matrix **after the holdout seal, on
`train_df` only**, then augments both partitions. The sealed holdout has never touched the matrix's
estimation sample, so the pure-migration challenger benchmark scored on it is honest.

- **Hybrid mode (default)**: the horizon cumulative PD implied by the obligor's current rating
  (`migration_pd`) feeds the reduced-form champion. The raw rating column is **swapped out** of the
  champion's features: `migration_pd` is an exact function of the rating, and keeping both would
  make the K-1 inference design perfectly collinear (dummy trap by another name).
- **Pure-migration benchmark**: `migration_pd` alone is scored on the sealed holdout as a labelled
  challenger benchmark (`migration.benchmark`, `deployed: false`) — quantifying what the agency
  matrix buys before fundamentals, ADR-0007 unchanged.
- **Serve-time recompute**: the fitted artifact is a plain JSON-serializable dict (spec + pd map +
  fill value) carried on `FittedModel`/`ScorerBundle`, so serving recomputes `migration_pd` from
  the raw rating exactly as in training — the same contract as transforms and Merton DD.

### 2. Agency-data statistics get first-class, reported treatment

- **NR adjustment**: transitions into withdrawn states are removed from the row denominator (the
  standard S&P/CreditPro treatment) and the withdrawal share is reported; a high share raises a
  finding rather than silently biasing the matrix.
- **Laplace smoothing + weighted PAVA rank-ordering** (`monotone_pd`, default on): zero-default
  high-grade rows make raw PD estimates non-monotone across the scale — a small-sample artifact,
  not a credit view. The default column is monotonized by weighted pool-adjacent-violators with
  live cells rescaled to keep rows stochastic; the run record reports raw-vs-adjusted and the
  adjusted grades as a LOW finding. If monotonization is disabled and the matrix still fails to
  rank-order, that is a MEDIUM finding — visible, never a crash, never a BLOCK (gates stay
  reserved for confirmed leakage and doc drift).
- **`next_rating_col` is outcome data**: excluded from features exactly like the target and
  `event_time_col` (datautil), because the end-of-window rating trivially encodes the default.

### 3. The loss forecast is a report; regime conditioning is its stress axis

`expected_loss_forecast` (EL = EAD × LGD × cumulative PD at the horizon, by rating band) and the
matrix-power term structure are **reports in the model payload** — like ADR-0008 §3 simulations,
they never feed champion selection. `migration.condition_col` estimates per-regime conditional
matrices (expansion/recession) so the forecast carries a baseline/downturn split — the
CreditMetrics-style complement to the covariate-shock `stress:` scenarios, which re-score the
champion and therefore cannot move a matrix.

### 4. The external-data decision is machine-generated design evidence

Ideate's transition-matrix framework entry is config-aware (rejected → available → candidate, with
stated reasons) and emits a `data-migration` unlock question when rating columns exist unconfigured.
Combined with the events-per-variable finding on a short internal history, the system itself
produces the documented case for licensing external data — SR 11-7 "alternatives considered / data
sufficiency" evidence, demonstrated end-to-end in `examples/rating_migration_loss/`.

## Consequences

- A rating panel is enough to run the full loss-forecast engagement offline; the champion stays a
  single interpretable GLM that *contains* the migration signal.
- Real licensed agency data drops in by pointing `data.path` at the extract and mapping
  `rating_col`/`next_rating_col`; the synthetic generator (`synth.make_rating_migration_dataset`)
  exists because CreditPro extracts cannot ship in a repo.
- The matrix inherits the run's reproducibility guarantees (seeded, train-only, artifacts under
  the run dir) and the review gate's docs↔code tracing like every other stage output.
