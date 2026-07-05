# Comprehensive showcase — the econometric & structural toolkit on a public-obligor book

One script demonstrates **every v0.4.0 capability** (ADR-0008) on a synthetic *public* C&I
portfolio — 2,000 obligors with traded-market observables, so, unlike the private middle-market
book in [`examples/commercial_credit/`](../commercial_credit/), **every framework applies**.

```bash
pip install -e .                                     # from the repo root
python examples/public_obligor_pd/run_demo.py        # add --save-sample to refresh sample_output/
```

Fully offline: solvers and seeded simulation live in the deterministic engine; no API key needed.

## The five steps (with real captured numbers)

### 1. Framework unlocks — the engine names what the data supports

Ideate runs *before* any capability is switched on. The design brief marks the hazard framework
**partial** and structural **available**, and emits unlock questions instead of assuming:

> `[data-event-time]` Column(s) default_quarter look like event timing — set `data.event_time_col`
> to unlock the discrete-time hazard families and a PD term structure.
> `[data-structural]` Market observables detected (equity_value, equity_vol) — enable the
> `structural:` config block to compute Merton distance-to-default.

### 2. Traditional GLM links — probit / cloglog with valid inference

A probit champion (CV AUC 0.820) with statsmodels p-values from the full-rank K-1 design — signs
match credit intuition (utilization +, leverage +, margin −). cloglog is the grouped-time
proportional-hazards link, the statistical bridge to step 3.

### 3. Discrete-time hazard — the PD term structure

With `data.event_time_col: default_quarter`, the survival family panel-expands obligor-quarters
*inside the estimator* (CV stays obligor-level — the survival-CV leak is impossible by
construction) and delivers what lifetime (CECL/IFRS 9) use cases actually need:

| Quarter | Marginal hazard | Cumulative PD |
|---|---|---|
| Q1 | 1.88% | 1.88% |
| Q2 | 1.45% | 3.18% |
| Q3 | 0.88% | 3.92% |
| Q4 | 1.30% | 4.96% |

Baseline-hazard period effects carry p-values (panel GLM inference); the holdout AUC (0.752) edges
the single-period probit reading (0.744).

### 4. Merton structural — distance-to-default hybrid + benchmark

The engine's KMV fixed-point solver turns equity value/vol + debt face into asset value/vol and
**DD** (mean 6.4, zero failed rows). Two uses, both shown:

- **Hybrid** (the deployed pattern): `merton_dd` enters the champion with a **−0.39 coefficient**
  (higher DD = safer — the economically right sign), recomputed target-hidden at serve time;
- **Pure-structural benchmark**: PD = N(−DD) alone scores holdout AUC 0.638 — real signal, but
  well below the hybrid champion's 0.749. The gap is the documented case for reduced-form + DD
  over pure structural on this book, exactly the "alternatives considered" evidence SR 11-7 wants.

### 5. Full pipeline — all families compete, simulation layer reports

All 7 families (logit/probit/cloglog, both hazard links, regularized logits) compete in the
ratchet; the full 8-stage pipeline runs green and the backtest stage reports:

- **OOT outcomes**: Gini 0.499, KS 0.490, PSI 0.104;
- **Vasicek portfolio** (LGD 45%, Basel ρ(PD)): EL 1.40%, VaR₀.₉₉₉ 8.6%, ES 9.8%, closed-form
  IRB K 7.1% — the loss distribution behind the PDs;
- **Macro stress**: baseline mean PD 3.12% → adverse 4.44% → severely adverse 10.07%,
  deterministically re-scored through the deployed scorer.

## sample_output/

Real artifacts captured from an actual run (`--save-sample`): the full `design_brief.md` (framework
assessment with all five frameworks resolved), `portfolio.json` (the loss distribution),
`stress.json` (scenario deltas), and `run_summary.txt` (the canonical verdict block).

## Where each capability lives

| Capability | Engine module | Config switch |
|---|---|---|
| probit / cloglog links | `modeling/fit.py::SMBinaryGLM` | in the default classification slate |
| Discrete-time hazard + term structure | `modeling/hazard.py` | `data.event_time_col` (+ `horizon_periods`) |
| Merton DD solver (hybrid + benchmark) | `modeling/structural.py` | `structural:` block |
| Vasicek portfolio + IRB capital | `modeling/simulate.py` | `portfolio:` block |
| Macro-scenario stress | `modeling/simulate.py` | `stress:` block |

Design rationale: [ADR-0008](../../docs/adr/0008-econometric-core-survival-structural-simulation.md).
