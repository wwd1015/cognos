# Worked example — a commercial C&I probability-of-default model

The commercial-risk showcase for the COGNOS agents' domain capabilities. One script follows the
arc of a real model-development engagement on a synthetic-but-realistic **C&I middle-market
portfolio** (2,000 obligors, quarterly origination vintages 2019Q1–2023Q4 spanning the 2020 stress
period, obligor financial ratios, facility characteristics, macro at origination — and, on
purpose, one classic post-outcome leak).

```bash
pip install -e .                                    # from the repo root
python examples/commercial_credit/run_demo.py       # add --save-sample to refresh sample_output/
```

Equivalent CLI one-liner (leak pre-dropped, design pre-answered): `cognos demo --task cni`

## The arc

1. **Explore catches the leak.** The raw pull includes `dpd_at_outcome` — days past due observed
   at the *end* of the outcome window. |corr| ≈ 0.99 with the default flag; explore flags it as a
   leakage suspect. (Playbook: `.claude/skills/cognos-explore/SKILL.md`.)

2. **Ideate triangulates with the MD.** Run before the sponsor answered the design brief, ideate
   emits **open design questions** instead of assuming: use case? horizon? default definition?
   segment? is the flagged field really in the information set? plus a low
   events-per-variable warning (93 defaults, 13 candidate features). See
   [`sample_output/open_questions_before_answers.md`](sample_output/open_questions_before_answers.md).

3. **The MD answers.** The config drops the leak (`data.drop_columns`) and fills the `design:`
   block (origination underwriting, 12-month PD, 90+ DPD/nonaccrual, C&I middle-market,
   interpretability **required**). Ideate now produces the full
   [`design_brief.md`](sample_output/design_brief.md):
   - **Data structure** — vintage panel, event support, EPV;
   - **Framework assessment** ("alternatives considered" for SR 11-7): reduced-form PD **primary**;
     discrete-time hazard **partial** (vintage panel → period-indexed logit, Shumway 2001);
     structural Merton **rejected** — no market observables for private obligors (Merton 1974,
     KMV); rating migration **rejected** — no rating history (CreditMetrics); ML **challenger
     only** because interpretability is required;
   - a **ranked hypothesis slate** with parsimonious feature sets up-weighted (low EPV).

4. **The full pipeline** runs with the econometric and simulation layers unlocked by the MD's
   answers: `data.event_time_col` (the default quarter) adds the **discrete-time hazard families**
   (`hazard_logit`/`hazard_cloglog`, Shumway 2001 — obligor-period panel expansion, PD term
   structure) to the slate alongside logit/**probit/cloglog**; the ratchet picks the champion on
   evidence, with the labelled ensemble challenger stated as "the price of interpretability" →
   **out-of-time SR 11-7 outcomes analysis** (Gini ≈ 0.48, KS ≈ 0.45, PSI ≈ 0.12 — the 2020
   vintages moved the population, and the report says so) → **Vasicek one-factor portfolio
   simulation** (EL/VaR/ES + closed-form Basel IRB capital) → **macro-scenario stress** (adverse
   and severely-adverse unemployment/GDP shocks re-scored deterministically; baseline ≈ 3.1% mean
   PD rising to ≈ 10.2% severely adverse) → independent validation → non-gating readiness report →
   OKF white paper → docs↔code review. Captured verdicts:
   [`sample_output/run_summary.txt`](sample_output/run_summary.txt).

   (Structural Merton stays **rejected — no market observables** for this private middle-market
   portfolio; that recorded rejection is the SR 11-7 "alternatives considered" evidence. On a
   public-obligor book, `synth.make_cni_portfolio_dataset(include_market=True)` plus the
   `structural:` config block turn on the distance-to-default hybrid + benchmark.)

## What each agent's enhancement contributes here

| Agent | Enhanced capability on display |
|---|---|
| explore | post-outcome leakage patterns, event-support profiling (skill pack) |
| ideate | data-structure assessment, framework selection (incl. hazard/structural unlock questions), MD triangulation, EPV-aware ranking |
| model | traditional-regression menu (logit/probit/cloglog + discrete-time hazard with PD term structure), interpretable champion + challenger gap, economic sign checks |
| backtest | OOT Gini/KS/calibration/PSI + Vasicek portfolio losses/IRB capital + macro stress deltas |
| validate | independent effective challenge; leakage would have been the hard BLOCK |
| comply | readiness report with human-only steps; fair-lending scope per ADR-0004 |
| document | design brief + rejected frameworks become "alternatives considered" |
| review | docs↔code drift gate over the generated bundle |

The per-agent domain playbooks live in `.claude/skills/cognos-<stage>/SKILL.md`; each stage agent
loads its playbook before judging engine output.

## sample_output/

Real artifacts captured from an actual run of this script (refresh with `--save-sample`):

- `open_questions_before_answers.md` — the MD-triangulation questions from step 2.
- `design_brief.md` — the full design brief from step 3.
- `run_summary.txt` — the canonical run-summary token block from step 4.
