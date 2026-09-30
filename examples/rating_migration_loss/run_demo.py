#!/usr/bin/env python3
"""COGNOS comprehensive showcase — rating-migration loss forecasting for a corporate credit book.

The business problem, exactly as a commercial/corporate bank meets it: the sponsor needs a
**loss-forecasting model for the large-corporate book** (CECL provisioning and stress testing)
built on the industry-standard **rating-migration** framework — but the bank's internal rating
history is a few years old, with too few defaults to calibrate anything. The industry answer is to
license a long external agency history (S&P CreditPro-style obligor-year rating data, proxied here
by a synthetic panel calibrated to the shape of the published S&P long-run averages) and let the
matrix carry the long-run default experience.

This demo walks that engagement through every COGNOS stage and shows the system reaching each
decision for auditable reasons:

  1. INTERNAL DATA ASSESSMENT. Run explore + ideate on the internal book alone. The engine itself
     surfaces why the plan fails: events-per-variable far below the rule of thumb, and a rating
     history too short for a through-the-cycle matrix. The sponsor's recorded decision: bring in
     the agency panel.
  2. FRAMEWORK UNLOCK. Same two stages on the S&P-style panel with nothing switched on: ideate
     marks the rating-transition framework "available" and emits the unlock question that names
     the exact config switch. The engine names the capability; the human decides.
  3. FULL PIPELINE. All eight stages with the migration block on: the cohort matrix is estimated
     on the training partition only (NR-adjusted, PAVA rank-ordered), the rating-implied PD feeds
     the champion as the hybrid feature, the pure-migration PD is scored on the sealed out-of-time
     holdout as a labelled challenger benchmark, and the by-rating expected-loss forecast is
     reported under the pooled and the recession-conditioned matrices — then backtest (SR 11-7
     outcomes analysis + Vasicek portfolio + macro stress), the independent validate gate, the
     non-gating compliance report, the white paper, and the docs<->code review gate.

Run:  python examples/rating_migration_loss/run_demo.py [--save-sample]
(Fully offline — deterministic engine, no API key. `--save-sample` refreshes sample_output/.)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from cognos import synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator

warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn deprecation noise
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")  # penalty=l1 chatter

HERE = Path(__file__).parent
SCALE = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

DESIGN = {
    "use_case": "portfolio loss forecasting for CECL provisioning and enterprise stress testing",
    "horizon": "1-year default, extended to a 3-year cumulative term structure via matrix powers",
    "default_definition": "agency default state D (payment default, distressed exchange, "
                          "or bankruptcy)",
    "segment": "large corporate (agency-rated universe, mapped to the internal master scale)",
    "interpretability": "required",
    "notes": "Internal history (2019+) is too short for a through-the-cycle matrix; licensed "
             "agency data carries the long-run default experience. Expected signs: leverage "
             "up-risk; coverage, margin, size down-risk.",
}

STRESS_SCENARIOS = [
    {"name": "adverse", "shocks": {"unemployment_rate": {"add": 3.0}}},
    {"name": "severely_adverse", "shocks": {"unemployment_rate": {"add": 6.0}}},
]


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _config(csv: str, *, migration: bool = True, stages: list[str] | None = None,
            max_candidates: int = 12, name: str = "corporate_migration_loss") -> CognosConfig:
    raw = {
        "name": name,
        "description": "Corporate loss forecasting via rating migration on agency (S&P-style) data",
        "task": "classification",
        "data": {"path": csv, "target": "default", "datetime_col": "asof",
                 "drop_columns": ["obligor_id", "regime", "ead", "next_rating"]},
        "design": DESIGN,
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": max_candidates, "cv_folds": 5, "holdout_fraction": 0.2},
        "compliance": {"regimes": ["SR11-7", "NIST-AI-RMF"], "risk_tier": "high",
                       "jurisdictions": ["US"]},
    }
    if migration:
        # next_rating moves from a manual drop to a *declared outcome column* the engine consumes.
        raw["data"]["drop_columns"] = ["obligor_id", "regime", "ead"]
        raw["migration"] = {
            "enabled": True, "rating_col": "rating", "next_rating_col": "next_rating",
            "rating_scale": SCALE, "default_state": "D", "withdrawn_states": ["NR"],
            "horizon_periods": 3, "condition_col": "regime", "lgd": 0.40, "ead_column": "ead",
        }
        raw["portfolio"] = {"enabled": True, "lgd": 0.40, "ead_column": "ead", "n_sims": 10000}
        raw["stress"] = {"enabled": True, "scenarios": STRESS_SCENARIOS}
    if stages:
        raw["stages"] = {"enabled": stages}
    return CognosConfig.from_dict(raw)


def step1_internal_assessment(workdir: Path) -> None:
    banner("1. INTERNAL DATA ASSESSMENT — why the internal book alone cannot carry it")
    internal = synth.make_rating_migration_dataset(
        n_obligors=260, start_year=2019, end_year=2023, seed=11, book="bank")
    csv = str(workdir / "internal_book.csv")
    internal.to_csv(csv, index=False)
    n_events = int(internal["default"].sum())
    print(f"  Internal book: {len(internal)} obligor-years (2019-2023), {n_events} defaults, "
          f"{internal['asof'].dt.year.nunique()} annual cohorts.")

    cfg = _config(csv, migration=False, stages=["explore", "ideate"], name="internal_only")
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_internal"))
    orch.run_stage("explore")
    res = orch.run_stage("ideate")

    print("\n  Engine findings on the internal-only plan:")
    for f in res.findings:
        print(f"    [{f.severity.value}] {f.message}")
    for q in res.payload["open_questions"]:
        if q["id"] in ("data-epv", "data-migration"):
            print(f"    open question [{q['id']}]: {q['question'][:150]}...")
    epv = res.payload["data_structure"].get("events_per_variable")
    print(f"\n  Events-per-variable ≈ {epv} (rule of thumb ≥10): a scorecard fit on this sample "
          "would be unstable,")
    print("  and 5 annual cohorts cannot pin down a through-the-cycle transition matrix "
          "(no full cycle observed).")
    print("  → Sponsor decision (recorded in the design brief): license the S&P long-run "
        "rating history and")
    print("    develop on the agency universe, mapped to the internal master scale.")


def step2_framework_unlock(workdir: Path, csv: str) -> None:
    banner("2. FRAMEWORK UNLOCK — ideate names the capability on the agency panel")
    cfg = _config(csv, migration=False, stages=["explore", "ideate"], name="agency_unlock")
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_unlock"))
    orch.run_stage("explore")
    res = orch.run_stage("ideate")
    for fw in res.payload["framework_assessment"]:
        print(f"  - {fw['label']}: applicable={fw['applicable']} role={fw['role']}")
    print("\n  Unlock questions for the sponsor:")
    for q in res.payload["open_questions"]:
        if q["id"].startswith("data-"):
            print(f"    [{q['id']}] {q['question'][:160]}...")
    print("\n  The rating-transition framework is 'available'; switching it on is the "
          "sponsor's call (step 3).")


def step3_full_pipeline(workdir: Path, csv: str):
    banner("3. FULL PIPELINE — eight stages with the migration block enabled")
    cfg = _config(csv)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_full"))
    summary = orch.run()
    for stage in cfg.stages.enabled:
        r = orch.ctx.get(stage)
        print(f"    - {stage:9s} {r.verdict.value:6s} {r.summary}" if r
              else f"    - {stage:9s} (not run)")

    mp = orch.ctx.require("model").payload
    mig = mp["migration"]
    est = mig["estimate"]

    print("\n  --- The estimated transition matrix (cohort method, NR-adjusted, train-only) ---")
    header = "        " + "".join(f"{s:>8s}" for s in est["states"])
    print(header)
    for i, s in enumerate(est["scale"]):
        row = est["matrix"][i]
        print(f"  {s:>4s}  " + "".join(f"{100 * v:7.2f}%" for v in row)
              + f"   (n={est['row_support'][s]})")
    d = est["diagnostics"]
    print(f"  rank-ordered={d['default_col_monotone']} (raw monotone="
          f"{d['raw_default_col_monotone']}, PAVA-adjusted grades: "
          f"{', '.join(d['monotonized_rows']) or 'none'}); NR share={est['withdrawn_share']:.1%}")

    print("\n  --- Cumulative PD term structure (matrix powers, 3-year horizon) ---")
    ts = mig["term_structure"]
    for s in est["scale"]:
        print(f"    {s:>4s}: " + "  ".join(f"Y{k + 1}={100 * v:6.3f}%"
                                           for k, v in enumerate(ts[s])))

    print("\n  --- Champion vs. the pure-migration challenger benchmark (sealed OOT holdout) ---")
    bm = mig["benchmark"]
    print(f"    Champion ({mp['champion']['family']}, hybrid: migration_pd + fundamentals): "
          f"holdout AUC={mp['holdout_metric']:.3f}")
    print(f"    Pure migration PD (deployed={bm['deployed']}): "
          f"holdout AUC={bm['holdout_roc_auc']:.3f} (Gini={bm['holdout_gini']:.3f})")
    print("    → the matrix alone rank-orders well; obligor fundamentals add discrimination "
          "within grade.")

    print("\n  --- Expected-loss forecast on the out-of-time book "
          f"(LGD={mig['loss_forecast']['lgd']}, 3y horizon) ---")
    fc = mig["loss_forecast"]
    print(f"    {'rating':>7s} {'EAD share':>10s} {'cum PD':>9s} {'EL':>14s}")
    for b in fc["pooled"]["bands"]:
        print(f"    {b['rating']:>7s} {b['ead_share']:>9.1%} {b['cumulative_pd']:>8.3%} "
              f"{b['expected_loss']:>14,.0f}")
    print(f"    Through-the-cycle EL rate: {fc['pooled']['expected_loss_rate']:.3%} of EAD")
    for regime, cond in fc["conditional"].items():
        print(f"    {regime}-conditioned matrix: EL rate={cond['expected_loss_rate']:.3%}")

    bt = orch.ctx.require("backtest").payload
    oa = bt.get("outcomes_analysis") or {}
    pa = bt.get("portfolio_analysis") or {}
    st = bt.get("stress_testing") or {}
    print("\n  --- Backtest: SR 11-7 outcomes analysis + portfolio + macro stress ---")
    print(f"    OOT outcomes: Gini={oa.get('gini'):.3f} KS={oa.get('ks'):.3f} "
          f"PSI={oa.get('psi'):.3f} ({oa.get('psi_label')})")
    if pa:
        print(f"    Vasicek portfolio (LGD={pa['lgd']}): EL={pa['expected_loss']:.4f} "
              f"VaR{pa['confidence']:.3f}={pa['var']:.4f} ES={pa['expected_shortfall']:.4f} "
              f"| IRB K={pa['irb_capital_mean']:.4f}")
    if st:
        print(f"    Macro stress (baseline mean PD={st['baseline']['mean_pd']:.3%}):")
        for sc in st["scenarios"]:
            print(f"      - {sc['name']}: mean PD={sc['mean_pd']:.3%} (Δ{sc['delta_pd']:+.3%})")

    print(f"\n  Final verdict: {summary.final_verdict.value}")
    print(f"  White paper (OKF bundle): {orch.ctx.docs_dir}")
    return orch, summary


def save_sample(orch: Orchestrator, summary) -> None:
    out = HERE / "sample_output"
    out.mkdir(exist_ok=True)
    for rel, name in (("stages/ideate/design_brief.md", "design_brief.md"),
                      ("stages/model/migration.json", "migration.json"),
                      ("stages/backtest/stress.json", "stress.json")):
        src = orch.ctx.run_dir / rel
        if src.exists():
            shutil.copy(src, out / name)
    (out / "run_summary.txt").write_text(summary.token_block() + "\n", encoding="utf-8")
    print(f"\n  Sample artifacts refreshed under {out}")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="cognos_migration_"))
    print(f"COGNOS rating-migration loss-forecast showcase. Working directory: {workdir}")
    agency = synth.make_rating_migration_dataset(n_obligors=450, start_year=1981, end_year=2023)
    csv = str(workdir / "agency_panel.csv")
    agency.to_csv(csv, index=False)
    print(f"Agency (S&P-style) panel: {len(agency)} obligor-years, "
          f"{agency['asof'].dt.year.nunique()} annual cohorts (1981-2023), "
          f"{int(agency['default'].sum())} defaults.")

    step1_internal_assessment(workdir)
    step2_framework_unlock(workdir, csv)
    orch, summary = step3_full_pipeline(workdir, csv)

    if "--save-sample" in sys.argv[1:]:
        save_sample(orch, summary)

    banner("DONE")
    print("Every number above is deterministic and offline: the matrix estimation, PAVA")
    print("rank-ordering, term structure, loss forecast, and simulations live in the engine;")
    print("LLM reasoning is strictly additive. Design decisions: docs/adr/0009.")


if __name__ == "__main__":
    from cognos.cli import quiet_numerics, utf8_console

    utf8_console()
    quiet_numerics()
    main()
