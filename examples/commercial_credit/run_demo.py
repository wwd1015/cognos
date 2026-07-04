#!/usr/bin/env python3
"""COGNOS worked example — a commercial C&I probability-of-default model, end to end.

This is the commercial-risk showcase for the enhanced agent capabilities. It follows the arc of a
real model-development engagement:

  1. EXPLORE catches a post-outcome field. The raw data pull includes ``dpd_at_outcome`` (days past
     due observed at the end of the outcome window) — a classic target leak. Explore flags it.
  2. IDEATE triangulates with the MD. Run before the sponsor has answered the design brief, ideate
     refuses to assume: it emits open design questions (use case? horizon? default definition?
     segment? and "is dpd_at_outcome really in the information set?").
  3. THE MD ANSWERS. The config drops the leaked column and fills the ``design:`` block. Ideate now
     produces a full design brief: data-structure assessment (vintage panel, event support),
     econometric framework assessment (reduced-form PD primary; discrete-time hazard partial;
     structural Merton rejected — no market observables for private obligors; trees as challengers
     because interpretability is required), and a ranked hypothesis slate.
  4. THE FULL PIPELINE runs: ratchet search → single interpretable champion → out-of-time SR 11-7
     outcomes analysis (Gini/KS, calibration, PSI) → independent validation → readiness report →
     OKF white paper → docs↔code review.

Run:  python examples/commercial_credit/run_demo.py [--save-sample]
(Requires `pip install -e .`. Runs fully offline with the deterministic engine; an LLM brain adds
proposed transforms and a design review on top, never instead.)

`--save-sample` refreshes examples/commercial_credit/sample_output/ with the real artifacts
(design brief, open questions, run summary) so the repository carries a captured example.
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

HERE = Path(__file__).parent

MD_ANSWERS = {
    "use_case": "origination underwriting",
    "horizon": "12-month PD",
    "default_definition": "90+ DPD or nonaccrual within 12 months of origination",
    "segment": "C&I middle-market",
    "interpretability": "required",
    "notes": "Signs must match credit intuition: leverage up-risk; coverage, liquidity, margin, "
             "size down-risk. Trees are benchmarks only.",
}


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _config(csv: str, *, drop: list[str], design: dict) -> CognosConfig:
    return CognosConfig.from_dict({
        "name": "cni_middle_market_pd",
        "description": "C&I middle-market 12-month PD (synthetic worked example)",
        "task": "classification",
        "data": {"path": csv, "target": "default", "datetime_col": "vintage",
                 "drop_columns": drop},
        "design": design,
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 12, "cv_folds": 5, "holdout_fraction": 0.2,
                   "ensemble": True},
        "compliance": {"regimes": ["SR11-7", "NIST-AI-RMF"], "risk_tier": "high",
                       "jurisdictions": ["US"]},
    })


def step1_explore_catches_the_leak(workdir: Path, csv: str) -> None:
    banner("1. EXPLORE — the raw data pull contains a post-outcome field")
    cfg = _config(csv, drop=["obligor_id"], design={})  # leak still present
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_naive"))
    res = orch.run_stage("explore")
    print(f"  verdict: {res.verdict.value} — {res.summary}")
    for f in res.findings:
        if f.category == "leakage":
            print(f"  LEAKAGE: {f.message}")
    print("  dpd_at_outcome is observed at the END of the outcome window — it is the outcome,")
    print("  not a predictor available at origination. The remedy is data.drop_columns.")
    return orch


def step2_ideate_triangulates(orch: Orchestrator) -> None:
    banner("2. IDEATE — before the MD answers, the design questions are OPEN")
    res = orch.run_stage("ideate")
    print(f"  verdict: {res.verdict.value} — {res.summary}\n")
    print("  Open questions for the sponsor (MD triangulation):")
    for q in res.payload["open_questions"]:
        print(f"    [{q['id']}] {q['question']}")
    print("\n  Ideate refuses to silently assume a horizon, default definition, segment, or that")
    print("  a flagged field is usable — the questions go to the human; answers go in `design:`.")


def step3_full_run_with_answers(workdir: Path, csv: str):
    banner("3. THE MD ANSWERS — design brief filled, leak dropped, full pipeline")
    cfg = _config(csv, drop=["obligor_id", "dpd_at_outcome"], design=MD_ANSWERS)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    summary = orch.run()

    ide = orch.ctx.require("ideate").payload
    print("  Data structure: "
          f"{ide['data_structure']['shape']}, {ide['data_structure']['n_rows']} obligors, "
          f"{ide['data_structure']['n_events']} defaults "
          f"(EPV≈{ide['data_structure']['events_per_variable']})")
    print("  Framework assessment (alternatives considered):")
    for fw in ide["framework_assessment"]:
        print(f"    - {fw['label']}: applicable={fw['applicable']} role={fw['role']}")
    print(f"  Remaining open questions: {[q['id'] for q in ide['open_questions']] or 'none'}")

    print("\n  Pipeline verdicts:")
    for stage in cfg.stages.enabled:
        res = orch.ctx.get(stage)
        if res is None:
            print(f"    - {stage:9s} (not run — halted earlier)")
            continue
        print(f"    - {stage:9s} {res.verdict.value:6s} {res.summary}")

    mp = orch.ctx.require("model").payload
    bt = orch.ctx.require("backtest").payload
    oa = bt.get("outcomes_analysis") or {}
    print(f"\n  Champion: {mp['champion']['family']} | CV roc_auc={mp['cv_mean']:.3f} "
          f"| sealed holdout={mp['holdout_metric']:.3f}")
    if mp.get("challenger_benchmark"):
        cb = mp["challenger_benchmark"]
        print(f"  Challenger benchmark ({cb['kind']}): {cb['benchmark_score']:.3f} vs champion "
              f"{cb['champion_single_score']:.3f} — the price of interpretability, stated.")
    print(f"  OOT outcomes ({bt['evaluation_sample']}): Gini={oa.get('gini'):.3f} "
          f"KS={oa.get('ks'):.3f} PSI={oa.get('psi'):.3f} ({oa.get('psi_label')})")
    print(f"  Final verdict: {summary.final_verdict.value}")
    print(f"  White paper (OKF bundle): {orch.ctx.docs_dir}")
    return orch, summary


def save_sample(orch: Orchestrator, summary, naive_orch: Orchestrator) -> None:
    """Refresh sample_output/ with the real artifacts of this run (committed to the repo)."""
    out = HERE / "sample_output"
    out.mkdir(exist_ok=True)
    shutil.copy(orch.ctx.run_dir / "stages" / "ideate" / "design_brief.md",
                out / "design_brief.md")
    ide = naive_orch.ctx.require("ideate").payload
    lines = ["# Open design questions (ideate, before the MD answered)", ""]
    lines += [f"- **[{q['id']}]** {q['question']}" for q in ide["open_questions"]]
    (out / "open_questions_before_answers.md").write_text("\n".join(lines) + "\n")
    (out / "run_summary.txt").write_text(summary.token_block() + "\n")
    print(f"\n  Sample artifacts refreshed under {out}")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="cognos_cni_"))
    print(f"COGNOS commercial-credit worked example. Working directory: {workdir}")
    df = synth.make_cni_portfolio_dataset(n=2000)  # includes the dpd_at_outcome leak on purpose
    csv = str(workdir / "cni_portfolio.csv")
    df.to_csv(csv, index=False)

    naive_orch = step1_explore_catches_the_leak(workdir, csv)
    step2_ideate_triangulates(naive_orch)
    orch, summary = step3_full_run_with_answers(workdir, csv)

    if "--save-sample" in sys.argv[1:]:
        save_sample(orch, summary, naive_orch)

    banner("DONE")
    print(f"All run artifacts are under {workdir}/runs — the design brief is at")
    print("  <run_dir>/stages/ideate/design_brief.md")
    print("Captured artifacts from a real run are committed under "
          "examples/commercial_credit/sample_output/.")


if __name__ == "__main__":
    main()
