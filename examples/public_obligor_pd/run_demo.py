#!/usr/bin/env python3
"""COGNOS comprehensive showcase — the econometric & structural toolkit on a public-obligor book.

One script exercises every v0.4.0 capability on a synthetic *public* C&I portfolio (2,000 obligors
with traded-market observables), where — unlike the private middle-market example — **every
framework applies**:

  1. FRAMEWORK UNLOCKS. Run ideate before the config switches anything on: the design brief marks
     the hazard framework "partial" and structural "available", and emits unlock questions
     (set data.event_time_col; enable the structural: block). The engine tells you what the data
     supports; the human decides.
  2. TRADITIONAL GLM LINKS. Fit probit and cloglog champions with full statsmodels inference —
     valid p-values from the K-1 design (cloglog = grouped-time proportional-hazards link).
  3. DISCRETE-TIME HAZARD. With event timing configured, fit the survival family: obligor-period
     panel expansion (inside the estimator, so CV stays obligor-level), panel GLM inference with
     baseline-hazard period effects, and the deliverable — a cumulative PD TERM STRUCTURE.
  4. MERTON STRUCTURAL (HYBRID). The engine's KMV solver turns equity value/vol + debt face into
     distance-to-default: `merton_dd` feeds the champion as an engineered feature (recomputed
     target-hidden at serve time), and the pure structural PD is scored on the sealed holdout as a
     labelled challenger benchmark.
  5. FULL PIPELINE + SIMULATION LAYER. All families compete in the ratchet; the champion flows to
     out-of-time SR 11-7 outcomes analysis, a Vasicek one-factor portfolio loss simulation
     (EL/VaR/ES + closed-form Basel IRB capital), and CCAR-flavoured macro stress scenarios.

Run:  python examples/public_obligor_pd/run_demo.py [--save-sample]
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

HERE = Path(__file__).parent

DESIGN = {
    "use_case": "portfolio surveillance and lifetime (CECL) loss estimation",
    "horizon": "12-month PD with a quarterly term structure",
    "default_definition": "90+ DPD or nonaccrual within 12 months of the observation point",
    "segment": "public C&I obligors (traded equity)",
    "interpretability": "required",
    "notes": "Term structure needed for lifetime provisioning; structural signal expected to add "
             "discrimination on the public book.",
}

STRUCTURAL = {"enabled": True, "equity_value_col": "equity_value",
              "equity_vol_col": "equity_vol", "debt_col": "debt_face"}

SCENARIOS = [
    {"name": "adverse",
     "shocks": {"unemployment_rate": {"add": 3.0}, "gdp_growth": {"add": -2.0}}},
    {"name": "severely_adverse",
     "shocks": {"unemployment_rate": {"add": 6.0}, "gdp_growth": {"set": -4.0}}},
]


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _config(csv: str, *, families: list[str] | None = None, unlock: bool = True,
            stages: list[str] | None = None, max_candidates: int = 14) -> CognosConfig:
    raw = {
        "name": "public_cni_pd",
        "description": "Public C&I obligor PD — econometric & structural showcase",
        "task": "classification",
        "data": {"path": csv, "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome"]},
        "design": DESIGN,
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": max_candidates, "cv_folds": 5, "holdout_fraction": 0.2,
                   "ensemble": True,
                   **({"model_families": families} if families else {})},
        "compliance": {"regimes": ["SR11-7", "NIST-AI-RMF"], "risk_tier": "high",
                       "jurisdictions": ["US"]},
    }
    if unlock:  # the switches the step-1 unlock questions point at
        raw["data"].update({"event_time_col": "default_quarter", "horizon_periods": 4})
        raw["structural"] = STRUCTURAL
        raw["portfolio"] = {"enabled": True, "lgd": 0.45, "n_sims": 10000}
        raw["stress"] = {"enabled": True, "scenarios": SCENARIOS}
    if stages:
        raw["stages"] = {"enabled": stages}
    return CognosConfig.from_dict(raw)


def step1_framework_unlocks(workdir: Path, csv: str) -> None:
    banner("1. FRAMEWORK UNLOCKS — ideate reports what the data supports")
    cfg = _config(csv, unlock=False, stages=["explore", "ideate"])
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_unlock"))
    orch.run_stage("explore")
    res = orch.run_stage("ideate")
    for fw in res.payload["framework_assessment"]:
        print(f"  - {fw['label']}: applicable={fw['applicable']} role={fw['role']}")
    print("\n  Unlock questions for the sponsor:")
    for q in res.payload["open_questions"]:
        if q["id"].startswith("data-"):
            print(f"    [{q['id']}] {q['question']}")
    print("\n  The engine names the capability; switching it on is the sponsor's call.")


def step2_glm_links(workdir: Path, csv: str) -> None:
    banner("2. TRADITIONAL GLM LINKS — probit / cloglog with valid inference")
    cfg = _config(csv, families=["probit", "cloglog"], stages=["explore", "ideate", "model"],
                  max_candidates=6)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_glm"))
    for s in ("explore", "ideate"):
        orch.run_stage(s)
    res = orch.run_stage("model")
    mp = res.payload
    print(f"  Champion: {mp['champion']['family']} | CV roc_auc={mp['cv_mean']:.3f} "
          f"| sealed holdout={mp['holdout_metric']:.3f}")
    pvals = {k: v for k, v in (mp["pvalues"] or {}).items() if k != "const"}
    top = sorted(pvals.items(), key=lambda kv: kv[1])[:4]
    print("  Most significant coefficients (statsmodels GLM on the K-1 design):")
    coefs = mp["coefficients"] or {}
    for name, p in top:
        print(f"    {name:24s} coef={coefs.get(name, float('nan')):+.3f}  p={p:.2e}")


def step3_hazard_term_structure(workdir: Path, csv: str) -> None:
    banner("3. DISCRETE-TIME HAZARD — survival panel, PD term structure")
    cfg = _config(csv, families=["hazard_cloglog"], stages=["explore", "ideate", "model"],
                  max_candidates=4)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_hazard"))
    for s in ("explore", "ideate"):
        orch.run_stage(s)
    res = orch.run_stage("model")
    mp = res.payload
    hz = mp["hazard"]
    ts = hz["term_structure"]
    print(f"  Champion: {mp['champion']['family']} (horizon={hz['horizon_periods']} quarters) "
          f"| CV roc_auc={mp['cv_mean']:.3f} | holdout={mp['holdout_metric']:.3f}")
    print("  PD term structure (portfolio mean):")
    for p, h, c in zip(ts["periods"], ts["mean_hazard"], ts["mean_cumulative_pd"], strict=False):
        print(f"    Q{p}: marginal hazard={h:.3%}   cumulative PD={c:.3%}")
    period_effects = [k for k in (mp["pvalues"] or {}) if k.startswith("_period")]
    print(f"  Baseline-hazard period effects with p-values: {len(period_effects)} "
          "(panel GLM inference)")


def step4_structural_hybrid(workdir: Path, csv: str) -> None:
    banner("4. MERTON STRUCTURAL — distance-to-default hybrid + benchmark")
    cfg = _config(csv, families=["logit"], stages=["explore", "ideate", "model"],
                  max_candidates=4)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_struct"))
    for s in ("explore", "ideate"):
        orch.run_stage(s)
    res = orch.run_stage("model")
    mp = res.payload
    st = mp["structural"]
    dd_coef = (mp["coefficients"] or {}).get("merton_dd")
    print(f"  Solver: mean DD={st['mean_dd']:.2f}, mean structural PD="
          f"{st['mean_structural_pd']:.3%}, failed rows={st['n_failed']}")
    print(f"  Hybrid: merton_dd in champion features={'merton_dd' in mp['champion']['features']}"
          + (f", coefficient={dd_coef:+.3f} (negative = higher DD is safer)" if dd_coef else ""))
    bm = st["benchmark"]
    print(f"  Pure-structural challenger benchmark (deployed={bm['deployed']}): "
          f"holdout AUC={bm['holdout_roc_auc']:.3f} (Gini={bm['holdout_gini']:.3f})")
    print(f"  Hybrid champion holdout AUC={mp['holdout_metric']:.3f} — the reduced-form model "
          "with the structural signal inside")


def step5_full_pipeline(workdir: Path, csv: str):
    banner("5. FULL PIPELINE — all families compete; portfolio + stress reported")
    cfg = _config(csv)
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs_full"))
    summary = orch.run()
    for stage in cfg.stages.enabled:
        r = orch.ctx.get(stage)
        print(f"    - {stage:9s} {r.verdict.value:6s} {r.summary}" if r
              else f"    - {stage:9s} (not run)")
    mp = orch.ctx.require("model").payload
    bt = orch.ctx.require("backtest").payload
    oa = bt.get("outcomes_analysis") or {}
    pa = bt.get("portfolio_analysis") or {}
    st = bt.get("stress_testing") or {}
    print(f"\n  Champion: {mp['champion']['family']} | CV={mp['cv_mean']:.3f} "
          f"| holdout={mp['holdout_metric']:.3f}")
    print(f"  OOT outcomes: Gini={oa.get('gini'):.3f} KS={oa.get('ks'):.3f} "
          f"PSI={oa.get('psi'):.3f} ({oa.get('psi_label')})")
    if pa:
        print(f"  Vasicek portfolio (LGD={pa['lgd']}): EL={pa['expected_loss']:.4f} "
              f"VaR{pa['confidence']:.3f}={pa['var']:.4f} ES={pa['expected_shortfall']:.4f} "
              f"| IRB K={pa['irb_capital_mean']:.4f}")
    if st:
        print(f"  Macro stress (baseline mean PD={st['baseline']['mean_pd']:.3%}):")
        for sc in st["scenarios"]:
            print(f"    - {sc['name']}: mean PD={sc['mean_pd']:.3%} (Δ{sc['delta_pd']:+.3%})")
    print(f"  Final verdict: {summary.final_verdict.value}")
    print(f"  White paper (OKF bundle): {orch.ctx.docs_dir}")
    return orch, summary


def save_sample(orch: Orchestrator, summary) -> None:
    out = HERE / "sample_output"
    out.mkdir(exist_ok=True)
    shutil.copy(orch.ctx.run_dir / "stages" / "ideate" / "design_brief.md",
                out / "design_brief.md")
    for rel, name in (("stages/backtest/portfolio.json", "portfolio.json"),
                      ("stages/backtest/stress.json", "stress.json")):
        src = orch.ctx.run_dir / rel
        if src.exists():
            shutil.copy(src, out / name)
    (out / "run_summary.txt").write_text(summary.token_block() + "\n", encoding="utf-8")
    print(f"\n  Sample artifacts refreshed under {out}")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="cognos_public_"))
    print(f"COGNOS public-obligor showcase. Working directory: {workdir}")
    df = synth.make_cni_portfolio_dataset(n=2000, include_market=True)
    csv = str(workdir / "public_cni.csv")
    df.to_csv(csv, index=False)

    step1_framework_unlocks(workdir, csv)
    step2_glm_links(workdir, csv)
    step3_hazard_term_structure(workdir, csv)
    step4_structural_hybrid(workdir, csv)
    orch, summary = step5_full_pipeline(workdir, csv)

    if "--save-sample" in sys.argv[1:]:
        save_sample(orch, summary)

    banner("DONE")
    print("Every capability above is deterministic and offline: solvers and seeded simulation in")
    print("the engine, LLM reasoning strictly additive. Design decisions: docs/adr/0008.")


if __name__ == "__main__":
    main()
