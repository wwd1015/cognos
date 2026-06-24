#!/usr/bin/env python3
"""End-to-end COGNOS demonstration on synthetic data.

Shows the pipeline working across three scenarios:

  1. AUTONOMOUS commercial credit model — the full 8-stage pipeline runs unattended and produces
     SR 11-7 outcomes analysis (Gini/KS, calibration, PSI) on an out-of-time sample, a non-gating
     model-risk readiness report, and a white paper (OKF bundle).
  2. The VALIDATION GATE firing — a model that uses a leaking feature is BLOCKed by the independent
     validation gate before it can be documented (confirmed leakage is the hard BLOCK).
  3. STAGE-BY-STAGE mode — each agent invoked individually (human-in-the-loop friendly).

Run:  python examples/end_to_end/run_demo.py
(Requires `pip install -e .`; optionally the IMPACT library for the real feature-table scoring path —
without it, COGNOS falls back to its built-in scorer automatically. The LLM-guided search and
LLM-driven ideation activate only when an LLM brain is configured; this demo runs the deterministic
engine.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cognos import synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator


def _write(df, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def show_run(ctx) -> None:
    for stage in ctx.config.stages.enabled:
        res = ctx.get(stage)
        if res is None:
            print(f"  - {stage:9s}  (not run — pipeline halted earlier)")
            continue
        print(f"  - {stage:9s} {res.verdict.value:6s}  {res.summary}")
        for f in res.findings[:2]:
            print(f"      finding: {f.line()}")


def demo_autonomous_commercial(workdir: Path) -> None:
    banner("1. AUTONOMOUS MODE — commercial credit model (full pipeline, out-of-time backtest)")
    csv = _write(synth.make_commercial_credit_dataset(n=1200), workdir / "commercial.csv")
    cfg = CognosConfig.from_dict({
        "name": "commercial_pd_demo",
        "description": "Synthetic commercial credit-default model",
        "task": "classification",
        "data": {"path": csv, "target": "default", "datetime_col": "vintage"},
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 12, "cv_folds": 5, "holdout_fraction": 0.2},
        "compliance": {"regimes": ["SR11-7", "NIST-AI-RMF"], "risk_tier": "high",
                       "jurisdictions": ["US"]},
    })
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    summary = orch.run()
    show_run(orch.ctx)
    bt = orch.ctx.require("backtest").payload
    oa = bt.get("outcomes_analysis") or {}
    print(f"\n  Final verdict: {summary.final_verdict.value}")
    print(f"  Evaluation sample: {bt['evaluation_sample']}")
    print(f"  Outcomes analysis: Gini={oa.get('gini'):.3f}, KS={oa.get('ks'):.3f}, "
          f"PSI={oa.get('psi'):.3f} ({oa.get('psi_label')})")
    print(f"  White paper (OKF bundle): {orch.ctx.docs_dir}")


def demo_validation_block_on_leakage(workdir: Path) -> None:
    banner("2. VALIDATION GATE — a leaking model is BLOCKed before documentation")
    df = synth.make_regression_dataset(n=600)
    df["leaky"] = df["target"] + np.random.default_rng(0).normal(0, 1e-3, len(df))  # target leak
    csv = _write(df, workdir / "leak.csv")
    cfg = CognosConfig.from_dict({
        "name": "leakage_demo", "task": "regression",
        "data": {"path": csv, "target": "target"},
        "metric": {"name": "rmse"}, "search": {"max_candidates": 8, "cv_folds": 3},
    })
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    summary = orch.run()
    show_run(orch.ctx)
    print(f"\n  Final verdict: {summary.final_verdict.value}")
    print("  The independent validation gate detected target leakage and halted the pipeline "
          "before the model could be documented or shipped.")


def demo_stage_by_stage(workdir: Path) -> None:
    banner("3. STAGE-BY-STAGE MODE — invoke each agent individually")
    csv = _write(synth.make_classification_dataset(n=500), workdir / "clf.csv")
    cfg = CognosConfig.from_dict({
        "name": "stepwise_demo", "task": "classification",
        "data": {"path": csv, "target": "target"},
        "metric": {"name": "roc_auc"}, "search": {"max_candidates": 10},
    })
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    for stage in ("explore", "ideate", "model", "backtest", "validate"):
        res = orch.run_stage(stage)
        print(f"  ran '{stage}' independently -> {res.verdict.value}: {res.summary}")
    print(f"\n  Each stage read the previous stages' on-disk artifacts from {orch.ctx.run_dir}")
    print("  (A human can inspect/approve between any two stages.)")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="cognos_demo_"))
    print(f"COGNOS end-to-end demo. Working directory: {workdir}")
    demo_autonomous_commercial(workdir)
    demo_validation_block_on_leakage(workdir)
    demo_stage_by_stage(workdir)
    banner("DONE")
    print(f"All run artifacts (models, diagnostics, OKF white papers) are under: {workdir}/runs")
    print("Tip: set an LLM brain (brain.kind: llm + ANTHROPIC_API_KEY) and search.guided: true to let "
          "the reasoning layer drive feature engineering and the search.")


if __name__ == "__main__":
    main()
