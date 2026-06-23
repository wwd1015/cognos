#!/usr/bin/env python3
"""End-to-end COGNOS demonstration on synthetic data.

Shows every stage of the pipeline working, in both operating modes:

  1. AUTONOMOUS mode on a clean regression problem — the full 8-stage pipeline runs unattended
     and produces a validated, documented model (an OKF white-paper bundle).
  2. The COMPLIANCE GATE firing — a synthetic credit model with an injected group disparity is
     BLOCKed by the fair-lending (disparate-impact) check before it can be documented.
  3. STAGE-BY-STAGE mode — each agent invoked individually (human-in-the-loop friendly).

Run:  python examples/end_to_end/run_demo.py
(Requires `pip install -e .` from the repo root, plus optionally the IMPACT library for the real
feature-table scoring path; without IMPACT, COGNOS falls back to its built-in scorer automatically.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
    """Print each stage's verdict, key metrics, and a couple of findings."""
    for stage in ctx.config.stages.enabled:
        res = ctx.get(stage)
        if res is None:
            print(f"  - {stage:9s}  (not run — pipeline halted earlier)")
            continue
        metrics = ", ".join(f"{k}={v}" for k, v in list(res.metrics.items())[:3])
        print(f"  - {stage:9s} {res.verdict.value:6s}  {res.summary}")
        if metrics:
            print(f"      metrics: {metrics}")
        for f in res.findings[:2]:
            print(f"      finding: {f.line()}")


def demo_autonomous_regression(workdir: Path) -> None:
    banner("1. AUTONOMOUS MODE — clean regression problem (full pipeline)")
    csv = _write(synth.make_regression_dataset(n=600), workdir / "regression.csv")
    cfg = CognosConfig.from_dict({
        "name": "house_prices_demo",
        "description": "Synthetic regression demo for COGNOS",
        "task": "regression",
        "data": {"path": csv, "target": "target"},
        "metric": {"name": "rmse"},
        "search": {"max_candidates": 16, "cv_folds": 5, "holdout_fraction": 0.2},
        "compliance": {"regimes": ["SR11-7", "NIST-AI-RMF"], "risk_tier": "medium",
                       "jurisdictions": ["US", "EU"]},
    })
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    summary = orch.run()
    show_run(orch.ctx)
    print(f"\n  Final verdict: {summary.final_verdict.value}")
    print(f"  Champion: {orch.ctx.require('model').metrics['champion']} "
          f"({cfg.metric.name}={summary.champion_metric:.4f})")
    print(f"  White paper (OKF bundle): {orch.ctx.docs_dir}")
    print(f"  Concepts: {[p.name for p in sorted(orch.ctx.docs_dir.glob('*.md'))]}")


def demo_compliance_block(workdir: Path) -> None:
    banner("2. COMPLIANCE GATE — credit model with injected disparity is BLOCKed")
    csv = _write(synth.make_credit_dataset(n=1000), workdir / "credit.csv")
    cfg = CognosConfig.from_dict({
        "name": "credit_default_demo",
        "description": "Synthetic credit-default model (fair-lending demo)",
        "task": "classification",
        "data": {"path": csv, "target": "default", "protected_attributes": ["group"]},
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 12, "cv_folds": 5},
        "compliance": {"fair_lending": True, "risk_tier": "high", "jurisdictions": ["US", "EU"]},
    })
    orch = Orchestrator(cfg, runs_root=str(workdir / "runs"))
    summary = orch.run()
    show_run(orch.ctx)
    comply = orch.ctx.require("comply").payload
    di = comply["fair_lending"].get("disparate_impact")
    print(f"\n  Final verdict: {summary.final_verdict.value}")
    print(f"  Disparate impact ratio: {di:.3f} (four-fifths threshold 0.80) -> "
          f"{'BLOCKED' if summary.final_verdict.value == 'BLOCK' else 'passed'}")
    print("  The pipeline correctly halted before shipping a model with a fair-lending violation.")


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
    demo_autonomous_regression(workdir)
    demo_compliance_block(workdir)
    demo_stage_by_stage(workdir)
    banner("DONE")
    print(f"All run artifacts (models, diagnostics, OKF white papers) are under: {workdir}/runs")


if __name__ == "__main__":
    main()
