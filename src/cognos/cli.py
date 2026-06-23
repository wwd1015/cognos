"""COGNOS command-line interface.

Exposes both operating modes plus per-stage invocation:
  cognos run        --config cognos.yaml [--interactive]   # full pipeline (autonomous or HITL)
  cognos run-stage  <stage> --config ... --run <run_id>     # one stage, individually invocable
  cognos demo       [--task regression|classification|credit|timeseries]
  cognos init / explain / report / list-runs / agents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CognosConfig

CONFIG_TEMPLATE = """\
name: my_model
description: "Describe the modeling problem."
task: regression          # regression | classification | ml_regression | ml_classification | timeseries
mode: autonomous          # autonomous | interactive

data:
  path: data.csv
  format: csv
  target: target
  features: []            # empty = all non-target/non-protected columns
  datetime_col: null      # set for timeseries / walk-forward
  protected_attributes: []  # excluded from features; used for fair-lending checks

metric:
  name: auto              # auto, or rmse/mae/r2/roc_auc/accuracy/f1/log_loss/...

search:
  max_candidates: 24
  cv_folds: 5
  holdout_fraction: 0.2
  random_state: 42
  ensemble: true

compliance:
  regimes: [SR11-7, NIST-AI-RMF]
  risk_tier: medium       # low | medium | high
  fair_lending: false
  jurisdictions: [US]

stages:
  enabled: [explore, ideate, model, backtest, validate, comply, document, review]
  gates: [validate, comply, review]
"""


def _load_config(path: str) -> CognosConfig:
    return CognosConfig.from_yaml(path)


def _console_gate(result) -> str:
    print("\n" + "=" * 60)
    print(f"GATE: {result.token_line()}")
    print(f"  {result.summary}")
    for f in result.findings[:10]:
        print(f"    - {f.line()}")
    if not sys.stdin.isatty():
        print("  (non-interactive stdin) -> auto-approve")
        return "approve"
    ans = input("Approve and continue? [y/N]: ").strip().lower()
    return "approve" if ans in ("y", "yes") else "reject"


def _cmd_init(args) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite).")
        return 1
    out.write_text(CONFIG_TEMPLATE)
    print(f"Wrote config template to {out}")
    return 0


def _cmd_explain(args) -> int:
    cfg = _load_config(args.config)
    print(f"COGNOS plan for project '{cfg.name}'")
    print(f"  task={cfg.task.value}  mode={cfg.mode.value}  metric={cfg.metric.name} ({cfg.metric.direction.value})")
    print(f"  target={cfg.data.target}  holdout={cfg.search.holdout_fraction}  budget={cfg.search.max_candidates} candidates")
    print(f"  stages: {' -> '.join(cfg.stages.enabled)}")
    print(f"  gates (may BLOCK): {', '.join(cfg.stages.gates)}")
    print(f"  compliance: regimes={cfg.compliance.regimes} risk_tier={cfg.compliance.risk_tier} "
          f"fair_lending={cfg.compliance.fair_lending} jurisdictions={cfg.compliance.jurisdictions}")
    return 0


def _cmd_run(args) -> int:
    from .orchestrator import Orchestrator

    cfg = _load_config(args.config)
    orch = Orchestrator(cfg, runs_root=args.runs_dir, run_id=args.run_id)
    summary = orch.run(interactive=args.interactive,
                       gate_handler=_console_gate if args.interactive else None)
    print(summary.token_block())
    print(f"\nRun directory: {orch.ctx.run_dir}")
    return 0 if summary.final_verdict.ok or summary.final_verdict.value in ("FAIL", "WARN") else 2


def _cmd_run_stage(args) -> int:
    from .orchestrator import Orchestrator

    cfg = _load_config(args.config)
    orch = Orchestrator(cfg, runs_root=args.runs_dir, run_id=args.run)
    result = orch.run_stage(args.stage)
    print(result.token_line())
    print(f"  {result.summary}")
    for f in result.findings:
        print(f"    - {f.line()}")
    print(f"\nRun directory: {orch.ctx.run_dir}")
    return 0


def _cmd_report(args) -> int:
    import json

    run_dir = Path(args.runs_dir or "runs") / args.run
    summ = run_dir / "summary.txt"
    if summ.exists():
        print(summ.read_text())
    else:
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            print(f"No run found at {run_dir}")
            return 1
        print(json.dumps(json.loads(manifest.read_text()), indent=2))
    return 0


def _cmd_list_runs(args) -> int:
    import json

    runs_dir = Path(args.runs_dir or "runs")
    if not runs_dir.exists():
        print(f"No runs directory at {runs_dir}")
        return 0
    for d in sorted(runs_dir.iterdir()):
        man = d / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            done = [k for k, v in m.get("stages", {}).items() if v]
            print(f"{d.name}  project={m.get('project')}  stages_done={len(done)}/{len(m.get('stages', {}))}")
    return 0


def _cmd_agents(args) -> int:
    from .orchestrator import _import_stages
    from .stages.base import STAGE_REGISTRY

    _import_stages()
    for name, cls in STAGE_REGISTRY.items():
        gate = " [gate]" if getattr(cls, "is_gate", False) else ""
        print(f"  {name}{gate}: {cls.description}")
    return 0


def _cmd_demo(args) -> int:
    from . import synth
    from .config import CognosConfig
    from .orchestrator import Orchestrator

    runs_dir = Path(args.runs_dir or "runs")
    data_dir = runs_dir / "_demo_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    gen = synth.GENERATORS[args.task]
    df = gen()
    csv = data_dir / f"{args.task}.csv"
    df.to_csv(csv, index=False)

    presets = {
        "regression": dict(task="regression", target="target", metric="rmse"),
        "classification": dict(task="classification", target="target", metric="roc_auc"),
        "timeseries": dict(task="timeseries", target="target", metric="rmse", datetime_col="date"),
        "credit": dict(task="classification", target="default", metric="roc_auc",
                       protected=["group"], fair_lending=True),
    }
    p = presets[args.task]
    raw = {
        "name": f"demo_{args.task}",
        "description": f"COGNOS synthetic {args.task} demo",
        "task": p["task"],
        "data": {"path": str(csv), "format": "csv", "target": p["target"],
                 "datetime_col": p.get("datetime_col"), "protected_attributes": p.get("protected", [])},
        "metric": {"name": p["metric"]},
        "compliance": {"fair_lending": p.get("fair_lending", False),
                       "jurisdictions": ["US", "EU"], "risk_tier": "high" if args.task == "credit" else "medium"},
    }
    cfg = CognosConfig.from_dict(raw)
    orch = Orchestrator(cfg, runs_root=str(runs_dir))
    summary = orch.run()
    print(summary.token_block())
    print(f"\nRun directory: {orch.ctx.run_dir}")
    print(f"White paper (OKF bundle): {orch.ctx.docs_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cognos", description="COGNOS — autonomous model-development agents.")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="write a config template")
    pi.add_argument("-o", "--output", default="cognos.yaml")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=_cmd_init)

    pe = sub.add_parser("explain", help="print the run plan without executing")
    pe.add_argument("--config", required=True)
    pe.set_defaults(func=_cmd_explain)

    pr = sub.add_parser("run", help="run the full pipeline")
    pr.add_argument("--config", required=True)
    pr.add_argument("--interactive", action="store_true", help="pause at gates for approval")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--runs-dir", default=None)
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("run-stage", help="run a single stage against an existing run")
    ps.add_argument("stage")
    ps.add_argument("--config", required=True)
    ps.add_argument("--run", required=True, help="run id")
    ps.add_argument("--runs-dir", default=None)
    ps.set_defaults(func=_cmd_run_stage)

    prep = sub.add_parser("report", help="print a run summary")
    prep.add_argument("--run", required=True)
    prep.add_argument("--runs-dir", default=None)
    prep.set_defaults(func=_cmd_report)

    pl = sub.add_parser("list-runs", help="list runs")
    pl.add_argument("--runs-dir", default=None)
    pl.set_defaults(func=_cmd_list_runs)

    pa = sub.add_parser("agents", help="list the stage agents")
    pa.set_defaults(func=_cmd_agents)

    pd = sub.add_parser("demo", help="run an end-to-end demo on synthetic data")
    pd.add_argument("--task", default="regression",
                    choices=["regression", "classification", "timeseries", "credit"])
    pd.add_argument("--runs-dir", default=None)
    pd.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
