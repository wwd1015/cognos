"""COGNOS command-line interface.

  cognos ui         [--port 8050]                              # the Dash workbench
  cognos run        --config cognos.yaml [--interactive] [--provider P]
  cognos demo       [--task commercial|cni|migration|...] [--interactive] [--provider P]
  cognos status     --run <run_id>                            # step table, gates, questions
  cognos gate       <gate> --run <run_id> --action accept|edit|override|send_back|approve|reject
  cognos answer     --run <run_id> --gap <id> --text "..."   # answer a sponsor question
  cognos retry      <step> --run <run_id>
  cognos run-stage  <stage> --config ... --run <run_id>       # one stage, individually invocable
  cognos providers | agents | init | explain | report | list-runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CognosConfig

CONFIG_TEMPLATE = """\
name: my_model
description: "Describe the modeling problem."
task: regression          # regression | classification | ml_regression | ml_classification | timeseries
mode: interactive         # interactive (pause at the review gates) | autonomous (agents' calls stand)

data:
  path: data.csv
  format: csv
  target: target
  features: []            # empty = all non-target/non-protected columns
  datetime_col: null      # set for timeseries / walk-forward
  protected_attributes: []  # excluded from features; used for fair-lending checks
  event_time_col: null    # 1-based period of the event (survival); unlocks the hazard families
  horizon_periods: null   # outcome window in periods; null = max observed event time

design:                   # the sponsor's (MD's) design brief — unanswered fields become
  use_case: ""            #   open questions at the design gate, never silent assumptions
  horizon: ""             #   e.g. origination | surveillance | CECL | IRB
  default_definition: ""  # e.g. "90+ DPD or nonaccrual"
  segment: ""             # e.g. "C&I middle-market"
  interpretability: required  # required | preferred | flexible
  notes: ""

agents:                   # who makes the recommendations (cognos providers lists them)
  provider: auto          # auto | heuristic | claude_cli | anthropic | openai | xai | ...
  model: null             # override the provider's default model
  max_retries: 2          # engine-check failures are fed back and retried
  budget_usd: 5.0         # spend cap per run (0 = unlimited)

workflow:
  gates: [gate_data, gate_design, gate_champion, gate_validation, gate_signoff]
  auto_challenge_loops: 2 # validator findings routed back automatically, at most N times

metric:
  name: auto              # auto, or rmse/mae/r2/roc_auc/accuracy/f1/log_loss/...

search:
  max_candidates: 24
  cv_folds: 5
  holdout_fraction: 0.2
  random_state: 42
  ensemble: true
  guided: false           # let the modeler agent propose extra experiments (engine keeps winners)

structural:               # Merton distance-to-default (needs market observables)
  enabled: false
  equity_value_col: null
  equity_vol_col: null
  debt_col: null

migration:                # rating-transition matrix (needs a rating history)
  enabled: false
  rating_col: null
  next_rating_col: null   # rating at the END of the outcome window (outcome data, never a feature)
  rating_scale: []
  default_state: D
  withdrawn_states: [NR]
  horizon_periods: 1
  monotone_pd: true
  condition_col: null
  lgd: 0.45
  ead_column: null

portfolio:                # Vasicek one-factor loss simulation + Basel IRB capital (reported)
  enabled: false
  lgd: 0.45
  asset_correlation: null

stress:                   # macro-scenario stress: shock covariates, re-score deterministically
  enabled: false
  scenarios: []           # [{name: adverse, shocks: {unemployment: {add: 3.0}}}]

compliance:
  regimes: [SR11-7, NIST-AI-RMF]
  risk_tier: medium       # low | medium | high
  fair_lending: false
  jurisdictions: [US]

stages:
  enabled: [explore, ideate, model, backtest, validate, comply, document, review]
  gates: [validate, review]   # verdict gates that may BLOCK
"""


def _load_config(path: str) -> CognosConfig:
    return CognosConfig.from_yaml(path)


# --- terminal gate review ------------------------------------------------------------------
def _gate_prompt(run_id: str, gate: str, root) -> bool:
    """Show a waiting gate in the terminal and take a decision. False = leave it waiting."""
    from . import service
    from .engine import GateError
    from .engine.graph import LABELS, STAGE_OF_GATE

    stage = STAGE_OF_GATE[gate]
    res = service.results(run_id, root).get(stage)
    print("\n" + "=" * 72)
    print(f"GATE {gate} — {LABELS[gate]}")
    if res is not None:
        print(f"  {res.token_line()}\n  {res.summary}")
        rec = (res.payload or {}).get("recommendation") or {}
        if rec:
            print(f"  Agent ({rec.get('agent')} via {rec.get('provider')}): "
                  f"{rec.get('output', {}).get('summary', '')}")
        for f in res.findings[:8]:
            print(f"    - {f.line()}")
    st = service.state(run_id, root)
    for g in st.open_gaps()[:6]:
        print(f"  ? [{g.id}] {g.question}")
    if not sys.stdin.isatty():
        if res is not None and res.verdict.value == "BLOCK":
            print("  (non-interactive stdin, BLOCK) -> left waiting; resolve with `cognos gate`.")
            return False
        action = "approve" if gate == "gate_signoff" else "accept"
        print(f"  (non-interactive stdin) -> {action}")
        service.submit_gate(run_id, gate, action, reason="non-interactive stdin", root=root,
                            background=False)
        return True
    choice = input("  [a]ccept  [s]end back  [q]uit (leave waiting) > ").strip().lower()
    try:
        if choice.startswith("a"):
            reason = input("  reason (optional) > ").strip()
            action = "approve" if gate == "gate_signoff" else "accept"
            service.submit_gate(run_id, gate, action, reason=reason, root=root, background=False)
            return True
        if choice.startswith("s"):
            target = stage if stage in ("explore", "ideate", "model") else (
                input("  send back to [explore/ideate/model] > ").strip() or "model")
            msg = input("  what should the agent reconsider? > ").strip()
            service.submit_gate(run_id, gate, "send_back", {"target": target, "message": msg},
                                reason=msg, root=root, background=False)
            return True
    except GateError as exc:
        print(f"  refused: {exc}")
        return True
    return False


def _drive(run_id: str, root, interactive: bool) -> int:
    from . import service
    from .engine.graph import GATES

    for _ in range(50):
        st = service.run_until_idle(run_id, root)
        waiting = [g for g in GATES if st.status_of(g) == "awaiting"]
        if not (interactive and waiting):
            break
        if not _gate_prompt(run_id, waiting[0], root):
            break
    eng = service.engine(run_id, root)
    summary = eng.summary()
    st = service.state(run_id, root)
    print(summary.token_block())
    print(f"status: {st.status}" + (f" ({st.halted_reason})" if st.halted_reason else ""))
    print(f"\nRun directory: {eng.run_dir}")
    return 0 if st.status in ("completed", "approved", "awaiting") else 2


# --- commands ------------------------------------------------------------------------------
def _cmd_init(args) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite).")
        return 1
    out.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"Wrote config template to {out}")
    return 0


def _cmd_explain(args) -> int:
    from .agents import providers

    cfg = _load_config(args.config)
    print(f"COGNOS plan for project '{cfg.name}'")
    print(f"  task={cfg.task.value}  mode={cfg.mode.value}  metric={cfg.metric.name} ({cfg.metric.direction.value})")
    print(f"  target={cfg.data.target}  holdout={cfg.search.holdout_fraction}  budget={cfg.search.max_candidates} candidates")
    print(f"  stages: {' -> '.join(cfg.stages.enabled)}")
    print(f"  human review gates: {', '.join(cfg.workflow.gates) or 'none'}")
    print(f"  verdict gates (may BLOCK): {', '.join(cfg.stages.gates)}")
    try:
        prov = providers.resolve(cfg.agents.provider, cfg.agents.model)
        print(f"  agents: {prov['id']} ({prov.get('model') or prov['kind']})")
    except providers.ProviderUnavailable as exc:
        print(f"  agents: {exc}")
    print(f"  compliance: regimes={cfg.compliance.regimes} risk_tier={cfg.compliance.risk_tier} "
          f"fair_lending={cfg.compliance.fair_lending} jurisdictions={cfg.compliance.jurisdictions}")
    return 0


def _cmd_run(args) -> int:
    from . import service

    cfg = _load_config(args.config)
    mode = "interactive" if args.interactive else cfg.mode.value
    run_id = args.run_id
    if run_id and (service.runs_root(args.runs_dir) / run_id / "state.json").exists():
        pass  # resume an existing run
    else:
        from .engine import Engine

        eng = Engine(cfg, run_id=run_id, runs_root=service.runs_root(args.runs_dir),
                     provider=args.provider, mode=mode)
        run_id = eng.run_id
    return _drive(run_id, args.runs_dir, interactive=mode == "interactive")


def _cmd_demo(args) -> int:
    from . import service

    cfg = service.demo_config(args.task, args.runs_dir)
    mode = "interactive" if args.interactive else "autonomous"
    run_id = service.create_run(cfg, mode=mode, provider=args.provider, root=args.runs_dir)
    rc = _drive(run_id, args.runs_dir, interactive=args.interactive)
    print(f"White paper (OKF bundle): {service.runs_root(args.runs_dir) / run_id / 'docs'}")
    return rc


def _cmd_run_stage(args) -> int:
    from .orchestrator import Orchestrator

    cfg = _load_config(args.config)
    orch = Orchestrator(cfg, runs_root=args.runs_dir, run_id=args.run, provider=args.provider)
    result = orch.run_stage(args.stage)
    print(result.token_line())
    print(f"  {result.summary}")
    for f in result.findings:
        print(f"    - {f.line()}")
    print(f"\nRun directory: {orch.engine.run_dir}")
    return 0


def _cmd_status(args) -> int:
    from . import service
    from .engine.graph import LABELS, STEPS

    st = service.state(args.run, args.runs_dir)
    print(f"run {st.run_id}  project={st.project}  mode={st.mode}  agents={st.provider}  "
          f"status={st.status}  spend=${st.spend_usd:.2f}")
    if st.halted_reason:
        print(f"  halted: {st.halted_reason}")
    for step in STEPS:
        s = st.steps[step]
        verdict = f" [{s.verdict}]" if s.verdict else ""
        print(f"  {step:<16} {s.status:<9}{verdict:<10} {LABELS[step]}")
    for g in st.gaps:
        print(f"  ? {g.id} ({g.status}) {g.question}" + (f" -> {g.answer}" if g.answer else ""))
    for c in st.challenges:
        print(f"  ! {c.id} {c.source}->{c.target_stage} ({c.status}) {c.message}")
    return 0


def _cmd_gate(args) -> int:
    from . import service
    from .engine import GateError

    payload = json.loads(args.payload) if args.payload else {}
    if args.target:
        payload["target"] = args.target
    if args.message:
        payload["message"] = args.message
    try:
        service.submit_gate(args.run, args.gate, args.action, payload, args.reason or "",
                            root=args.runs_dir, background=False)
    except (GateError, KeyError) as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_answer(args) -> int:
    from . import service
    from .engine import GateError

    try:
        service.answer_gap(args.run, args.gap, args.text or "", assume=args.assume,
                           root=args.runs_dir, background=False)
    except GateError as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_retry(args) -> int:
    from . import service
    from .engine import GateError

    try:
        service.retry(args.run, args.step, args.runs_dir, background=False)
    except GateError as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_providers(args) -> int:
    from .agents import providers

    for p in providers.public_list():
        mark = "available" if p["available"] else "-"
        print(f"  {p['id']:<11} {mark:<10} {p['kind']:<14} {p.get('model') or '':<22} {p['label']}")
    try:
        print(f"\nauto resolves to: {providers.resolve('auto')['id']}")
    except providers.ProviderUnavailable as exc:
        print(f"\n{exc}")
    return 0


def _cmd_report(args) -> int:
    run_dir = Path(args.runs_dir or "runs") / args.run
    summ = run_dir / "summary.txt"
    if summ.exists():
        print(summ.read_text(encoding="utf-8"))
    else:
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            print(f"No run found at {run_dir}")
            return 1
        print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))
    return 0


def _cmd_list_runs(args) -> int:
    from . import service

    rows = service.list_runs(args.runs_dir)
    if not rows:
        print(f"No runs under {service.runs_root(args.runs_dir)}")
    for r in rows:
        wait = f" waiting={r['waiting_on']}" if r["waiting_on"] else ""
        print(f"{r['run_id']}  project={r['project']}  status={r['status']}  "
              f"agents={r['provider']}{wait}")
    return 0


def _cmd_agents(args) -> int:
    from .agents.contracts import FRIENDLY, STAGE_AGENT
    from .orchestrator import _import_stages
    from .stages.base import STAGE_REGISTRY

    _import_stages()
    for name, cls in STAGE_REGISTRY.items():
        gate = " [verdict gate]" if getattr(cls, "is_gate", False) else ""
        agent = STAGE_AGENT.get(name)
        who = f"  agent: {FRIENDLY[agent]}" if agent else "  agent: none (deterministic)"
        print(f"  {name}{gate}: {cls.description}\n  {who}")
    return 0


def _cmd_ui(args) -> int:
    try:
        from .ui.app import run_server
    except ImportError as exc:
        print(f"The UI needs the [ui] extra: pip install -e '.[ui]'  ({exc})")
        return 1
    run_server(host=args.host, port=args.port, runs_dir=args.runs_dir, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cognos",
                                description="COGNOS — agents recommend, you decide, the engine disposes.")
    sub = p.add_subparsers(dest="command", required=True)

    def runs(sp):
        sp.add_argument("--runs-dir", default=None)

    pu = sub.add_parser("ui", help="launch the Dash workbench")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--port", type=int, default=8050)
    pu.add_argument("--debug", action="store_true")
    runs(pu)
    pu.set_defaults(func=_cmd_ui)

    pi = sub.add_parser("init", help="write a config template")
    pi.add_argument("-o", "--output", default="cognos.yaml")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=_cmd_init)

    pe = sub.add_parser("explain", help="print the run plan without executing")
    pe.add_argument("--config", required=True)
    pe.set_defaults(func=_cmd_explain)

    pr = sub.add_parser("run", help="run the workflow (resumes when --run-id exists)")
    pr.add_argument("--config", required=True)
    pr.add_argument("--interactive", action="store_true", help="review each gate in the terminal")
    pr.add_argument("--provider", default=None, help="agent backend (see `cognos providers`)")
    pr.add_argument("--run-id", default=None)
    runs(pr)
    pr.set_defaults(func=_cmd_run)

    pd = sub.add_parser("demo", help="run end to end on synthetic data")
    pd.add_argument("--task", default="commercial",
                    choices=["regression", "classification", "timeseries", "credit", "commercial",
                             "cni", "migration"])
    pd.add_argument("--interactive", action="store_true")
    pd.add_argument("--provider", default="heuristic")
    runs(pd)
    pd.set_defaults(func=_cmd_demo)

    ps = sub.add_parser("run-stage", help="run a single stage against an existing run")
    ps.add_argument("stage")
    ps.add_argument("--config", required=True)
    ps.add_argument("--run", required=True, help="run id")
    ps.add_argument("--provider", default=None)
    runs(ps)
    ps.set_defaults(func=_cmd_run_stage)

    pst = sub.add_parser("status", help="show a run's steps, questions and challenges")
    pst.add_argument("--run", required=True)
    runs(pst)
    pst.set_defaults(func=_cmd_status)

    pg = sub.add_parser("gate", help="decide a waiting gate")
    pg.add_argument("gate")
    pg.add_argument("--run", required=True)
    pg.add_argument("--action", required=True,
                    choices=["accept", "edit", "override", "send_back", "approve", "reject"])
    pg.add_argument("--reason", default="")
    pg.add_argument("--target", default=None, help="send_back: explore | ideate | model")
    pg.add_argument("--message", default=None, help="send_back: what the agent should reconsider")
    pg.add_argument("--payload", default=None, help="JSON, e.g. '{\"champion\": \"c3\"}'")
    runs(pg)
    pg.set_defaults(func=_cmd_gate)

    pa = sub.add_parser("answer", help="answer (or accept as an assumption) a sponsor question")
    pa.add_argument("--run", required=True)
    pa.add_argument("--gap", required=True)
    pa.add_argument("--text", default="")
    pa.add_argument("--assume", action="store_true")
    runs(pa)
    pa.set_defaults(func=_cmd_answer)

    prt = sub.add_parser("retry", help="retry a failed step")
    prt.add_argument("step")
    prt.add_argument("--run", required=True)
    runs(prt)
    prt.set_defaults(func=_cmd_retry)

    pv = sub.add_parser("providers", help="list agent backends and availability")
    pv.set_defaults(func=_cmd_providers)

    prep = sub.add_parser("report", help="print a run summary")
    prep.add_argument("--run", required=True)
    runs(prep)
    prep.set_defaults(func=_cmd_report)

    pl = sub.add_parser("list-runs", help="list runs")
    runs(pl)
    pl.set_defaults(func=_cmd_list_runs)

    pag = sub.add_parser("agents", help="list the stages and the agent behind each")
    pag.set_defaults(func=_cmd_agents)
    return p


def quiet_numerics() -> None:
    """Entry points only: statsmodels' rank-deficiency and convergence chatter is reported in the
    diagnostics battery already; keep it off the console (a library never touches global filters)."""
    import warnings

    warnings.filterwarnings("ignore", module=r"statsmodels(\..*)?")


def main(argv: list[str] | None = None) -> int:
    quiet_numerics()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
