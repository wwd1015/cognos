"""COGNOS command-line interface.

  cognos ui         [--port 8050]                              # the Dash workbench
  cognos run        --config cognos.yaml [--interactive] [--provider P]
  cognos run        --data loans.csv --intent intent.md      # no profile: the Data Analyst
                                                              #   proposes the target
  cognos run        --data loans.csv financials.csv ...       # several sources: it also proposes
                                                              #   the join; you confirm at the gate
                    [--intent intent.md] [--support FILE ...]            # new model development
                    [--kind update --prior FILE ... | --prior-run ID]    # model update
  cognos intent-template [--kind new|update] [-o intent.md]   # the document a run starts from
  cognos demo       [--task commercial|cni|migration|...] [--interactive] [--provider P]
  cognos status     --run <run_id>                            # step table, gates, questions
  cognos gate       <gate> --run <run_id> --action accept|edit|override|send_back|approve|reject
  cognos answer     --run <run_id> --gap <id> --text "..."   # answer a sponsor question
  cognos retry      <step> --run <run_id> [--stuck]           # --stuck: take over a step a dead
                                                              #   process left running
  cognos run-stage  <stage> --config ... --run <run_id>       # one stage, individually invocable
  cognos plugins                                              # tools per stage + data sources
  cognos delete-run <run_id> [<run_id> ...] [--yes]           # remove runs for good (asks first)
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
                          #   (omit it to infer the task from the confirmed target)
mode: interactive         # interactive (pause at the review gates) | autonomous (agents' calls stand)

data:
  path: data.csv          # a file; or use source: for a database (cognos plugins lists connectors)
  format: csv
  # source:               # instead of path
  #   kind: snowflake     # file | sqlite | snowflake | a plugin's connector
  #   table: RISK.LOANS.ORIGINATIONS      # or query: "SELECT ... FROM ..."  (one read-only SELECT)
  #   limit: null
  #   options: {warehouse: ANALYTICS_WH, role: env:MY_ROLE}   # never a secret: account, user and
  #                       #   the password / key come from SNOWFLAKE_* environment variables
  # sources:              # or several inputs of any kind; the engine measures how they connect,
  #   - {name: loans, kind: snowflake, table: RISK.CREDIT.LOANS}     # the Data Analyst
  #   - {name: financials, kind: file, path: financials.csv}        # proposes the join and you
  # base: loans           #   confirm it at the data gate. base / join state it instead:
  # join:
  #   - {left: loans, left_on: obligor_id, right: financials, right_on: obligor_id, many: aggregate}
  target: target          # leave empty ("") and the Data Analyst proposes the dependent
                          #   variable from the business intent; you confirm it at the data gate
  features: []            # empty = all non-target/non-protected columns
  datetime_col: null      # set for timeseries / walk-forward
  protected_attributes: []  # excluded from features; used for fair-lending checks
  event_time_col: null    # 1-based period of the event (survival); unlocks the hazard families
  horizon_periods: null   # outcome window in periods; null = max observed event time

engagement:               # the development mode and the documents the run starts from
  kind: new               # new (complete new model development) | update (change an existing model)
  intent: null            # the business intent document / update request
                          #   (template: cognos intent-template [--kind update] -o intent.md)
  supporting: []          # background material the Intake Analyst may read
  prior_artifacts: []     # update: the existing model's white paper, code, validation reports
  prior_run: null         # update: an earlier COGNOS run of that model (id or directory)

analysis:                 # exploratory analysis the Data Analyst may request at explore
  rounds: 2               # request rounds before its recommendation
  max_requests: 8         # analyses per round (registered tools, or Python it writes)
  allow_code: true        # agent-written Python: run restricted, kept as an artifact, shown at
                          #   the data gate, printed in the white paper, re-run at validation
plugins: []               # modules exposing register(registry): more tools and data sources

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
  gates: [gate_intent, gate_data, gate_design, gate_candidates, gate_champion, gate_validation,
          gate_signoff]
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

# The development plan (ADR-0016). Anything left at auto is recommended from the shape of the
# data and the intended use, proposed by the Design Lead and confirmed at the design gate.
evaluation:               # how the data are split to build, compare and confirm models
  test: auto              # out_of_time | random | none: the pool is compared on this sample
  test_fraction: null     # null => search.holdout_fraction
  holdout_fraction: null  # sealed final sample, scored once for the champion (0 = none)
  validation: auto        # none | kfold | stratified_kfold | group_kfold | rolling_origin | sliding_window
  folds: null
  group_col: null         # group_kfold: rows of one entity stay in one fold
  horizon: null           # rolling_origin: periods per validation block
  rank_by: []             # e.g. [{metric: rmse, sample: validation}, {metric: r2, sample: in_sample}]
  pool_size: 4            # candidates carried to outcomes analysis

factors:                  # the factor library and the single-factor screen (rules run in this order)
  transforms: [auto]      # none | lag | diff | pct | yoy_diff | yoy_pct | ma | log | sqrt | square
  lags: [1, 2, 3, 4]
  max_missing: 0.2
  max_p_value: 0.10
  enforce_sign: true
  expected_signs: {}      # {column: "+" | "-"}
  best_per_source: 2      # transformations kept per explored column
  max_correlation: 0.8
  top_n: 12
  force_in: []
  force_out: []

selection:                # from single factors to multi-factor models
  methods: [auto]         # grid | forward | backward | stepwise | best_subset (one or several)
  criterion: auto         # validation | in_sample | aic | bic | adj_r2
  max_factors: 4
  must_include: []        # factors every model carries
  max_p_value: null       # every coefficient of an accepted model
  enforce_sign: true
  max_correlation: 0.8

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
  regimes: [SR26-2, NIST-AI-RMF]
  risk_tier: medium       # low | medium | high
  fair_lending: false
  jurisdictions: [US]

stages:
  enabled: [intake, explore, ideate, model, backtest, validate, comply, document, review]
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

    if gate == "gate_source":
        return _data_prompt(run_id, root)
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
    for g in st.open_gaps()[:8]:
        print(f"  ? [{g.id}] {g.question}")
    if gate == "gate_intent" and sys.stdin.isatty() and _interview(run_id, st, root):
        return True  # answers recorded: intake re-reads the brief, then this gate re-opens
    if not sys.stdin.isatty():
        if res is not None and res.verdict.value == "BLOCK":
            print("  (non-interactive stdin, BLOCK) -> left waiting; resolve with `cognos gate`.")
            return False
        action = "approve" if gate == "gate_signoff" else "accept"
        print(f"  (non-interactive stdin) -> {action}")
        try:
            service.submit_gate(run_id, gate, action, reason="non-interactive stdin", root=root,
                                background=False)
        except GateError as exc:
            print(f"  refused: {exc}")
            return False
        return True
    choice = input("  [a]ccept  [s]end back  [q]uit (leave waiting) > ").strip().lower()
    try:
        if choice.startswith("a"):
            reason = input("  reason (optional) > ").strip()
            action = "approve" if gate == "gate_signoff" else "accept"
            service.submit_gate(run_id, gate, action, reason=reason, root=root, background=False)
            return True
        if choice.startswith("s"):
            target = stage if stage in ("intake", "explore", "ideate", "model") else (
                input("  send back to [explore/ideate/model] > ").strip() or "model")
            msg = input("  what should the agent reconsider? > ").strip()
            service.submit_gate(run_id, gate, "send_back", {"target": target, "message": msg},
                                reason=msg, root=root, background=False)
            return True
    except GateError as exc:
        print(f"  refused: {exc}")
        return True
    return False


def _data_prompt(run_id: str, root) -> bool:
    """The run waits for its data (gate_source). In a terminal, choose catalog tables and add
    files."""
    from . import service
    from .engine import GateError

    info = service.data_info(run_id, root, with_catalog=True)
    print("\n" + "=" * 72)
    print("GATE gate_source — Provide the data")
    if info["guide"]:
        print("  What this kind of model is usually built on:")
        print("\n".join(f"    {line}" for line in info["guide"].strip().splitlines()))
    checked = info.get("preselected") or []
    for t in (info.get("catalog") or {}).get("tables", []):
        print(f"  [{'x' if t['id'] in checked else ' '}] {t['id']:<22} {t.get('title', '')}")
    if not sys.stdin.isatty():
        print(f"  (non-interactive stdin) -> left waiting; give it with "
              f"`cognos data --run {run_id} --catalog` (the checked tables) or --file PATH.")
        return False
    picked = input("  catalog tables, separated by spaces (Enter: the checked ones, - for "
                   "none) > ").split()
    picked = [] if picked == ["-"] else (picked or checked)
    files = input("  data files to add, separated by spaces (Enter: none) > ").split()
    if not picked and not files:
        return False
    try:
        service.submit_gate(run_id, "gate_source", "accept",
                            service.data_payload(picked, files, root=root), root=root,
                            background=False)
    except GateError as exc:
        print(f"  refused: {exc}")
    return True


def _interview(run_id: str, st, root) -> bool:
    """The Intake Analyst's interview in the terminal: one prompt per open question (Enter skips).
    True when at least one answer was recorded."""
    from . import service
    from .engine import GateError

    gaps = [g for g in st.open_gaps() if g.stage == "intake"]
    if not gaps:
        return False
    print("  The Intake Analyst needs the sponsor's answers (Enter skips a question):")
    answers = {}
    for g in gaps:
        text = input(f"  [{g.id}] {g.question}\n    > ").strip()
        if text:
            answers[g.id] = text
    if not answers:
        return False
    try:
        service.submit_gate(run_id, "gate_intent", "edit", {"answers": answers}, root=root,
                            background=False)
    except GateError as exc:
        print(f"  refused: {exc}")
        return False
    return True


def _engagement_args(args) -> dict:
    """The engagement the command line asks for, on top of the profile's."""
    kind = args.kind or ("update" if (args.prior or args.prior_run) else None)
    return {"kind": kind, "intent": args.intent, "supporting": args.support,
            "prior_artifacts": args.prior, "prior_run": args.prior_run}


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


def _cmd_intent_template(args) -> int:
    from . import service

    text = service.intent_template(args.kind, args.name)
    if not args.output:
        print(text)
        return 0
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite).")
        return 1
    out.write_text(text, encoding="utf-8")
    print(f"Wrote the {'model update request' if args.kind == 'update' else 'business intent'} "
          f"template to {out}. Fill it in and start a run with --intent {out}.")
    return 0


def _cmd_plugins(args) -> int:
    from . import service

    info = service.plugin_list(_load_config(args.config).plugins if args.config else ())
    print("Tools (the agent of each listed stage may request these):")
    for t in info["tools"]:
        params = f"({', '.join(t['params'])})" if t["params"] else "()"
        print(f"  {t['name']}{params}  [{t['origin']}]  stages: {', '.join(t['stages'])}"
              f"\n      {t['description']}"
              + ("" if t["available"] else f"\n      unavailable: {t['note']}"))
    print("Data sources (data.source.kind):")
    for s in info["sources"]:
        print(f"  {s['kind']:<10} {s['label']}  [{s['origin']}]"
              + ("" if s["available"] else f"\n      unavailable: {s['note']}"))
    print(f"Plugins loaded: {', '.join(info['plugins']) or 'none'}")
    for p in info["problems"]:
        print(f"  ! {p['plugin']}: {p['error']}")
    return 0


def _cmd_explain(args) -> int:
    from .agents import providers

    cfg = _load_config(args.config)
    print(f"COGNOS plan for project '{cfg.name}'")
    eng = cfg.engagement
    print(f"  development mode: {eng.kind}  intent document: {eng.intent or 'none'}"
          + (f"  prior model: {len(eng.prior_artifacts)} artifact(s)"
             + (f" + run {eng.prior_run}" if eng.prior_run else "") if eng.kind == "update" else ""))
    if cfg.task is None or not cfg.data.target:
        print(f"  target={cfg.data.target or 'proposed by the Data Analyst, confirmed at the data gate'}"
              f"  task={cfg.task.value if cfg.task else 'inferred from the target'}  mode={cfg.mode.value}")
    else:
        print(f"  task={cfg.task.value}  mode={cfg.mode.value}  metric={cfg.metric.name} "
              f"({cfg.metric.direction.value})")
    print(f"  target={cfg.data.target or '(open)'}  holdout={cfg.search.holdout_fraction}  budget={cfg.search.max_candidates} candidates")
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


def _cmd_data(args) -> int:
    from . import service
    from .engine import GateError

    lines = list(args.sql or [])
    if args.table or args.query:
        lines.append(f"snowflake = {args.query or args.table}")
    try:
        catalog = args.catalog
        if catalog == []:  # --catalog alone: what the form starts with checked
            catalog = service.data_info(args.run, args.runs_dir, with_catalog=True).get(
                "preselected", [])
        if catalog is None and not args.file and not lines:
            print("refused: give --catalog TABLE..., --file FILE..., or --sql / --table / --query")
            return 1
        payload = service.data_payload(catalog or [], args.file or [], "\n".join(lines),
                                       datetime_col=args.date_col, target=args.target,
                                       root=args.runs_dir)
        service.submit_gate(args.run, "gate_source", "accept", payload, root=args.runs_dir,
                            background=False)
    except (GateError, KeyError) as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_catalog(args) -> int:
    from . import catalog

    if args.init:
        try:
            path = catalog.init(args.runs_dir)
        except FileExistsError as exc:
            print(f"refused: {exc}")
            return 1
        print(f"wrote {path}, a copy of the default catalog: edit it to list your team's tables")
        return 0
    cat = catalog.load(args.runs_dir)
    path, origin = catalog.in_force(args.runs_dir)
    print(cat.title or "Data catalog")
    if origin == "default":
        print(f"  the default catalog shipped with COGNOS ({path})\n  to use your team's tables, "
              f"write {catalog.team_file(args.runs_dir)} (cognos catalog --init)")
    else:
        print(f"  your team's catalog ({path})")
    for t in cat.tables:
        mark = "*" if t.id in cat.defaults else " "
        print(f"\n{mark} {t.id:<22} {t.title}")
        print(f"  {'':<22} {catalog.summary_line({**t.model_dump(), 'summary': catalog.measure(t, args.runs_dir, fetch=args.measure)})}")
        if t.description:
            print(f"  {'':<22} {t.description}")
    if cat.links:
        print("\nDocumented links:")
        for lk in cat.links:
            on = ", ".join(f"{a} = {b}" for a, b in zip(lk.left_on, lk.right_on, strict=False))
            print(f"  {lk.left} -> {lk.right} on {on}" + (f"  ({lk.note})" if lk.note else ""))
    print("\n* preselected for a run started from scratch")
    return 0


def _cmd_templates(args) -> int:
    from . import service

    for t in service.list_templates(args.runs_dir):
        saved = f"  saved by {t['saved_by']} from {t['from_run']}" if t["origin"] == "team" else ""
        print(f"{t['ref']:<40} {t['title']}{saved}")
        print(f"{'':<40} {t['description']}")
    return 0


def _cmd_save_template(args) -> int:
    from . import service
    from .engine import GateError

    try:
        ref = service.save_template(args.run, args.title, args.description, root=args.runs_dir)
    except (GateError, KeyError) as exc:
        print(f"refused: {exc}")
        return 1
    print(f"saved: {ref}  (start from it with `cognos run --template {ref}`)")
    return 0


def _cmd_run(args) -> int:
    from . import service

    try:
        if not args.config and not args.data and not args.template:
            print("refused: give --template or --config, or --data to start from a data file "
                  "alone")
            return 1
        if args.template and args.config:
            print("refused: give --template or --config, not both")
            return 1
        files = list(args.data or [])
        if len(files) > 1:  # several files: joined at the data stage (linking.py)
            from . import linking

            spec: dict = linking.named_sources(files)
            keys = {"path": None, "source": None, **spec}
        elif files:
            spec = {"kind": "file", "path": files[0]}
            keys = {"path": None, "source": spec, "sources": []}
        if args.template:  # without --data, the run asks for it after the intent
            base = service.template_config(None if args.template == "scratch" else args.template,
                                           args.name or "", root=args.runs_dir)
        else:
            base = (_load_config(args.config) if args.config else
                    service.config_from_data(Path(files[0]).stem, spec, root=args.runs_dir))
        if (args.config or args.template) and files:
            base = service.load_config({**base.model_dump(mode="json"), "data": {
                **base.data.model_dump(mode="json"), **keys}})
        cfg = service.with_engagement(base, _engagement_args(args))
        if args.model:
            cfg = cfg.model_copy(update={"agents": cfg.agents.model_copy(
                update={"model": args.model})})
    except (ValueError, KeyError) as exc:
        print(f"refused: {exc}")
        return 1
    mode = "interactive" if args.interactive else cfg.mode.value
    run_id = args.run_id
    if run_id and (service.runs_root(args.runs_dir) / run_id / "state.json").exists():
        pass  # resume an existing run
    else:
        from .engine import Engine

        try:
            eng = Engine(cfg, run_id=run_id, runs_root=service.runs_root(args.runs_dir),
                         provider=args.provider, mode=mode)
        except (FileNotFoundError, ValueError) as exc:  # a document the engagement names is missing
            print(f"refused: {exc}")
            return 1
        run_id = eng.run_id
    return _drive(run_id, args.runs_dir, interactive=mode == "interactive")


def _cmd_demo(args) -> int:
    from . import service

    cfg = service.demo_config(args.task, args.runs_dir)
    mode = "interactive" if args.interactive else "autonomous"
    run_id = service.create_run(cfg, mode=mode, provider=args.provider, model=args.model,
                                root=args.runs_dir)
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
    from .engine.process import next_action
    nxt = next_action(st)
    print(f"  next: {nxt['text']}")
    if st.package is not None:
        print(f"  package v{st.package.version} {st.package.status} {st.package.digest[:12]}")
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
                            root=args.runs_dir, background=False, seat=args.seat)
    except (GateError, KeyError) as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_answer(args) -> int:
    from . import service
    from .engine import GateError

    try:
        service.answer_gap(args.run, args.gap, args.text or "", assume=args.assume,
                           limitation=args.limitation, root=args.runs_dir, background=False)
    except GateError as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_retry(args) -> int:
    from . import service
    from .engine import GateError

    try:
        service.retry(args.run, args.step, args.runs_dir, background=False, stuck=args.stuck)
    except GateError as exc:
        print(f"refused: {exc}")
        return 1
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_providers(args) -> int:
    from .agents import providers

    for p in providers.public_list():
        mark = "available" if p["available"] else "-"
        print(f"  {p['id']:<11} {mark:<10} {p['kind']:<14} {p.get('model') or '':<22} {p['label']}")
        if len(p["models"]) > 1:
            print(f"  {'':<11} {'':<10} {'':<14} also: {', '.join(p['models'][1:])}")
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


def _cmd_delete_run(args) -> int:
    from . import service

    try:
        rows = service.deletion_preview(args.runs, args.runs_dir)
    except (KeyError, ValueError) as exc:
        print(exc.args[0] if exc.args else exc)
        return 1
    for r in rows:
        signed = f"  SIGNED PACKAGES: {r['packages']}" if r["packages"] else ""
        print(f"{r['run_id']}  project={r['project']}  status={r['status']}  "
              f"{r['size_bytes'] / 1e6:.1f} MB{signed}")
    print("This removes each run's results, documents, decisions and audit log for good. "
          "Export first (cognos export <run_id>) to keep a copy.")
    if any(r.get("keeps_line") for r in rows):
        print("A run of a model with trial runs keeps one line in the model's record.")
    for r in rows:
        if r["blocked"]:
            print(f"Cannot be deleted now: {r['run_id']} ({r['blocked']}).")
    if not args.yes and input(f"Delete {len(rows)} run(s)? Type 'delete' to confirm: ") != "delete":
        print("Nothing deleted.")
        return 1
    try:
        service.delete_runs(args.runs, args.runs_dir)
    except RuntimeError as exc:
        print(exc)
        return 1
    print(f"Deleted {len(rows)} run(s).")
    return 0


def _cmd_trial(args) -> int:
    from . import service
    from .engine import GateError

    try:
        new_id = service.trial_run(args.run, args.gate, args.note, root=args.runs_dir)
    except (GateError, FileNotFoundError) as exc:
        print(exc)
        return 1
    print(f"Trial run {new_id}: a copy of {args.run} waiting at {args.gate}.")
    print("Decide the gate differently (cognos gate ...). It stops before validation; "
          f"`cognos promote --run {new_id}` makes it the main run.")
    return 0


def _cmd_promote(args) -> int:
    from . import service
    from .engine import GateError

    try:
        service.promote_trial(args.run, args.reason, root=args.runs_dir, background=False)
    except (GateError, FileNotFoundError) as exc:
        print(exc)
        return 1
    print(f"{args.run} is now the main run of its model. The run it replaced stays on record.")
    return _drive(args.run, args.runs_dir, interactive=False)


def _cmd_trials(args) -> int:
    from . import service, trials

    try:
        rec = service.trial_record(args.run, args.runs_dir)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    for r in rec["lines"]:
        print("  " + trials.describe(r))
    opened = len(rec["holdout_scored"])
    print(f"{rec['n_runs']} run(s) of this model; the test sample was used "
          f"{rec['test_evaluations']} time(s) across them; sealed holdout "
          + (f"scored for {opened} champion(s)." if opened else "not opened."))
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


def _cmd_compare(args) -> int:
    from . import service
    from .engine.graph import LABELS

    try:
        c = service.compare(args.run_a, args.run_b, args.runs_dir)
    except FileNotFoundError as exc:
        print(f"No run found: {exc.filename}")
        return 1
    a, b = c["a"], c["b"]

    def num(v) -> str:
        return "—" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"A  {a['run_id']}  {a['project']}  {a['status']}  champion={a['champion']}")
    print(f"B  {b['run_id']}  {b['project']}  {b['status']}  champion={b['champion']}")
    for line in c["caveats"]:
        print(f"  note: {line}")
    changed = [r for r in c["metrics"] if r["delta"] != 0]
    print(f"\nResults (B relative to A; {len(c['metrics']) - len(changed)} metric(s) unchanged)")
    for r in changed:
        delta = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        verdict = {"a": "A better", "b": "B better"}.get(r["better"], "")
        print(f"  {LABELS[r['stage']]:<26} {r['label']:<34} {num(r['a']):>12} {num(r['b']):>12} {delta:>10}  {verdict}")
    for v in c["verdicts"]:
        if v["a"] != v["b"]:
            print(f"  verdict {v['label']}: {v['a']} -> {v['b']}")
    decided = [d for d in c["decisions"] if d["different"]]
    if decided or c["overrides"] or c["config"]:
        print("\nDecided differently")
    for d in decided:
        def say(x):
            return "—" if x is None else f"{x['action']} ({x['actor']})" + (f": {x['reason']}" if x["reason"] else "")
        print(f"  {d['label']}: A {say(d['a'])} | B {say(d['b'])}")
    for o in c["overrides"]:
        print(f"  {o['what']}: A {o['a']} | B {o['b']}")
    for k in c["config"]:
        print(f"  config {k['key']}: A {k['a']} | B {k['b']}")
    f = c["findings"]
    if f["new"] or f["gone"]:
        print(f"\nFindings: {len(f['new'])} only in B, {len(f['gone'])} only in A")
        for tag, rows in (("+", f["new"]), ("-", f["gone"])):
            for r in rows[:10]:
                print(f"  {tag} {r['severity']:<8} {LABELS[r['stage']]}: {r['message']}")
    return 0


def _cmd_export(args) -> int:
    from . import service

    try:
        path = service.export_run(args.run, args.out, args.runs_dir, agent_io=args.with_agent_io)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    print(f"Exported {path}  ({path.stat().st_size / 1024:.0f} KB; the data directory is never included)")
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
    pr.add_argument("--config", default=None)
    pr.add_argument("--data", default=None, nargs="+", metavar="FILE",
                    help="a data file (.csv, .parquet, .xlsx); without --config the Data Analyst "
                         "proposes the target. Several files: it also proposes how they join")
    pr.add_argument("--template", default=None,
                    help="start from a template (see `cognos templates`), or 'scratch'; the "
                         "run asks for the data after the intent unless --data gives it")
    pr.add_argument("--name", default=None, help="with --template: the model's name")
    pr.add_argument("--interactive", action="store_true", help="review each gate in the terminal")
    pr.add_argument("--provider", default=None, help="agent backend (see `cognos providers`)")
    pr.add_argument("--model", default=None,
                    help="the provider's model for this run's agents, e.g. sonnet")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--kind", choices=["new", "update"], default=None,
                    help="development mode: a new model, or an update of an existing one")
    pr.add_argument("--intent", default=None,
                    help="the business intent document / update request (see intent-template)")
    pr.add_argument("--support", action="append", default=[], metavar="FILE",
                    help="background document for the Intake Analyst (repeatable)")
    pr.add_argument("--prior", action="append", default=[], metavar="FILE",
                    help="update: an artifact of the existing model, e.g. its white paper or "
                         "code (repeatable)")
    pr.add_argument("--prior-run", default=None,
                    help="update: an earlier COGNOS run of the existing model (id or directory)")
    runs(pr)
    pr.set_defaults(func=_cmd_run)

    pt = sub.add_parser("intent-template",
                        help="write the business intent document (or update request) to fill in")
    pt.add_argument("--kind", choices=["new", "update"], default="new")
    pt.add_argument("--name", default="", help="the model's name, for the title")
    pt.add_argument("-o", "--output", default=None, help="write to a file instead of stdout")
    pt.add_argument("--force", action="store_true")
    pt.set_defaults(func=_cmd_intent_template)

    pd = sub.add_parser("demo", help="run end to end on synthetic data")
    pd.add_argument("--task", default="commercial",
                    choices=["regression", "classification", "timeseries", "credit", "commercial",
                             "cni", "migration", "loss_forecast", "linked"])
    pd.add_argument("--interactive", action="store_true")
    pd.add_argument("--provider", default="heuristic")
    pd.add_argument("--model", default=None, help="the provider's model, e.g. sonnet")
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
    pg.add_argument("--target", default=None, help="send_back: intake | explore | ideate | model")
    pg.add_argument("--message", default=None, help="send_back: what the agent should reconsider")
    pg.add_argument("--payload", default=None, help="JSON, e.g. '{\"pool\": [\"c1a2b3c\", \"c4d5e6f\"]}' or "
                                                  "'{\"champion\": \"c1a2b3c\"}'")
    pg.add_argument("--seat", default=None,
                    choices=["developer", "reviewer", "approver"],
                    help="who is acting; omit to act as the seat that owns the gate")
    runs(pg)
    pg.set_defaults(func=_cmd_gate)

    ptr = sub.add_parser("trial", help="copy a run at a developer gate to decide it "
                                       "differently (the run itself is not changed)")
    ptr.add_argument("--run", required=True)
    ptr.add_argument("--gate", required=True,
                     choices=["gate_intent", "gate_source", "gate_data", "gate_design",
                              "gate_candidates", "gate_champion"])
    ptr.add_argument("--note", required=True, help="what the trial run is for (kept on record)")
    runs(ptr)
    ptr.set_defaults(func=_cmd_trial)

    ppm = sub.add_parser("promote", help="make a trial run the main run of its model")
    ppm.add_argument("--run", required=True)
    ppm.add_argument("--reason", default="")
    runs(ppm)
    ppm.set_defaults(func=_cmd_promote)

    ptl = sub.add_parser("trials", help="list every run of a run's model, deleted trial runs "
                                        "included")
    ptl.add_argument("--run", required=True)
    runs(ptl)
    ptl.set_defaults(func=_cmd_trials)

    pdt = sub.add_parser("data", help="give a run its data (it asks once the intent is "
                                      "confirmed)")
    pdt.add_argument("--run", required=True)
    pdt.add_argument("--catalog", nargs="*", default=None, metavar="TABLE",
                     help="tables from the data catalog (the first is the proposed base "
                          "table); alone, the ones the template starts with")
    pdt.add_argument("--file", nargs="+", default=None, metavar="FILE",
                     help="data files to add; several sources are joined at the data stage")
    pdt.add_argument("--sql", action="append", default=None, metavar="'NAME = TABLE|SELECT'",
                     help="a Snowflake table or read-only query to add (repeatable)")
    pdt.add_argument("--table", default=None, help="a Snowflake table")
    pdt.add_argument("--query", default=None, help="a read-only Snowflake query")
    pdt.add_argument("--target", default=None,
                     help="the dependent variable (else the Data Analyst proposes it)")
    pdt.add_argument("--date-col", default=None, help="the date column")
    runs(pdt)
    pdt.set_defaults(func=_cmd_data)

    pcat = sub.add_parser("catalog", help="list the data catalog: its tables, what each "
                                          "holds and how they join")
    pcat.add_argument("--measure", action="store_true",
                      help="also read database tables to measure them (files always are)")
    pcat.add_argument("--init", action="store_true",
                      help="write the runs folder's own catalog, a copy of the default, to edit")
    runs(pcat)
    pcat.set_defaults(func=_cmd_catalog)

    ptp = sub.add_parser("templates", help="list the built-in and team templates")
    runs(ptp)
    ptp.set_defaults(func=_cmd_templates)

    psv = sub.add_parser("save-template", help="keep a finished run's setup as a team template")
    psv.add_argument("--run", required=True)
    psv.add_argument("--title", required=True)
    psv.add_argument("--description", default="")
    runs(psv)
    psv.set_defaults(func=_cmd_save_template)

    pa = sub.add_parser("answer", help="answer (or accept as an assumption) a sponsor question")
    pa.add_argument("--run", required=True)
    pa.add_argument("--gap", required=True)
    pa.add_argument("--text", default="")
    pa.add_argument("--assume", action="store_true",
                    help="record it as an assumption instead of answering")
    pa.add_argument("--limitation", action="store_true",
                    help="record it as a model limitation instead of answering")
    runs(pa)
    pa.set_defaults(func=_cmd_answer)

    prt = sub.add_parser("retry", help="retry a failed step")
    prt.add_argument("step")
    prt.add_argument("--run", required=True)
    prt.add_argument("--stuck", action="store_true",
                     help="take over a step left 'running' by a process that died")
    runs(prt)
    prt.set_defaults(func=_cmd_retry)

    pp = sub.add_parser("plugins", help="list analysis tools, data sources and loaded plugins")
    pp.add_argument("--config", default=None, help="also load the profile's plugins: list")
    pp.set_defaults(func=_cmd_plugins)

    pv = sub.add_parser("providers", help="list agent backends and availability")
    pv.set_defaults(func=_cmd_providers)

    prep = sub.add_parser("report", help="print a run summary")
    prep.add_argument("--run", required=True)
    runs(prep)
    prep.set_defaults(func=_cmd_report)

    pc = sub.add_parser("compare", help="what one run decided differently from another, and what it changed")
    pc.add_argument("run_a")
    pc.add_argument("run_b")
    runs(pc)
    pc.set_defaults(func=_cmd_compare)

    pe = sub.add_parser("export", help="one zip of a run's documents, results, decisions and audit log")
    pe.add_argument("run")
    pe.add_argument("-o", "--out", default=None, help="a directory or a .zip path (default: <runs>/_exports/)")
    pe.add_argument("--with-agent-io", action="store_true", help="include every agent prompt and raw output")
    runs(pe)
    pe.set_defaults(func=_cmd_export)

    pl = sub.add_parser("list-runs", help="list runs")
    runs(pl)
    pl.set_defaults(func=_cmd_list_runs)

    pd_ = sub.add_parser("delete-run", help="delete runs for good (asks for confirmation)")
    pd_.add_argument("runs", nargs="+", metavar="run_id")
    pd_.add_argument("--yes", action="store_true", help="do not ask (for scripts)")
    runs(pd_)
    pd_.set_defaults(func=_cmd_delete_run)

    pag = sub.add_parser("agents", help="list the stages and the agent behind each")
    pag.set_defaults(func=_cmd_agents)
    return p


def quiet_numerics() -> None:
    """Entry points only: statsmodels' rank-deficiency and convergence chatter is reported in the
    diagnostics battery already; keep it off the console (a library never touches global filters)."""
    import warnings

    warnings.filterwarnings("ignore", module=r"statsmodels(\..*)?")


def utf8_console() -> None:
    """Entry points only: Windows consoles and redirected pipes default to a legacy codepage that
    cannot print the ≈ / ↔ / Δ in stage summaries; never crash on output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    quiet_numerics()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
