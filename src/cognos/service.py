"""The service layer — the single API boundary the CLI and the UI use.

Everything a front end needs: create a run (from a template, a profile or a synthetic demo
preset, as a new model development or a model update with its intent and prior-model
documents), give it its data once the intent is confirmed, keep a finished run as a template, advance it
(synchronously or in the background), submit gate decisions, answer questions, retry, and read
state, results, events and the agent audit. One :class:`~cognos.engine.Engine` per run is cached
in-process so a background run and the UI callbacks share its in-flight bookkeeping. A REST adapter,
if ever needed, is a thin wrapper over these functions.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from .artifacts import StageResult
from .config import CognosConfig
from .engine import Engine, RunState, events
from .engine.gates import GateError
from .engine.graph import LABELS, STAGES

_engines: dict[str, Engine] = {}
_guard = threading.Lock()

DEMO_PRESETS: dict[str, dict[str, Any]] = {
    "regression": dict(task="regression", target="target", metric="rmse"),
    "classification": dict(task="classification", target="target", metric="roc_auc"),
    "timeseries": dict(task="timeseries", target="target", metric="rmse", datetime_col="date"),
    "credit": dict(task="classification", target="default", metric="roc_auc",
                   protected=["group"], fair_lending=True),
    "commercial": dict(task="classification", target="default", metric="roc_auc",
                       datetime_col="vintage"),
    "cni": dict(task="classification", target="default", metric="roc_auc",
                datetime_col="vintage", drop=["obligor_id", "dpd_at_outcome"],
                event_time_col="default_quarter", horizon_periods=4,
                portfolio={"enabled": True, "lgd": 0.45, "n_sims": 10000},
                stress={"enabled": True, "scenarios": [
                    {"name": "adverse",
                     "shocks": {"unemployment_rate": {"add": 3.0}, "gdp_growth": {"add": -2.0}}},
                    {"name": "severely_adverse",
                     "shocks": {"unemployment_rate": {"add": 6.0}, "gdp_growth": {"set": -4.0}}},
                ]},
                objective="Rank commercial and industrial middle-market borrowers by their "
                          "risk of default over the next 12 months, to support origination "
                          "underwriting, and size the portfolio's loss under stress.",
                factors={"expected_signs": {
                    "debt_to_ebitda": "+", "utilization_rate": "+", "unemployment_rate": "+",
                    "interest_coverage": "-", "current_ratio": "-", "operating_margin": "-",
                    "log_total_assets": "-", "collateral_coverage": "-", "gdp_growth": "-",
                    "revenue_growth": "-"}},
                design={"use_case": "origination underwriting", "horizon": "12-month PD",
                        "default_definition": "90+ DPD or nonaccrual within 12 months",
                        "segment": "C&I middle-market", "interpretability": "required"}),
    "migration": dict(task="classification", target="default", metric="roc_auc",
                      datetime_col="asof", drop=["obligor_id", "regime", "ead"],
                      migration={"enabled": True, "rating_col": "rating",
                                 "next_rating_col": "next_rating",
                                 "rating_scale": ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"],
                                 "condition_col": "regime", "lgd": 0.40, "ead_column": "ead"},
                      portfolio={"enabled": True, "lgd": 0.40, "ead_column": "ead",
                                 "n_sims": 10000},
                      stress={"enabled": True, "scenarios": [
                          {"name": "adverse", "shocks": {"unemployment_rate": {"add": 3.0}}}]},
                      design={"use_case": "loss forecasting (CECL / stress testing)",
                              "horizon": "1-year default, multi-year via matrix powers",
                              "default_definition": "agency default state D "
                                                    "(payment default / bankruptcy)",
                              "segment": "large corporate (agency-rated universe)",
                              "interpretability": "required"}),
}
# A quarterly loss rate on macro drivers: the series is short and the drivers act through lags
# and changes, so this is where the development plan shows — a rolling forecast origin, a
# factor library cut by the screen, forward selection checked against best subset, and a pool
# of candidates told apart by how they forecast the held-back quarters and the scenarios.
DEMO_PRESETS["loss_forecast"] = dict(
    task="timeseries", target="nco_rate", metric="rmse", datetime_col="date",
    factors={"expected_signs": {"unemployment_rate": "+", "bbb_spread": "+",
                                "gdp_growth": "-", "hpi": "-", "cre_price_index": "-"}},
    selection={"max_factors": 3},
    stress={"enabled": True, "scenarios": [
        {"name": "adverse",
         "shocks": {"unemployment_rate": {"add": 2.0}, "bbb_spread": {"add": 1.0},
                    "hpi": {"mul": 0.93}}},
        {"name": "severely_adverse",
         "shocks": {"unemployment_rate": {"add": 4.5}, "bbb_spread": {"add": 2.5},
                    "hpi": {"mul": 0.82}}}]},
    design={"use_case": "loss forecasting (CECL / stress testing)",
            "horizon": "9-quarter forecast of the quarterly net charge-off rate",
            "default_definition": "net charge-offs over average loans, annualized, in percent",
            "segment": "commercial real estate portfolio",
            "interpretability": "required"})
# The C&I portfolio again, arriving as three sources of two kinds (two CSV files and a SQLite
# table): the Data Analyst proposes how they join and the developer confirms it.
DEMO_PRESETS["linked"] = {**DEMO_PRESETS["cni"], "generator": "cni", "linked": True}
DEMO_LABELS = {
    "cni": "C&I PD scorecard (obligors by vintage: stepwise logit, sealed holdout, hazard "
           "challengers, stress)",
    "loss_forecast": "Loss forecasting (quarterly series: lags and changes, forward selection, "
                     "rolling origin)",
    "linked": "C&I portfolio from three linked sources (the Data Analyst proposes the join)",
    "commercial": "Commercial PD (vintage panel, out-of-time)",
    "migration": "Rating migration (agency transition matrix, loss forecast)",
    "credit": "Consumer-style credit with fair-lending scan",
    "classification": "Generic classification",
    "regression": "Generic regression",
    "timeseries": "Time series",
}


def runs_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("COGNOS_RUNS_DIR") or "runs")


def demo_config(preset: str, root: str | Path | None = None, *, n: int | None = None,
                search_budget: int | None = None) -> CognosConfig:
    """A ready-to-run profile over freshly generated synthetic data."""
    from . import synth

    if preset not in DEMO_PRESETS:
        raise KeyError(f"unknown demo preset {preset!r}; choose from {sorted(DEMO_PRESETS)}")
    data_dir = runs_root(root) / "_demo_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p = DEMO_PRESETS[preset]
    gen = synth.GENERATORS[p.get("generator", preset)]
    df = gen(n=n) if n else gen()
    csv = data_dir / f"{preset}.csv"
    if p.get("linked"):
        data_spec = _linked_demo_sources(df, data_dir)
    else:
        df.to_csv(csv, index=False)
        data_spec = {"path": str(csv), "format": "csv"}
    # The demo sponsor's intent document: the preset's design brief on the template. Presets
    # without one leave those sections empty, so the interview has something to ask.
    from . import engagement as eg

    intent = data_dir / f"{preset}_intent.md"
    design = p.get("design", {})
    intent.write_text(eg.render_template("new", {
        "objective": p.get("objective") or
                     f"Demonstrate a COGNOS model development on synthetic data: "
                     f"{DEMO_LABELS.get(preset, preset)}.",
        **{k: design.get(k, "") for k in eg.CORE},
        "interpretability": design.get("interpretability", ""),
        "data_sources": f"Synthetic data from cognos.synth ({preset}).",
    }, name=f"demo_{preset}"), encoding="utf-8")
    raw: dict[str, Any] = {
        "name": f"demo_{preset}",
        "description": f"COGNOS synthetic {preset} demo — {DEMO_LABELS.get(preset, preset)}",
        "task": p["task"],
        "data": {**data_spec, "target": p["target"],
                 "datetime_col": p.get("datetime_col"), "protected_attributes": p.get("protected", []),
                 "drop_columns": p.get("drop", []), "event_time_col": p.get("event_time_col"),
                 "horizon_periods": p.get("horizon_periods")},
        "engagement": {"kind": "new", "intent": str(intent)},
        "design": p.get("design", {}),
        "migration": p.get("migration", {}),
        "portfolio": p.get("portfolio", {}),
        "stress": p.get("stress", {}),
        "evaluation": p.get("evaluation", {}),
        "factors": p.get("factors", {}),
        "selection": p.get("selection", {}),
        "metric": {"name": p["metric"]},
        "compliance": {"fair_lending": p.get("fair_lending", False),
                       "jurisdictions": ["US"] if preset in ("commercial", "cni", "migration",
                                                               "loss_forecast")
                       else ["US", "EU"],
                       "risk_tier": "high" if preset in ("credit", "commercial", "cni", "migration",
                                                         "loss_forecast")
                       else "medium"},
        "runs_dir": str(runs_root(root)),
    }
    if search_budget:
        raw["search"] = {"max_candidates": search_budget}
    return CognosConfig.from_dict(raw)


def _linked_demo_sources(df: Any, data_dir: Path) -> dict[str, Any]:
    """Split the C&I demo data the way it would arrive in practice: the loan book (a CSV with
    the outcome), obligor financials (a database table, a few obligors missing) and a macro
    series by vintage (a CSV)."""
    import sqlite3

    firm = ["debt_to_ebitda", "interest_coverage", "current_ratio", "operating_margin",
            "revenue_growth", "log_total_assets", "sector", "region"]
    macro = ["unemployment_rate", "gdp_growth"]
    loans = data_dir / "linked_loans.csv"
    df.drop(columns=firm + macro).to_csv(loans, index=False)
    series = data_dir / "linked_macro.csv"
    df.groupby("vintage", as_index=False)[macro].mean().to_csv(series, index=False)
    db = data_dir / "linked_financials.db"
    db.unlink(missing_ok=True)
    financials = df[["obligor_id", *firm]].iloc[: int(len(df) * 0.97)]
    con = sqlite3.connect(db)
    try:
        financials.to_sql("obligor_financials", con, index=False)
    finally:
        con.close()
    return {"sources": [
        {"name": "loans", "kind": "file", "path": str(loans)},
        {"name": "financials", "kind": "sqlite", "path": str(db), "table": "obligor_financials"},
        {"name": "macro", "kind": "file", "path": str(series)},
    ]}


def load_config(source: CognosConfig | str | Path | dict) -> CognosConfig:
    if isinstance(source, CognosConfig):
        return source
    if isinstance(source, dict):
        return CognosConfig.from_dict(source)
    return CognosConfig.from_yaml(source)


# --- data sources and plugins --------------------------------------------------------------------
def plugin_list(modules: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Installed analysis tools, data sources (with availability) and plugin load problems."""
    from . import plugins

    return plugins.public_list(modules)


def config_from_data(name: str, source: dict[str, Any], *, description: str = "",
                     root: str | Path | None = None, **extra: Any) -> CognosConfig:
    """A profile for "here is my data": no target, no task. The Data Analyst proposes the
    dependent variable from the business intent and the model developer confirms it at the data
    gate. ``source`` is a ``data.source`` mapping (``{"kind": "file", "path": ...}``,
    ``{"kind": "snowflake", "table": ...}``)."""
    import re

    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name or "model").strip("_").lower() or "model"
    # {"sources": [...]}: several inputs, joined at the data stage (linking.py)
    data = {"sources": source["sources"]} if "sources" in source else {"source": source}
    return CognosConfig.from_dict({"name": slug, "description": description,
                                   "data": data, "runs_dir": str(runs_root(root)),
                                   **extra})


def preview_source(source: dict[str, Any], rows: int = 5) -> dict[str, Any]:
    """Columns and the first rows of a source, to check a connection before starting a run."""
    from . import datasources

    cfg = config_from_data("preview", {**source, "limit": rows})
    df, provenance = datasources.load(cfg)
    return {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            "rows": json.loads(df.head(rows).to_json(orient="records", date_format="iso")),
            "source": provenance}


# --- engagements: the development mode and its documents ---------------------------------------
def intent_template(kind: str = "new", name: str = "") -> str:
    """The business intent document (or model update request) to fill in before a run."""
    from . import engagement as eg

    return eg.render_template(kind, name=name)


def save_upload(filename: str, content: str | bytes, root: str | Path | None = None) -> str:
    """Keep an uploaded document under ``<runs>/_uploads/`` until a run copies it in. ``content``
    is bytes or a browser data URL (``data:...;base64,...``). Returns the saved path."""
    import base64
    import re
    import uuid

    if isinstance(content, str):
        content = base64.b64decode(content.split(",", 1)[1] if content.startswith("data:")
                                   else content)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename or "document").name).strip("._")
    dest = runs_root(root) / "_uploads" / uuid.uuid4().hex[:10] / (name or "document")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return str(dest)


def with_engagement(source: CognosConfig | str | Path | dict,
                    engagement: dict[str, Any] | None = None) -> CognosConfig:
    """The profile with its engagement replaced or extended: ``kind`` (new | update), ``intent``,
    ``supporting``, ``prior_artifacts``, ``prior_run``. Keys left out keep the profile's value."""
    cfg = load_config(source)
    given = {k: v for k, v in (engagement or {}).items() if v not in (None, [], "")}
    if not given:
        return cfg
    raw = cfg.model_dump(mode="json")
    raw["engagement"] = {**raw.get("engagement", {}), **given}
    if raw["engagement"].get("kind") != "update":  # a new development reads no prior model
        raw["engagement"].update(prior_artifacts=[], prior_run=None)
    return CognosConfig.from_dict(raw)


# --- runs ------------------------------------------------------------------------------------
def create_run(source: CognosConfig | str | Path | dict, *, mode: str = "interactive",
               provider: str | None = None, model: str | None = None,
               root: str | Path | None = None,
               engagement: dict[str, Any] | None = None) -> str:
    """``model`` picks one of the provider's models for the run's agents (``agents.model``)."""
    cfg = with_engagement(source, engagement)
    if model:
        cfg = cfg.model_copy(update={"agents": cfg.agents.model_copy(update={"model": model})})
    eng = Engine(cfg, runs_root=runs_root(root), provider=provider, mode=mode)
    with _guard:
        _engines[str(eng.run_dir.resolve())] = eng
    return eng.run_id


def engine(run_id: str, root: str | Path | None = None) -> Engine:
    run_dir = runs_root(root) / run_id
    key = str(run_dir.resolve())
    with _guard:
        eng = _engines.get(key)
        if eng is None:
            if not RunState.exists(run_dir):
                raise KeyError(f"no v1 run at {run_dir}")
            eng = Engine.load(run_dir)
            _engines[key] = eng
        return eng


def start(run_id: str, root: str | Path | None = None) -> None:
    """Advance the run in the background (returns immediately)."""
    engine(run_id, root).start()


def run_until_idle(run_id: str, root: str | Path | None = None) -> RunState:
    return engine(run_id, root).run_until_idle()


def submit_gate(run_id: str, gate: str, action: str, payload: dict | None = None,
                reason: str = "", *, root: str | Path | None = None,
                background: bool = True, seat: str | None = None,
                by: str | None = None) -> RunState:
    eng = engine(run_id, root)
    state = eng.submit_gate(gate, action, payload, reason, seat=seat, by=by)
    if background:
        eng.start()
    return state


def answer_gap(run_id: str, gap_id: str, answer: str, *, assume: bool = False,
               limitation: bool = False, root: str | Path | None = None,
               background: bool = True, seat: str | None = None,
               by: str | None = None) -> RunState:
    eng = engine(run_id, root)
    state = eng.answer_gap(gap_id, answer, assume=assume, limitation=limitation, seat=seat,
                           by=by)
    if background:
        eng.start()
    return state


def retry(run_id: str, step: str, root: str | Path | None = None,
          background: bool = True, *, stuck: bool = False) -> RunState:
    eng = engine(run_id, root)
    state = eng.retry(step, stuck=stuck)
    if background:
        eng.start()
    return state


def reopen(run_id: str, gate: str, root: str | Path | None = None,
           *, seat: str | None = None) -> RunState:
    return engine(run_id, root).reopen(gate, seat=seat)


# --- trial runs (ADR-0017) ---------------------------------------------------------------------
def _developer(seat: str | None, what: str) -> None:
    if seat not in (None, "", "developer"):
        from .engine.process import seat_label

        raise GateError(f"{what} is the model developer's. You are acting as the "
                        f"{seat_label(seat)}.")


def trial_gates(run_id: str, root: str | Path | None = None) -> list[str]:
    """The decisions of this run a trial run can be made from now."""
    from . import trials

    return trials.branch_gates(state(run_id, root))


def trial_run(run_id: str, gate: str, note: str, *, root: str | Path | None = None,
              seat: str | None = None, by: str | None = None) -> str:
    """Make a trial run: a copy of ``run_id`` that waits at ``gate`` to be decided
    differently. The run it was made from is not touched. Returns the trial run's id."""
    from . import trials

    _developer(seat, "A trial run")
    if is_busy(run_id, root):
        raise GateError("A step is running; wait for the run to pause.")
    by = by or whoami()
    try:
        new_id = trials.branch(runs_root(root), run_id, gate, note=note, by=by)
    except trials.TrialError as exc:
        raise GateError(str(exc)) from None
    eng = engine(new_id, root)
    with eng.lock:
        eng._save(eng.state)  # the run's status, by the engine's own rule
    events.publish(eng.run_dir, "run_created",
                   f"Trial run made from {run_id} by {by}, to decide "
                   f"“{LABELS[gate]}” differently: {note}", by=by)
    events.publish(runs_root(root) / run_id, "trial",
                   f"Trial run {new_id} made from “{LABELS[gate]}” by {by}: {note}", by=by)
    return new_id


def promote_trial(run_id: str, note: str = "", *, root: str | Path | None = None,
                  seat: str | None = None, by: str | None = None,
                  background: bool = True) -> RunState:
    """Make a trial run the main run of its model. It then goes on to validation; the run it
    replaces stays on record as a trial run."""
    from . import trials

    _developer(seat, "Choosing the main run")
    by = by or whoami()
    base = runs_root(root)
    main = next((m.run_id for m in trials.members(base, trials.model_of(state(run_id, root)))
                 if not m.is_trial), None)
    if is_busy(run_id, root) or (main and is_busy(main, root)):
        raise GateError("A step is running; wait for the run to pause.")
    try:
        replaced = trials.promote(base, run_id, note=note, by=by)
    except trials.TrialError as exc:
        raise GateError(str(exc)) from None
    eng = engine(run_id, root)
    events.publish(eng.run_dir, "trial", f"Made the main run by {by}"
                   + (f", replacing {replaced}" if replaced else "")
                   + (f": {note}" if note else "."), by=by)
    if replaced:
        events.publish(base / replaced, "trial",
                       f"No longer the main run: {run_id} was made the main run by {by}.", by=by)
        old = engine(replaced, root)
        with old.lock:
            old._save(old.state)
    with eng.lock:
        eng._save(eng.state)
    if background:
        eng.start()
    return eng.state


def trial_record(run_id: str, root: str | Path | None = None) -> dict[str, Any]:
    """Every run of the model this run belongs to, deleted ones included, with how often the
    test sample was used across them. Computed on read."""
    from . import trials

    return trials.record_for(runs_root(root) / run_id)


def model_record(model: str, root: str | Path | None = None) -> dict[str, Any]:
    """The record of a model by its id (the first run of it), whether or not that run's files
    are still there."""
    from . import trials

    return trials.record(runs_root(root), model)


def trial_stamp(run_id: str, root: str | Path | None = None) -> int:
    """Changes whenever the record of this run's model does (a page re-reads it then)."""
    from . import trials

    try:
        st = state(run_id, root)
        return trials.record_path(runs_root(root), trials.model_of(st)).stat().st_mtime_ns
    except OSError:
        return 0


def whoami() -> str:
    """The person this process acts for (``COGNOS_USER``, else the login name)."""
    from . import identity

    return identity.whoami()


def state(run_id: str, root: str | Path | None = None) -> RunState:
    return RunState.load(runs_root(root) / run_id)


def is_busy(run_id: str, root: str | Path | None = None) -> bool:
    try:
        return engine(run_id, root).busy
    except KeyError:
        return False


def results(run_id: str, root: str | Path | None = None) -> dict[str, StageResult]:
    out = {}
    for stage in STAGES:
        path = runs_root(root) / run_id / "stages" / stage / "result.json"
        if path.exists():
            out[stage] = StageResult.model_validate_json(path.read_text(encoding="utf-8"))
    return out


def _results_in(run_dir: Path, name: str = "result.json") -> dict[str, StageResult]:
    out = {}
    for stage in STAGES:
        path = run_dir / "stages" / stage / name
        try:
            out[stage] = StageResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            continue
    return out


def _config_dict(run_dir: Path) -> dict | None:
    import yaml

    try:
        return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError):
        return None


def compare(run_a: str, run_b: str, root: str | Path | None = None) -> dict[str, Any]:
    """Run ``run_b`` against ``run_a``: what was decided differently and what it did to the
    results. Read from disk; nothing is stored."""
    from . import compare as cmp

    da, db = runs_root(root) / run_a, runs_root(root) / run_b
    return cmp.compare_runs(RunState.load(da), _results_in(da), _config_dict(da),
                            RunState.load(db), _results_in(db), _config_dict(db))


def step_changes(run_id: str, stage: str, root: str | Path | None = None) -> dict[str, Any] | None:
    """What the last re-run of ``stage`` changed against the result it replaced (None on a first
    run). The engine keeps the replaced result as ``result.prev.json``."""
    from . import compare as cmp

    d = runs_root(root) / run_id
    return cmp.step_changes(stage, _results_in(d, "result.prev.json").get(stage), _results_in(d).get(stage))


def previous_run(run_id: str, root: str | Path | None = None) -> str | None:
    """The most recent earlier run of the same project (the natural thing to compare with)."""
    rows = [r for r in list_runs(root) if not r["legacy"]]
    me = next((r for r in rows if r["run_id"] == run_id), None)
    if me is None:
        return None
    def born(rid: str) -> tuple[int, str]:
        # config.yaml is written once, when the run is created: finer than the id's one-second stamp
        try:
            return (runs_root(root) / rid / "config.yaml").stat().st_mtime_ns, rid
        except OSError:
            return 0, rid

    older = [r["run_id"] for r in rows if r["project"] == me["project"] and born(r["run_id"]) < born(run_id)]
    return max(older, key=born) if older else None


def export_run(run_id: str, dest: str | Path | None = None, root: str | Path | None = None, *,
               agent_io: bool = False) -> Path:
    """One zip of the run's documents, results, decisions and audit log (never the data)."""
    from .export import export_run as _export

    return _export(runs_root(root) / run_id, dest, agent_io=agent_io)


def run_events(run_id: str, since: float = 0.0, root: str | Path | None = None,
               limit: int = 300) -> list[dict]:
    return events.history(runs_root(root) / run_id, since=since, limit=limit)


def audit(run_id: str, root: str | Path | None = None) -> list[dict]:
    from .agents.runner import read_audit

    return read_audit(runs_root(root) / run_id)


def agent_io(run_id: str, call_id: str, root: str | Path | None = None) -> dict[str, Any]:
    d = runs_root(root) / run_id / "agents"
    out: dict[str, Any] = {}
    for kind in ("input", "output"):
        p = d / f"{call_id}.{kind}.json"
        if p.exists():
            out[kind] = json.loads(p.read_text(encoding="utf-8"))
    return out


def read_artifact(run_id: str, relpath: str, root: str | Path | None = None) -> str | None:
    path = (runs_root(root) / run_id / relpath).resolve()
    base = (runs_root(root) / run_id).resolve()
    if base not in path.parents or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def list_runs(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Every run under the runs root, newest first. v0.x runs (no state.json) are listed as
    ``legacy`` and read-only."""
    base = runs_root(root)
    if not base.exists():
        return []
    rows = []
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if RunState.exists(d):
            try:
                st = RunState.load(d)
            except Exception:
                continue
            champ = None
            # The champion is chosen on the outcomes analysis of the pool. A run recorded
            # before candidate pools carries it on the model stage.
            for stage in ("backtest", "model"):
                recorded = d / "stages" / stage / "result.json"
                if champ is None and recorded.exists():
                    try:
                        got = json.loads(recorded.read_text(encoding="utf-8"))
                        champ = ((got.get("payload") or {}).get("champion_name")
                                 or got["metrics"].get("champion"))
                    except Exception:
                        champ = None
            waiting = [s for s, v in st.steps.items() if v.status == "awaiting"]
            kind = "new"
            try:  # the development mode, as intake recorded it
                kind = json.loads((d / "stages" / "intake" / "result.json").read_text(
                    encoding="utf-8"))["payload"].get("kind") or "new"
            except Exception:
                pass
            lin = st.lineage
            rows.append({"run_id": d.name, "project": st.project, "status": st.status,
                         "kind": kind,
                         "model": lin.model if lin else d.name,
                         "role": ("trial" if lin.trial else "main") if lin else "",
                         "trial_of": lin.parent if lin else None,
                         "trial_gate": lin.gate if lin else None,
                         "trial_note": lin.note if lin else "",
                         "trial_ended": bool(lin and lin.trial and st.status == "completed"),
                         "mode": st.mode, "provider": st.provider, "updated_at": st.updated_at,
                         "champion": champ, "waiting_on": waiting[0] if waiting else None,
                         "spend_usd": st.spend_usd, "legacy": False})
        elif (d / "manifest.json").exists():
            try:
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({"run_id": d.name, "project": m.get("project", ""), "status": "legacy",
                         "kind": "new",
                         "mode": m.get("mode", ""), "provider": "-",
                         "updated_at": m.get("created_at", ""), "champion": None,
                         "waiting_on": None, "spend_usd": 0.0, "legacy": True})
    return sorted(rows, key=lambda r: r["updated_at"] or "", reverse=True)


def _run_dir(run_id: str, root: str | Path | None = None) -> Path:
    """The directory of a run that exists under the runs root. Refuses anything that is not a
    plain run id (a path, a working folder such as ``_uploads``), so a delete can never leave
    the runs root."""
    base = runs_root(root)
    name = str(run_id or "")
    if not name or name != Path(name).name or name.startswith(("_", ".")):
        raise ValueError(f"not a run id: {run_id!r}")
    d = base / name
    if not d.is_dir() or not (RunState.exists(d) or (d / "manifest.json").exists()):
        raise KeyError(f"no run {name} under {base}")
    return d


def deletion_preview(run_ids: list[str], root: str | Path | None = None) -> list[dict[str, Any]]:
    """What deleting these runs would remove: shown to the person before they confirm.
    ``blocked`` says why a run cannot be deleted now (this process is still working on it)."""
    from . import trials

    base = runs_root(root)
    out = []
    for run_id in run_ids:
        d = _run_dir(run_id, root)
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        row = {"run_id": d.name, "project": "", "status": "legacy", "size_bytes": size,
               "packages": len(list((d / "packages").glob("v*.json"))), "blocked": "",
               "keeps_line": False}
        if RunState.exists(d):
            try:
                st = RunState.load(d)
                row.update(project=st.project, status=st.status,
                           keeps_line=st.lineage is not None)
                # The main run of a model with trial runs goes last: its trial runs are
                # read against it. Deleting it together with all of them is allowed.
                held = trials.can_delete(base, st)
                others = {m.run_id for m in trials.members(base, trials.model_of(st))} - {d.name}
                if held and not others <= set(run_ids):
                    row["blocked"] = held
            except Exception:
                row["status"] = "unreadable"
            if is_busy(d.name, root):
                row["blocked"] = "a step is running in this workbench"
        out.append(row)
    return out


def delete_runs(run_ids: list[str], root: str | Path | None = None, *,
                by: str | None = None) -> list[dict[str, Any]]:
    """Delete runs from disk, for good: results, documents, decisions, the audit log and any
    signed package. The caller confirms first (``deletion_preview``); a run this process is
    still working on is refused. A run of a model with trial runs keeps one line in the
    model's record (what it tried, how far it went, how often it used the test sample):
    its files go, the fact that it was run does not. Returns what was removed."""
    from . import trials

    rows = deletion_preview(run_ids, root)
    busy = [r for r in rows if r["blocked"]]
    if busy:
        raise RuntimeError("cannot be deleted now: "
                           + "; ".join(f"{r['run_id']} ({r['blocked']})" for r in busy))
    by = by or whoami()
    for r in rows:
        d = _run_dir(r["run_id"], root)
        trials.keep_line(d.parent, d.name, by=by)
        with _guard:
            _engines.pop(str(d.resolve()), None)
        shutil.rmtree(d)
        (d.parent / "_locks" / f"{d.name}.lock").unlink(missing_ok=True)
    return rows


# --- templates (ADR-0018) ------------------------------------------------------------------
BUILTIN_TEMPLATES = Path(__file__).parent / "templates"
# How the built-ins are offered: PD first, then loss, then LGD and EAD. One not named goes last.
_BUILTIN_ORDER = ["pd_logistic_scorecard", "pd_structural_simulation", "rating_migration",
                  "loss_forecast_timeseries", "lgd_regression", "ead_ccf_regression"]
_CARRIED = ("task", "mode", "design", "metric", "search", "evaluation", "factors", "selection",
            "analysis", "plugins", "structural", "migration", "compliance", "impact", "backtest",
            "portfolio", "stress", "workflow", "stages")


def team_templates(root: str | Path | None = None) -> Path:
    """Where a team keeps the templates it saved: beside its runs, so everyone on one runs
    folder sees them (ADR-0015)."""
    return runs_root(root) / "_templates"


def list_templates(root: str | Path | None = None) -> list[dict[str, Any]]:
    """The built-in templates, then the team's, each as ``{"ref", "title", "description",
    "origin", "guide", "task", ...}``. ``ref`` is what :func:`template_config` takes."""
    out = []
    for origin, folder in (("built-in", BUILTIN_TEMPLATES), ("team", team_templates(root))):
        found = sorted(folder.glob("*.yaml")) if folder.exists() else []
        if origin == "built-in":
            found.sort(key=lambda f: (_BUILTIN_ORDER + [f.stem]).index(f.stem))
        for p in found:
            try:
                cfg = CognosConfig.from_yaml(p)
            except Exception:
                continue  # an unreadable file is not offered
            info = cfg.template
            out.append({"ref": f"{origin}:{p.stem}", "id": p.stem, "origin": origin,
                        "title": (info.title if info else "") or cfg.name,
                        "description": cfg.description, "guide": cfg.data.guide,
                        "task": cfg.task.value if cfg.task else "from the data",
                        "setup": template_setup(cfg),
                        "saved_by": info.saved_by if info else "",
                        "saved_at": info.saved_at if info else "",
                        "from_run": info.from_run if info else "", "path": str(p)})
    return out


_TASKS = {"classification": "a yes / no outcome", "regression": "a continuous outcome",
          "timeseries": "a series over time"}
_METRICS = {"roc_auc": "AUC", "rmse": "RMSE"}
_TESTS = {"out_of_time": "the latest periods (out of time)", "random": "a random share",
          "none": "none"}


def template_setup(cfg: CognosConfig) -> list[tuple[str, str]]:
    """What a template sets up, as (label, text) lines a person reads before choosing it. Only
    what the template states; anything left to the run is not listed."""
    out = []
    d = cfg.design
    if d.use_case:
        out.append(("For", d.use_case))
    if d.horizon:
        out.append(("Horizon", d.horizon))
    model = []
    if cfg.structural.enabled:
        model.append("Merton distance to default, alongside a scored model")
    if cfg.migration.enabled:
        model.append("rating transition matrices, alongside a scored model")
    if cfg.search.model_families:
        model.append(", ".join(cfg.search.model_families))
    if cfg.task:
        model.append(f"predicts {_TASKS.get(cfg.task.value, cfg.task.value)}")
    if cfg.metric.name != "auto":
        model.append(f"ranked on {_METRICS.get(cfg.metric.name, cfg.metric.name)}")
    if model:
        out.append(("Model", "; ".join(model)))
    plan = []
    if cfg.evaluation.test != "auto":
        plan.append(f"test sample: {_TESTS[cfg.evaluation.test]}")
    methods = [m for m in cfg.selection.methods if m != "auto"]
    if methods:
        plan.append(f"{' and '.join(m.replace('_', ' ') for m in methods)} selection, "
                    f"up to {cfg.selection.max_factors} factors")
    if cfg.factors.expected_signs:
        n = len(cfg.factors.expected_signs)
        plan.append(f"expected signs on {n} driver{'s' if n > 1 else ''}")
    if plan:
        out.append(("Development", "; ".join(plan)))
    after = []
    if cfg.stress.enabled and cfg.stress.scenarios:
        after.append("stress scenarios: " + ", ".join(
            str(s.get("name", "")).replace("_", " ") for s in cfg.stress.scenarios))
    if cfg.portfolio.enabled:
        after.append("portfolio loss simulation")
    if after:
        out.append(("Also", "; ".join(after)))
    c = cfg.compliance
    out.append(("Governance", f"{', '.join(c.regimes)}; {c.risk_tier} risk tier"))
    return out


def _slug(text: str, fallback: str = "model") -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", text or "").strip("_").lower() or fallback


def template_config(ref: str | None, name: str = "", *,
                    root: str | Path | None = None) -> CognosConfig:
    """The profile a run starts from: a template (``ref`` from :func:`list_templates`) or, with
    no ref, nothing but the defaults. Either way it has no data; the developer gives it at
    gate_source once the intent is confirmed."""
    if not ref:
        return CognosConfig.from_dict({"name": _slug(name), "mode": "interactive",
                                       "runs_dir": str(runs_root(root))})
    origin, _, tid = ref.partition(":")
    folder = {"built-in": BUILTIN_TEMPLATES, "team": team_templates(root)}.get(origin)
    path = folder / f"{tid}.yaml" if folder is not None and tid == Path(tid).name else None
    if path is None or not path.exists():
        raise KeyError(f"no template {ref!r}")
    raw = CognosConfig.from_yaml(path).model_dump(mode="json")
    raw["template"] = {**(raw.get("template") or {}), "id": tid, "origin": origin}
    raw.update(name=_slug(name, raw["name"]), runs_dir=str(runs_root(root)))
    return CognosConfig.from_dict(raw)


def save_template(run_id: str, title: str, description: str = "", *,
                  root: str | Path | None = None, by: str | None = None) -> str:
    """Keep a finished run's setup as a team template: the template it started from with the
    decisions the run took laid over it (design answers, intended use, the development plan
    as edited, the model families). Nothing about its data goes in, except a line in the
    guide saying what it was built on. Returns the template's ref."""
    from .context import RunContext
    from .engine.gates import describe_data
    from .engine.state import utcnow
    from .identity import whoami

    title = " ".join(str(title or "").split())
    if not title:
        raise GateError("Give the template a title: it is what a person picks when starting a run.")
    run_dir = _run_dir(run_id, root)
    st = RunState.load(run_dir)
    if st.status not in ("completed", "approved") or st.is_trial:
        raise GateError("A template is saved from a main run that has gone through the whole "
                        "process.")
    cfg = CognosConfig.from_yaml(run_dir / "config.yaml")
    ctx = RunContext(cfg, run_id=run_id, runs_root=run_dir.parent)
    eff, ov = ctx.config, st.overrides
    raw = {k: v for k, v in eff.model_dump(mode="json").items() if k in _CARRIED}
    for section, fields in (ov.plan or {}).items():
        raw[section] = {**raw.get(section, {}), **fields}
    families = list(dict.fromkeys(s["family"] for s in ov.slate or [] if s.get("family")))
    if families:
        raw["search"] = {**raw["search"], "model_families": families}
    built_on = [f"Saved from run {run_id} ({utcnow()[:10]}), built on "
                f"{describe_data(ov.data or eff.data.model_dump(mode='json'))}"]
    if eff.data.target:
        built_on.append(f"with {eff.data.target} as the dependent variable")
    guide = "\n\n".join(x for x in (cfg.data.guide.strip(), ", ".join(built_on) + ".") if x)
    slug = _slug(title, "template")
    dest = team_templates(root) / f"{slug}.yaml"
    # The catalog tables the run was built on become the template's starting selection.
    last = st.last_decision("gate_source")
    used = list((last.payload or {}).get("catalog") or []) if last else []
    raw.update(name=slug, description=description.strip() or cfg.description,
               data={"guide": guide, "catalog": used or list(cfg.data.catalog)},
               template={"id": slug, "title": title, "origin": "team", "saved_by": by or whoami(),
                         "saved_at": utcnow(), "from_run": run_id})
    tpl = CognosConfig.from_dict(raw)  # refuses anything a run could not start from
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:  # exclusive: two people saving the same title never overwrite each other
        with open(dest, "x", encoding="utf-8") as fh:
            fh.write(f"# A team template saved from run {run_id} by {tpl.template.saved_by} "
                     f"(ADR-0018).\n")
            import yaml

            yaml.safe_dump(tpl.model_dump(mode="json", exclude_defaults=True,
                                          exclude={"runs_dir", "agents", "engagement"}),
                           fh, sort_keys=False, allow_unicode=True)
    except FileExistsError:
        raise GateError(f"The team already has a template called {title!r}; choose another "
                        "title.") from None
    return f"team:{slug}"


def data_info(run_id: str, root: str | Path | None = None, *,
              with_catalog: bool = False) -> dict[str, Any]:
    """What a run's page says about its data: the template's note on what this kind of model
    is built on, and the data in force (the profile's, or what was given at gate_source).
    ``with_catalog`` adds the data catalog (``catalog``) and the tables the form starts with
    checked (``preselected``): the last choice, else the template's, else the defaults."""
    from .engine.gates import describe_data

    run_dir = _run_dir(run_id, root)
    cfg = CognosConfig.from_yaml(run_dir / "config.yaml")
    st = RunState.load(run_dir)
    given = st.overrides.data
    if not given and cfg.data.given:
        given = cfg.data.model_dump(mode="json")
    info: dict[str, Any] = {"guide": cfg.data.guide,
                            "given": describe_data(given) if given else ""}
    if with_catalog:
        from . import catalog

        try:
            info["catalog"] = catalog.listing(root)
        except Exception as exc:  # an unreadable catalog leaves the form for added data
            info["catalog"], info["catalog_error"] = None, str(exc)
            return info
        last = st.last_decision("gate_source")
        chosen = list((last.payload or {}).get("catalog") or []) if last else []
        known = {t["id"] for t in info["catalog"]["tables"]}
        info["preselected"] = ([i for i in chosen if i in known] if last
                               else catalog.preselected(cfg, root))
    return info


def catalog_line(table: dict[str, Any]) -> str:
    """One line on what a catalog table (as :func:`catalog_listing` gives it) holds."""
    from . import catalog

    return catalog.summary_line(table)


def catalog_listing(root: str | Path | None = None) -> dict[str, Any]:
    """The data catalog: every table with its description, what it holds and its links."""
    from . import catalog

    return catalog.listing(root)


def data_payload(catalog: list[str] = (), files: list[str] = (), lines: str = "", *,
                 datetime_col: str | None = None, target: str | None = None,
                 root: str | Path | None = None) -> dict[str, Any]:
    """The gate_source payload for what the workbench form describes: catalog tables, data
    files added, and Snowflake lines (``name = TABLE`` or ``name = SELECT ...``)."""
    from . import catalog as catalog_mod
    from . import linking

    try:
        added = linking.named_sources(list(files), lines or "", at_least=0)["sources"]
        data = catalog_mod.data_spec(list(catalog), added, datetime_col=datetime_col or None,
                                     root=root)
    except (KeyError, ValueError) as exc:  # LinkError is a ValueError
        raise GateError(str(exc).strip("'\"")) from exc
    return {"catalog": list(catalog), "data": data,
            **({"target": target} if target else {})}


def provide_data(run_id: str, data: dict[str, Any] | None = None, *,
                 catalog: list[str] = (), added: list[dict[str, Any]] = (),
                 datetime_col: str | None = None, target: str | None = None,
                 root: str | Path | None = None, by: str | None = None,
                 background: bool = True) -> RunState:
    """Give a run its data at gate_source: tables from the data ``catalog`` (ids, the first is
    the proposed base table) and/or ``added`` sources (``{"name", "kind", "path" | "table" |
    "query"}``), or a ``data`` mapping as it is (``{"source": {...}}`` or
    ``{"sources": [...]}``)."""
    payload: dict[str, Any] = {"catalog": list(catalog)}
    if catalog or added:
        from . import catalog as catalog_mod

        try:
            data = catalog_mod.data_spec(list(catalog), list(added), datetime_col=datetime_col,
                                         root=root)
        except (KeyError, ValueError) as exc:
            raise GateError(str(exc).strip("'\"")) from exc
    elif data is not None and datetime_col:
        data = {**data, "datetime_col": datetime_col}
    payload["data"] = data or {}
    if target:
        payload["target"] = target
    return submit_gate(run_id, "gate_source", "accept", payload, root=root, by=by,
                       background=background)


def agent_model(run_id: str, root: str | Path | None = None) -> str | None:
    """The model a run's agents use: the run's choice, else its provider's default."""
    from .agents import providers

    eng = engine(run_id, root)
    p = providers.all_providers().get(eng.state.provider) or {}
    if p.get("kind") in (None, "heuristic", "replay"):
        return None
    return eng.config.agents.model or p.get("model")


def provider_list() -> list[dict[str, Any]]:
    from .agents import providers

    return providers.public_list()
