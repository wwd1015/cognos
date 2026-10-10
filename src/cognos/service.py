"""The service layer — the single API boundary the CLI and the UI use.

Everything a front end needs: create a run (from a profile or a synthetic demo preset, as a new
model development or a model update with its intent and prior-model documents), advance it
(synchronously or in the background), submit gate decisions, answer questions, retry, and read
state, results, events and the agent audit. One :class:`~cognos.engine.Engine` per run is cached
in-process so a background run and the UI callbacks share its in-flight bookkeeping. A REST adapter,
if ever needed, is a thin wrapper over these functions.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .artifacts import StageResult
from .config import CognosConfig
from .engine import Engine, RunState, events
from .engine.graph import STAGES

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
DEMO_LABELS = {
    "commercial": "Commercial PD (vintage panel, out-of-time)",
    "cni": "C&I portfolio (hazard term structure, portfolio sim, stress)",
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
    gen = synth.GENERATORS[preset]
    df = gen(n=n) if n else gen()
    csv = data_dir / f"{preset}.csv"
    df.to_csv(csv, index=False)
    p = DEMO_PRESETS[preset]
    # The demo sponsor's intent document: the preset's design brief on the template. Presets
    # without one leave those sections empty, so the interview has something to ask.
    from . import engagement as eg

    intent = data_dir / f"{preset}_intent.md"
    design = p.get("design", {})
    intent.write_text(eg.render_template("new", {
        "objective": f"Demonstrate a COGNOS model development on synthetic data: "
                     f"{DEMO_LABELS.get(preset, preset)}.",
        **{k: design.get(k, "") for k in eg.CORE},
        "interpretability": design.get("interpretability", ""),
        "data_sources": f"Synthetic data from cognos.synth ({preset}).",
    }, name=f"demo_{preset}"), encoding="utf-8")
    raw: dict[str, Any] = {
        "name": f"demo_{preset}",
        "description": f"COGNOS synthetic {preset} demo — {DEMO_LABELS.get(preset, preset)}",
        "task": p["task"],
        "data": {"path": str(csv), "format": "csv", "target": p["target"],
                 "datetime_col": p.get("datetime_col"), "protected_attributes": p.get("protected", []),
                 "drop_columns": p.get("drop", []), "event_time_col": p.get("event_time_col"),
                 "horizon_periods": p.get("horizon_periods")},
        "engagement": {"kind": "new", "intent": str(intent)},
        "design": p.get("design", {}),
        "migration": p.get("migration", {}),
        "portfolio": p.get("portfolio", {}),
        "stress": p.get("stress", {}),
        "metric": {"name": p["metric"]},
        "compliance": {"fair_lending": p.get("fair_lending", False),
                       "jurisdictions": ["US"] if preset in ("commercial", "cni", "migration")
                       else ["US", "EU"],
                       "risk_tier": "high" if preset in ("credit", "commercial", "cni", "migration")
                       else "medium"},
        "runs_dir": str(runs_root(root)),
    }
    if search_budget:
        raw["search"] = {"max_candidates": search_budget}
    return CognosConfig.from_dict(raw)


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
    return CognosConfig.from_dict({"name": slug, "description": description,
                                   "data": {"source": source}, "runs_dir": str(runs_root(root)),
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
               provider: str | None = None, root: str | Path | None = None,
               engagement: dict[str, Any] | None = None) -> str:
    cfg = with_engagement(source, engagement)
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
                background: bool = True, seat: str | None = None) -> RunState:
    eng = engine(run_id, root)
    state = eng.submit_gate(gate, action, payload, reason, seat=seat)
    if background:
        eng.start()
    return state


def answer_gap(run_id: str, gap_id: str, answer: str, *, assume: bool = False,
               root: str | Path | None = None, background: bool = True,
               seat: str | None = None) -> RunState:
    eng = engine(run_id, root)
    state = eng.answer_gap(gap_id, answer, assume=assume, seat=seat)
    if background:
        eng.start()
    return state


def retry(run_id: str, step: str, root: str | Path | None = None,
          background: bool = True) -> RunState:
    eng = engine(run_id, root)
    state = eng.retry(step)
    if background:
        eng.start()
    return state


def reopen(run_id: str, gate: str, root: str | Path | None = None,
           *, seat: str | None = None) -> RunState:
    return engine(run_id, root).reopen(gate, seat=seat)


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
            model = d / "stages" / "model" / "result.json"
            if model.exists():
                try:
                    champ = json.loads(model.read_text(encoding="utf-8"))["metrics"].get("champion")
                except Exception:
                    champ = None
            waiting = [s for s, v in st.steps.items() if v.status == "awaiting"]
            kind = "new"
            try:  # the development mode, as intake recorded it
                kind = json.loads((d / "stages" / "intake" / "result.json").read_text(
                    encoding="utf-8"))["payload"].get("kind") or "new"
            except Exception:
                pass
            rows.append({"run_id": d.name, "project": st.project, "status": st.status,
                         "kind": kind,
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


def list_profiles(directory: str | Path = "projects") -> list[dict[str, str]]:
    d = Path(directory)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.y*ml")):
        try:
            cfg = CognosConfig.from_yaml(p)
            out.append({"path": str(p), "name": cfg.name, "description": cfg.description,
                        "task": cfg.task.value if cfg.task else "from the data"})
        except Exception:
            continue
    return out


def provider_list() -> list[dict[str, Any]]:
    from .agents import providers

    return providers.public_list()
