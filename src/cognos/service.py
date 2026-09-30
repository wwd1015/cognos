"""The service layer — the single API boundary the CLI and the UI use.

Everything a front end needs: create a run (from a profile or a synthetic demo preset), advance it
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
    raw: dict[str, Any] = {
        "name": f"demo_{preset}",
        "description": f"COGNOS synthetic {preset} demo — {DEMO_LABELS.get(preset, preset)}",
        "task": p["task"],
        "data": {"path": str(csv), "format": "csv", "target": p["target"],
                 "datetime_col": p.get("datetime_col"), "protected_attributes": p.get("protected", []),
                 "drop_columns": p.get("drop", []), "event_time_col": p.get("event_time_col"),
                 "horizon_periods": p.get("horizon_periods")},
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


# --- runs ------------------------------------------------------------------------------------
def create_run(source: CognosConfig | str | Path | dict, *, mode: str = "interactive",
               provider: str | None = None, root: str | Path | None = None) -> str:
    cfg = load_config(source)
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
                background: bool = True) -> RunState:
    eng = engine(run_id, root)
    state = eng.submit_gate(gate, action, payload, reason)
    if background:
        eng.start()
    return state


def answer_gap(run_id: str, gap_id: str, answer: str, *, assume: bool = False,
               root: str | Path | None = None, background: bool = True) -> RunState:
    eng = engine(run_id, root)
    state = eng.answer_gap(gap_id, answer, assume=assume)
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


def reopen(run_id: str, gate: str, root: str | Path | None = None) -> RunState:
    return engine(run_id, root).reopen(gate)


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
            rows.append({"run_id": d.name, "project": st.project, "status": st.status,
                         "mode": st.mode, "provider": st.provider, "updated_at": st.updated_at,
                         "champion": champ, "waiting_on": waiting[0] if waiting else None,
                         "spend_usd": st.spend_usd, "legacy": False})
        elif (d / "manifest.json").exists():
            try:
                m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({"run_id": d.name, "project": m.get("project", ""), "status": "legacy",
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
                        "task": cfg.task.value})
        except Exception:
            continue
    return out


def provider_list() -> list[dict[str, Any]]:
    from .agents import providers

    return providers.public_list()
