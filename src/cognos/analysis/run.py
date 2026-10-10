"""The analysis executor: turn the Data Analyst's requests into kept results.

A request names a registered tool (with parameters) or carries a script. Each one becomes a
record in the stage payload and two artifacts under ``stages/explore/analyses/``: ``<id>.json``
(summary, table, chart specs) and, for a script, ``<id>.py`` — the code as written, with a
header saying who wrote it and why. A request that fails is recorded as failed, with the reason;
it never fails the stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import charts, sandbox

DIR = "stages/explore/analyses"


def digest(result: dict[str, Any]) -> str:
    """Hash of what an analysis found (values, tables, charts): equal digests mean a re-run
    reproduced it."""
    body = {k: result.get(k) for k in ("summary", "tables", "charts")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _numbers(summary: dict[str, Any]) -> dict[str, Any]:
    return {str(k)[:60]: charts._cell(v) for k, v in list((summary or {}).items())[:12]}


def _run_tool(tool, df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    if tool.needs_target and not env.get("target"):
        raise ValueError("this tool needs the target, which is not decided yet")
    raw = tool.fn(df, params, env) or {}
    chart = raw.get("chart")
    return {"title": str(raw.get("title") or tool.name)[:160], "summary": _numbers(raw.get("summary")),
            "tables": [{"name": "result", **raw["table"]}] if raw.get("table") else [],
            "charts": [charts.normalize(chart)] if chart else []}


def _run_code(code: str, data_path: Path, env: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    out = sandbox.run(code, data_path, target=env.get("target") or "", timeout_s=timeout_s)
    if not out.get("ok"):
        raise RuntimeError(out.get("error") or "the script failed")
    specs = []
    for spec in out.get("charts", [])[:6]:
        try:
            specs.append(charts.normalize(spec))
        except ValueError:
            continue  # a malformed chart is dropped; the values and tables still stand
    return {"title": "", "summary": _numbers(out.get("values")), "tables": out.get("tables", [])[:6],
            "charts": specs, "stdout": out.get("stdout", "")}


def header(record: dict[str, Any], run_id: str) -> str:
    return ("# COGNOS analysis script — a model-development artifact.\n"
            f"# Run: {run_id}   Analysis: {record['id']}   Stage: explore\n"
            f"# Written by: {record['author']}   Executed by: the COGNOS engine (restricted)\n"
            f"# Purpose: {' '.join(record['purpose'].split())}\n"
            f"# SHA-256 of the code below: {record['code_sha256']}\n"
            "# Inputs: df (the run's dataset snapshot), TARGET. Outputs: emit_value / emit_table /"
            " emit_chart.\n\n")


def code_of(text: str) -> str:
    """A saved script without its header (what was hashed and what is re-run)."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[i + 1:] if i < len(lines) and not lines[i].strip() else lines[i:])


def execute(ctx, requests: list[dict[str, Any]], df: pd.DataFrame, env: dict[str, Any], *,
            start: int, author: str) -> list[dict[str, Any]]:
    """Run each request; write its artifacts; return the records (small, payload-safe)."""
    from .. import plugins

    reg = plugins.registry(ctx.config.plugins)
    cfg = ctx.config.analysis
    data_path = ctx.data_dir / "dataset.parquet"
    records = []
    for i, req in enumerate(requests, start):
        code = (req.get("code") or "").strip()
        rec: dict[str, Any] = {
            "id": f"a{i}", "kind": "code" if code else "tool", "tool": req.get("tool") or "",
            "params": dict(req.get("params") or {}), "purpose": req.get("purpose") or "",
            "status": "ok", "error": "", "title": "", "summary": {}, "n_tables": 0, "n_charts": 0,
            "author": author if code else "engine", "origin": "agent" if code else "cognos",
            "code_path": "", "code_sha256": "", "result_digest": "", "artifact": f"{DIR}/{{}}.json"}
        rec["artifact"] = f"{DIR}/{rec['id']}.json"
        result: dict[str, Any] = {"summary": {}, "tables": [], "charts": []}
        try:
            if code:
                rec["code_sha256"] = sandbox.sha256(code)
                rec["code_path"] = f"{DIR}/{rec['id']}.py"
                ctx.save_text(rec["code_path"], header(rec, ctx.run_id) + code + "\n", kind="code")
                if not cfg.allow_code:
                    raise PermissionError("agent-written code is switched off (analysis.allow_code)")
                result = _run_code(code, data_path, env, cfg.code_timeout_s)
            else:
                tool = reg.tools.get(rec["tool"])
                if tool is None:
                    raise KeyError(f"unknown tool {rec['tool']!r}")
                rec["origin"] = tool.origin
                result = _run_tool(tool, df, rec["params"], env)
        except Exception as exc:  # a failed analysis is a record, never a failed stage
            rec["status"], rec["error"] = "error", f"{type(exc).__name__}: {exc}"[:500]
        rec["title"] = result.get("title") or rec["purpose"][:120] or rec["tool"] or rec["id"]
        rec["summary"] = result.get("summary", {})
        rec["n_tables"], rec["n_charts"] = len(result.get("tables", [])), len(result.get("charts", []))
        rec["result_digest"] = digest(result) if rec["status"] == "ok" else ""
        ctx.save_json(rec["artifact"], {**rec, **result})
        records.append(rec)
    return records


def load_result(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads((Path(run_dir) / record["artifact"]).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def reproduce(ctx, record: dict[str, Any]) -> tuple[bool, str]:
    """Re-run a kept script from its saved code and compare what it finds with what was
    recorded. (True, "") when it reproduces."""
    try:
        code = code_of(ctx.resolve(record["code_path"]).read_text(encoding="utf-8")).strip()
    except FileNotFoundError:
        return False, "the script file is missing"
    if sandbox.sha256(code) != record.get("code_sha256"):
        return False, "the script on disk is not the script that was recorded (hash differs)"
    env = {"target": ctx.config.data.target}
    try:
        result = _run_code(code, ctx.data_dir / "dataset.parquet", env,
                           ctx.config.analysis.code_timeout_s)
    except Exception as exc:
        return False, f"it no longer runs: {exc}"[:300]
    result["title"] = ""
    if digest(result) != record.get("result_digest"):
        return False, "it runs but its results differ from the recorded ones"
    return True, ""
