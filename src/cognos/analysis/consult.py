"""Stage tools: how the agent of any stage uses a tool it was not built with.

Before a stage's agent makes its recommendation, the engine offers it the tools registered for
that stage (built in or from a plugin). The agent *requests* runs (the ``<agent>_tools`` step, a
``ToolRequestOutput``); the engine executes them, keeps every result under
``stages/<stage>/analyses/``, turns failed checks into findings, and hands the results back as
facts (``tools.<stage>.<id>.<key>``) the recommendation can cite. Agents still execute nothing.

A stage with no available tool skips the step entirely: no agent call, no change in behaviour.
The Data Analyst's own request step (``data_scout``, which may also write code) is in
``stages/explore.py``.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

from ..artifacts import Finding, Severity
from . import run as runner

SLIM = ("id", "tool", "params", "purpose", "status", "error", "summary", "checks")
_SEVERITY = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH}
# Stages whose agents choose before the outcome is known: they are not shown what an earlier
# run of the model being updated scored.
_BLIND = ("ideate", "model")


def slim(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool runs as an agent sees them."""
    return [{k: r.get(k) for k in SLIM} for r in records]


def findings(records: list[dict[str, Any]]) -> list[Finding]:
    """Every failed check of every tool run, as a finding of the stage."""
    out = []
    for r in records:
        for c in r.get("checks") or []:
            if c.get("passed"):
                continue
            out.append(Finding(
                id=f"tool-{r['id']}-{c['name']}".replace(" ", "-")[:80],
                severity=_SEVERITY.get(c.get("severity"), Severity.MEDIUM),
                category=f"tool/{r['tool']}", location=r.get("artifact"),
                message=f"{r['tool']} ({r.get('origin', 'cognos')}): check '{c['name']}' failed"
                        + (f" — {c['detail']}" if c.get("detail") else "") + "."))
    return out


def _missing(ctx, tool, extra: dict[str, Any]) -> str:
    """Why this run cannot give ``tool`` what it needs ("" when it can)."""
    if tool.needs_target and not ctx.config.data.target:
        return "the target is not decided yet"
    for need in tool.needs:
        if need in extra:
            continue
        if need == "holdout" and not (ctx.data_dir / "holdout.parquet").exists():
            return "this run has no sealed holdout"
        if need == "train" and not (ctx.data_dir / "train.parquet").exists():
            return "the development sample has not been split yet"
        if need == "model" and not ((ctx.get("model") and ctx.get("model").payload) or {}).get(
                "scorer_path"):
            return "no champion has been fitted yet"
    return ""


def _inputs(ctx, stage: str, extra: dict[str, Any]):
    """``inputs(tool) -> (df, env)``: exactly what the tool declared, loaded once."""
    from ..plugins import TOOL_STAGES

    cache: dict[str, Any] = {}
    cfg = ctx.config
    explore = ctx.get("explore")
    base = {"stage": stage, "target": cfg.data.target, "task": cfg.task.value if cfg.task else None,
            "features": list(ctx.profile().get("features", [])) if explore is not None
            and explore.payload else [],
            "datetime_col": cfg.data.datetime_col, "metric": cfg.metric.name}

    def load(need: str) -> Any:
        if need in extra:
            return extra[need]
        if need not in cache:
            if need == "data":
                cache[need] = ctx.load_dataset()
            elif need in ("train", "holdout"):
                cache[need] = ctx.load_df(f"data/{need}.parquet")
            elif need == "model":
                cache[need] = ctx.require("model").payload["scorer_path"]
            elif need == "documents":
                try:
                    cache[need] = json.loads(ctx.resolve("stages/intake/corpus.json").read_text(
                        encoding="utf-8")).get("documents", [])
                except (FileNotFoundError, ValueError):
                    cache[need] = []
            elif need == "results":
                out = {}
                for s in TOOL_STAGES[:TOOL_STAGES.index(stage)]:
                    res = ctx.get(s)
                    if res is not None and res.payload:
                        out[s] = {k: v for k, v in res.payload.items()
                                  if not (stage in _BLIND and k == "prior")}
                cache[need] = out
        return cache[need]

    def inputs(tool):
        env = dict(base)
        df = None
        for need in tool.needs:
            value = load(need)
            if need == "data":
                df = value
            elif need == "model":
                from ..runtime.score import score_frame

                env["model_path"] = value
                env["score"] = lambda frame, path=value: score_frame(frame, path)
            else:
                env[need] = value
        return df, env

    return inputs


def consult(ctx, agent: str, brief: dict[str, Any] | None = None, *, res=None,
            inputs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Let ``agent`` request the tools registered for its stage and run them. Returns the runs
    as the agent should see them (pass them on as ``tool_runs`` in its recommendation context).
    ``brief`` is the small stage context the agent needs to choose; ``res`` (the stage's
    provisional result) receives a finding per failed check; ``inputs`` supplies a declared need
    the run directory does not hold yet (intake's ``documents``)."""
    from .. import plugins
    from ..agents.contracts import AGENT_STAGE, FRIENDLY
    from ..agents.runner import AgentRunError
    from ..engine import events

    stage = AGENT_STAGE[agent]
    extra = dict(inputs or {})
    records: list[dict[str, Any]] = []
    ctx.tool_runs[stage] = records
    shutil.rmtree(ctx.resolve(runner.dir_for(stage)), ignore_errors=True)
    reg = plugins.registry(ctx.config.plugins)
    unavailable = reg.unavailable_for(stage)
    tools = []
    for t in reg.tools_for(stage):
        why = _missing(ctx, t, extra)
        if why:
            unavailable.append({"name": t.name, "description": t.description, "origin": t.origin,
                                "note": why})
        else:
            tools.append(t)
    ctx.tools_unavailable[stage] = unavailable
    cfg = ctx.config.analysis
    if not tools or not cfg.stage_tools:
        return []
    catalog = [{"name": t.name, "description": t.description, "params": t.params,
                "origin": t.origin} for t in tools]
    resolve = _inputs(ctx, stage, extra)
    for round_no in range(1, cfg.rounds + 1):
        try:
            out = ctx.recommend(f"{agent}_tools", {
                **(brief or {}), "round": round_no, "tools": catalog,
                "max_requests": cfg.max_requests, "tool_runs": slim(records),
            }, fresh={stage: res} if res is not None else None)
        except AgentRunError as exc:
            if res is not None:
                res.add_finding(Finding(
                    id="tool-requests-failed", severity=Severity.LOW, category="tool",
                    message=f"The {FRIENDLY[agent]}'s tool requests could not be obtained: {exc}"))
            break
        reqs = [{"purpose": r.purpose, "tool": r.tool,
                 "params": {p.name: p.value for p in r.params}} for r in out.requests]
        if reqs:
            records += runner.execute(ctx, reqs, None, {}, start=len(records) + 1, author="engine",
                                      stage=stage, prefix="t", inputs=resolve)
            events.publish(ctx.run_dir, "progress",
                           f"Ran {len(reqs)} tool request(s) for the {FRIENDLY[agent]}.",
                           step=stage)
        if out.done or not reqs:
            break
    if res is not None:
        for f in findings(records):
            res.add_finding(f)
    return slim(records)


def recorded(ctx, stage: str) -> list[dict[str, Any]]:
    """The tool runs of ``stage``: this process's, else what its last result recorded."""
    if stage in ctx.tool_runs:
        return ctx.tool_runs[stage]
    res = ctx.get(stage)
    return list(((res.payload if res is not None else None) or {}).get("tool_runs") or [])
