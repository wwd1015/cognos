"""Plugins: how COGNOS is extended without editing it.

A plugin is a module with a ``register(registry)`` function. It may add:

- **tools** (``registry.add_tool``): a named, documented function that returns numbers, a table,
  a chart and (for a test) pass/fail checks. A tool names the stages it serves; the agent of
  each of those stages can ask for it by name, and the engine runs it and keeps the result. A
  tool is ordinary reviewed code, so its numbers are the engine's numbers. The prompts that
  guide an agent stay inside COGNOS: a plugin adds what an agent can *ask for*, never how it
  is told to think.
- **data sources** (``registry.add_source``): a connector for ``data.source.kind``.

Plugins are found in three places: Python entry points in the group ``cognos.plugins``, the
profile's ``plugins:`` list, and the ``COGNOS_PLUGINS`` environment variable (comma-separated
module names). A plugin that fails to import or register is reported (``cognos plugins``) and
skipped: it is unavailable, never a crash.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import datasources

# What a tool may be handed, and the first stage at which each input exists. Nothing before
# ``backtest`` can receive the sealed holdout or the fitted champion: the modeler chooses blind.
NEEDS = {"data": "explore", "documents": "intake", "results": "intake", "train": "model",
         "holdout": "backtest", "model": "backtest"}
TOOL_STAGES = ("intake", "explore", "ideate", "model", "backtest", "validate", "comply",
               "document")


@dataclass
class AnalysisTool:
    """One thing a stage agent may ask the engine to run. ``fn(df, params, env) -> result``.

    ``result`` is ``{"title", "summary": {name: number|text}, "table": {"columns", "rows"},
    "chart": <chart spec, see analysis/charts.py>, "checks": [{"name", "passed", "severity",
    "detail"}]}`` (every key optional). A failed check becomes a finding of the stage (severity
    ``low`` / ``medium`` / ``high``; a tool can never block a run).

    ``stages`` are the stages whose agent may request it. ``needs`` are the inputs it is handed
    (see ``NEEDS``): ``df`` is the dataset snapshot when ``data`` is among them, else ``None``;
    the rest arrive in ``env`` (``train``, ``holdout``, ``score``, ``model_path``, ``documents``,
    ``results``) beside ``stage``, ``target``, ``task``, ``features``, ``datetime_col`` and
    ``metric``. ``params`` maps each parameter name to its description; values arrive as text.
    ``available`` returns ``(ok, why)``: a tool whose dependency is missing is listed with the
    reason and never offered to an agent."""

    name: str
    description: str
    fn: Callable[[Any, dict[str, str], dict[str, Any]], dict[str, Any]]
    params: dict[str, str] = field(default_factory=dict)
    needs_target: bool = False
    origin: str = "cognos"
    stages: tuple[str, ...] = ("explore",)
    needs: tuple[str, ...] = ("data",)
    available: Callable[[], tuple[bool, str]] | None = None
    version: str = ""

    def problems(self) -> list[str]:
        """Why this tool cannot be registered as declared (empty when it can)."""
        out = [f"unknown stage {s!r}" for s in self.stages if s not in TOOL_STAGES]
        out += [f"unknown input {n!r}" for n in self.needs if n not in NEEDS]
        for stage in (s for s in self.stages if s in TOOL_STAGES):
            early = [n for n in self.needs if n in NEEDS
                     and TOOL_STAGES.index(NEEDS[n]) > TOOL_STAGES.index(stage)]
            if early:
                out.append(f"stage {stage!r} cannot be given {early} (not available that early)")
        return out

    def status(self) -> tuple[bool, str]:
        if self.available is None:
            return True, ""
        try:
            ok, why = self.available()
            return bool(ok), str(why or "")
        except Exception as exc:  # an availability probe that raises means "not available"
            return False, f"{type(exc).__name__}: {exc}"


Tool = AnalysisTool  # the general name; AnalysisTool is kept for plugins written against v1.2


@dataclass
class Registry:
    tools: dict[str, AnalysisTool] = field(default_factory=dict)
    sources: dict[str, datasources.DataSource] = field(default_factory=dict)
    problems: list[dict[str, str]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    _origin: str = "cognos"

    def add_tool(self, tool: AnalysisTool) -> None:
        """Register a tool. A later registration under the same name replaces the earlier one
        (how a plugin supersedes a built-in placeholder)."""
        tool.stages, tool.needs = tuple(tool.stages), tuple(tool.needs)
        bad = tool.problems()
        if bad:
            raise ValueError(f"tool {tool.name!r}: " + "; ".join(bad))
        if self._origin != "cognos":
            tool.origin = self._origin
        self.tools[tool.name] = tool

    def tools_for(self, stage: str) -> list[AnalysisTool]:
        """The tools the agent of ``stage`` may be offered: registered for it and available."""
        return [t for t in self.tools.values() if stage in t.stages and t.status()[0]]

    def unavailable_for(self, stage: str) -> list[dict[str, str]]:
        out = []
        for t in self.tools.values():
            ok, why = t.status()
            if stage in t.stages and not ok:
                out.append({"name": t.name, "description": t.description, "origin": t.origin,
                            "note": why})
        return out

    def add_source(self, source: datasources.DataSource) -> None:
        if self._origin != "cognos":
            source.origin = self._origin
        self.sources[source.kind] = source


_cache: dict[tuple[str, ...], Registry] = {}


def _entry_points() -> list[tuple[str, Callable[[Registry], None]]]:
    from importlib.metadata import entry_points

    out = []
    for ep in entry_points(group="cognos.plugins"):
        out.append((ep.name, ep))
    return out


def registry(modules: list[str] | tuple[str, ...] = (), *, refresh: bool = False) -> Registry:
    """The built-ins plus every plugin that loads. Cached per set of extra modules."""
    env = [m.strip() for m in os.environ.get("COGNOS_PLUGINS", "").split(",") if m.strip()]
    names = tuple(dict.fromkeys([*env, *modules]))
    if not refresh and names in _cache:
        return _cache[names]
    from .analysis import tools as builtin_tools

    reg = Registry()
    for source in datasources.BUILTIN:
        reg.add_source(source)
    builtin_tools.register(reg)
    from .integrations import impact_tools

    impact_tools.register(reg)

    def run(name: str, load: Callable[[], Any]) -> None:
        reg._origin = name
        try:
            target = load()
            hook = getattr(target, "register", target)
            if not callable(hook):
                raise TypeError("a plugin must expose register(registry)")
            hook(reg)
            reg.loaded.append(name)
        except Exception as exc:  # a broken plugin is reported, never fatal
            reg.problems.append({"plugin": name, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            reg._origin = "cognos"

    try:
        for ep_name, ep in _entry_points():
            run(ep_name, ep.load)
    except Exception as exc:  # entry-point metadata itself unreadable
        reg.problems.append({"plugin": "entry points", "error": f"{type(exc).__name__}: {exc}"})
    for name in names:
        run(name, lambda name=name: importlib.import_module(name))
    _cache[names] = reg
    return reg


def public_list(modules: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """What is installed, for ``cognos plugins`` and the workbench."""
    reg = registry(modules)
    sources = []
    for s in reg.sources.values():
        ok, why = s.available()
        sources.append({"kind": s.kind, "label": s.label, "origin": s.origin, "available": ok,
                        "note": why})
    tools = []
    for t in reg.tools.values():
        ok, why = t.status()
        tools.append({"name": t.name, "description": t.description, "params": t.params,
                      "needs_target": t.needs_target, "origin": t.origin,
                      "stages": list(t.stages), "needs": list(t.needs), "version": t.version,
                      "available": ok, "note": why})
    return {
        "tools": tools,
        "sources": sources, "plugins": list(reg.loaded), "problems": list(reg.problems),
    }
