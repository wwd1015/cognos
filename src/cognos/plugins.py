"""Plugins: how COGNOS is extended without editing it.

A plugin is a module with a ``register(registry)`` function. It may add:

- **analysis tools** (``registry.add_tool``): a named, documented function over the dataset that
  returns numbers, a table and a chart. The Data Analyst can ask for any registered tool by name;
  the engine runs it and keeps the result. A tool is ordinary reviewed code, so its numbers are
  the engine's numbers.
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


@dataclass
class AnalysisTool:
    """One analysis the Data Analyst may request. ``fn(df, params, env) -> result`` where
    ``result`` is ``{"title", "summary": {name: number|text}, "table": {"columns", "rows"},
    "chart": <chart spec, see analysis/charts.py>}`` (every key optional). ``env`` carries
    ``target``, ``task``, ``features`` and ``datetime_col``. ``params`` maps each parameter name
    to its description; values arrive as text."""

    name: str
    description: str
    fn: Callable[[Any, dict[str, str], dict[str, Any]], dict[str, Any]]
    params: dict[str, str] = field(default_factory=dict)
    needs_target: bool = False
    origin: str = "cognos"


@dataclass
class Registry:
    tools: dict[str, AnalysisTool] = field(default_factory=dict)
    sources: dict[str, datasources.DataSource] = field(default_factory=dict)
    problems: list[dict[str, str]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    _origin: str = "cognos"

    def add_tool(self, tool: AnalysisTool) -> None:
        if self._origin != "cognos":
            tool.origin = self._origin
        self.tools[tool.name] = tool

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
    return {
        "tools": [{"name": t.name, "description": t.description, "params": t.params,
                   "needs_target": t.needs_target, "origin": t.origin}
                  for t in reg.tools.values()],
        "sources": sources, "plugins": list(reg.loaded), "problems": list(reg.problems),
    }
