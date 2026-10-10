"""Chart specs: a small, plain-JSON description of a chart.

Tools and agent-written scripts produce specs, not figures: the core package has no plotting
dependency, a spec is a text artifact that travels in the export, and the workbench draws every
chart in its own theme (``ui/charts.py::from_spec``).

    {"kind": "bar" | "line" | "scatter" | "histogram" | "heatmap",
     "title": str, "x_title": str, "y_title": str, "y_format": "" | "%",
     "x": [...], "series": [{"name": str, "y": [...]}],      # bar / line / scatter / histogram
     "y": [...], "z": [[...]]}                               # heatmap: x columns, y rows, z values
"""

from __future__ import annotations

import math
from typing import Any

KINDS = ("bar", "line", "scatter", "histogram", "heatmap")
MAX_POINTS = 400
MAX_SERIES = 8


def _cell(v: Any) -> Any:
    """A JSON-safe scalar: numbers rounded, NaN as None, everything else as text."""
    if v is None or isinstance(v, bool | str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 6)


def cells(values: Any) -> list[Any]:
    return [_cell(v) for v in list(values)[:MAX_POINTS]]


def normalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and bound a spec. Raises ValueError with the reason."""
    if not isinstance(spec, dict):
        raise ValueError("a chart is a mapping")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"chart kind must be one of {list(KINDS)}, not {kind!r}")
    out: dict[str, Any] = {"kind": kind, "title": str(spec.get("title") or "")[:160],
                           "x_title": str(spec.get("x_title") or "")[:80],
                           "y_title": str(spec.get("y_title") or "")[:80],
                           "y_format": "%" if spec.get("y_format") == "%" else ""}
    out["x"] = cells(spec.get("x") or [])
    if kind == "heatmap":
        out["y"] = cells(spec.get("y") or [])[:40]
        out["x"] = out["x"][:40]
        out["z"] = [cells(row)[:40] for row in list(spec.get("z") or [])[:40]]
        if not out["z"] or len(out["z"]) != len(out["y"]) or any(len(r) != len(out["x"])
                                                                 for r in out["z"]):
            raise ValueError("a heatmap needs z with one row per y and one value per x")
        return out
    series = spec.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("a chart needs at least one series")
    out["series"] = []
    for s in series[:MAX_SERIES]:
        y = cells(s.get("y") or [])
        if len(y) != len(out["x"]):
            raise ValueError("every series needs one y per x")
        out["series"].append({"name": str(s.get("name") or "")[:60], "y": y})
    if not out["x"]:
        raise ValueError("a chart needs x values")
    return out


def table(df: Any, max_rows: int = 50) -> dict[str, Any]:
    """A DataFrame as a small table artifact."""
    head = df.head(max_rows)
    return {"columns": [str(c) for c in head.columns],
            "rows": [[_cell(v) for v in row] for row in head.itertuples(index=False)],
            "n_rows": int(len(df))}
