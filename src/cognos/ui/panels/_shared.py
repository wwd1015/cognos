"""Small helpers several stage pages share: gate control ids, a result's recommendation and
findings, and how a candidate or a word is shown."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import html

from ...artifacts import StageResult


def _field(gate: str, field: str) -> dict:
    return {"type": "gate-field", "gate": gate, "field": field}


def _act(gate: str, action: str) -> dict:
    return {"type": "gate-act", "gate": gate, "action": action}


def _rec(res: StageResult | None) -> dict | None:
    return (res.payload or {}).get("recommendation") if res is not None else None


def _findings(res: StageResult | None) -> list[dict]:
    return [f.model_dump(mode="json") for f in res.findings] if res is not None else []


def _summary_line(label: str, value: Any) -> Any:
    return html.Div([html.Span(label, className="sum-k"), html.Span(value, className="sum-v")],
                    className="sum-row")


def _words(v: Any) -> str:
    return str(v).replace("_", " ")


def _who(c: dict) -> Any:
    """A candidate as it is referred to on every page: its name, the id in a tooltip (the id
    is what decisions store; a person tells models apart by name)."""
    return dmc.Tooltip(dmc.Text(c.get("name") or c.get("label") or c["id"], size="sm", fw=600,
                                className="model-name"),
                       label=f"id {c['id']}", openDelay=300, withArrow=True)
