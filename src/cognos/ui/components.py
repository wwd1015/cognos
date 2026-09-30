"""Small, reusable workbench pieces (Dash Mantine Components)."""

from __future__ import annotations

import math
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..agents.contracts import FRIENDLY
from .theme import RUN, STEP, VERDICT


def icon(name: str, size: int = 16, **kw) -> DashIconify:
    return DashIconify(icon=name, width=size, **kw)


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "—"
    return f"{f:,.{nd}f}".rstrip("0").rstrip(".") if abs(f) < 1e6 else f"{f:,.0f}"


def pct(v: Any, nd: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(f) else f"{f:.{nd}%}"


def verdict_badge(verdict: str | None, size: str = "md") -> dmc.Badge | None:
    if not verdict:
        return None
    color, ic, label = VERDICT.get(verdict, ("gray", "tabler:point", verdict))
    return dmc.Badge(label, color=color, variant="light", size=size, leftSection=icon(ic, 14))


def step_badge(status: str, size: str = "sm") -> dmc.Badge:
    color, ic, label = STEP.get(status, ("gray", "tabler:point", status))
    return dmc.Badge(label, color=color, variant="light", size=size, leftSection=icon(ic, 12))


def run_badge(status: str, size: str = "md") -> dmc.Badge:
    color, label = RUN.get(status, ("gray", status))
    return dmc.Badge(label, color=color, variant="filled" if status in ("awaiting",) else "light",
                     size=size)


def kpi(label: str, value: str, hint: str | None = None) -> dmc.Paper:
    return dmc.Paper([
        dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600, lts=0.4),
        dmc.Text(value, fz=22, fw=650, mt=2, style={"lineHeight": 1.2}),
        dmc.Text(hint, size="xs", c="dimmed", mt=2) if hint else None,
    ], withBorder=True, radius="md", p="sm")


def kpis(items: list[tuple[str, str, str | None]]) -> dmc.SimpleGrid:
    return dmc.SimpleGrid([kpi(*i) for i in items], cols={"base": 2, "sm": 3, "lg": len(items)},
                          spacing="sm")


def section(title: str, *children, right=None, description: str | None = None) -> dmc.Card:
    head = dmc.Group([
        dmc.Stack([dmc.Text(title, fw=650, size="md"),
                   dmc.Text(description, size="xs", c="dimmed") if description else None], gap=0),
        right,
    ], justify="space-between", align="flex-start", mb="sm")
    return dmc.Card([head, *children])


def table(head: list[str], rows: list[list[Any]], *, striped: bool = True,
          max_height: int | None = None) -> Any:
    # Cells may be components, so the table is built from rows (Table.data takes plain values only).
    t = dmc.Table([
        dmc.TableThead(dmc.TableTr([dmc.TableTh(h) for h in head])),
        dmc.TableTbody([dmc.TableTr([dmc.TableTd(c if c is not None else "") for c in row])
                        for row in rows]),
    ], striped=striped, highlightOnHover=True, withTableBorder=False, verticalSpacing="xs",
        fz="sm", tabularNums=True)
    if max_height:
        return dmc.ScrollArea(t, mah=max_height, type="auto")
    return dmc.ScrollArea(t, type="auto", offsetScrollbars=True)


def markdown(text: str) -> dmc.TypographyStylesProvider:
    return dmc.TypographyStylesProvider(dcc.Markdown(text or "", link_target="_blank"),
                                        className="cognos-md")


def graph(fig, placeholder: str | None = None, **kw) -> Any:
    """A chart, or the placeholder text when there is nothing to plot. (Never use ``graph(...) or
    x``: Dash components define ``__len__`` over their children, so a Graph is falsy.)"""
    if fig is None:
        return empty(placeholder) if placeholder else None
    return dcc.Graph(figure=fig, config={"displaylogo": False, "responsive": True,
                                         "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, **kw)


def empty(text: str) -> dmc.Text:
    return dmc.Text(text, c="dimmed", size="sm", fs="italic")


def findings_table(findings: list[dict]) -> Any:
    if not findings:
        return empty("No findings.")
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "gray", "INFO": "gray"}
    rows = []
    for f in sorted(findings, key=lambda f: order.get(f["severity"], 9)):
        source = "Agent" if str(f["id"]).startswith(("val-", "oa-")) else "Engine"
        rows.append([
            dmc.Badge(f["severity"].title(), color=colors.get(f["severity"], "gray"), variant="light",
                      size="sm"),
            dmc.Text(source, size="xs", c="dimmed"),
            dmc.Text(f.get("category", ""), size="xs"),
            dmc.Stack([dmc.Text(f["message"], size="sm"),
                       dmc.Text(f.get("suggestion") or "", size="xs", c="dimmed")
                       if f.get("suggestion") else None], gap=2),
        ])
    return table(["Severity", "Source", "Category", "Finding"], rows)


def recommendation_card(rec: dict | None, *, extra=None, title: str | None = None) -> dmc.Card:
    """The agent's recommendation, labelled with who produced it and how."""
    if not rec:
        return section(title or "Agent recommendation", empty("No recommendation recorded."))
    out = rec.get("output") or {}
    agent = FRIENDLY.get(rec.get("agent", ""), rec.get("agent", "Agent"))
    backend = rec.get("provider", "")
    model = rec.get("model") or ""
    meta = dmc.Group([
        dmc.Badge(f"{backend}" + (f" · {model}" if model and model not in (backend, "heuristic",
                                                                           "replay") else ""),
                  variant="outline", size="sm", color="gray"),
        dmc.Badge(f"attempt {rec.get('attempts', 1)}", variant="outline", size="sm", color="gray")
        if rec.get("attempts", 1) > 1 else None,
        dmc.Badge("reused (override only)", variant="outline", size="sm", color="gray")
        if rec.get("reused") else None,
    ], gap=6)
    body = [
        dmc.Group([dmc.ThemeIcon(icon("tabler:sparkles", 16), variant="light", radius="xl"),
                   dmc.Text(agent, fw=650), meta], gap="xs", mb="xs"),
        dmc.Text(out.get("summary", ""), size="sm"),
    ]
    if out.get("uncertainties"):
        body.append(dmc.Alert(dmc.List([dmc.ListItem(u) for u in out["uncertainties"]], size="sm"),
                              title="Uncertainties", color="gray", variant="light", mt="sm",
                              icon=icon("tabler:help-circle")))
    if out.get("responses_to_challenges"):
        body.append(dmc.Alert(dmc.Stack([
            dmc.Text([dmc.Code(r["challenge_id"]), " ",
                      dmc.Badge("changed" if r["changed_recommendation"] else "kept",
                                size="xs", variant="light",
                                color="blue" if r["changed_recommendation"] else "gray"),
                      " ", r["response"]], size="sm")
            for r in out["responses_to_challenges"]], gap=4),
            title="Responses to challenges", color="violet", variant="light", mt="sm",
            icon=icon("tabler:message-reply")))
    if extra is not None:
        body.append(html.Div(extra, style={"marginTop": 12}))
    return dmc.Card(body, className="cognos-rec")


def status_icon(status: str) -> Any:
    color, ic, _ = STEP.get(status, ("gray", "tabler:point", status))
    return dmc.ThemeIcon(icon(ic, 14, className="spin" if status == "running" else None),
                         color=color, variant="light", radius="xl", size="sm")
