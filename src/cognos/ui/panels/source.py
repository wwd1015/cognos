"""The data (gate_source, ADR-0018; the data catalog, ADR-0020)."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ... import service
from ...engine.graph import LABELS
from ...engine.state import RunState
from ..components import (
    icon,
    table,
)
from ._shared import _act, _field
from .gate import gate_block


def _src_upload(field: str, hint: str, multiple: bool) -> Any:
    """A drop zone; the callback saves what is dropped and keeps the paths in a hidden field."""
    return html.Div([
        dcc.Upload(id={"type": "src-upload", "field": field}, multiple=multiple,
                   className="cognos-upload",
                   children=dmc.Group([icon("tabler:upload", 16), dmc.Text(hint, size="xs")],
                                      gap="xs", justify="center")),
        html.Div(id={"type": "src-names", "field": field}),
        dmc.TextInput(id=_field("gate_source", field), value="", style={"display": "none"}),
    ])


FIT = {"fits": ("fits", "src-fit fits"), "partly": ("check", "src-fit partly"),
       "unclear": ("unclear", "src-fit unclear")}


def _fit_badge(f: dict | None) -> Any:
    if not f:
        return None
    label, cls = FIT.get(f["level"], (f["level"], "src-fit"))
    return html.Span(label, className=cls, title=" ".join(f.get("reasons") or []) or None)


def _fit_reasons(f: dict | None) -> Any:
    if not f or not f.get("reasons"):
        return None
    who = ("The engine's check against the intent" if f.get("origin") == "engine"
           else f"From the catalog source {f.get('origin')}")
    return html.Div([html.Div(who + " (advice; you decide)", className="src-who"),
                     html.Ul([html.Li(r) for r in f["reasons"]])], className="src-reasons")


def _grain(t: dict) -> str:
    m = t.get("summary") or {}
    parts = [t.get("grain") or ""]
    if m.get("measured"):
        parts.append(f"{m['n_rows']:,} rows")
    return " · ".join(p for p in parts if p)


def catalog_rows(cards: list[dict], fits: list[dict] | None = None) -> list:
    """The tables chosen, one line each (the first is the base table): open a line to read
    what the table holds and why it fits the intent, or not."""
    if not cards:
        return [html.Div("No table chosen yet. Search the catalog below, or add data further "
                         "down.", className="src-none")]
    fit = {f["table"]: f for f in fits or []}
    rows = []
    for n, t in enumerate(cards):
        m = t.get("summary") or {}
        cols = [[dmc.Text(c, ff="monospace", size="xs"), dmc.Text(text, size="xs")]
                for c, text in t.get("columns", {}).items()]
        cols += [[dmc.Text(c, ff="monospace", size="xs"),
                  dmc.Text("not described in the catalog", size="xs", c="dimmed")]
                 for c in m.get("undescribed", [])]
        owner = " · ".join(x for x in (t.get("owner"), t.get("refresh") and
                                       f"refreshed {t['refresh']}", t.get("location")) if x)
        lead = (html.Span("base", className="src-base", title="The base table: one row per "
                          "thing the model scores")
                if n == 0 else
                html.Button("make base", id={"type": "src-base", "table": t["id"]}, n_clicks=0,
                            className="src-make-base", title="Make this the base table"))
        rows.append(html.Details([
            html.Summary([
                lead,
                html.Span(t["id"], className="src-id"),
                html.Span(t.get("title") or "", className="src-title"),
                html.Span(_grain(t), className="src-grain"),
                _fit_badge(fit.get(t["id"])),
                html.Button("×", id={"type": "src-remove", "table": t["id"]}, n_clicks=0,
                            className="src-x", title=f"Remove {t['id']}",
                            **{"aria-label": f"Remove {t['id']}"}),
            ], className="src-sum"),
            html.Div([
                html.P(t.get("description", "")),
                html.Div(service.catalog_line(t), className="src-meta"),
                html.Div(owner, className="src-meta") if owner else None,
                *(html.Span(f"outcome: {o}", className="src-tag") for o in t.get("outcomes", [])),
                _fit_reasons(fit.get(t["id"])),
                html.Details([html.Summary(f"What the columns mean ({len(cols)})"),
                              table(["Column", "Meaning"], cols)], className="src-cols")
                if cols else None,
            ], className="src-detail"),
        ], className="src-row"))
    return rows


def catalog_results(cards: list[dict], chosen: list[str], *, text: str = "") -> list:
    """Tables found in the catalog: click one to add it."""
    if not cards:
        return [html.Div(f"Nothing in the catalog matches “{text}”." if text else
                         "The catalog is empty.", className="src-none")]
    return [html.Div([
        html.Button([html.Span(t["id"], className="src-id"),
                     html.Span(t.get("title") or "", className="src-title"),
                     html.Span(_grain(t), className="src-grain"),
                     html.Span("added" if t["id"] in chosen else "+ add", className="src-add")],
                    id={"type": "src-add", "table": t["id"]}, n_clicks=0,
                    className="src-hit", disabled=t["id"] in chosen,
                    title=t.get("description") or None)
        for t in cards], className="src-hits")]


def catalog_origin(cat: dict) -> str:
    """Which catalog is in force: the default shipped with COGNOS, the runs folder's own, or
    a catalog source a plugin adds (ADR-0022)."""
    if cat.get("kind") == "default":
        return ("The default catalog shipped with COGNOS, with synthetic data. To use your "
                f"team's tables, write {cat.get('team_file')} (cognos catalog --init starts it "
                "from this one).")
    if cat.get("kind") == "team":
        return f"Your team's catalog: {cat.get('file')}"
    return f"From the catalog source {cat.get('name')} ({cat.get('origin')})."


def source_form(info: dict | None = None) -> list:
    """Where the data comes from: tables from the data catalog (the chosen ones in a short
    list, the rest found by searching) and any data to add (files, Snowflake tables,
    read-only queries). The engine measures how they connect and the Data agent proposes the
    join at explore."""
    info = info or {}
    f = "gate_source"
    cat = info.get("catalog")
    if cat:
        picked = list(info.get("preselected") or [])
        n = cat.get("n_tables")
        catalog_part = [
            dmc.Group([dmc.Text(cat.get("title") or "The data catalog", fw=600, size="sm"),
                       dmc.Text(f"{n} tables" if n else "", size="xs", c="dimmed")],
                      gap="xs", align="baseline"),
            dmc.Text(catalog_origin(cat), size="xs", c="dimmed"),
            dmc.Text("The tables to build on, as the template proposes. The first is the base "
                     "table (one row per thing the model scores). Open a line to read what the "
                     "table holds; the badge says whether it fits the intent.", size="xs",
                     c="dimmed"),
            dcc.Store(id="src-pick", data=picked),
            html.Div(catalog_rows(info.get("cards") or [], info.get("fit")), id="src-chosen",
                     className="src-chosen"),
            dmc.Group([
                dmc.TextInput(id="src-search", placeholder="Search tables, columns, outcomes…",
                              leftSection=icon("tabler:search", 14), debounce=250, size="sm",
                              style={"flex": 1}),
                dmc.Button("Browse all", id="src-browse", n_clicks=0, variant="subtle",
                           size="sm", color="gray"),
            ], gap="xs", mt=4),
            html.Div(id="src-results"),
            dmc.TextInput(id=_field(f, "catalog"), value=json.dumps(picked),
                          style={"display": "none"}),
        ]
    else:
        catalog_part = [dmc.Alert(
            "The data catalog cannot be read" + (f": {info['catalog_error']}"
                                                 if info.get("catalog_error") else ".")
            + " Add the data below.", color="yellow", variant="light",
            icon=icon("tabler:alert-triangle"))]
    return [
        *catalog_part,
        html.Details([
            html.Summary("Add data the catalog does not have (optional)", className="gate-fold"),
            dmc.Text("The Data agent measures how it connects to the tables above and proposes "
                     "the join; you confirm it at the data gate.", size="xs", c="dimmed"),
            _src_upload("add_files", "Drop files, or click to choose (.csv, .xlsx, .parquet)",
                        multiple=True),
            dmc.Textarea(id=_field(f, "add_sql"), label="Snowflake tables or read-only queries",
                         autosize=True, minRows=2,
                         placeholder="one per line:\ncovenants = RISK.CREDIT.COVENANTS\n"
                                     "watchlist = SELECT obligor_id, as_of_date, on_watch FROM ..."),
            dmc.Text("Snowflake connects with the SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER and password "
                     "or key in this server's environment. Nothing secret is stored in the run.",
                     size="xs", c="dimmed"),
            dmc.Group([
                dmc.TextInput(id=_field(f, "target"), label="Dependent variable (optional)",
                              description="Left empty, the Data agent proposes it", w=260),
                dmc.TextInput(id=_field(f, "datetime_col"), label="Date column (optional)",
                              description="Left empty: the base table's, from the catalog",
                              w=260),
            ], align="flex-start", mt="xs"),
        ], className="gate-more src-more"),
    ]


def guide_text(guide: str) -> Any:
    """A template's data guide as text: wrapped lines joined, ``- `` lines as a list."""
    blocks: list[tuple[str, list[str]]] = []  # ("p" | "li", words)
    for line in guide.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            blocks.append(("li", [line[2:]]))
        elif blocks and blocks[-1][0] == "li" and not blocks[-1][1][-1].endswith((";", ".")):
            blocks[-1][1].append(line)  # a wrapped list item
        elif blocks and blocks[-1][0] == "p" and not blocks[-1][1][-1].endswith((":", ".")):
            blocks[-1][1].append(line)
        else:
            blocks.append(("p", [line]))
    out: list[Any] = []
    for kind, words in blocks:
        text = " ".join(words)
        if kind == "li":
            if not out or not isinstance(out[-1], dmc.List):
                out.append(dmc.List([], size="xs", spacing=2, withPadding=True))
            out[-1].children.append(dmc.ListItem(text))
        else:
            out.append(dmc.Text(text, size="xs"))
    return dmc.Stack(out, gap=4)


def source_panel(state: RunState, seat: str = "developer",
                 info: dict | None = None) -> Any:
    """Provide the data: the form while the run waits for it, what was given once it has.
    ``info`` is ``service.data_info(..., with_catalog=True)``: the template's note on the data,
    the data in force, the data catalog and the tables to start with checked."""
    info = info or {}
    status = state.status_of("gate_source")
    if status in ("pending", "skipped"):
        return None
    guide = (dmc.Alert(guide_text(info["guide"]),
                       title="What this kind of model is usually built on", color="gray",
                       variant="light", icon=icon("tabler:database"))
             if info.get("guide") else None)
    if status == "done" and state.last_decision("gate_source") is None:
        return dmc.Card([dmc.Group([
            dmc.Text(LABELS["gate_source"], fw=650),
            dmc.Text(f"The data came with the run: {info.get('given') or 'see the profile'}.",
                     size="sm", c="dimmed")], gap="sm")], className="cognos-gate")
    body = [guide, *source_form(info)] if status == "awaiting" else []
    block = gate_block("gate_source", state, None, seat=seat, send_back=False, body=body,
                       primary=[dmc.Button("Use this data", id=_act("gate_source", "accept"),
                                           leftSection=icon("tabler:database-import"))])
    if status == "done" and info.get("given"):
        return dmc.Stack([dmc.Text(["Data in use: ", dmc.Text(info["given"], span=True, fw=600)],
                                   size="sm"), block], gap=4)
    return block
