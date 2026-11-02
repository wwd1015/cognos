"""The Ask window (ADR-0021): a floating chat with the agent that owns the stage page.

Pure functions of the thread, like the panels: the callbacks in ``app.py`` call the service and
hand the turns here. Asking changes nothing in the run; a suggested send-back is copied into the
gate's form, and a person still sends it.
"""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ..agents.contracts import FRIENDLY
from ..engine.graph import LABELS
from .components import icon

# What a person can ask on each page, to show the kind of question that works.
EXAMPLES = {
    "intake": ["Which questions must the sponsor answer?", "What did the documents state?"],
    "explore": ["Why is a column excluded?", "How do the sources join?", "Why this target?"],
    "ideate": ["Why is this specification ranked first?", "Which frameworks were rejected?"],
    "model": ["Why are these models in the pool?", "Which factors were screened out?"],
    "backtest": ["Why this champion?", "How stable is it over time?"],
    "validate": ["Which findings matter most?", "What would clear the verdict?"],
    "comply": ["What must a person do before use?", "Which evidence is missing?"],
    "document": ["Where does this number come from?", "What are the limitations?"],
}


def dock() -> Any:
    """The button and the window, fixed to the bottom right of the workspace."""
    window = html.Div([
        html.Div([
            html.Div([html.Div(id="ask-title", className="ask-title"),
                      html.Div(id="ask-sub", className="ask-sub")]),
            dmc.ActionIcon(icon("tabler:x", 16), id="ask-close", variant="subtle", color="gray",
                           n_clicks=0, **{"aria-label": "Close"}),
        ], className="ask-head"),
        html.Div(id="ask-thread", className="ask-thread"),
        html.Div(id="ask-error"),
        html.Div([
            dmc.Textarea(id="ask-input", placeholder="Ask about this page…", autosize=True,
                         minRows=1, maxRows=4, style={"flex": 1}),
            dmc.Button("Ask", id="ask-send", n_clicks=0, color="oxblood", size="sm"),
        ], className="ask-compose"),
        html.Div(id="ask-foot", className="ask-foot"),
    ], id="ask-window", className="ask-window", hidden=True)
    return html.Div([
        dcc.Store(id="ask-open", data=False),
        dcc.Store(id="ask-version", data=0),
        window,
        dmc.Button("Ask", id="ask-fab", n_clicks=0, className="ask-fab", color="oxblood",
                   radius="xl", leftSection=icon("tabler:message-question", 16)),
    ], id="ask-dock", className="ask-dock", hidden=True)


def heading(agent: str, stage: str) -> tuple[str, str, str]:
    """Title, subtitle and footer for the window on a stage's page."""
    name = FRIENDLY[agent]
    return (name, f"Answers about the {LABELS.get(stage, stage)} page",
            f"Sees only what the {name} sees. Asking changes nothing in the run; the thread is "
            "kept with it.")


def thread(turns: list[dict[str, Any]], agent: str, stage: str, *, gate_open: bool) -> list:
    """The conversation, oldest first. ``gate_open``: the page's gate is waiting, so a suggested
    send-back can be copied into its form."""
    if not turns:
        return [html.Div([
            html.P("Ask what something on this page means, why it was proposed, or where a "
                   "number comes from."),
            html.Div([html.Button(q, id={"type": "ask-example", "q": q}, n_clicks=0,
                                  className="ask-example")
                      for q in EXAMPLES.get(stage, [])], className="ask-examples"),
        ], className="ask-empty")]
    who = FRIENDLY[agent].upper()
    out: list = []
    for i, t in enumerate(turns):
        out.append(html.Div([
            html.Div(t["question"]),
            html.Div(f"{t.get('by') or ''} · {(t.get('at') or '')[11:16]}", className="ask-meta"),
        ], className="ask-me"))
        body: list = [html.Div(who, className="ask-who"),
                      html.Div(t["answer"], className="ask-text")]
        if t.get("cites"):
            body.append(html.Div([html.Span(f"{c['id']} = {c['value']}", className="ask-fact")
                                  for c in t["cites"]], className="ask-facts"))
        if t.get("suggested_challenge"):
            body.append(html.Div([
                html.Div("Suggested send-back", className="ask-who"),
                html.Div(t["suggested_challenge"]),
                dmc.Button("Use in send-back", id={"type": "ask-use", "turn": i}, n_clicks=0,
                           size="compact-xs", variant="light", color="oxblood", mt=6,
                           leftSection=icon("tabler:arrow-back-up", 12))
                if gate_open else html.Div("The gate is not open, so nothing can be sent back "
                                           "now.", className="ask-meta"),
            ], className="ask-suggest"))
        out.append(html.Div(body, className="ask-agent"))
    return out
