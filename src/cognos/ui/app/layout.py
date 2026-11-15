"""The page shell: the masthead, the app layout, the notices and the workspace's column split."""

from __future__ import annotations

import time
from typing import Any

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, html

from ... import __version__, service
from ..components import (
    icon,
)
from ..theme import MANTINE_THEME


def header() -> dmc.AppShellHeader:
    """The masthead: the app tile over a double rule, and a few quiet controls."""
    return dmc.AppShellHeader(html.Div(html.Div([
        dcc.Link([html.Img(src="/assets/app-icon.svg", alt="COGNOS", className="wm-tile wm-tile-light"),
                  html.Img(src="/assets/app-icon-dark.svg", alt="COGNOS", className="wm-tile wm-tile-dark"),
                  html.Span("Model development workbench", className="masthead-eyebrow")],
                 href="/", className="masthead-brand", title="COGNOS: all runs"),
        html.Div([
            dcc.Link("All runs", href="/", className="masthead-link"),
            html.Span(["as ", html.B(service.whoami())], className="masthead-meta",
                      title="Recorded on your decisions and answers (COGNOS_USER, else your "
                            "login name)."),
            html.Span(f"v{__version__}", className="masthead-meta"),
            dmc.Switch(id="theme-switch", onLabel=icon("tabler:moon", 13),
                       offLabel=icon("tabler:sun", 13), size="sm", color="dark"),
        ], className="masthead-tools"),
    ], className="masthead-row"), className="masthead"))


def build_layout() -> Any:
    return dmc.MantineProvider(
        id="mantine", theme=MANTINE_THEME, forceColorScheme="light",
        children=[
            dcc.Location(id="url"),
            dcc.Store(id="theme-store", storage_type="local"),
            dcc.Interval(id="poll", interval=1000),
            dmc.NotificationContainer(id="notify", position="top-right", autoClose=5000),
            dmc.AppShell([
                header(),
                dmc.AppShellMain(html.Div(id="page", className="cognos-page")),
            ], header={"height": 72}, padding="md"),
        ])


def column_layout(focus: bool | None) -> tuple[bool, int]:
    """(stage list shown, main column span of 24). Focus hides the stage list so the page
    gets the full width; the live activity opens from its button on every view."""
    left = focus is not True
    return left, 24 - (6 if left else 0)


def notice(title: str, message: str, color: str) -> dict:
    return {"action": "show", "title": title, "message": message, "color": color,
            "id": f"n{time.time_ns()}"}


def register(app: Dash) -> None:
    """The light or dark page, kept in the browser."""
    @app.callback(Output("mantine", "forceColorScheme"), Output("theme-store", "data"),
                  Input("theme-switch", "checked"), prevent_initial_call=True)
    def set_theme(dark):
        scheme = "dark" if dark else "light"
        return scheme, {"scheme": scheme}

    @app.callback(Output("theme-switch", "checked"), Input("theme-store", "modified_timestamp"),
                  State("theme-store", "data"))
    def load_theme(_, data):
        return bool(data and data.get("scheme") == "dark")
