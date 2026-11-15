"""The URL to the page."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import Dash, Input, Output

from ... import service
from ..panels import (
    compare_page,
)
from .runs import runs_page
from .workspace import workspace_page


def register(app: Dash) -> None:
    @app.callback(Output("page", "children"), Input("url", "pathname"))
    def route(path):
        path = path or "/"
        if path.startswith("/run/"):
            run_id = path.split("/")[2]
            try:
                service.state(run_id)
            except FileNotFoundError:
                return dmc.Container(dmc.Alert(f"No run {run_id}.", color="red"), py="xl")
            return workspace_page(run_id)
        if path.startswith("/compare/"):
            parts = path.strip("/").split("/")
            try:
                return dmc.Container(dmc.Stack(compare_page(service.compare(parts[1], parts[2])), gap="md"),
                                     size="xl", py="md")
            except (IndexError, FileNotFoundError):
                return dmc.Container(dmc.Alert("Pick two existing runs to compare (from the Runs page).",
                                               color="red"), py="xl")
        return runs_page()
