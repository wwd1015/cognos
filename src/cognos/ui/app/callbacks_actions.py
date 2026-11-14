"""Callbacks of the run's actions: a trial run, the main run, a team template, an agent call."""

from __future__ import annotations

import json

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ... import service
from ...engine import GateError
from ...engine.graph import (
    GATE_OF_STAGE,
    LABELS,
)
from .layout import notice
from .workspace import trial_gate_options


def register(app: Dash) -> None:
    @app.callback(Output("trial-modal", "opened"), Output("trial-gate", "data"),
                  Output("trial-gate", "value"), Output("trial-note", "value"),
                  Output("trial-feedback", "children"),
                  Input("trial-open", "n_clicks"), Input("trial-cancel", "n_clicks"),
                  State("ws-run", "data"), State("ws-step", "data"), prevent_initial_call=True)
    def open_trial(opened, cancelled, run_id, step):
        if ctx.triggered_id == "trial-open" and opened:
            options = trial_gate_options(run_id)
            values = [o["value"] for o in options]
            # Start on the decision of the page being read, else the latest one reached.
            here = GATE_OF_STAGE.get(step or "")
            return (True, options, here if here in values else (values[-1] if values else None),
                    "", None)
        if ctx.triggered_id == "trial-cancel" and cancelled:
            return False, no_update, no_update, no_update, None
        return (no_update,) * 5

    @app.callback(Output("url", "pathname", allow_duplicate=True),
                  Output("trial-feedback", "children", allow_duplicate=True),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Input("trial-go", "n_clicks"), State("trial-gate", "value"),
                  State("trial-note", "value"), State("ws-run", "data"), State("ws-seat", "data"),
                  prevent_initial_call=True)
    def make_trial(clicks, gate, note, run_id, seat):
        if not clicks:
            return no_update, no_update, no_update
        try:
            new_id = service.trial_run(run_id, gate or "", note or "", seat=seat)
        except (GateError, OSError) as exc:
            return no_update, dmc.Alert(str(exc), color="red", variant="light"), no_update
        return f"/run/{new_id}", None, [notice(
            "Trial run made", f"Decide “{LABELS[gate]}” differently here. The run you came "
            "from is unchanged.", "oxblood")]

    @app.callback(Output("promote-modal", "opened"), Output("promote-note", "value"),
                  Output("promote-feedback", "children"),
                  Input("promote-open", "n_clicks"), Input("promote-cancel", "n_clicks"),
                  prevent_initial_call=True)
    def open_promote(opened, cancelled):
        if ctx.triggered_id == "promote-open" and opened:
            return True, "", None
        if ctx.triggered_id == "promote-cancel" and cancelled:
            return False, no_update, None
        return no_update, no_update, no_update

    @app.callback(Output("promote-modal", "opened", allow_duplicate=True),
                  Output("promote-feedback", "children", allow_duplicate=True),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Output("ws-key", "data", allow_duplicate=True),
                  Input("promote-go", "n_clicks"), State("promote-note", "value"),
                  State("ws-run", "data"), State("ws-seat", "data"), prevent_initial_call=True)
    def promote(clicks, note, run_id, seat):
        if not clicks:
            return (no_update,) * 4
        try:
            service.promote_trial(run_id, note or "", seat=seat)
        except (GateError, OSError) as exc:
            return no_update, dmc.Alert(str(exc), color="red", variant="light"), no_update, no_update
        return False, None, [notice("This is now the main run",
                                    "It goes on to independent validation.", "oxblood")], None

    @app.callback(Output("tpl-modal", "opened"), Output("tpl-title", "value"),
                  Output("tpl-feedback", "children"),
                  Input("tpl-open", "n_clicks"), Input("tpl-cancel", "n_clicks"),
                  prevent_initial_call=True)
    def open_template(opened, cancelled):
        if ctx.triggered_id == "tpl-open" and opened:
            return True, "", None
        if ctx.triggered_id == "tpl-cancel" and cancelled:
            return False, no_update, None
        return no_update, no_update, no_update

    @app.callback(Output("tpl-modal", "opened", allow_duplicate=True),
                  Output("tpl-feedback", "children", allow_duplicate=True),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Input("tpl-go", "n_clicks"), State("tpl-title", "value"),
                  State("tpl-description", "value"), State("ws-run", "data"),
                  prevent_initial_call=True)
    def save_template(clicks, title, description, run_id):
        if not clicks:
            return (no_update,) * 3
        try:
            service.save_template(run_id, title or "", description or "")
        except (GateError, OSError, ValueError) as exc:
            return no_update, dmc.Alert(str(exc), color="red", variant="light"), no_update
        return False, None, [notice("Template saved", f"“{title}” is under Team templates when "
                                    "a run is started.", "oxblood")]

    @app.callback(Output("io-modal", "opened"), Output("io-body", "children"),
                  Input({"type": "audit-row", "call": ALL}, "n_clicks"), State("ws-run", "data"),
                  prevent_initial_call=True)
    def show_io(clicks, run_id):
        trig = ctx.triggered_id
        if not trig or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        io = service.agent_io(run_id, trig["call"])
        inp, out = io.get("input") or {}, io.get("output") or {}
        return True, dmc.Tabs([
            dmc.TabsList([dmc.TabsTab("Output", value="out"), dmc.TabsTab("Context slice", value="ctx"),
                          dmc.TabsTab("Prompt", value="prompt"), dmc.TabsTab("System", value="sys")]),
            dmc.TabsPanel(dmc.Code(json.dumps(out, indent=1, default=str), block=True), value="out"),
            dmc.TabsPanel(dmc.Code(json.dumps(inp.get("context"), indent=1, default=str), block=True),
                          value="ctx"),
            dmc.TabsPanel(dmc.Code(inp.get("prompt", ""), block=True), value="prompt"),
            dmc.TabsPanel(dmc.Code(inp.get("system", ""), block=True), value="sys"),
        ], value="out")
