"""Callbacks of the Ask window (ADR-0021)."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ... import service
from ...ask import agent_for
from ...engine.graph import (
    GATE_OF_STAGE,
)
from .. import ask as ask_ui
from .layout import notice
from .workspace import auto_step


def register(app: Dash) -> None:
    # --- Ask: a floating chat with the page's agent (ADR-0021) ------------------------------
    @app.callback(Output("ask-open", "data"), Input("ask-fab", "n_clicks"),
                  Input("ask-close", "n_clicks"), State("ask-open", "data"),
                  prevent_initial_call=True)
    def ask_toggle(_fab, _close, is_open):
        return ctx.triggered_id == "ask-fab" and not is_open

    @app.callback(Output("ask-dock", "hidden"), Output("ask-window", "hidden"),
                  Output("ask-title", "children"), Output("ask-sub", "children"),
                  Output("ask-foot", "children"), Output("ask-thread", "children"),
                  Input("ws-tab", "value"), Input("ws-step", "data"), Input("ws-key", "data"),
                  Input("ask-open", "data"), Input("ask-version", "data"),
                  State("ws-run", "data"))
    def ask_window(tab, step, _key, is_open, _version, run_id):
        hold = (no_update,) * 4
        if not run_id or tab != "workflow":
            return (True, True, *hold)
        st = service.state(run_id)
        sel = step or auto_step(st)
        agent = agent_for(sel)
        if agent is None:
            return (True, True, *hold)
        if not is_open:
            return (False, True, *hold)
        gate = GATE_OF_STAGE.get(sel)
        return (False, False, *ask_ui.heading(agent, sel),
                ask_ui.thread(service.ask_thread(run_id, sel), agent, sel,
                              gate_open=bool(gate) and st.status_of(gate) == "awaiting"))

    @app.callback(Output("ask-version", "data"), Output("ask-input", "value"),
                  Output("ask-error", "children"),
                  Input("ask-send", "n_clicks"), State("ask-input", "value"),
                  State("ws-run", "data"), State("ws-step", "data"), State("ask-version", "data"),
                  running=[(Output("ask-send", "loading"), True, False)],
                  prevent_initial_call=True)
    def ask_send(_n, text, run_id, step, version):
        sel = step or auto_step(service.state(run_id))
        try:
            service.ask(run_id, sel, text or "")
        except Exception as exc:  # no valid answer, a spent budget, an empty question
            return no_update, no_update, dmc.Text(str(exc), size="xs", c="red", px="sm")
        return (version or 0) + 1, "", None

    @app.callback(Output("ask-input", "value", allow_duplicate=True),
                  Input({"type": "ask-example", "q": ALL}, "n_clicks"), prevent_initial_call=True)
    def ask_example(clicks):
        if not any(clicks or []):
            return no_update
        return ctx.triggered_id["q"]

    @app.callback(
        Output({"type": "gate-field", "gate": ALL, "field": "message"}, "value",
               allow_duplicate=True),
        Output({"type": "gate-sendback", "gate": ALL}, "value"),
        Output("notify", "sendNotifications", allow_duplicate=True),
        Input({"type": "ask-use", "turn": ALL}, "n_clicks"),
        State("ws-run", "data"), State("ws-step", "data"), prevent_initial_call=True)
    def ask_use(clicks, run_id, step):
        fields, panels = ctx.outputs_list[0], ctx.outputs_list[1]
        if not any(clicks or []) or not fields:
            return no_update, no_update, no_update
        sel = step or auto_step(service.state(run_id))
        turns = service.ask_thread(run_id, sel)
        i = ctx.triggered_id["turn"]
        text = turns[i].get("suggested_challenge", "") if i < len(turns) else ""
        return ([text] * len(fields), ["send"] * len(panels),
                [notice("Copied into the send-back", "Read it, edit it if needed, and press "
                        "Send back to send it.", "oxblood")])
