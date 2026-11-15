"""Callbacks of the run workspace: the poll that redraws it, the column split, the activity
drawer, focus and the stage rail."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ... import service
from ...engine.graph import (
    GATE_OF_STAGE,
    STAGE_AFTER_GATE,
)
from ..components import (
    icon,
)
from ..panels import (
    stage_panel,
)
from .layout import column_layout
from .workspace import activity, audit_tab, auto_step, questions_tab, rail, run_header


def register(app: Dash) -> None:
    @app.callback(
        Output("ws-head", "children"), Output("ws-rail", "children"),
        Output("ws-activity", "children"), Output("ws-content", "children"),
        Output("ws-key", "data"),
        Input("poll", "n_intervals"), Input("ws-step", "data"), Input("ws-tab", "value"),
        Input("theme-store", "data"), Input("ws-seat", "data"),
        State("ws-run", "data"), State("ws-key", "data"))
    def refresh_workspace(_, step, tab, theme, seat, run_id, key):
        if not run_id:
            return (no_update,) * 5
        st = service.state(run_id)
        scheme = (theme or {}).get("scheme", "light")
        seat = seat or "developer"
        sel = step or auto_step(st)
        gate = GATE_OF_STAGE.get(sel)
        pkg = f"{st.package.version}:{st.package.status}" if st.package else ""
        if tab == "workflow":
            asked = ",".join(f"{g.id}:{g.status}" for g in st.gaps if g.stage == sel)
            sent = ",".join(f"{c.id}:{c.status}" for c in st.challenges if c.target_stage == sel)
            before = STAGE_AFTER_GATE.get(sel)
            content_key = (f"wf|{sel}|{st.status_of(sel)}|{st.steps[sel].runs}|"
                           f"{st.status_of(gate) if gate else ''}|"
                           f"{st.status_of(before) if before else ''}|{len(st.decisions)}|"
                           f"{scheme}|{seat}|{pkg}|{asked}|{sent}")
        elif tab == "questions":
            content_key = "q|" + "|".join(f"{g.id}:{g.status}" for g in st.gaps) + "|" + "|".join(
                f"{c.id}:{c.status}" for c in st.challenges) + f"|{len(st.decisions)}|{scheme}|{seat}|{pkg}|{service.trial_stamp(run_id)}"
        else:
            content_key = f"a|{len(service.audit(run_id))}|{scheme}"
        full_key = f"{st.version}|{content_key}"
        if key == full_key:
            return (no_update,) * 5
        content = no_update
        if not key or key.split("|", 1)[1] != content_key:
            if tab == "workflow":
                res = service.results(run_id).get(sel)
                info = (service.data_info(run_id, with_catalog=st.status_of("gate_source")
                                          == "awaiting")
                        if sel in STAGE_AFTER_GATE else None)
                content = dmc.Stack(stage_panel(sel, res, st, scheme,
                                                service.step_changes(run_id, sel), seat, info),
                                    gap="md")
            elif tab == "questions":
                content = questions_tab(st, seat)
            else:
                content = audit_tab(run_id)
        return (run_header(st), rail(st, sel if tab == "workflow" else None), activity(run_id),
                content, full_key)

    @app.callback(Output("ws-left-col", "style"), Output("ws-main-col", "span"),
                  Input("ws-focus", "data"))
    def widen(focus):
        left, main = column_layout(focus)
        return {} if left else {"display": "none"}, {"base": 24, "md": main}

    @app.callback(Output("ws-activity-drawer", "opened"), Input("ws-activity-open", "n_clicks"),
                  prevent_initial_call=True)
    def open_activity(_clicks):
        return True

    @app.callback(Output("ws-focus", "data"), Output("ws-focus-toggle", "children"),
                  Output("ws-focus-tip", "label"),
                  Input("ws-focus-toggle", "n_clicks"), State("ws-focus", "data"))
    def focus(_clicks, focused):
        focused = focused is True  # an older browser may hold the per-side choice
        if ctx.triggered_id == "ws-focus-toggle":
            focused = not focused
        return (focused, icon("tabler:layout-sidebar-left-"
                              + ("expand" if focused else "collapse"), 18),
                "Show the stage list" if focused else "Hide the stage list")

    @app.callback(Output("ws-step", "data"), Output("ws-tab", "value"),
                  Input({"type": "rail", "step": ALL}, "n_clicks"), prevent_initial_call=True)
    def pick_step(clicks):
        if not ctx.triggered_id or not any(clicks or []):
            return no_update, no_update
        return ctx.triggered_id["step"], "workflow"
