"""Callbacks of the gates: the seat, decisions, re-opening or retrying, and answers to the
sponsor's questions."""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ... import service
from ...engine import GateError
from ...engine.graph import (
    LABELS,
)
from .layout import notice
from .payloads import gate_payload


def register(app: Dash) -> None:
    @app.callback(Output("ws-seat", "data"), Input("seat-switch", "value"))
    def choose_seat(value):
        return value or "developer"

    @app.callback(
        Output("notify", "sendNotifications", allow_duplicate=True),
        Output("ws-key", "data", allow_duplicate=True),
        Input({"type": "gate-act", "gate": ALL, "action": ALL}, "n_clicks"),
        State({"type": "gate-field", "gate": ALL, "field": ALL}, "value"),
        State({"type": "gate-field", "gate": ALL, "field": ALL}, "id"),
        State("ws-run", "data"), State("ws-seat", "data"), prevent_initial_call=True)
    def gate_action(clicks, values, ids, run_id, seat):
        trig = ctx.triggered_id
        if not trig or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        gate, action = trig["gate"], trig["action"]
        fields = {i["field"]: v for i, v in zip(ids, values, strict=False) if i["gate"] == gate}
        try:
            payload, reason = gate_payload(gate, action, fields, run_id)
            service.submit_gate(run_id, gate, action, payload, reason, seat=seat)
        except GateError as exc:
            return [notice("Decision refused", str(exc), "red")], no_update
        return [notice("Decision recorded", f"{LABELS[gate]}: {action.replace('_', ' ')}", "oxblood")], None


def register_answers(app: Dash) -> None:
    @app.callback(
        Output("notify", "sendNotifications", allow_duplicate=True),
        Output("ws-key", "data", allow_duplicate=True),
        Input({"type": "reopen", "gate": ALL}, "n_clicks"),
        Input({"type": "retry", "step": ALL}, "n_clicks"),
        State("ws-run", "data"), State("ws-seat", "data"), prevent_initial_call=True)
    def reopen_or_retry(reopen_clicks, retry_clicks, run_id, seat):
        trig = ctx.triggered_id
        if not trig or not any(c for c in (reopen_clicks or []) + (retry_clicks or []) if c):
            return no_update, no_update
        try:
            if trig["type"] == "reopen":
                service.reopen(run_id, trig["gate"], seat=seat, by=service.whoami())
                return [notice("Gate re-opened", "Revise the decision below.", "violet")], None
            service.retry(run_id, trig["step"], by=service.whoami())
            return [notice("Retrying", LABELS[trig["step"]], "blue")], None
        except GateError as exc:
            return [notice("Not possible", str(exc), "red")], no_update

    @app.callback(
        Output("notify", "sendNotifications", allow_duplicate=True),
        Output("ws-key", "data", allow_duplicate=True),
        Input({"type": "gap-act", "gap": ALL, "action": ALL}, "n_clicks"),
        State({"type": "gap-text", "gap": ALL}, "value"),
        State({"type": "gap-text", "gap": ALL}, "id"),
        State("ws-run", "data"), State("ws-seat", "data"), prevent_initial_call=True)
    def answer(clicks, texts, ids, run_id, seat):
        trig = ctx.triggered_id
        if not trig or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        text = next((v for i, v in zip(ids, texts, strict=False) if i["gap"] == trig["gap"]), "") or ""
        assume, limitation = trig["action"] == "assume", trig["action"] == "limitation"
        if not (assume or limitation) and not text.strip():
            return [notice("Answer needed", "Type an answer, or record it as an assumption or "
                           "a limitation.", "yellow")], no_update
        try:
            service.answer_gap(run_id, trig["gap"], text, assume=assume, limitation=limitation,
                               seat=seat)
        except GateError as exc:
            return [notice("Not possible", str(exc), "red")], no_update
        said = ("Recorded as a limitation. Validation will ask how it is mitigated."
                if limitation else
                "Recorded as an assumption. Validation will ask how sensitive the result is "
                "to it." if assume else "The stage that raised it will re-run.")
        return [notice("Recorded", said, "oxblood")], None
