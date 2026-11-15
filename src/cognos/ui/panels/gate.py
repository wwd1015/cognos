"""The gate block: the decision as one card, with send-back."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import html

from ... import service
from ...artifacts import StageResult
from ...engine.graph import LABELS, SEND_BACK_TARGETS
from ...engine.process import SEAT_OF_GATE, seat_label
from ...engine.state import RunState
from ..components import (
    empty,
    icon,
    step_badge,
)
from ._shared import _act, _field


def _send_back(gate: str, choose_target: bool) -> Any:
    """Send-back controls. Only the validation gate chooses a target stage; the others send the
    work back to the stage they review."""
    target = (dmc.Select(id=_field(gate, "target"), label="Send back to",
                         data=[{"value": t, "label": LABELS[t]} for t in SEND_BACK_TARGETS],
                         value="model", w=240, allowDeselect=False)
              if choose_target else None)
    return dmc.Accordion([dmc.AccordionItem([
        dmc.AccordionControl("Challenge / send back", icon=icon("tabler:arrow-back-up")),
        dmc.AccordionPanel(dmc.Stack([
            target,
            dmc.Textarea(id=_field(gate, "message"), label="What should the agent reconsider?",
                         placeholder="e.g. 'Leverage enters with the wrong sign — prefer a "
                                     "candidate without the collinear ratio.'",
                         autosize=True, minRows=2),
            dmc.Group([dmc.Button("Send back", id=_act(gate, "send_back"), color="oxblood",
                                  variant="light", leftSection=icon("tabler:arrow-back-up"))]),
        ], gap="xs")),
    ], value="send")], id={"type": "gate-sendback", "gate": gate}, value=None,
        variant="contained", mt="sm")


def gate_block(gate: str, state: RunState, res: StageResult | None, body: list | None = None,
               primary: list | None = None, *, send_back: bool = True,
               seat: str = "developer", proposal: str | None = None,
               evidence: list[tuple[str, str]] | None = None, fold: str | None = None,
               alternative: list | None = None) -> Any:
    """The human decision for a stage: a card when awaiting, a stamp when decided.

    ``seat`` is the role the workbench is acting as. A page that belongs to another seat
    stays readable and its controls are absent; the engine refuses the action too.

    A choice among things the agent already ranked reads as one card: ``proposal`` (what
    accepting means, in a line), ``evidence`` (engine figures as chips), the reason and the
    buttons. With ``fold``, ``body`` and ``alternative`` (controls for choosing differently,
    shown only to the owning seat) open behind a button of that name instead of filling the
    card. A gate whose decision is a set of settings passes no ``fold`` and keeps its form.
    """
    status = state.status_of(gate)
    last = state.last_decision(gate)
    owner = SEAT_OF_GATE.get(gate, "")
    mine = seat == owner
    header = dmc.Group([
        dmc.Group([dmc.ThemeIcon(icon("tabler:user-check", 16), variant="light", radius="xl",
                                 color="oxblood"),
                   dmc.Text(f"{LABELS[gate]} — {seat_label(owner)}", fw=650)], gap="xs"),
        step_badge(status),
    ], justify="space-between", mb="xs")
    if status == "skipped":
        return None
    if status in ("pending", "running"):
        return dmc.Card([header, empty("Opens when the stage finishes.")], className="cognos-gate")
    if status == "stale":
        return dmc.Card([header, empty("The stage is re-running; this gate re-opens with the new "
                                       "recommendation.")], className="cognos-gate")
    replaced = service.replaced(state) if status == "awaiting" else ""
    if replaced:  # a main run another run replaced: the engine refuses the decision here
        return dmc.Card([header, empty(f"Not decided here: {replaced}.")],
                        className="cognos-gate")
    if status == "blocked":
        return dmc.Card([header, dmc.Alert("The reviewed stage BLOCKed and this run is autonomous. "
                                           "Re-run interactively to send the work back.",
                                           color="red", variant="light")], className="cognos-gate")
    if status == "done":
        if last and last.seat == "express":
            who = "express preparation (not a signature)"
        elif last and last.seat:
            who = f"the {seat_label(last.seat)}"
        elif last and last.actor == "human":
            who = "a person"
        else:
            who = "express preparation"
        rec = [html.Div([
            html.Span(" · ".join(x for x in (
                last.action.replace("_", " "), last.by or who,
                last.at[:16].replace("T", " ") + " UTC") if x),
                      # a person's decision carries the seal; express preparation is not a signature
                      className="gate-stamp" + (" sealed" if last.actor == "human" else "")),
            dmc.Text([dmc.Text(last.action.replace("_", " "), fw=650, span=True), f" by {who}",
                      f" — {last.reason}" if last.reason else ""], size="sm"),
        ], className="gate-done")] if last else [empty("Decided.")]
        revise = (dmc.Group([dmc.Button("Revise decision", id={"type": "reopen", "gate": gate},
                                        variant="subtle", size="xs",
                                        leftSection=icon("tabler:edit"))], mt="xs")
                  if mine else dmc.Text(f"The {seat_label(owner)} can revise this.",
                                        size="xs", c="dimmed", mt="xs"))
        return dmc.Card([header, *rec, revise], className="cognos-gate")
    lead = [html.Div(proposal, className="gate-proposal") if proposal else None,
            html.Div([html.Span([f"{k} ", html.B(v)]) for k, v in evidence],
                     className="gate-evidence") if evidence else None]
    lead = [x for x in lead if x is not None]

    def folded(extra: list) -> list:
        if not fold:
            return [*(body or []), *extra]
        return [html.Details([html.Summary(fold, className="gate-fold"), *(body or []), *extra],
                             className="gate-more")]

    if not mine:
        return dmc.Card([
            header, *lead, *folded([]),
            dmc.Alert(f"This decision belongs to the {seat_label(owner)}. "
                      "Switch seat to act. The evidence above stays open to read.",
                      color="gray", variant="light", mt="sm",
                      icon=icon("tabler:lock")),
        ], className="cognos-gate cognos-gate-open")
    # awaiting, and this seat owns it
    parts = [header, *lead, *folded([dmc.Group(alternative, mt="sm", gap="sm")]
                                    if alternative else []),
             dmc.Textarea(id=_field(gate, "reason"), label="Why, in a line (kept in the decision log)",
                          autosize=True, minRows=1, mt="sm"),
             dmc.Group(primary or [], mt="sm", gap="sm")]
    if send_back:
        parts.append(_send_back(gate, choose_target=gate == "gate_validation"))
    return dmc.Card(parts, className="cognos-gate cognos-gate-open")
