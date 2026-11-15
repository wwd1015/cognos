"""Questions for the sponsor and challenges, shown on the stage page that raised them."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import html

from ...engine.graph import LABELS
from ...engine.process import CORE_DESIGN
from ...engine.state import RunState
from ..components import (
    icon,
    section,
    table,
)

SPONSOR_NOTE = ("The sponsor is the business owner who asked for this model (the author of the "
                "intent document). The model developer records the sponsor's answer here; an "
                "agent never answers for them.")


GAP_STATUS = {"answered": "answered", "assumed": "assumption", "limitation": "limitation"}
# What each way of closing a question does, shown wherever questions are answered.
GAP_OUTCOMES = [
    ("Answer", "The sponsor's answer. It fills the brief and the stage that raised it runs "
               "again with it, as does everything after."),
    ("Accept as assumption", "No answer: the model rests on it. Nothing re-runs. It is listed "
                             "under assumptions in the white paper, and independent validation "
                             "asks how sensitive the result is to it."),
    ("Flag as limitation", "No answer: the model does not cover it. Nothing re-runs. It is "
                           "listed under limitations in the white paper, and independent "
                           "validation asks how it is mitigated."),
]


def gap_outcomes() -> Any:
    return dmc.Accordion([dmc.AccordionItem([
        dmc.AccordionControl("What each choice does", icon=icon("tabler:help-circle")),
        dmc.AccordionPanel(dmc.Stack([
            *[dmc.Text([dmc.Text(name + ". ", fw=650, span=True), what], size="sm")
              for name, what in GAP_OUTCOMES],
            dmc.Text("An open question never stops a stage. Only the required ones (use case, "
                     "horizon, default definition, segment) stop the final sign-off, and those "
                     "can only be answered. Type a note before choosing assumption or "
                     "limitation to say what is assumed or what is not covered.",
                     size="sm", c="dimmed"),
        ], gap=6)),
    ], value="outcomes")], variant="default", mb="xs")


def gap_required(g) -> bool:
    """Use, horizon, default definition and segment cannot be assumed: sign-off refuses while
    one is open. Every other question may be answered or accepted as an assumption."""
    return g.design_field in CORE_DESIGN


def gap_cell(g, seat: str = "developer") -> Any:
    """One question with its answer controls (open) or its recorded answer."""
    required = gap_required(g)
    if g.status == "open":
        ctrl = dmc.Group([
            dmc.TextInput(id={"type": "gap-text", "gap": g.id}, style={"flex": 1},
                          placeholder="The sponsor's answer"),
            dmc.Button("Answer", id={"type": "gap-act", "gap": g.id, "action": "answer"},
                       size="xs", disabled=seat != "developer"),
            dmc.Text("Cannot be assumed.", size="xs", c="dimmed") if required else
            dmc.Button("Accept as assumption",
                       id={"type": "gap-act", "gap": g.id, "action": "assume"},
                       size="xs", variant="subtle", disabled=seat != "developer"),
            None if required else
            dmc.Button("Flag as limitation",
                       id={"type": "gap-act", "gap": g.id, "action": "limitation"},
                       size="xs", variant="subtle", disabled=seat != "developer"),
        ], gap="xs", wrap="nowrap")
    else:
        ctrl = dmc.Text([dmc.Badge(GAP_STATUS.get(g.status, g.status), size="xs",
                                   variant="light", color="gray"), " ",
                         g.answer or "",
                         *([html.Span(f" — {g.answered_by}", className="stamp")]
                           if g.answered_by else [])], size="sm")
    return dmc.Stack([dmc.Text(g.question, size="sm"), ctrl], gap=4)


def gap_need(g) -> Any:
    if g.status != "open":
        return dmc.Badge(GAP_STATUS.get(g.status, g.status), color="gray", variant="outline",
                         size="xs")
    if gap_required(g):
        return dmc.Badge("required", color="oxblood", variant="filled", size="xs",
                         leftSection=icon("tabler:alert-circle", 11))
    return dmc.Badge("optional", color="gray", variant="light", size="xs")


def _open_first(gaps: list) -> list:
    return sorted(gaps, key=lambda g: (g.status != "open", not gap_required(g)))


def questions_summary(gaps: list) -> str:
    open_ = [g for g in gaps if g.status == "open"]
    required = sum(1 for g in open_ if gap_required(g))
    if not open_:
        return "Every question here is closed."
    if not required:
        return (f"{len(open_)} open. None stops the run: answer them, or accept them as "
                "recorded assumptions or limitations.")
    return (f"{len(open_)} open. {required} must be answered before sign-off (they cannot be "
            "assumed)." + (" The others do not stop the run: answer them, or accept them as "
                           "recorded assumptions or limitations." if len(open_) > required else ""))


def stage_questions(stage: str, state: RunState, seat: str = "developer") -> Any:
    """The questions this stage raised for the sponsor. While the intent gate is open, intake
    shows its own as the interview there (one place to answer, not two)."""
    gaps = [g for g in state.gaps if g.stage == stage]
    if stage == "intake" and state.status_of("gate_intent") == "awaiting":
        return None
    if not gaps:
        return None
    rows = [[gap_need(g), gap_cell(g, seat)] for g in _open_first(gaps)]
    return section(f"Questions from this stage ({sum(1 for g in gaps if g.status == 'open')} open)",
                   dmc.Text(questions_summary(gaps), size="sm", mb="xs"),
                   gap_outcomes(),
                   table(["", "Question for the business sponsor"], rows, max_height=420),
                   description=SPONSOR_NOTE + " An answer re-runs this stage.")


def challenge_rows(challenges: list) -> list[list]:
    return [[dmc.Code(c.id), c.source, LABELS.get(c.target_stage, c.target_stage),
             dmc.Badge(c.status, size="xs", variant="light",
                       color="oxblood" if c.status == "open" else "gray"),
             dmc.Stack([dmc.Text(c.message, size="sm"),
                        dmc.Text(f"↳ {c.response}", size="xs", c="dimmed") if c.response else None],
                       gap=2)] for c in challenges]


def stage_challenges(stage: str, state: RunState) -> Any:
    mine = [c for c in state.challenges if c.target_stage == stage]
    if not mine:
        return None
    return section(f"Challenges to this stage ({len(mine)})",
                   table(["Id", "From", "Sent to", "Status", "Challenge / response"],
                         challenge_rows(mine)),
                   description="Work sent back to this stage's agent, by a person at a gate or "
                               "by the Validation agent. The agent must answer each one.")
