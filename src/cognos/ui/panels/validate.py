"""The validate page."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc

from ...artifacts import StageResult
from ...engine.state import RunState
from .. import charts
from ..components import (
    findings_table,
    fmt,
    graph,
    icon,
    kpis,
    recommendation_card,
    section,
)
from ._shared import _act, _findings, _rec
from .gate import gate_block


def validate_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    v = p.get("validation_agent") or p.get("validator") or {}  # "validator": before ADR-0023
    verdict = res.verdict.value
    banner = dmc.Alert(
        "Confirmed target leakage: the model is invalid as built. Send the work back — a BLOCK "
        "cannot be accepted." if verdict == "BLOCK" else
        f"Engine rubric and independent challenge: {verdict}.",
        color={"PASS": "green", "WARN": "yellow", "FAIL": "orange", "BLOCK": "red"}.get(verdict, "gray"),
        variant="light", title=f"Validation verdict: {verdict}",
        icon=icon({"BLOCK": "tabler:ban", "FAIL": "tabler:alert-octagon"}.get(verdict, "tabler:shield-check")))
    rec_extra = dmc.Stack([
        dmc.Text(v.get("assessment", ""), size="sm"),
        dmc.Group([dmc.Text("Recommendation:", size="sm", c="dimmed"),
                   dmc.Badge(str(v.get("recommendation", "—")).replace("_", " "), variant="light")]),
        dmc.List([dmc.ListItem(c) for c in v.get("conditions", [])], size="sm")
        if v.get("conditions") else None,
    ], gap="xs") if v else None
    loops = state.loops.get("validation_agent", 0)
    routed = p.get("routed_findings") or []
    gate = gate_block("gate_validation", state, res, seat=seat,
                      proposal=(f"Verdict {verdict}: the Validation agent recommends "
                                f"{str(v.get('recommendation', '—')).replace('_', ' ')}."
                                if v else f"Verdict {verdict}."),
                      evidence=[("rubric", fmt(p.get("overall_score"), 2)),
                                ("blockers", fmt(len(p.get("blockers", [])))),
                                ("findings", fmt(len(res.findings))),
                                ("challenge loops", f"{loops}")], primary=[
        dmc.Button("Accept (record the risk)", id=_act("gate_validation", "accept"),
                   leftSection=icon("tabler:check"), disabled=verdict == "BLOCK"),
        dmc.Button("Reject the model", id=_act("gate_validation", "reject"), color="red",
                   variant="light", leftSection=icon("tabler:x")),
    ])
    return [
        banner,
        kpis([("Rubric score", fmt(p.get("overall_score"), 2), "mean of five axes"),
              ("Blockers", fmt(len(p.get("blockers", []))), None),
              ("Findings", fmt(len(res.findings)), None),
              ("Challenge loops", f"{loops}", f"{len(routed)} routed this pass")]),
        recommendation_card(_rec(res), extra=rec_extra),
        gate,
        section("Rubric", graph(charts.rubric(p.get("rubric") or {}, scheme))),
        trials_section(p.get("trials")),
        section("Findings", findings_table(_findings(res)),
                description="Engine findings re-derive risk from artifacts; agent findings "
                            "(val-*) are the Validation agent's challenge."),
    ]


def trials_section(tried: dict | None) -> Any:
    """The runs of this model as validation read them: every trial run looked at the test
    sample, so the count that matters is the model's."""
    from ...trials import describe

    tried = tried or {}
    if not tried.get("n_other_runs"):
        return None
    opened = len(tried.get("holdout_scored") or [])
    return section(
        f"Runs of this model ({tried['n_runs']})",
        kpis([("Other runs", fmt(tried["n_other_runs"]),
               f"{tried['n_deleted']} deleted, kept on record" if tried.get("n_deleted") else
               "each tried a decision differently"),
              ("Test sample used", fmt(tried["test_evaluations"]),
               f"times across all runs, {fmt(tried['test_evaluations_this_run'])} in this one"),
              ("Sealed holdout", f"scored for {opened}" if opened else "none",
               "champion(s); a trial run cannot open it")]),
        dmc.List([dmc.ListItem(describe(r)) for r in tried.get("lines") or []], size="sm",
                 spacing=6),
        description="Recorded when this validation ran. The live list is on the Run record.")
