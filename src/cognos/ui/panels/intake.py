"""The intake page."""

from __future__ import annotations

import dash_mantine_components as dmc

from ...artifacts import StageResult
from ...engine.graph import LABELS
from ...engine.state import RunState
from ..components import (
    empty,
    findings_table,
    fmt,
    icon,
    kpis,
    recommendation_card,
    section,
    table,
)
from ._shared import _act, _field, _findings, _rec
from .gate import gate_block

_BASIS = {"stated": ("document", "oxblood"), "answered": ("answered", "teal"),
          "profile": ("profile", "blue"), "inferred": ("inferred", "yellow"),
          "missing": ("open", "gray")}


def intake_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    brief = p.get("brief", [])
    blocking = {q["id"]: q for q in p.get("questions", []) if q.get("blocking")}
    why = {q["id"]: q.get("why_it_matters") for q in p.get("questions", [])}
    gaps = [g for g in state.gaps if g.stage == "intake"]
    open_gaps = [g for g in gaps if g.status == "open"]
    q_inputs = [dmc.Textarea(
        id=_field("gate_intent", f"answer::{g.id}"), autosize=True, minRows=1,
        label=g.question + ("  (blocking)" if g.id in blocking else ""),
        description=why.get(g.id) or "", placeholder="The sponsor's answer") for g in open_gaps]
    interview = (dmc.Stack([dmc.Text("Interview — what the sponsor still has to say", fw=600,
                                     size="sm"), *q_inputs], gap="xs")
                 if q_inputs else dmc.Text("The interview is finished: nothing is open.",
                                           size="sm", c="dimmed"))
    gate = gate_block("gate_intent", state, res, seat=seat, body=[interview], primary=[
        dmc.Button("Submit answers", id=_act("gate_intent", "edit"),
                   leftSection=icon("tabler:message-reply"), disabled=not q_inputs),
        dmc.Button("Confirm the intent", id=_act("gate_intent", "accept"),
                   variant="light" if q_inputs else "filled", leftSection=icon("tabler:check")),
    ])
    # A design answer given after intake lives on the run, not in this stage's result.
    later = {f: v for f, v in state.overrides.design.items() if v}
    rows = []
    for b in brief:
        value, basis = b.get("value") or "", b.get("basis", "missing")
        if later.get(b["field"]) and later[b["field"]] != value:
            value, basis = later[b["field"]], "answered"
        label, color = _BASIS.get(basis, (basis, "gray"))
        rows.append([dmc.Text(b["label"] + (" *" if b.get("required") else ""), size="sm", fw=500),
                     dmc.Text(value, size="sm") if value else dmc.Text("—", size="sm", c="dimmed"),
                     dmc.Badge(label, color=color, variant="light", size="sm")])
    up = p.get("update") or {}
    prior = p.get("prior") or {}
    run = prior.get("run") or {}
    update_sections = []
    if up:
        update_sections = [
            section("Update request",
                    dmc.Text([dmc.Text("Scope: ", fw=600, span=True),
                              str(up.get("scope", "")).replace("_", " "), " — ",
                              up.get("rationale", "")], size="sm", mb="xs"),
                    table(["Requested change", "Type", "Enters at"],
                          [[c["change"], c["type"].replace("_", " "),
                            LABELS.get(c["affects"], "—")] for c in up.get("change_items", [])])
                    if up.get("change_items") else empty("No change was listed."),
                    description="The engine re-estimates in every case. The scope tells the "
                                "design how far from the existing specification to look."),
            section("The existing model",
                    kpis([("Family", up.get("incumbent_family") or "not identified", None),
                          ("Artifacts", fmt(prior.get("n_artifacts")), None),
                          ("Code files", fmt(prior.get("n_code_files")), None),
                          ("Earlier COGNOS run", "yes" if run else "none",
                           run.get("run_id"))]),
                    table(["Family named in the artifacts", "Mentions"],
                          [[dmc.Text(m["family"], ff="monospace", size="sm"), fmt(m["mentions"])]
                           for m in prior.get("family_mentions", [])])
                    if prior.get("family_mentions") else None,
                    table(["Recorded by the earlier run", ""],
                          [["Champion", str(run.get("champion_label") or run.get("champion_family"))],
                           [f"CV {run.get('metric')}", fmt(run.get("cv_mean"))],
                           [f"Holdout {run.get('metric')}", fmt(run.get("holdout_metric"))],
                           ["Validation", str(run.get("validation_verdict") or "—")]])
                    if run else None,
                    description="Counted and copied from the artifacts, not interpreted."),
        ]
    doc_rows = [[dmc.Text(d["name"], ff="monospace", size="sm"), d["role"],
                 "read" if d["readable"] else dmc.Text(f"not readable — {d['note']}", size="xs",
                                                       c="red"),
                 dmc.Code((d.get("sha256") or "")[:12] or "—")] for d in p.get("documents", [])]
    n_decided = sum(1 for b in brief if b.get("value") or later.get(b["field"]))
    return [
        kpis([("Development mode", p.get("kind_label", "—"), None),
              ("Intent", str(p.get("clarity", "—")).replace("_", " "), "the Intake agent's read"),
              ("Brief decided", f"{n_decided} / {len(brief)}", "* = needed to start"),
              ("Open questions", fmt(len(open_gaps)),
               f"{sum(1 for g in open_gaps if g.id in blocking)} blocking")]),
        recommendation_card(_rec(res)),
        gate,
        section("Engagement brief", table(["Field", "Sponsor position", "Source"], rows),
                description=p.get("objective")),
        *update_sections,
        section("Documents received", table(["Document", "Role", "Text", "SHA-256"], doc_rows)
                if doc_rows else empty("None. The brief rests on the project profile and the "
                                       "sponsor's answers.")),
        section("Findings", findings_table(_findings(res))) if res.findings else None,
    ]
