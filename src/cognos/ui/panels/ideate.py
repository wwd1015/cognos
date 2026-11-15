"""The ideate page: the design gate."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ... import service
from ...artifacts import StageResult
from ...engine.process import SEAT_OF_GATE
from ...engine.state import RunState
from .. import transform_review
from ..components import (
    fmt,
    icon,
    kpis,
    markdown,
    recommendation_card,
    section,
    table,
)
from ._shared import _act, _field, _rec, _summary_line, _words
from .gate import gate_block


def ideate_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    """The design page in seven steps (ADR-0025, ``ui/design_steps.py``): the model form, how the
    data are cut, the variables, the variable rules, the single-factor analysis, the model
    rules and ranking, then the
    summary and the decision. Each step is confirmed before the next opens; the design gate
    reads every field wherever it sits."""
    from .. import design_steps as dz

    p = res.payload or {}
    ds = p.get("data_structure") or {}
    plan = p.get("plan") or {}
    editable = (state.status_of("gate_design") == "awaiting"
                and seat == SEAT_OF_GATE.get("gate_design"))
    screen = p.get("screen")
    exceptions = {n: {"keep": e["keep"], "reason": e["reason"]}
                  for n, e in state.overrides.screen_exceptions.items()}
    names = [r["name"] for r in (screen or {}).get("rows") or []]
    tx = design_transforms(p, state, seat)
    explored = service.results(state.run_id).get("explore")
    review = ((explored.payload if explored is not None else None) or {}).get(
        "transform_review") or {}
    form_rows = dz.forms_rows(review, state.overrides.transforms)
    # every other variable the screen sees enters as it is: a category, a flag, an engineered one
    reviewed = {r["column"] for r in form_rows}
    plain = sorted({(r.get("source") or r["name"]) for r in (screen or {}).get("rows") or []}
                   - reviewed) or list(review.get("as_they_are") or [])
    editable_design = (state.status_of("gate_design") == "awaiting"
                       and seat == SEAT_OF_GATE.get("gate_design"))
    forms = dz.forms_table(form_rows, plain,
                           (p.get("screen") or {}).get("rows"),
                           (p.get("plan") or {}).get("selection", {}).get("must_include"),
                           editable_design) if form_rows else None
    n_plain = len(plain)
    n_tx = (f"{len(form_rows) + n_plain} variables → "
            f"{sum(len(r['forms']) for r in form_rows) + n_plain} forms"
            if form_rows else str(len(p.get("transforms") or {})))
    open_gaps = [g for g in state.gaps if g.stage == "ideate" or g.reentry == "ideate"
                 or g.design_field]  # a design question still open from the intake interview
    q_inputs = [dmc.TextInput(id=_field("gate_design", f"answer::{g.id}"), label=g.question,
                              value=g.answer or "", disabled=g.status != "open",
                              description=f"fills design.{g.design_field}" if g.design_field else
                              ("answered" if g.status != "open" else None))
                for g in open_gaps]
    brief = service.read_artifact(state.run_id, "stages/ideate/design_brief.md", None) or ""
    proposed = p.get("proposed_transforms") or []

    steps = [dz.step(1, [
        recommendation_card(_rec(res), title="The Design agent's recommendation"),
        *dz.model_form(p, editable, reason=next(
            (r["reason"] for r in state.overrides.framework_roles.values()), ""),
            proposed=state.overrides.proposed_forms),
        html.Details([html.Summary(f"Transforms the Design agent proposed ({len(proposed)})",
                                   className="gate-fold"),
                      table(["Name", "Expression", "Rationale"],
                            [[t["name"], dmc.Code(t["expr"]), t.get("rationale", "")]
                             for t in proposed])], className="gate-more") if proposed else None,
    ], summary=dz.model_form_summary(p), editable=editable,
        description="What kind of model this is, set against the alternatives.")]
    if plan:
        steps += [
            dz.step(2, dz.data_treatment(plan, editable),
                    summary=dz.treatment_summary(plan["evaluation"]), editable=editable,
                    description="How the rows are cut: where models are fitted, compared and "
                                "confirmed."),
            dz.step(3, [*dz.variables(tx, plan, names, editable, forms=forms),
                        None if forms else dmc.MultiSelect(id=_field("gate_design", "plan::factors.transforms"),
                                        label="Extra forms tried for the other columns",
                                        description=(
                                            "A column with its own transformation from the "
                                            "data gate enters only in that form. Every other "
                                            "column enters as it is and also in these forms; "
                                            "each is screened like any variable and the "
                                            "per-column rule keeps the stronger. "
                                            + ((plan.get("reasons") or {}).get("transforms")
                                               or "")),
                                        size="xs",
                                        value=list(plan["factors"].get("transforms") or []),
                                        disabled=not editable, mt="sm", maw=640,
                                        data=[{"value": t, "label": _words(t)}
                                              for t in (plan.get("options") or {}).get(
                                                  "transforms") or [] if t != "none"])],
                    summary=dz.variables_summary(n_tx, plan["selection"].get("must_include")),
                    editable=editable,
                    description="What goes in: each transformation's parameters, the forms "
                                "tried for the rest, and what every model must carry."),
            dz.step(4, dz.variable_rules(plan, editable),
                    summary=dz.variable_rules_summary(plan), editable=editable,
                    description="What a variable's strength is and the rules that keep it. "
                                "Set before the single-factor analysis is seen."),
            dz.step(5, [
                dmc.TextInput(id=_field("gate_design", "screen_exceptions"),
                              value=json.dumps(exceptions), style={"display": "none"}),
                dcc.Store(id="sf-data", data=None if editable else
                          {"screen": screen, "exceptions": exceptions}),
                dcc.Store(id="sf-selected", data="all"),
                dcc.Store(id="sf-chosen", data=None),
                dcc.Store(id="sf-editable", data=editable),
                dcc.Loading(html.Div(
                    [dmc.Text("Drawn when the variable rules are confirmed.", size="sm",
                              c="dimmed")] if editable else
                    dz.single_factor(screen, exceptions, False), id="sf-body"),
                    type="dot", color="var(--mantine-color-oxblood-6)", delay_show=250),
            ], summary=dz.single_factor_summary(screen, len(exceptions)), editable=editable,
                description="Every variable fitted alone. The funnel shows each rule's toll; "
                            "click a bar to see who it removed and why, a variable to make an "
                            "exception."),
            dz.step(6, dz.model_rules(plan, editable), summary=dz.model_rules_summary(plan),
                    editable=editable,
                    description="When a model is kept, how models are built and how they are "
                                "ranked. Set with the single-factor analysis in view: these "
                                "act on combinations of variables."),
        ]
    revisions = service.screen_revisions(state.run_id)
    summary = html.Div([
        _summary_line("Model form", dz.model_form_summary(p)),
        _summary_line("Data treatment", dz.treatment_summary(plan["evaluation"]))
        if plan else None,
        _summary_line("Variable rules", dz.variable_rules_summary(plan)) if plan else None,
        _summary_line("Single-factor analysis",
                         dz.single_factor_summary(screen, len(exceptions))),
        _summary_line("Model rules and ranking", dz.model_rules_summary(plan)) if plan else None,
        _summary_line("Rules revised after a look", f"{len(revisions)} time(s), on record "
                         "for validation") if revisions else None,
    ], className="sum-block")
    gate = gate_block("gate_design", state, res, seat=seat, body=[
        summary,
        html.Details([html.Summary("The design brief: this design written up for the record",
                                   className="gate-fold"),
                      dmc.Text("The file the documentation and the export keep "
                               "(stages/ideate/design_brief.md). It restates the steps above; "
                               "nothing in it is a separate choice.", size="xs", c="dimmed"),
                      markdown(brief)], className="gate-more") if brief else None,
        dmc.Text("The steps above as they stand are what “Apply my design” sends; “Accept "
                 "design” keeps the engine's recommendation for every step.", size="xs",
                 c="dimmed", mt="xs") if editable else None,
        dmc.Stack(q_inputs, gap="xs", mt="sm") if q_inputs else None,
    ], primary=[
        dmc.Button("Accept design", id=_act("gate_design", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply my design", id=_act("gate_design", "edit"), variant="light",
                   leftSection=icon("tabler:edit")),
    ])
    steps.append(dz.step(7, [gate], summary="", editable=editable,
                         description="Confirm steps 1 to 6 first. Nothing above is recorded "
                                     "until the design is decided."))
    return [
        html.Div(f"{state.run_id}|{res.payload.get('plan_summary', '')}|"
                 f"{state.status_of('gate_design')}|{len(state.decisions)}", id="dz-key",
                 style={"display": "none"}),
        html.Div(str(n_tx), id="dz-tx-count", style={"display": "none"}),
        dcc.Store(id="dz-done", data=None),
        kpis([("Sample shape", str(ds.get("shape", "—")).replace("_", " "), None),
              ("Events", fmt(ds.get("n_events")), "rare-class count"),
              ("Events / variable", fmt(ds.get("events_per_variable"), 1), "≥ 10 rule of thumb"),
              ("Open questions", fmt(len([g for g in state.gaps if g.status == "open"])), None)]),
        *steps,
    ]


def design_transforms(p: dict, state: RunState, seat: str) -> Any:
    """Step two of each column's transformation (ADR-0024): the variables decided at the data
    gate, one at a time, with the parameters the engine fitted on the development sample and
    a refit that redraws only the chart. None when nothing is transformed."""
    entries = p.get("transforms") or {}
    if not entries:
        return None
    from ...modeling import column_transforms as ct

    explored = service.results(state.run_id).get("explore")
    review = ((explored.payload if explored is not None else None) or {}).get(
        "transform_review") or {}
    kinds = {k: v for k, v in (review.get("kinds") or {}).items()
             if k not in ("as_is", "library")}
    editable = (state.status_of("gate_design") == "awaiting"
                and seat == SEAT_OF_GATE.get("gate_design"))
    own = {}
    for key, spec in ct.decided(state.overrides.transforms).items():
        spec = {k: v for k, v in spec.items() if k != "column"}
        if key in entries and spec != ct.normalize(spec["kind"]):
            own[key] = spec  # set by a person, not the defaults
    data = {"kinds": {**kinds, **{k: ct.KINDS[k]["label"] for k in ct.REVIEWED}},
            "columns": entries, "order": list(entries)}
    return section(
        "Forms to fit",
        transform_review.block(
            "txp", data, own, editable=editable, excluded=[],
            lead=f"{len(entries)} form(s) need fitting: shapes placed with the outcome in view, "
                 "and expressions to test. Check each one and change its parameters if you "
                 "need to; “Accept design” keeps the engine's settings. Logs, lags and changes "
                 "are used as the data gate decided them."),
        description="Fitted on the development sample at this gate, then again by the model "
                    "stage on its own development sample.")
