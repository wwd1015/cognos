"""One panel per stage: the engine's evidence beside the agent's recommendation, and the human gate.

Every panel is a pure function of (stage result, run state, color scheme, run id) so it can be
rendered and tested without a browser. Gate controls carry pattern-matching ids
(``{"type": "gate-field" | "gate-act", "gate": ..., ...}``) that one callback in ``app.py`` reads.
"""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc

from .. import service
from ..artifacts import StageResult
from ..engine.graph import LABELS, SEND_BACK_TARGETS
from ..engine.process import SEAT_OF_GATE, preparation_of, seat_label
from ..engine.state import RunState
from . import charts
from .components import (
    empty,
    findings_table,
    fmt,
    graph,
    icon,
    kpis,
    markdown,
    pct,
    recommendation_card,
    run_badge,
    section,
    step_badge,
    table,
    verdict_badge,
)

STAGE_BLURB = {
    "intake": "The engine reads the business intent and, for an update, the existing model's "
              "artifacts; the Intake Analyst fills the brief and interviews the sponsor where the "
              "goal is not clear.",
    "explore": "The engine fetches and profiles the data and runs the analyses the Data Analyst "
               "asks for; the analyst names the dependent variable, the features worth "
               "considering and the columns that may not be inputs.",
    "ideate": "The engine assesses structure and frameworks; the Design Lead ranks the slate and "
              "raises what only the sponsor can decide.",
    "model": "The engine searches under a frozen metric; the Modeler picks the champion from the "
             "admissible set — before the sealed holdout is scored.",
    "backtest": "The engine scores the champion out of sample; the Outcomes Analyst reads the "
                "discrimination, calibration and stability.",
    "validate": "The engine's rubric re-derives risk from artifacts; the Independent Validator adds "
                "an effective challenge — it never sees the modeler's reasoning.",
    "comply": "A non-gating readiness report: evidence organized, human-only steps listed.",
    "document": "The white paper as an OKF bundle. The Technical Writer drafts prose; every number "
                "is rendered from a recorded fact.",
    "review": "The engine verifies every docs↔code anchor; stale references BLOCK.",
}


def _field(gate: str, field: str) -> dict:
    return {"type": "gate-field", "gate": gate, "field": field}


def _act(gate: str, action: str) -> dict:
    return {"type": "gate-act", "gate": gate, "action": action}


def _rec(res: StageResult | None) -> dict | None:
    return (res.payload or {}).get("recommendation") if res is not None else None


def _findings(res: StageResult | None) -> list[dict]:
    return [f.model_dump(mode="json") for f in res.findings] if res is not None else []


# --- gate block -----------------------------------------------------------------------------
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
            dmc.Group([dmc.Button("Send back", id=_act(gate, "send_back"), color="violet",
                                  variant="light", leftSection=icon("tabler:arrow-back-up"))]),
        ], gap="xs")),
    ], value="send")], variant="contained", mt="sm")


def gate_block(gate: str, state: RunState, res: StageResult | None, body: list | None = None,
               primary: list | None = None, *, send_back: bool = True,
               seat: str = "developer") -> Any:
    """The human decision for a stage: form when awaiting, record when decided.

    ``seat`` is the role the workbench is acting as. A page that belongs to another seat
    stays readable and its controls are absent; the engine refuses the action too.
    """
    status = state.status_of(gate)
    last = state.last_decision(gate)
    owner = SEAT_OF_GATE.get(gate, "")
    mine = seat == owner
    header = dmc.Group([
        dmc.Group([dmc.ThemeIcon(icon("tabler:user-check", 16), variant="light", radius="xl",
                                 color="violet"),
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
        rec = [dmc.Text([dmc.Text(last.action.replace("_", " "), fw=650, span=True),
                         f" by {who}", f" — {last.reason}" if last and last.reason else ""],
                        size="sm")] if last else [empty("Decided.")]
        revise = (dmc.Group([dmc.Button("Revise decision", id={"type": "reopen", "gate": gate},
                                        variant="subtle", size="xs",
                                        leftSection=icon("tabler:edit"))], mt="xs")
                  if mine else dmc.Text(f"The {seat_label(owner)} can revise this.",
                                        size="xs", c="dimmed", mt="xs"))
        return dmc.Card([header, *rec, revise], className="cognos-gate")
    if not mine:
        return dmc.Card([
            header, *(body or []),
            dmc.Alert(f"This decision belongs to the {seat_label(owner)}. "
                      "Switch seat to act. The evidence above stays open to read.",
                      color="gray", variant="light", mt="sm",
                      icon=icon("tabler:lock")),
        ], className="cognos-gate cognos-gate-open")
    # awaiting, and this seat owns it
    parts = [header, *(body or []),
             dmc.Textarea(id=_field(gate, "reason"), label="Reason (recorded in the decision log)",
                          autosize=True, minRows=1, mt="sm"),
             dmc.Group(primary or [], mt="sm", gap="sm")]
    if send_back:
        parts.append(_send_back(gate, choose_target=gate == "gate_validation"))
    return dmc.Card(parts, className="cognos-gate cognos-gate-open")


# --- intake -------------------------------------------------------------------------------
_BASIS = {"stated": ("document", "indigo"), "answered": ("answered", "teal"),
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
              ("Intent", str(p.get("clarity", "—")).replace("_", " "), "the Intake Analyst's read"),
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


# --- explore ------------------------------------------------------------------------------
def analysis_card(run_id: str, a: dict, scheme: str) -> Any:
    """One analysis: the question, what it found, its chart and table and, for a script, the
    code exactly as the analyst wrote it."""
    try:
        result = json.loads(service.read_artifact(run_id, a["artifact"]) or "{}")
    except ValueError:
        result = {}
    is_code = a["kind"] == "code"
    by = (dmc.Badge("agent-written code", color="grape", variant="light", size="sm",
                    leftSection=icon("tabler:code", 12)) if is_code else
          dmc.Badge(a.get("tool") or "tool", color="indigo", variant="light", size="sm",
                    leftSection=icon("tabler:tool", 12)))
    plugin = (dmc.Badge(f"plugin: {a['origin']}", color="teal", variant="outline", size="sm")
              if not is_code and a.get("origin") not in (None, "cognos") else None)
    parts: list[Any] = [
        dmc.Group([dmc.Group([dmc.Code(a["id"]), by, plugin], gap=6),
                   None if a["status"] == "ok" else
                   dmc.Badge("did not run", color="red", variant="light", size="sm",
                             leftSection=icon("tabler:alert-triangle", 12))],
                  justify="space-between"),
        dmc.Text(a.get("title") or a["purpose"], fw=600, size="sm", mt=6),
        dmc.Text(a["purpose"], size="xs", c="dimmed"),
    ]
    if a["status"] != "ok":
        parts.append(dmc.Alert(a.get("error") or "No result.", color="red", variant="light",
                               mt="xs", p="xs"))
    for spec in result.get("charts", []):
        parts.append(graph(charts.from_spec(spec, scheme)))
    if a.get("summary"):
        parts.append(dmc.Group([dmc.Badge(f"{k}: {fmt(v, 3) if not isinstance(v, str) else v}",
                                          variant="default", size="sm", tt="none")
                                for k, v in list(a["summary"].items())[:8]], gap=4, mt=4))
    folds = []
    for t in result.get("tables", [])[:3]:
        folds.append(dmc.AccordionItem([
            dmc.AccordionControl(f"Table: {t.get('name', 'result')} ({t.get('n_rows', len(t['rows']))} rows)"),
            dmc.AccordionPanel(table(t["columns"], [[fmt(v, 4) if isinstance(v, float) else
                                                     ("" if v is None else str(v)) for v in row]
                                                    for row in t["rows"][:25]], max_height=260)),
        ], value=f"t-{t.get('name', '')}"))
    if is_code:
        code = service.read_artifact(run_id, a["code_path"]) or ""
        folds.append(dmc.AccordionItem([
            dmc.AccordionControl(f"Code to review — {a['code_path'].rsplit('/', 1)[-1]} · "
                                 f"sha256 {a['code_sha256'][:12]}", icon=icon("tabler:code")),
            dmc.AccordionPanel(dmc.Code(code, block=True)),
        ], value="code"))
    if folds:
        parts.append(dmc.Accordion(folds, variant="separated", mt="xs",
                                   value="code" if is_code else None))
    # (a card keeps its own height: a tall neighbour in the grid must not stretch its chart)
    return dmc.Card([p for p in parts if p is not None], p="md", style={"alignSelf": "start"})


def explore_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    ts = p.get("target_summary") or {}
    suspects = set(p.get("leakage_suspects", []))
    corr = {c["feature"]: c["corr"] for c in p.get("top_correlations", [])}
    rec = _rec(res) or {}
    out = rec.get("output") or {}
    decisions = {d["column"]: d for d in out.get("column_decisions", [])}
    rows = []
    for f in p.get("features", []):
        d = decisions.get(f)
        rows.append([
            dmc.Group([dmc.Text(f, size="sm", ff="monospace"),
                       dmc.Badge("leakage suspect", color="red", size="xs", variant="light")
                       if f in suspects else None], gap=6),
            p.get("dtypes", {}).get(f, ""),
            pct(p.get("missing", {}).get(f, 0.0)),
            fmt(corr.get(f), 3),
            (dmc.Badge(d["decision"], color="red" if d["decision"] == "exclude" else "gray",
                       variant="light", size="sm") if d else ""),
        ])
    decision_rows = [[dmc.Text(d["column"], ff="monospace", size="sm"),
                      dmc.Badge(d["decision"], color="red" if d["decision"] == "exclude" else "gray",
                                variant="light", size="sm"), d["reason"]]
                     for d in out.get("column_decisions", [])]
    rec_extra = dmc.Stack([
        table(["Column", "Decision", "Reason"], decision_rows) if decision_rows else None,
        dmc.List([dmc.ListItem(c["statement"]) for c in out.get("data_quality", [])], size="sm")
        if out.get("data_quality") else None,
    ], gap="xs")
    analyses = p.get("analyses") or []
    scripts = [a for a in analyses if a["kind"] == "code"]
    src = p.get("source") or {}
    fixed = p.get("target_source") == "profile"
    how = {"profile": "fixed in the project profile", "decision": "set at the data gate",
           "agent": "proposed by the Data Analyst"}.get(p.get("target_source", ""), "")
    target_options = ([{"value": c["column"], "label": f"{c['column']} — {c['why']}"}
                       for c in p.get("target_candidates", [])]
                      or [{"value": p.get("target", ""), "label": p.get("target", "")}])
    if p.get("target") and p["target"] not in [o["value"] for o in target_options]:
        target_options.insert(0, {"value": p["target"], "label": p["target"]})
    cand_rows = [[dmc.Text(c["column"], ff="monospace", size="sm"),
                  dmc.Badge({"+": "rises", "-": "falls", "nonlinear": "non-linear",
                             "unknown": "unknown"}[c["relationship"]], variant="light", size="sm",
                            color="gray"),
                  fmt(corr.get(c["column"]), 3), dmc.Text(c["rationale"], size="xs")]
                 for c in p.get("feature_candidates", [])]
    gate = gate_block("gate_data", state, res, seat=seat, body=[
        dmc.Select(id=_field("gate_data", "target"), label="Dependent variable",
                   data=target_options, value=p.get("target"), allowDeselect=False,
                   disabled=fixed, searchable=True,
                   description="Fixed in the project profile." if fixed else
                   "Changing it re-runs the exploration against the new target."),
        dmc.Alert(f"{len(scripts)} analysis script(s) written by the Data Analyst are part of "
                  "what you are accepting. Read the code under “Analyses” below: it goes to "
                  "validation and into the white paper as it stands.", color="grape",
                  variant="light", icon=icon("tabler:code"), mt="xs") if scripts else None,
        dmc.MultiSelect(id=_field("gate_data", "exclude"), label="Columns to exclude from modeling",
                        data=[{"value": c, "label": c} for c in p.get("features", [])],
                        value=list(state.overrides.exclude_columns
                                   if state.last_decision("gate_data")
                                   else p.get("recommended_exclusions", [])),
                        searchable=True, clearable=True,
                        description="Pre-filled with the Data Analyst's recommendation."),
    ], primary=[
        dmc.Button("Accept recommendation", id=_act("gate_data", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply my changes", id=_act("gate_data", "edit"), variant="light",
                   leftSection=icon("tabler:edit")),
    ])
    return [
        kpis([("Rows", fmt(p.get("n_rows")), None),
              ("Features", fmt(len(p.get("features", []))), None),
              ("Event rate", pct(ts.get("positive_rate")) if "positive_rate" in ts else "—",
               "positive share of the target"),
              ("Leakage suspects", fmt(len(suspects)), "|corr| ≥ 0.98 with the target")]),
        recommendation_card(rec, extra=rec_extra),
        gate,
        section("Dependent variable",
                dmc.Group([dmc.Text(p.get("target", "—"), ff="monospace", fw=650, size="lg"),
                           dmc.Badge(p.get("task", ""), variant="light", size="sm"),
                           dmc.Badge(how, variant="outline", size="sm", color="gray", tt="none")],
                          gap="xs"),
                dmc.Text(p.get("target_rationale") or "", size="sm", mt=4)
                if p.get("target_rationale") else None,
                description="The outcome the business intent describes.") if p.get("target") else None,
        section("Features to consider", table(["Feature", "Expected", "Corr.", "Why"], cand_rows),
                description="The Data Analyst's candidates for the design stage: hypotheses, "
                            "not a selection.") if cand_rows else None,
        section(f"Analyses ({len(analyses)})",
                dmc.SimpleGrid([analysis_card(state.run_id, a, scheme) for a in analyses],
                               cols={"base": 1, "lg": 2}, spacing="md"),
                description="Requested by the Data Analyst, run by the engine. Tools are "
                            "reviewed code; a script is code the agent wrote for this run.")
        if analyses else None,
        section("Data source",
                table(["Connector", "Location", "Rows × columns", "Fetched", "Snapshot SHA-256"],
                      [[src.get("connector", "—"),
                        dmc.Text(src.get("query") or src.get("location") or "—", size="xs",
                                 ff="monospace"),
                        f"{fmt(src.get('n_rows'))} × {fmt(src.get('n_cols'))}",
                        str(src.get("fetched_at", ""))[:19].replace("T", " "),
                        dmc.Code(str(src.get("snapshot_sha256", ""))[:12])]]),
                description="Fetched once; every stage reads this snapshot.") if src else None,
        section("Feature profile", table(["Feature", "Type", "Missing", "Corr. with target",
                                          "Agent"], rows, max_height=380)),
        section("Engine findings", findings_table(_findings(res))),
    ]


# --- ideate -------------------------------------------------------------------------------
def ideate_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    ds = p.get("data_structure") or {}
    choices = {c["framework"]: c for c in p.get("framework_choices", [])}
    decision_color = {"primary": "indigo", "candidate": "blue", "challenger": "grape",
                      "rejected": "gray"}
    fw_rows = []
    for f in p.get("framework_assessment", []):
        ch = choices.get(f["framework"], {})
        fw_rows.append([
            dmc.Text(f["label"], size="sm", fw=500),
            str(f["applicable"]),
            dmc.Badge(ch.get("decision", f["role"]),
                      color=decision_color.get(ch.get("decision", ""), "gray"), variant="light",
                      size="sm"),
            dmc.Text(ch.get("reason") or f["reason"], size="xs"),
        ])
    slate_rows = [[h["id"], dmc.Text(h["family"], ff="monospace", size="sm"), h["feature_strategy"],
                   h["role"], fmt(h["priority"], 2), dmc.Text(h["rationale"], size="xs")]
                  for h in p.get("hypotheses", [])]
    open_gaps = [g for g in state.gaps if g.stage == "ideate" or g.reentry == "ideate"
                 or g.design_field]  # a design question still open from the intake interview
    q_inputs = [dmc.TextInput(id=_field("gate_design", f"answer::{g.id}"), label=g.question,
                              value=g.answer or "", disabled=g.status != "open",
                              description=f"fills design.{g.design_field}" if g.design_field else
                              ("answered" if g.status != "open" else None))
                for g in open_gaps]
    gate = gate_block("gate_design", state, res, seat=seat, body=[
        dmc.CheckboxGroup(dmc.Stack([dmc.Checkbox(label=f"{h['id']} · {h['family']} / "
                                                        f"{h['feature_strategy']} ({h['role']})",
                                                  value=h["id"])
                                     for h in p.get("hypotheses", [])], gap=4),
                          id=_field("gate_design", "keep"), label="Specifications to search",
                          description="Untick to drop a specification from the slate.",
                          value=[h["id"] for h in p.get("hypotheses", [])]),
        dmc.Stack(q_inputs, gap="xs", mt="sm") if q_inputs else None,
    ], primary=[
        dmc.Button("Accept design", id=_act("gate_design", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply edits / answers", id=_act("gate_design", "edit"), variant="light",
                   leftSection=icon("tabler:edit")),
    ])
    brief = service.read_artifact(state.run_id, "stages/ideate/design_brief.md",
                                  None) or ""
    return [
        kpis([("Sample shape", str(ds.get("shape", "—")).replace("_", " "), None),
              ("Events", fmt(ds.get("n_events")), "rare-class count"),
              ("Events / variable", fmt(ds.get("events_per_variable"), 1), "≥ 10 rule of thumb"),
              ("Open questions", fmt(len([g for g in state.gaps if g.status == "open"])), None)]),
        recommendation_card(_rec(res)),
        gate,
        section("Framework assessment — alternatives considered",
                table(["Framework", "Applicable", "Decision", "Reason"], fw_rows)),
        section("Ranked specification slate",
                table(["#", "Family", "Features", "Role", "Priority", "Rationale"], slate_rows,
                      max_height=360)),
        section("Proposed transforms", table(["Name", "Expression", "Rationale"],
                                             [[t["name"], dmc.Code(t["expr"]), t.get("rationale", "")]
                                              for t in p.get("proposed_transforms", [])])
                if p.get("proposed_transforms") else empty("None proposed.")),
        dmc.Accordion([dmc.AccordionItem([dmc.AccordionControl("Design brief (markdown)"),
                                          dmc.AccordionPanel(markdown(brief))], value="brief")],
                      variant="separated"),
    ]


# --- model --------------------------------------------------------------------------------
def model_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    metric = p.get("metric", "metric")
    ledger = json.loads(service.read_artifact(state.run_id, "stages/model/ledger.json",
                                              None) or "[]")
    rec = _rec(res) or {}
    out = rec.get("output") or {}
    recommended = out.get("champion")
    adm = p.get("admissible_set") or []
    adm_rows = []
    for c in adm:
        tags = [dmc.Badge("champion", color="indigo", size="xs")
                if c["id"] == p.get("champion_id") else None,
                dmc.Badge("agent pick", color="blue", size="xs", variant="light")
                if c["id"] == recommended else None,
                dmc.Badge("ratchet best", color="gray", size="xs", variant="outline")
                if c.get("ratchet_champion") else None,
                dmc.Badge("challenger only", color="grape", size="xs", variant="light")
                if c["role"] == "challenger" else None]
        adm_rows.append([dmc.Code(c["id"]), dmc.Text(c["family"], ff="monospace", size="sm"),
                         c["n_features"], f"{c['cv_mean']:.4f} ± {c['cv_std']:.4f}",
                         dmc.Group(tags, gap=4)])
    signs = [[dmc.Text(s["feature"], ff="monospace", size="sm"), s["expected"], s["observed"],
              dmc.Badge(s["assessment"].replace("_", " "),
                        color={"consistent": "green", "wrong_sign": "red"}.get(s["assessment"], "gray"),
                        variant="light", size="sm")]
             for s in out.get("sign_checks", [])]
    diag = p.get("diagnostics") or {}
    diag_rows = [[t["name"], t["category"],
                  "skipped" if t.get("passed") is None else ("passed" if t["passed"] else "failed"),
                  fmt(t.get("pvalue"), 4), dmc.Text(t.get("interpretation", ""), size="xs")]
                 for t in diag.get("tests", [])]
    gate = gate_block("gate_champion", state, res, seat=seat, body=[
        dmc.RadioGroup(dmc.Stack([dmc.Radio(label=f"{c['id']} · {c['label']} — CV {c['cv_mean']:.4f}",
                                            value=c["id"], disabled=c["role"] == "challenger")
                                  for c in adm], gap=4),
                       id=_field("gate_champion", "champion"), label="Champion",
                       description="Admissible set: within one CV standard error of the best.",
                       value=p.get("champion_id")),
    ], primary=[
        dmc.Button("Accept champion", id=_act("gate_champion", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Override with selected", id=_act("gate_champion", "override"), variant="light",
                   color="orange", leftSection=icon("tabler:switch-horizontal")),
    ])
    bench = []
    for key, label in (("challenger_benchmark", "Ensemble ceiling"), ("structural", "Merton structural"),
                       ("migration", "Rating migration")):
        b = p.get(key) or {}
        if key == "challenger_benchmark" and b:
            bench.append([label, fmt(b.get("benchmark_score")), "not deployed"])
        elif (b.get("benchmark") or {}):
            bench.append([label, f"AUC {fmt(b['benchmark'].get('holdout_roc_auc'))}", "not deployed"])
    return [
        kpis([("Champion", f"{p.get('champion_id', '—')} · {(p.get('champion') or {}).get('family', '—')}",
               f"chosen by {p.get('champion_source', 'agent')}"),
              (f"CV {metric}", f"{fmt(p.get('cv_mean'))}", f"± {fmt(p.get('cv_std'))}"),
              (f"Holdout {metric}", fmt(p.get("holdout_metric")),
               f"evaluated {p.get('holdout_evaluations', 1)}×"),
              ("Candidates tried", fmt(p.get("n_candidates_tried")), None)]),
        recommendation_card(rec, extra=table(["Feature", "Expected", "Observed", "Assessment"], signs)
                            if signs else None),
        gate,
        section("Admissible set", table(["Id", "Family", "Features", f"CV {metric}", ""], adm_rows),
                description="Statistically indistinguishable from the best; the modeler chose "
                            "without seeing the sealed holdout."),
        section("Experiment ledger", graph(charts.ledger(ledger, metric, p.get("champion_label"), scheme),
                                            "No ledger.")),
        section("Champion effects", graph(charts.coefficients(p.get("coefficients"), p.get("pvalues"),
                                                              p.get("feature_importances"), scheme),
                                           "This family exposes no per-feature effects.")),
        section("PD term structure", graph(charts.term_structure((p.get("hazard") or {})
                                                                 .get("term_structure"), scheme)))
        if p.get("hazard") else None,
        section("Statistical battery", table(["Test", "Category", "Result", "p-value", "Reading"],
                                             diag_rows, max_height=320)),
        section("Challenger benchmarks", table(["Benchmark", "Score", "Status"], bench))
        if bench else None,
        section("Findings", findings_table(_findings(res))),
    ]


# --- backtest -----------------------------------------------------------------------------
def backtest_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    p = res.payload or {}
    oa = p.get("outcomes_analysis") or {}
    interp = p.get("interpretation") or {}
    port = p.get("portfolio_analysis") or {}
    stress = (p.get("stress_testing") or {}).get("scenarios") or []
    return [
        kpis([("Gini", fmt(oa.get("gini"), 3), "discrimination"),
              ("KS", fmt(oa.get("ks"), 3), None),
              ("Calibration error", fmt(oa.get("expected_calibration_error"), 3), "mean |obs − pred|"),
              ("PSI", fmt(oa.get("psi"), 3), oa.get("psi_label"))] if oa else
             [(f"OOS {p.get('oos_metric_name', '')}", fmt(p.get("oos_metric")), p.get("evaluation_sample")),
              ("Scored rows", fmt(p.get("scored_rows")), None)]),
        recommendation_card(_rec(res), extra=dmc.SimpleGrid([
            dmc.Stack([dmc.Text(k.title(), size="xs", c="dimmed", fw=600), dmc.Text(v, size="sm")], gap=2)
            for k, v in interp.items()], cols={"base": 1, "md": 3}) if interp else None),
        section("Calibration by score band", graph(charts.calibration(oa.get("calibration_table") or [],
                                                                     scheme), "n/a"),
                description=f"Evaluation sample: {p.get('evaluation_sample', '—')}"
                            + (" (IMPACT)" if p.get("used_impact") else " (built-in scorer)")),
        section("Portfolio loss simulation", kpis([
            ("Expected loss", pct(port.get("expected_loss"), 2), "share of EAD"),
            (f"VaR {fmt(port.get('confidence'), 3)}", pct(port.get("var"), 2), None),
            ("Expected shortfall", pct(port.get("es"), 2), None)])) if port else None,
        section("Stress scenarios", table(["Scenario", "Mean PD", "Δ PD"],
                                          [[s["name"], pct(s.get("mean_pd"), 2),
                                            f"{s.get('delta_pd', 0):+.2%}"] for s in stress]))
        if stress else None,
        section("Findings", findings_table(_findings(res))),
    ]


# --- validate -----------------------------------------------------------------------------
def validate_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    v = p.get("validator") or {}
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
    loops = state.loops.get("validator", 0)
    routed = p.get("routed_findings") or []
    gate = gate_block("gate_validation", state, res, seat=seat, primary=[
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
        section("Findings", findings_table(_findings(res)),
                description="Engine findings re-derive risk from artifacts; agent findings "
                            "(val-*) are the validator's challenge."),
    ]


# --- comply / document / review ------------------------------------------------------------
def comply_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    p = res.payload or {}
    r = p.get("readiness") or {}
    sr = p.get("sr11_7") or {}
    nist = p.get("nist_ai_rmf") or {}
    status_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    return [
        dmc.Alert(r.get("narrative", ""), title=f"Readiness: {str(r.get('level', '—')).replace('_', ' ')}",
                  color="blue", variant="light", icon=icon("tabler:clipboard-check")),
        recommendation_card(_rec(res), extra=dmc.List([dmc.ListItem(a["statement"])
                                                       for a in r.get("priority_actions", [])], size="sm")
                            if r.get("priority_actions") else None),
        section("SR 11-7 evidence", table(["Element", "Status", "Evidence"], [
            [k.replace("_", " "), dmc.Badge(e["status"], color=status_color.get(e["status"], "gray"),
                                            variant="light", size="sm"),
             dmc.Text(e["evidence"], size="xs")] for k, e in sr.items()])),
        section("NIST AI RMF", table(["Function", "Status", "Evidence"], [
            [k.upper(), dmc.Badge(e["status"], color=status_color.get(e["status"], "gray"),
                                  variant="light", size="sm"), dmc.Text(e["evidence"], size="xs")]
            for k, e in nist.items()])),
        section("Outstanding human steps", dmc.List([dmc.ListItem(s)
                                                    for s in p.get("outstanding_human_steps", [])],
                                                   size="sm")),
    ]


def document_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    root = None  # the app points service.runs_root at its runs dir (COGNOS_RUNS_DIR)
    tabs = [("narrative", "Narrative", "docs/narrative.md"),
            ("decisions", "Decision log", "docs/decisions.md"),
            ("card", "Model card", "docs/model_card.md"),
            ("paper", "White paper", "stages/document/whitepaper.md")]
    p = res.payload or {}
    return [
        kpis([("Concepts", fmt(p.get("n_concepts")), "OKF bundle"),
              ("Code anchors", fmt(len(p.get("code_links", []))), "verified by review"),
              ("Bundle", "docs/", None)]),
        recommendation_card(_rec(res)),
        dmc.Tabs([
            dmc.TabsList([dmc.TabsTab(label, value=key) for key, label, _ in tabs]),
            *[dmc.TabsPanel(dmc.Paper(markdown(_strip_frontmatter(
                service.read_artifact(state.run_id, path, root) or "_Not written yet._")),
                p="lg", withBorder=True, mt="sm"), value=key) for key, _, path in tabs],
        ], value="narrative", variant="outline"),
    ]


def review_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    gate = gate_block("gate_signoff", state, res, seat=seat, primary=[
        dmc.Button("Approve & sign off", id=_act("gate_signoff", "approve"), color="green",
                   leftSection=icon("tabler:signature"), disabled=res.verdict.value == "BLOCK"),
        dmc.Button("Reject", id=_act("gate_signoff", "reject"), color="red", variant="light",
                   leftSection=icon("tabler:x")),
    ], send_back=False)
    sealed = state.package
    package_note = None
    if sealed is not None and sealed.status == "sealed":
        prep = " Express prepared the analysis." if sealed.preparation == "express" else ""
        package_note = dmc.Alert(
            f"Package v{sealed.version} is sealed ({sealed.digest[:12]}).{prep} "
            "A later edit supersedes it; the file is not rewritten. Approval is not deployment.",
            color="green", variant="light", icon=icon("tabler:rosette-discount-check"))
    elif state.status == "completed" and sealed is None:
        express = state.mode == "autonomous" or preparation_of(state) == "express"
        package_note = dmc.Alert(
            ("Express finished this analysis and accepted the gates. That is not a signature. "
             if express else
             "This analysis finished without a sealed package. ")
            + "An approver seals a package from here, once the design brief is answered.",
            color="yellow", variant="light", icon=icon("tabler:signature"))
    return [
        dmc.Group([verdict_badge(res.verdict.value, "lg"),
                   dmc.Text(res.summary, size="sm")], gap="sm"),
        package_note,
        kpis([(k.replace("_", " ").title(), fmt(v), None) for k, v in list(res.metrics.items())[:4]])
        if res.metrics else None,
        gate,
        section("Consistency findings", findings_table(_findings(res))),
        section("Checked anchors", dmc.Code("\n".join(p.get("checked_links", [])[:40]) or "—",
                                            block=True)) if p.get("checked_links") else None,
    ]


PANELS = {"intake": intake_panel, "explore": explore_panel, "ideate": ideate_panel, "model": model_panel,
          "backtest": backtest_panel, "validate": validate_panel, "comply": comply_panel,
          "document": document_panel, "review": review_panel}


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            return text[end + 4:].lstrip()
    return text


def _change_badge(better: str | None) -> Any:
    """How a number moved from A (or the previous run of a step) to B. Icon and word, never color
    alone; no badge where no direction is claimed."""
    if better == "b":
        return dmc.Badge("Improved", color="green", variant="light", size="sm",
                         leftSection=icon("tabler:trending-up", 12))
    if better == "a":
        return dmc.Badge("Worse", color="orange", variant="light", size="sm",
                         leftSection=icon("tabler:trending-down", 12))
    return None


def _delta(v: Any) -> str:
    return "—" if v is None else "0" if v == 0 else f"{v:+,.4f}".rstrip("0").rstrip(".")


def _value(v: Any) -> Any:
    if isinstance(v, list | tuple):
        return ", ".join(str(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return dmc.Code(json.dumps(v, default=str))
    return fmt(v)


def rerun_note(stage: str, state: RunState, changes: dict | None) -> Any:
    """Why this stage ran again and what that changed, once it has. None on a first run."""
    step = state.steps[stage]
    if not step.rerun_reason or step.status in ("stale", "running", "pending") or step.runs < 2:
        return None
    head = dmc.Text(["Ran again because — ", dmc.Text(step.rerun_reason, span=True, fw=600)], size="sm")
    if changes is None:
        return dmc.Alert(head, color="gray", variant="light", icon=icon("tabler:history"))
    if changes["unchanged"]:
        body = dmc.Text("The re-run changed nothing here: same verdict, same metrics.", size="sm", c="dimmed")
    else:
        moved = [r for r in changes["metrics"] if r["delta"] != 0]
        lines = []
        if changes["champion"]:
            lines.append(dmc.Text(["Champion: ", dmc.Code(str(changes["champion"]["a"])), " → ",
                                   dmc.Code(str(changes["champion"]["b"]))], size="sm"))
        if changes["verdict"]["a"] != changes["verdict"]["b"]:
            lines.append(dmc.Group([dmc.Text("Verdict:", size="sm"), verdict_badge(changes["verdict"]["a"], "sm"),
                                    dmc.Text("→", size="sm"), verdict_badge(changes["verdict"]["b"], "sm")], gap=6))
        f = changes["findings"]
        if f["new"] or f["gone"]:
            lines.append(dmc.Text(f"Findings: {len(f['new'])} new, {len(f['gone'])} no longer raised.", size="sm"))
        body = dmc.Stack([*lines, table(["Metric", "Before", "After", "Change", ""], [
            [r["label"], _value(r["a"]), _value(r["b"]), _delta(r["delta"]), _change_badge(r["better"])]
            for r in moved]) if moved else None], gap="xs")
    return dmc.Alert(dmc.Stack([head, body], gap="xs"), color="gray", variant="light",
                     icon=icon("tabler:history"), title="What the re-run changed")


def compare_page(c: dict) -> list:
    """Run B against run A: what was decided differently, and what it did to the results. A pure
    function of ``service.compare``'s dictionary."""
    def card(tag: str, r: dict) -> Any:
        return dmc.Card([
            dmc.Group([dmc.Badge(tag, variant="filled", color="dark", size="lg", radius="sm"),
                       dmc.Text(r["project"], fw=650), run_badge(r["status"], "sm")], gap="xs"),
            dcc.Link(dmc.Text(r["run_id"], size="xs", ff="monospace"), href=f"/run/{r['run_id']}"),
            dmc.Text(["Champion ", dmc.Code(str(r["champion"] or "—"))], size="sm", mt="xs"),
            dmc.Text(f"{r['mode']} · agents: {r['provider']}" + (f" · metric: {r['metric']}" if r["metric"] else ""),
                     size="xs", c="dimmed", mt=4),
        ], p="md")

    a, b = c["a"], c["b"]
    head = dmc.Group([
        dmc.Stack([dmc.Title("Compare runs", order=2),
                   dmc.Text("B relative to A. Every number is read from the two runs' recorded results; "
                            "“improved” is only claimed where the metric has a direction.",
                            size="sm", c="dimmed")], gap=2),
        dcc.Link(dmc.Button("Swap A and B", variant="subtle", leftSection=icon("tabler:arrows-exchange")),
                 href=f"/compare/{b['run_id']}/{a['run_id']}"),
    ], justify="space-between", align="flex-start")

    decided = [[d["label"], *[
        "—" if x is None else dmc.Stack([dmc.Text(f"{x['action'].replace('_', ' ')} · {'you' if x['actor'] == 'human' else 'auto'}", size="sm"),
                                         dmc.Text(x["reason"], size="xs", c="dimmed") if x["reason"] else None], gap=0)
        for x in (d["a"], d["b"])]] for d in c["decisions"] if d["different"]]
    decided += [[dmc.Stack([dmc.Text(o["what"], size="sm"), dmc.Text(o["note"], size="xs", c="dimmed") if o["note"] else None], gap=0),
                 _value(o["a"]), _value(o["b"])] for o in c["overrides"]]
    decided += [[dmc.Code(k["key"]), _value(k["a"]), _value(k["b"])] for k in c["config"]]

    moved = [r for r in c["metrics"] if r["delta"] != 0]
    same = len(c["metrics"]) - len(moved)
    verdicts = [[v["label"], verdict_badge(v["a"], "sm") if v["a"] else "—", verdict_badge(v["b"], "sm") if v["b"] else "—"]
                for v in c["verdicts"] if v["a"] != v["b"]]
    f = c["findings"]

    def finding_rows(rows: list[dict]) -> list:
        return [[dmc.Badge(r["severity"].title(), variant="light", color="gray", size="sm"), LABELS[r["stage"]],
                 dmc.Text(r["message"], size="sm")] for r in rows[:12]]

    return [
        head,
        *[dmc.Alert(text, color="yellow", variant="light", icon=icon("tabler:alert-triangle")) for text in c["caveats"]],
        dmc.SimpleGrid([card("A", a), card("B", b)], cols={"base": 1, "sm": 2}, spacing="md"),
        section("Decided differently", table(["What", "A", "B"], decided) if decided
                else empty("Nothing: the same decisions, overrides and configuration."),
                description="Gate decisions, the overrides they produced, and the project configuration."),
        section("What it did to the results",
                table(["Stage", "Metric", "A", "B", "Change", ""], [
                    [LABELS[r["stage"]], r["label"], _value(r["a"]), _value(r["b"]), _delta(r["delta"]),
                     _change_badge(r["better"])] for r in moved]) if moved
                else empty("No metric differs between the two runs."),
                dmc.Text(f"{same} metric(s) are identical in both runs.", size="xs", c="dimmed", mt="xs") if same else None,
                description="Only the metrics that differ. Change is B minus A."),
        section("Verdicts", table(["Stage", "A", "B"], verdicts) if verdicts
                else empty("Every stage reached the same verdict in both runs.")),
        section("Findings", dmc.Stack([
            dmc.Text(f"Only in B ({len(f['new'])})", size="sm", fw=600),
            table(["Severity", "Stage", "Finding"], finding_rows(f["new"])) if f["new"] else empty("None."),
            dmc.Text(f"Only in A ({len(f['gone'])})", size="sm", fw=600, mt="sm"),
            table(["Severity", "Stage", "Finding"], finding_rows(f["gone"])) if f["gone"] else empty("None."),
        ], gap=4), description="Matched by stage and finding id."),
    ]


def stage_panel(stage: str, res: StageResult | None, state: RunState, scheme: str,
                changes: dict | None = None, seat: str = "developer") -> list:
    """The full panel for a stage, including running/failed/pending placeholders. ``changes`` is
    ``service.step_changes`` for this stage (what its last re-run changed), when there is one."""
    status = state.status_of(stage)
    head = dmc.Group([
        dmc.Stack([dmc.Title(LABELS[stage], order=3),
                   dmc.Text(STAGE_BLURB[stage], size="sm", c="dimmed", maw=760)], gap=2),
        dmc.Group([verdict_badge(res.verdict.value) if res is not None and status != "running"
                   else None, step_badge(status, "md")], gap="xs"),
    ], justify="space-between", align="flex-start")
    if status == "failed":
        msg = state.steps[stage].message or "The step failed."
        return [head, dmc.Alert(msg, title="This step failed", color="red", variant="light",
                                icon=icon("tabler:alert-circle")),
                dmc.Group([dmc.Button("Retry", id={"type": "retry", "step": stage},
                                      leftSection=icon("tabler:refresh"))])]
    if res is None or status in ("pending", "running") and res is None:
        body = (dmc.Stack([dmc.Group([dmc.Loader(size="sm", type="dots"),
                                      dmc.Text("Working — the agent's recommendation and the "
                                               "engine's evidence appear here when it finishes.",
                                               size="sm", c="dimmed")]),
                           dmc.Skeleton(h=90, radius="md"), dmc.Skeleton(h=220, radius="md")])
                if status == "running" else empty("Not started yet."))
        return [head, body]
    notice = None
    if status == "running":
        notice = dmc.Alert("Re-running with new inputs; the evidence below is from the previous run.",
                           color="blue", variant="light", icon=icon("tabler:loader-2"))
    elif status == "stale":
        why = state.steps[stage].rerun_reason
        notice = dmc.Alert(["This stage will re-run because — ", dmc.Text(why, span=True, fw=600)] if why
                           else "An earlier input changed; this stage will re-run.",
                           title="Out of date: the evidence below predates that decision" if why else None,
                           color="yellow", variant="light", icon=icon("tabler:refresh-alert"))
    else:
        notice = rerun_note(stage, state, changes)
    body = PANELS[stage](res, state, scheme, seat)
    return [head, notice, *[b for b in body if b is not None]]
