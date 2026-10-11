"""One panel per stage: the engine's evidence beside the agent's recommendation, and the human gate.

Every panel is a pure function of (stage result, run state, color scheme, run id) so it can be
rendered and tested without a browser. Gate controls carry pattern-matching ids
(``{"type": "gate-field" | "gate-act", "gate": ..., ...}``) that one callback in ``app.py`` reads.
"""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from .. import service
from ..artifacts import StageResult
from ..engine.graph import LABELS, SEND_BACK_TARGETS
from ..engine.process import CORE_DESIGN, SEAT_OF_GATE, preparation_of, seat_label
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
    "model": "The engine follows the development plan: every factor fitted alone, the library "
             "cut to a shortlist, multi-factor models built by the plan's methods and ranked. The "
             "Modeler recommends the pool of candidates that goes on; no champion is chosen here.",
    "backtest": "The engine scores every pool member on the test sample and under each scenario; "
                "the Outcomes Analyst reads the comparison and recommends the champion.",
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


# --- questions and challenges ---------------------------------------------------------------
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
                               "by the independent validator. The agent must answer each one.")


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
            dmc.Group([dmc.Button("Send back", id=_act(gate, "send_back"), color="oxblood",
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
SOURCES = {"builtin": ("Built-in tool", "oxblood", "tabler:tool"),
           "plugin": ("Plugin tool", "teal", "tabler:plug"),
           "script": ("Agent-written script", "grape", "tabler:code")}


def analysis_source(a: dict) -> str:
    """Where an analysis came from: reviewed COGNOS code, reviewed plugin code, or code the
    agent wrote for this run."""
    if a["kind"] == "code":
        return "script"
    return "builtin" if a.get("origin") in (None, "", "cognos") else "plugin"


def source_badge(a: dict) -> Any:
    kind = analysis_source(a)
    label, color, ic = SOURCES[kind]
    if kind == "plugin":
        label = f"{label} ({a['origin']})"
    return dmc.Badge(label, color=color, variant="light", size="sm", leftSection=icon(ic, 12))


def source_counts(analyses: list[dict]) -> Any:
    """How many analyses came from each source, as a line of labelled counts."""
    n = {k: sum(1 for a in analyses if analysis_source(a) == k) for k in SOURCES}
    return dmc.Group([dmc.Group([dmc.Badge(SOURCES[k][0], color=SOURCES[k][1], variant="light",
                                           size="sm", leftSection=icon(SOURCES[k][2], 12)),
                                 dmc.Text(str(n[k]), size="sm", ff="monospace")], gap=4)
                      for k in SOURCES if n[k]], gap="lg", mb="sm")


def analysis_card(run_id: str, a: dict, scheme: str) -> Any:
    """One analysis: the question, what it found, its chart and table and, for a script, the
    code exactly as the analyst wrote it."""
    try:
        result = json.loads(service.read_artifact(run_id, a["artifact"]) or "{}")
    except ValueError:
        result = {}
    is_code = a["kind"] == "code"
    parts: list[Any] = [
        dmc.Group([dmc.Group([dmc.Code(a["id"]), source_badge(a),
                              None if is_code else dmc.Text(a.get("tool") or "", size="xs",
                                                            ff="monospace", c="dimmed")], gap=6),
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
    for c in a.get("checks") or []:
        parts.append(dmc.Group([
            dmc.Badge("passed" if c["passed"] else f"failed · {c['severity']}",
                      color="green" if c["passed"] else "red", variant="light", size="sm",
                      leftSection=icon("tabler:check" if c["passed"] else "tabler:x", 12)),
            dmc.Text(c["name"], size="xs", fw=600),
            dmc.Text(c.get("detail") or "", size="xs", c="dimmed")], gap=6, mt=4))
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
        # Closed until asked for: the gate's notice says there is code to read, and a script
        # opened by default pushes the other analyses off the page.
        parts.append(dmc.Accordion(folds, variant="separated", mt="xs"))
    # (a card keeps its own height: a tall neighbour in the grid must not stretch its chart)
    return dmc.Card([p for p in parts if p is not None], className="cognos-tile",
                    style={"alignSelf": "start"})


def tools_section(res: StageResult, state: RunState, scheme: str) -> list:
    """What the stage's agent asked the engine to run (plugin or built-in tools), and the tools
    registered for the stage that could not be run. Empty for a stage with neither."""
    p = res.payload or {}
    runs, missing = p.get("tool_runs") or [], p.get("tools_unavailable") or []
    out: list[Any] = []
    if runs:
        out.append(section(
            f"Tools run ({len(runs)})",
            dmc.SimpleGrid([analysis_card(state.run_id, a, scheme) for a in runs],
                           cols={"base": 1, "lg": 2}, spacing="md"),
            description="Requested by this stage's agent, run by the engine. A tool is reviewed "
                        "code from COGNOS or a plugin; a failed check is a finding of this stage."))
    if missing:
        out.append(section(
            "Tools not run",
            table(["Tool", "From", "Why not"],
                  [[dmc.Code(t["name"]), t.get("origin", ""), t.get("note", "")] for t in missing]),
            description="Registered for this stage but unavailable in this run."))
    return out


def _link_label(lk: dict) -> str:
    on = ", ".join(f"{a} = {b}" for a, b in zip(lk["a_on"], lk["b_on"], strict=True))
    return (f"{lk['id']}: {lk['a']} – {lk['b']} on {on} ({lk['relation']}, "
            f"{max(lk['a_in_b'], lk['b_in_a']):.0%} matched)")


def linking_section(linked: dict | None) -> Any:
    """Several sources: what they are, the join as the engine ran it, and the links it measured
    but did not use."""
    if not linked:
        return None
    plan, report = linked["plan"], linked["report"]
    why = (linked.get("rationale") or {})
    reasons = why.get("joins") or {}
    unmatched = why.get("unmatched") or {}
    how = {"profile": "stated in the project profile", "decision": "set at the data gate",
           "agent": "proposed by the Data Analyst"}.get(plan["source"], "")
    used = {s["link"] for s in plan["steps"]}
    sources = table(["Source", "Connector", "Location", "Rows × columns", "Role"], [
        [dmc.Text(t["name"], ff="monospace", size="sm", fw=600), t.get("connector") or "—",
         dmc.Text(str(t.get("query") or t.get("location") or "—"), size="xs", ff="monospace"),
         f"{fmt(t['n_rows'])} × {fmt(t['n_cols'])}",
         dmc.Badge("base", color="oxblood", variant="light", size="sm") if t["name"] == plan["base"]
         else dmc.Badge("not joined", color="gray", variant="outline", size="sm")
         if t["name"] in linked.get("left_out", []) else dmc.Badge("joined", color="gray",
                                                                  variant="light", size="sm")]
        for t in linked["tables"]])
    steps = table(["Added", "On", "Rows matched", "Unmatched", "Rows per key", "Columns"], [
        [dmc.Text(s["right"], ff="monospace", size="sm"),
         dmc.Text("; ".join(s["on"]), ff="monospace", size="xs"),
         dmc.Badge(f"{s['match_rate']:.1%}", variant="light", size="sm",
                   color="green" if s["match_rate"] >= 0.95 else
                   "yellow" if s["match_rate"] >= 0.8 else "red"),
         (f"{fmt(s['dropped_rows'])} dropped" if s.get("dropped_rows") else
          "none" if s["match_rate"] >= 1 else "kept, missing values"),
         {"none": "one", "aggregate": "several, aggregated",
          "first": "several, first kept"}.get(s["reduced"], s["reduced"]),
         fmt(len(s["columns_added"]))]
        for s in report["steps"]]) if report["steps"] else empty("No table was joined to the base.")
    spare = [lk for lk in linked["links"] if lk["id"] not in used]
    notes = [*why.get("concerns", []),
             *(f"{t['table']} is left out: {t['reason']}" for t in why.get("left_out", []))]
    return section(
        "Sources and join", sources,
        dmc.Group([dmc.Text("Base table", size="sm", c="dimmed"),
                   dmc.Text(plan["base"], ff="monospace", fw=650),
                   dmc.Badge(how, variant="outline", size="sm", color="gray", tt="none")],
                  gap="xs", mt="md"),
        dmc.Text(why["base"], size="sm", mt=4) if why.get("base") else None,
        html.Div(steps, style={"marginTop": 12}),
        dmc.List([dmc.ListItem(reasons[s["link"]]) for s in report["steps"]
                  if reasons.get(s["link"])], size="xs", c="dimmed", mt="xs")
        if any(reasons.get(s["link"]) for s in report["steps"]) else None,
        dmc.List([dmc.ListItem([html.B(f"{s['right']}, unmatched rows: "), unmatched[s["link"]]])
                  for s in report["steps"] if s["match_rate"] < 1 and unmatched.get(s["link"])],
                 size="sm", mt="xs")
        if any(s["match_rate"] < 1 and unmatched.get(s["link"]) for s in report["steps"])
        else None,
        dmc.Alert(dmc.List([dmc.ListItem(n) for n in notes], size="sm"),
                  title="Check before accepting", color="yellow", variant="light", mt="sm",
                  icon=icon("tabler:alert-triangle")) if notes else None,
        dmc.Text("Other links the engine measured: " + "; ".join(_link_label(lk) for lk in spare),
                 size="xs", c="dimmed", mt="sm") if spare else None,
        description=f"{len(linked['tables'])} sources joined into one modelling table of "
                    f"{fmt(report['n_rows'])} rows (the base table has "
                    f"{fmt(report.get('base_rows', report['n_rows']))}). A join never adds or "
                    "repeats a row.")


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
    linked = p.get("linking") or None
    join_fixed = bool(linked) and linked["plan"]["source"] == "profile"
    join_fields = [] if not linked else [
        dmc.Select(id=_field("gate_data", "join_base"), label="Base table",
                   data=[t["name"] for t in linked["tables"]], value=linked["plan"]["base"],
                   allowDeselect=False, disabled=join_fixed,
                   description="Its rows are the modelling observations."),
        dmc.MultiSelect(id=_field("gate_data", "join_links"), label="Links to join on",
                        data=[{"value": lk["id"], "label": _link_label(lk)}
                              for lk in linked["links"]],
                        value=[s["link"] for s in linked["plan"]["steps"]], disabled=join_fixed,
                        clearable=True,
                        description="Stated in the project profile." if join_fixed else
                        "Every link the engine measured. Changing the join rebuilds the dataset "
                        "and re-runs the exploration."),
        dmc.Select(id=_field("gate_data", "join_many"), label="When a key matches several rows",
                   data=[{"value": "aggregate", "label": "Aggregate them (numbers averaged)"},
                         {"value": "first", "label": "Keep the first row"}],
                   value=next((s["many"] for s in linked["plan"]["steps"]
                               if s["many"] in ("aggregate", "first")), "aggregate"),
                   allowDeselect=False, disabled=join_fixed),
        dmc.Select(id=_field("gate_data", "join_unmatched"),
                   label="Base rows with no match",
                   data=[{"value": "plan", "label": "As recommended for each table (below)"},
                         {"value": "drop", "label": "Drop them for every table (counted and "
                                                    "reported)"},
                         {"value": "keep", "label": "Keep them for every table, with missing "
                                                    "values"}],
                   value="plan", allowDeselect=False, disabled=join_fixed,
                   description="Dropping changes the population; keeping leaves missing "
                               "values, which most model families fail on."),
    ]
    gate = gate_block("gate_data", state, res, seat=seat, body=[
        *join_fields,
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
        linking_section(linked),
        section("Features to consider", table(["Feature", "Expected", "Corr.", "Why"], cand_rows),
                description="The Data Analyst's candidates for the design stage: hypotheses, "
                            "not a selection.") if cand_rows else None,
        section(f"Analyses ({len(analyses)})",
                source_counts(analyses),
                dmc.SimpleGrid([analysis_card(state.run_id, a, scheme) for a in analyses],
                               cols={"base": 1, "lg": 2}, spacing="md"),
                description="Requested by the Data Analyst, run by the engine. Each one says "
                            "where it came from: a built-in tool and a plugin tool are reviewed "
                            "code; a script is code the agent wrote for this run, kept with its "
                            "hash and re-run at validation.")
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
    decision_color = {"primary": "oxblood", "candidate": "blue", "challenger": "grape",
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
    plan = p.get("plan") or {}
    gate = gate_block("gate_design", state, res, seat=seat, body=[
        dmc.CheckboxGroup(dmc.Stack([dmc.Checkbox(label=f"{h['id']} · {h['family']} / "
                                                        f"{h['feature_strategy']} ({h['role']})",
                                                  value=h["id"])
                                     for h in p.get("hypotheses", [])], gap=4),
                          id=_field("gate_design", "keep"), label="Specifications to search",
                          description="Untick to drop a specification from the slate.",
                          value=[h["id"] for h in p.get("hypotheses", [])]),
        dmc.Stack(q_inputs, gap="xs", mt="sm") if q_inputs else None,
        plan_form(plan) if plan else None,
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
        plan_section(plan) if plan else None,
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


# --- the development plan (shown at design, restated at model) ------------------------------
_SOURCE_COLOR = {"human": "oxblood", "profile": "blue", "agent": "grape", "engine": "gray"}
_SOURCE_LABEL = {"human": "developer", "profile": "profile", "agent": "Design Lead",
                 "engine": "engine"}


def _words(v: Any) -> str:
    return str(v).replace("_", " ")


def _plan_rows(plan: dict) -> list[list[Any]]:
    ev, sel, fc = plan["evaluation"], plan["selection"], plan["factors"]
    reasons, sources = plan.get("reasons") or {}, plan.get("sources") or {}

    def row(label: str, value: str, field: str) -> list[Any]:
        src = sources.get(field)
        return [dmc.Text(label, size="sm", fw=500), dmc.Text(value, size="sm"),
                dmc.Badge(_SOURCE_LABEL.get(src, src), color=_SOURCE_COLOR.get(src, "gray"),
                          variant="light", size="xs") if src else "",
                dmc.Text(reasons.get(field, ""), size="xs")]

    rank = ", then ".join(f"{r['metric']} ({_words(r['sample'])})" for r in ev.get("rank_by") or [])
    screen = [f"missing ≤ {pct(fc['max_missing'], 0)}" if fc.get("max_missing") is not None else None,
              f"p ≤ {fmt(fc['max_p_value'], 2)}" if fc.get("max_p_value") is not None else None,
              "expected sign" if fc.get("enforce_sign") else None,
              f"best {fc['best_per_source']} per column" if fc.get("best_per_source") else None,
              f"|corr| ≤ {fmt(fc['max_correlation'], 2)}" if fc.get("max_correlation") is not None
              else None,
              f"top {fc['top_n']}" if fc.get("top_n") else None]
    limits = [f"at most {sel['max_factors']} factors",
              f"every p ≤ {fmt(sel['max_p_value'], 2)}" if sel.get("max_p_value") is not None else None,
              "expected signs" if sel.get("enforce_sign") else None,
              f"|corr| ≤ {fmt(sel['max_correlation'], 2)} between factors"
              if sel.get("max_correlation") is not None else None,
              f"always {', '.join(sel['must_include'])}" if sel.get("must_include") else None]
    return [
        row("Test sample", _words(ev["test"]) + (f", {pct(ev['test_fraction'], 0)} of the rows"
                                                  if ev["test"] != "none" else ""), "test"),
        row("Validation", _words(ev["validation"])
            + (f", {ev['folds']} block(s)" if ev["validation"] != "none" else "")
            + (f" of {ev['horizon']} period(s)" if ev.get("horizon") else ""), "validation"),
        row("Sealed holdout", f"{pct(ev['holdout_fraction'], 0)} of the rows"
            if ev.get("holdout_fraction") else "none", "holdout_fraction"),
        row("Ranking", rank, "rank_by"),
        row("Factor library", ", ".join(_words(t) for t in fc.get("transforms") or []) or "as explored",
            "factors.transforms" if "factors.transforms" in sources else "transforms"),
        row("Single-factor screen", "; ".join(x for x in screen if x) or "no rule",
            next((k for k in sources if k.startswith("factors.") and k != "factors.transforms"),
                 "factors")),
        row("Selection methods", " + ".join(_words(m) for m in sel.get("methods") or []), "methods"),
        row("A step is kept when", f"{_words(sel['criterion'])} improves", "criterion"),
        row("A model is accepted when", "; ".join(x for x in limits if x), "max_factors"),
        row("Pool", f"{ev['pool_size']} candidate(s) go to outcomes analysis", "pool_size"),
    ]


def plan_section(plan: dict) -> Any:
    """The development plan in force: every choice, who set it and why."""
    return section("Development plan",
                   table(["Choice", "In force", "Set by", "Reason"], _plan_rows(plan)),
                   *[dmc.Alert(note, color="yellow", variant="light", mt="xs")
                     for note in plan.get("ignored") or []],
                   description="How the data are split, how the factor library is shortened and "
                               "how single factors become models. Recommended from the shape of "
                               "the data and the intended use; edit it in the review below.")


def plan_form(plan: dict) -> Any:
    """The plan as form fields. A field left as it is changes nothing; ``gate_payload`` sends
    only what differs from the plan on the page."""
    ev, sel, fc = plan["evaluation"], plan["selection"], plan["factors"]
    opts = plan.get("options") or {}

    def pick(field: str, label: str, value: Any, choices: list[str], **kw) -> Any:
        return dmc.Select(id=_field("gate_design", f"plan::{field}"), label=label, value=value,
                          data=[{"value": c, "label": _words(c)} for c in choices],
                          allowDeselect=False, size="xs", **kw)

    def num(field: str, label: str, value: Any, **kw) -> Any:
        return dmc.NumberInput(id=_field("gate_design", f"plan::{field}"), label=label,
                               value=value if value is not None else "", size="xs", **kw)

    rank = list(ev.get("rank_by") or []) + [{}]
    metrics, samples = opts.get("metric") or [], opts.get("sample") or []
    return dmc.Accordion([dmc.AccordionItem([
        dmc.AccordionControl("Edit the development plan"),
        dmc.AccordionPanel(dmc.Stack([
            dmc.Text("Samples", size="xs", fw=650, c="dimmed"),
            dmc.SimpleGrid([
                pick("evaluation.test", "Test sample", ev["test"], opts.get("test") or []),
                num("evaluation.test_fraction", "Test share of rows", ev["test_fraction"],
                    min=0, max=0.6, step=0.05, decimalScale=2),
                num("evaluation.holdout_fraction", "Sealed holdout share (0 = none)",
                    ev["holdout_fraction"], min=0, max=0.4, step=0.05, decimalScale=2),
                pick("evaluation.validation", "Validation", ev["validation"],
                     opts.get("validation") or []),
                num("evaluation.folds", "Validation blocks", ev["folds"], min=2, max=20),
                num("evaluation.horizon", "Periods per block (rolling)", ev.get("horizon"), min=1),
            ], cols={"base": 2, "md": 3}),
            dmc.Text("Ranking", size="xs", fw=650, c="dimmed"),
            dmc.SimpleGrid([
                pick("rank.0.metric", "Rank by", rank[0].get("metric"), metrics),
                pick("rank.0.sample", "read on", rank[0].get("sample"), samples),
                dmc.Select(id=_field("gate_design", "plan::rank.1.metric"), label="then by",
                           value=rank[1].get("metric"), size="xs", clearable=True,
                           data=[{"value": m, "label": m} for m in metrics]),
                pick("rank.1.sample", "read on", rank[1].get("sample") or "in_sample", samples),
                num("evaluation.pool_size", "Candidates in the pool", ev["pool_size"], min=1,
                    max=8),
            ], cols={"base": 2, "md": 5}),
            dmc.Text("Single-factor screen (applied in this order; empty = off)", size="xs",
                     fw=650, c="dimmed"),
            dmc.SimpleGrid([
                num("factors.max_missing", "Missing at most", fc.get("max_missing"), min=0, max=1,
                    step=0.05, decimalScale=2),
                num("factors.max_p_value", "p-value at most", fc.get("max_p_value"), min=0, max=1,
                    step=0.01, decimalScale=3),
                num("factors.best_per_source", "Best per column", fc.get("best_per_source"), min=1),
                num("factors.max_correlation", "Correlation at most", fc.get("max_correlation"),
                    min=0, max=1, step=0.05, decimalScale=2),
                num("factors.top_n", "Shortlist size", fc.get("top_n"), min=1),
                dmc.Select(id=_field("gate_design", "plan::factors.enforce_sign"),
                           label="Drop a wrong sign", size="xs", allowDeselect=False,
                           value="yes" if fc.get("enforce_sign") else "no",
                           data=[{"value": "yes", "label": "yes"}, {"value": "no", "label": "no"}]),
            ], cols={"base": 2, "md": 6}),
            dmc.MultiSelect(id=_field("gate_design", "plan::factors.transforms"),
                            label="Transformations in the library", size="xs",
                            value=list(fc.get("transforms") or []),
                            data=[{"value": t, "label": _words(t)}
                                  for t in (opts.get("transforms") or []) if t != "none"]),
            dmc.Text("From single factors to models", size="xs", fw=650, c="dimmed"),
            dmc.SimpleGrid([
                dmc.MultiSelect(id=_field("gate_design", "plan::selection.methods"),
                                label="Selection methods (one or several)", size="xs",
                                value=list(sel.get("methods") or []),
                                data=[{"value": m, "label": _words(m)}
                                      for m in opts.get("methods") or []]),
                pick("selection.criterion", "A step is kept when this improves", sel["criterion"],
                     opts.get("criterion") or []),
                num("selection.max_factors", "Factors at most", sel["max_factors"], min=1, max=12),
                num("selection.max_p_value", "Every p-value at most", sel.get("max_p_value"),
                    min=0, max=1, step=0.01, decimalScale=3),
            ], cols={"base": 2, "md": 4}),
        ], gap="xs")),
    ], value="plan")], variant="separated", mt="sm")


def _sign_badge(sign: str, expected: str | None) -> Any:
    if sign not in ("+", "-"):
        return dmc.Text("—", size="sm", c="dimmed")
    if expected in ("+", "-") and expected != sign:
        return dmc.Badge(f"{sign}  (expected {expected})", color="red", variant="light", size="sm",
                         leftSection=icon("tabler:alert-triangle", 12))
    return dmc.Text(sign, size="sm", ff="monospace")


# --- model --------------------------------------------------------------------------------
def model_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    plan = p.get("plan") or {}
    first = (p.get("rank_by") or [{}])[0]
    score_label = f"{first.get('metric', 'score')} ({_words(first.get('sample', 'validation'))})"
    ledger = json.loads(service.read_artifact(state.run_id, "stages/model/ledger.json",
                                              None) or "[]")
    rec = _rec(res) or {}
    out = rec.get("output") or {}
    recommended = set(out.get("pool") or [])
    pool_ids = list(p.get("pool_ids") or [])
    leaders = p.get("leaders") or []
    expected = p.get("expected_signs") or {}

    def tags(c: dict) -> Any:
        return dmc.Group([
            dmc.Badge("in the pool", color="oxblood", size="xs") if c["id"] in pool_ids else None,
            dmc.Badge("Modeler", color="blue", size="xs", variant="light")
            if c["id"] in recommended else None,
            dmc.Badge("within 1 SE", color="gray", size="xs", variant="outline")
            if c.get("within_one_se") else None,
            dmc.Badge("challenger only", color="grape", size="xs", variant="light")
            if c.get("role") == "challenger" else None,
            dmc.Badge(f"{len(c['violations'])} constraint(s) broken", color="red", size="xs",
                      variant="light", leftSection=icon("tabler:alert-triangle", 12))
            if c.get("violations") else None], gap=4)

    leader_rows = [[c.get("rank") or "", dmc.Code(c["id"]),
                    dmc.Text(c["family"], ff="monospace", size="sm"),
                    dmc.Text(", ".join(c["features"]), size="xs", ff="monospace"),
                    fmt(c.get("score")), dmc.Text(" + ".join(_words(m) for m in c.get("methods") or []),
                                                  size="xs"),
                    tags(c)] for c in leaders]
    pool_rows = [[dmc.Code(m["id"]), dmc.Text(m["label"], size="sm"),
                  dmc.Text(", ".join(m["features"]), size="xs", ff="monospace"),
                  fmt(m.get("score")), fmt(m.get("max_p_value"), 3),
                  f"{m['diagnostics_passed']}/{m['diagnostics_run']}",
                  dmc.Text("; ".join(m.get("violations") or []) or "none", size="xs")]
                 for m in p.get("pool") or []]
    signs = [[dmc.Text(s["feature"], ff="monospace", size="sm"), s["expected"], s["observed"],
              dmc.Badge(s["assessment"].replace("_", " "),
                        color={"consistent": "green", "wrong_sign": "red"}.get(s["assessment"], "gray"),
                        variant="light", size="sm")]
             for s in out.get("sign_checks", [])]
    factor_rows = [[dmc.Text(r["name"], ff="monospace", size="sm"),
                    dmc.Text(_words(r.get("transform", "level"))
                             + (f", lag {r['lag']}" if r.get("lag") else ""), size="xs"),
                    _sign_badge(r.get("sign", "n/a"), expected.get(r["name"])
                                or expected.get(r["source"])),
                    fmt(r.get("p_value"), 4), fmt(r.get("in_sample")), fmt(r.get("validation")),
                    dmc.Text(r.get("reason") or "", size="xs")]
                   for r in p.get("single_factor") or []]
    screen_rows = [[_words(r["rule"]), fmt(r["setting"]) if not isinstance(r["setting"], list)
                    else ", ".join(r["setting"]), fmt(r["before"]), fmt(r["after"]),
                    fmt(r["before"] - r["after"])] for r in p.get("screen") or []]
    path_blocks = []
    for path in p.get("paths") or []:
        steps = [[s["step"], s["action"], dmc.Text(s.get("factor") or "", ff="monospace", size="sm"),
                  dmc.Text(", ".join(s["factors"]), size="xs", ff="monospace"), fmt(s.get("criterion")),
                  f"{s['change']:+.4g}" if s.get("change") is not None else "",
                  f"{s['tried']} tried, {s['rejected_by_constraints']} rejected"]
                 for s in path.get("steps") or []]
        path_blocks.append(dmc.Stack([
            dmc.Group([dmc.Text(_words(path["method"]).capitalize(), fw=650, size="sm"),
                       dmc.Badge(path["family"], variant="light", color="gray", size="xs"),
                       dmc.Text(f"criterion: {_words(path['criterion'])}", size="xs", c="dimmed")],
                      gap="xs"),
            table(["Step", "Action", "Factor", "Factors in the model", "Criterion", "Change",
                   "Considered"], steps) if steps else None,
            dmc.Text(path.get("note") or "", size="xs", c="dimmed"),
        ], gap=4, mb="sm"))
    samples = p.get("samples") or {}
    lib = p.get("library") or {}
    known = {c["id"] for c in leaders}
    gate = gate_block("gate_candidates", state, res, seat=seat, body=[
        dmc.CheckboxGroup(dmc.Stack([
            dmc.Checkbox(label=f"{c['id']} · {c['label']} — {score_label} {fmt(c.get('score'))}"
                               + (" · breaks a constraint" if c.get("violations") else ""),
                         value=c["id"]) for c in leaders], gap=4),
            id=_field("gate_candidates", "pool"), label="Candidate pool",
            description="Tick the candidates that go on to outcomes analysis. Each is fitted, "
                        "tested and compared on the test sample; the champion is chosen there.",
            value=[i for i in pool_ids if i in known]),
        dmc.TextInput(id=_field("gate_candidates", "extra"), mt="sm",
                      label="Add candidates from the ledger (ids, comma separated)",
                      value=", ".join(i for i in pool_ids if i not in known),
                      description="Any model the search evaluated can join the pool by its id "
                                  "(stages/model/ledger.tsv)."),
    ], primary=[
        dmc.Button("Accept the pool", id=_act("gate_candidates", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Set the pool to the ticked candidates", id=_act("gate_candidates", "edit"),
                   variant="light", leftSection=icon("tabler:edit")),
    ])
    bench = []
    for key, label in (("challenger_benchmark", "Ensemble ceiling"), ("structural", "Merton structural"),
                       ("migration", "Rating migration")):
        b = p.get(key) or {}
        if key == "challenger_benchmark" and b:
            bench.append([label, fmt(b.get("benchmark_score")), "not deployed"])
        elif (b.get("benchmark") or {}):
            bench.append([label, f"AUC {fmt(b['benchmark'].get('test_roc_auc'))} (test sample)",
                          "not deployed"])
    return [
        kpis([("Factors", fmt(lib.get("n_factors")), f"from {fmt(lib.get('n_sources'))} column(s)"),
              ("Shortlisted", fmt(len(p.get("shortlist") or [])), "after the single-factor screen"),
              ("Models evaluated", fmt(p.get("n_candidates_tried")),
               " + ".join(_words(m) for m in (plan.get("selection") or {}).get("methods") or [])),
              ("Pool", fmt(len(pool_ids)), f"set by the {p.get('pool_source', 'engine')}")]),
        recommendation_card(rec, extra=table(["Factor", "Expected", "Observed", "Assessment"], signs)
                            if signs else None),
        gate,
        section("Samples", kpis([
            ("Development", fmt(samples.get("development")), "models are fitted here"),
            ("Validation", f"{len(samples.get('validation_blocks') or [])} block(s)",
             _words((plan.get("evaluation") or {}).get("validation", "—"))),
            ("Test", fmt(samples.get("test")), "the pool is compared here"),
            ("Sealed holdout", fmt(samples.get("holdout")) if samples.get("holdout") else "none",
             "scored once, after the choice")]),
            *[dmc.Text(n, size="xs", c="dimmed") for n in [samples.get("validation_note"),
                                                             *(lib.get("notes") or [])] if n],
            description=p.get("plan_summary")),
        section("Single-factor screen",
                table(["Rule", "Setting", "Factors in", "Factors out", "Dropped"], screen_rows),
                dmc.Text("Shortlist", size="xs", fw=650, c="dimmed", mt="sm"),
                table(["Factor", "Transformation", "Sign", "p-value", "In sample", "Validation", ""],
                      factor_rows, max_height=360),
                description="Every factor fitted alone on the development sample. The rule and "
                            "the reason for each dropped factor are in "
                            "stages/model/single_factor.json."),
        section("Multi-factor selection", *path_blocks,
                description="Each method's path: what was added or dropped at each step and what "
                            "it did to the criterion.") if path_blocks else None,
        section("Leaders", table(["#", "Id", "Family", "Factors", score_label, "Found by", ""],
                                 leader_rows, max_height=420),
                description="The top of the ranking, one line per distinct specification."),
        section("Candidate pool", table(["Id", "Specification", "Factors", score_label, "Largest p",
                                         "Diagnostics", "Constraints broken"], pool_rows),
                description="Each member is fitted on the development sample, tested and saved. "
                            "None is the champion yet."),
        section("Experiment ledger", graph(charts.ledger(ledger, first.get("metric", "score"),
                                                         leaders[0]["label"] if leaders else None,
                                                         scheme), "No ledger.")),
        section("Challenger benchmarks", table(["Benchmark", "Score", "Status"], bench))
        if bench else None,
        section("Findings", findings_table(_findings(res))),
    ]


# --- backtest -----------------------------------------------------------------------------
def _forecast_spec(run_id: str, p: dict) -> dict | None:
    """Actual and forecast per pool member on the comparison sample, as a chart spec."""
    if not p.get("forecasts_path") or p.get("outcomes_analysis"):
        return None
    try:
        data = json.loads(service.read_artifact(run_id, p["forecasts_path"], None) or "null")
    except ValueError:
        data = None
    if not data or not data.get("actual"):
        return None
    series = [{"name": "actual", "y": data["actual"]}]
    for cid, values in list(data["forecast"].items())[:7]:
        series.append({"name": cid + (" (champion)" if cid == p.get("champion_id") else ""),
                       "y": values})
    return {"kind": "line", "x": data["x"], "series": series, "x_title": data.get("x_label", ""),
            "y_title": "actual and forecast"}


def backtest_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    oa = p.get("outcomes_analysis") or {}
    fa = p.get("forecast_analysis") or {}
    interp = p.get("interpretation") or {}
    port = p.get("portfolio_analysis") or {}
    stress = (p.get("stress_testing") or {}).get("scenarios") or []
    cands = p.get("candidates") or []
    metric = p.get("oos_metric_name", "")
    sample = _words(p.get("evaluation_sample", "test"))
    champion = p.get("champion_id")
    keys = [k for k in (metric, "rmse", "mae", "r2", "bias", "gini", "ks",
                        "expected_calibration_error", "psi")
            if any(k in ((c.get("outcomes") or {}) | (c.get("metrics") or {})) for c in cands)]
    keys = list(dict.fromkeys(keys))
    labels = {"expected_calibration_error": "calibration error", "r2": "R²"}
    cand_rows = [[dmc.Group([dmc.Code(c["id"]),
                             dmc.Badge("champion", color="oxblood", size="xs")
                             if c["id"] == champion else None,
                             dmc.Badge("analyst pick", color="blue", size="xs", variant="light")
                             if c["id"] == p.get("recommended_id") and c["id"] != champion
                             else None], gap=4),
                  dmc.Text(", ".join(c.get("features") or []), size="xs", ff="monospace"),
                  fmt(c.get("development_score")),
                  *[fmt(((c.get("outcomes") or {}) | (c.get("metrics") or {})).get(k)) for k in keys],
                  dmc.Text("; ".join(c.get("flags") or []) or (c.get("error") or ""), size="xs")]
                 for c in cands]
    names = list(dict.fromkeys(s["name"] for c in cands for s in c.get("scenarios") or []))
    scen_rows = [[dmc.Code(c["id"]), fmt(c.get("scenario_baseline")),
                  *[next((f"{fmt(s['mean'])}  ({s['delta']:+.4g})" for s in c.get("scenarios") or []
                          if s["name"] == n), "—") for n in names]]
                 for c in cands if c.get("scenarios")]
    gate = gate_block("gate_champion", state, res, seat=seat, body=[
        dmc.RadioGroup(dmc.Stack([
            dmc.Radio(label=f"{c['id']} · {c['label']} — {sample} {metric} "
                            f"{fmt((c.get('metrics') or {}).get(metric))}",
                      value=c["id"], disabled=c.get("role") == "challenger" or bool(c.get("error")))
            for c in cands], gap=4),
            id=_field("gate_champion", "champion"), label="Champion",
            description="Chosen on how the candidates forecast, not only on how they fitted.",
            value=champion),
    ], primary=[
        dmc.Button("Accept the champion", id=_act("gate_champion", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Choose the selected candidate", id=_act("gate_champion", "override"),
                   variant="light", color="orange", leftSection=icon("tabler:switch-horizontal")),
    ])
    head = [("Champion", str(champion or "—"), f"chosen by the {p.get('champion_source', 'engine')}"),
            ("Candidates compared", fmt(len(cands)), f"on the {sample} sample")]
    if oa:
        head += [("Gini", fmt(oa.get("gini"), 3), "discrimination"),
                 ("Calibration error", fmt(oa.get("expected_calibration_error"), 3),
                  "mean |obs − pred|")]
    else:
        head += [(f"{sample.capitalize()} {metric}", fmt(p.get("oos_metric")), None),
                 ("Bias", f"{fa['bias']:+.4g}" if fa.get("bias") is not None else "—",
                  "mean forecast − mean outcome")]
    return [
        kpis(head),
        recommendation_card(_rec(res), extra=dmc.Stack([
            dmc.SimpleGrid([
                dmc.Stack([dmc.Text(k.title(), size="xs", c="dimmed", fw=600), dmc.Text(v, size="sm")],
                          gap=2) for k, v in interp.items()], cols={"base": 1, "md": 3})
            if interp else None], gap="xs")),
        gate,
        section("Candidates side by side",
                table(["Candidate", "Factors", "Development", *[labels.get(k, k) for k in keys], ""],
                      cand_rows),
                description=f"Every pool member scored on the {sample} sample, best first on "
                            f"{metric}. The test sample has been used "
                            f"{fmt(p.get('test_evaluations'))} time(s) in this run"
                            + ("; a sealed holdout is scored at validation."
                               if p.get("sealed_holdout") else
                               ", and no sealed holdout was reserved: the champion's result here "
                               "is also what it was chosen on.")),
        section("Forecasts against outcomes",
                graph(charts.from_spec(_forecast_spec(state.run_id, p), scheme)),
                description="Candidates that rank together on the development sample can part "
                            "ways here.") if _forecast_spec(state.run_id, p) else None,
        section("Scenarios, candidate by candidate",
                table(["Candidate", "Baseline", *names], scen_rows),
                description="Mean forecast under each scenario (change on the baseline).")
        if scen_rows else None,
        section("Champion: calibration by score band",
                graph(charts.calibration(oa.get("calibration_table") or [], scheme), "n/a"),
                description=f"Evaluation sample: {sample}"
                            + (" (IMPACT)" if p.get("used_impact") else " (built-in scorer)"))
        if oa else None,
        section("Champion: portfolio loss simulation", kpis([
            ("Expected loss", pct(port.get("expected_loss"), 2), "share of EAD"),
            (f"VaR {fmt(port.get('confidence'), 3)}", pct(port.get("var"), 2), None),
            ("Expected shortfall", pct(port.get("es"), 2), None)])) if port else None,
        section("Champion: stress scenarios", table(["Scenario", "Mean PD", "Δ PD"],
                                                    [[s["name"], pct(s.get("mean_pd"), 2),
                                                      f"{s.get('delta_pd', 0):+.2%}"] for s in stress]))
        if stress and oa else None,
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
    return [head, notice, stage_questions(stage, state, seat),
            *[b for b in body if b is not None], *tools_section(res, state, scheme),
            stage_challenges(stage, state)]
