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
    ], value="send")], value=None, variant="contained", mt="sm")


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


# --- the data (gate_source, ADR-0018; the data catalog, ADR-0020) ---------------------------
def _src_upload(field: str, hint: str, multiple: bool) -> Any:
    """A drop zone; the callback saves what is dropped and keeps the paths in a hidden field."""
    return html.Div([
        dcc.Upload(id={"type": "src-upload", "field": field}, multiple=multiple,
                   className="cognos-upload",
                   children=dmc.Group([icon("tabler:upload", 16), dmc.Text(hint, size="xs")],
                                      gap="xs", justify="center")),
        html.Div(id={"type": "src-names", "field": field}),
        dmc.TextInput(id=_field("gate_source", field), value="", style={"display": "none"}),
    ])


def _catalog_table(t: dict) -> Any:
    m = t.get("summary") or {}
    cols = [[dmc.Text(c, ff="monospace", size="xs"), dmc.Text(text, size="xs")]
            for c, text in t.get("columns", {}).items()]
    cols += [[dmc.Text(c, ff="monospace", size="xs"),
              dmc.Text("not described in the catalog", size="xs", c="dimmed")]
             for c in m.get("undescribed", [])]
    owner = " · ".join(x for x in (t.get("owner"), t.get("refresh") and
                                   f"refreshed {t['refresh']}", t.get("location")) if x)
    return dmc.Stack([
        dmc.Checkbox(value=t["id"], label=dmc.Group([
            dmc.Text(t.get("title") or t["id"], fw=600, size="sm"),
            dmc.Text(t["id"], ff="monospace", size="xs", c="dimmed"),
            *(dmc.Badge(f"outcome: {o}", variant="light", color="gray", size="xs", tt="none")
              for o in t.get("outcomes", []))], gap="xs")),
        html.Div(dmc.Stack([
            dmc.Text(t.get("description", ""), size="xs"),
            dmc.Text(service.catalog_line(t), size="xs", c="dimmed"),
            dmc.Text(owner, size="xs", c="dimmed") if owner else None,
            dmc.Accordion([dmc.AccordionItem([
                dmc.AccordionControl(dmc.Text(f"What the columns mean ({len(cols)})", size="xs")),
                dmc.AccordionPanel(table(["Column", "Meaning"], cols)),
            ], value="cols")], variant="default", chevronPosition="left") if cols else None,
        ], gap=2), style={"paddingLeft": 30}),
    ], gap=4, className="cognos-catalog-table")


def catalog_origin(cat: dict) -> str:
    """Which catalog is in force: the default shipped with COGNOS, or the runs folder's own."""
    if cat.get("origin") == "default":
        return ("The default catalog shipped with COGNOS, with synthetic data. To use your "
                f"team's tables, write {cat.get('team_file')} (cognos catalog --init starts it "
                "from this one).")
    return f"Your team's catalog: {cat.get('file')}"


def source_form(info: dict | None = None) -> list:
    """Where the data comes from: tables from the team's data catalog, with what each holds,
    and any data to add (files, Snowflake tables, read-only queries). The engine measures how
    they connect and the Data Analyst proposes the join at explore."""
    info = info or {}
    f = "gate_source"
    cat = info.get("catalog")
    if cat and cat.get("tables"):
        links = cat.get("links") or []
        catalog_part = [
            dmc.Text(cat.get("title") or "The data catalog", fw=600, size="sm"),
            dmc.Text(catalog_origin(cat), size="xs", c="dimmed"),
            dmc.Text("Checked: what this kind of model is usually built on. The first one checked "
                     "is proposed as the base table (one row per thing the model scores).",
                     size="xs", c="dimmed"),
            dmc.CheckboxGroup(id=_field(f, "catalog"), value=info.get("preselected") or [],
                              children=dmc.Stack([_catalog_table(t) for t in cat["tables"]],
                                                 gap="sm", mt="xs")),
            dmc.Accordion([dmc.AccordionItem([
                dmc.AccordionControl(dmc.Text(f"How the tables join, as documented "
                                              f"({len(links)})", size="xs"),
                                     icon=icon("tabler:link")),
                dmc.AccordionPanel(table(["Table", "Joins", "On", "Note"], [
                    [dmc.Text(lk["left"], ff="monospace", size="xs"),
                     dmc.Text(lk["right"], ff="monospace", size="xs"),
                     dmc.Text(", ".join(f"{a} = {b}" for a, b in zip(
                         lk["left_on"], lk["right_on"], strict=False)), ff="monospace",
                         size="xs"),
                     dmc.Text(lk.get("note") or "", size="xs")] for lk in links])),
            ], value="links")], variant="default") if links else None,
        ]
    else:
        catalog_part = [dmc.Alert(
            "The data catalog cannot be read" + (f": {info['catalog_error']}"
                                                 if info.get("catalog_error") else ".")
            + " Add the data below.", color="yellow", variant="light",
            icon=icon("tabler:alert-triangle"))]
    return [
        *catalog_part,
        dmc.Text("Add data (optional)", fw=600, size="sm", mt="md"),
        dmc.Text("Data the catalog does not have. The Data Analyst measures how it connects to "
                 "the tables above and proposes the join; you confirm it at the data gate.",
                 size="xs", c="dimmed"),
        _src_upload("add_files", "Drop files, or click to choose (.csv, .xlsx, .parquet)",
                    multiple=True),
        dmc.Textarea(id=_field(f, "add_sql"), label="Snowflake tables or read-only queries",
                     autosize=True, minRows=2,
                     placeholder="one per line:\ncovenants = RISK.CREDIT.COVENANTS\n"
                                 "watchlist = SELECT obligor_id, as_of_date, on_watch FROM ..."),
        dmc.Text("Snowflake connects with the SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER and password or "
                 "key in this server's environment. Nothing secret is stored in the run.",
                 size="xs", c="dimmed"),
        dmc.Group([
            dmc.TextInput(id=_field(f, "target"), label="Dependent variable (optional)",
                          description="Left empty, the Data Analyst proposes it", w=260),
            dmc.TextInput(id=_field(f, "datetime_col"), label="Date column (optional)",
                          description="Left empty: the base table's, from the catalog", w=260),
        ], align="flex-start", mt="xs"),
    ]


def guide_text(guide: str) -> Any:
    """A template's data guide as text: wrapped lines joined, ``- `` lines as a list."""
    blocks: list[tuple[str, list[str]]] = []  # ("p" | "li", words)
    for line in guide.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            blocks.append(("li", [line[2:]]))
        elif blocks and blocks[-1][0] == "li" and not blocks[-1][1][-1].endswith((";", ".")):
            blocks[-1][1].append(line)  # a wrapped list item
        elif blocks and blocks[-1][0] == "p" and not blocks[-1][1][-1].endswith((":", ".")):
            blocks[-1][1].append(line)
        else:
            blocks.append(("p", [line]))
    out: list[Any] = []
    for kind, words in blocks:
        text = " ".join(words)
        if kind == "li":
            if not out or not isinstance(out[-1], dmc.List):
                out.append(dmc.List([], size="xs", spacing=2, withPadding=True))
            out[-1].children.append(dmc.ListItem(text))
        else:
            out.append(dmc.Text(text, size="xs"))
    return dmc.Stack(out, gap=4)


def source_panel(state: RunState, seat: str = "developer",
                 info: dict | None = None) -> Any:
    """Provide the data: the form while the run waits for it, what was given once it has.
    ``info`` is ``service.data_info(..., with_catalog=True)``: the template's note on the data,
    the data in force, the data catalog and the tables to start with checked."""
    info = info or {}
    status = state.status_of("gate_source")
    if status in ("pending", "skipped"):
        return None
    guide = (dmc.Alert(guide_text(info["guide"]),
                       title="What this kind of model is usually built on", color="gray",
                       variant="light", icon=icon("tabler:database"))
             if info.get("guide") else None)
    if status == "done" and state.last_decision("gate_source") is None:
        return dmc.Card([dmc.Group([
            dmc.Text(LABELS["gate_source"], fw=650),
            dmc.Text(f"The data came with the run: {info.get('given') or 'see the profile'}.",
                     size="sm", c="dimmed")], gap="sm")], className="cognos-gate")
    body = [guide, *source_form(info)] if status == "awaiting" else []
    block = gate_block("gate_source", state, None, seat=seat, send_back=False, body=body,
                       primary=[dmc.Button("Use this data", id=_act("gate_source", "accept"),
                                           leftSection=icon("tabler:database-import"))])
    if status == "done" and info.get("given"):
        return dmc.Stack([dmc.Text(["Data in use: ", dmc.Text(info["given"], span=True, fw=600)],
                                   size="sm"), block], gap=4)
    return block


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


_RELATION = {"one-to-one": "1 → 1", "one-to-many": "1 → many", "many-to-one": "many → 1",
             "many-to-many": "many ↔ many"}


def _link_chip_text(lk: dict) -> str:
    on = ", ".join(a if a == b else f"{a} = {b}"
                   for a, b in zip(lk["a_on"], lk["b_on"], strict=True))
    return (f"{on} · {max(lk['a_in_b'], lk['b_in_a']):.0%}"
            f" · {_RELATION.get(lk['relation'], lk['relation'])}")


def join_canvas(linked: dict, target: str | None, chosen: list[str], *,
                disabled: bool = False) -> Any:
    """The join as a canvas: the tables where they are, the links the engine measured drawn
    between them. A chip on a link adds it to or removes it from the join (it writes the
    `join_links` field); a tile's button makes it the base table (`join_base`). The drawing and
    the dragging are `assets/join_canvas.js`; the colours of a chip come from `chip_class`."""
    plan = linked["plan"]
    proposed = {s["link"] for s in plan["steps"]}
    keys: dict[str, set[str]] = {}
    for lk in linked["links"]:
        keys.setdefault(lk["a"], set()).update(lk["a_on"])
        keys.setdefault(lk["b"], set()).update(lk["b_on"])
    others = [t for t in linked["tables"] if t["name"] != plan["base"]]
    base = next(t for t in linked["tables"] if t["name"] == plan["base"])
    # the base table on top, the others in rows of three under it: each link gets its own lane
    cols = min(3, len(others)) or 1
    width = 16 + cols * 224
    place = {base["name"]: ((width - 214) // 2, 16)}
    for i, t in enumerate(others):
        row, col = divmod(i, 3)
        place[t["name"]] = (12 + col * 224, 380 + row * 230)
    height = 380 + ((len(others) + 2) // 3 or 1) * 230
    tiles = []
    for t in [base, *others]:
        mine = keys.get(t["name"], set()) | set(t.get("unique_columns") or [])
        cols = sorted(t["columns"], key=lambda c: (c not in mine, c != target))
        shown = [html.Li(c, className="jc-key" if c in mine else
                         "jc-target" if c == target else None) for c in cols[:5]]
        if len(cols) > 5:
            shown.append(html.Li(f"+ {len(cols) - 5} more", className="jc-more"))
        x, y = place[t["name"]]
        tiles.append(html.Div([
            html.Header([html.B(t["name"]), html.Span(f"{fmt(t['n_rows'])} rows")]),
            html.Ul(shown),
            html.Button("base table" if t["name"] == plan["base"] else "Make base",
                        id={"type": "jc-base", "table": t["name"]}, className="jc-make-base",
                        disabled=disabled, n_clicks=0),
        ], id={"type": "jc-tile", "table": t["name"]},
            className="jc-tile" + (" base" if t["name"] == plan["base"] else ""),
            style={"left": f"{x}px", "top": f"{y}px"}, title=t.get("description") or None,
            **{"data-table": t["name"]}))
    chips = [html.Button(_link_chip_text(lk),
                         id={"type": "jc-link", "link": lk["id"],
                             "kind": "proposed" if lk["id"] in proposed else "measured"},
                         className=chip_class(lk["id"] in chosen, lk["id"] in proposed),
                         disabled=disabled, n_clicks=0,
                         title=_link_label(lk) + (" · suggested by the Data Analyst"
                                                  if lk["id"] in proposed else ""),
                         **{"data-a": lk["a"], "data-b": lk["b"], "data-link": lk["id"]})
             for lk in linked["links"]]
    return html.Div([
        html.Div(html.Div([html.Div(className="jc-svg-host"), *tiles, *chips],
                          className="jc-canvas",
                          style={"height": f"{height}px", "minWidth": f"{width}px"}),
                 className="jc-wrap"),
        html.Div([html.Span([html.I(className="m"), "Measured by the engine"]),
                  html.Span([html.I(className="p"), "Suggested by the Data Analyst"]),
                  html.Span([html.I(className="c"), "In your join"]),
                  html.Span("Drag a table to arrange. Click a link to add or remove it.")],
                 className="jc-legend"),
    ], className="jc")


def chip_class(chosen: bool, proposed: bool) -> str:
    """How a link reads on the canvas: in the join (and whether the agent suggested it), or
    only measured, or suggested and left out."""
    if chosen:
        return "jc-chip chosen" + (" suggested" if proposed else "")
    return "jc-chip " + ("proposed" if proposed else "measured")


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
    chosen_links = [s["link"] for s in linked["plan"]["steps"]] if linked else []
    join_fields = [] if not linked else [
        dmc.Text("The join", fw=600, size="sm"),
        dmc.Text("Stated in the project profile." if join_fixed else
                 "The tables, the base table framed in ink, and every link the engine measured "
                 "with how well it matches. Changing the join rebuilds the dataset and re-runs "
                 "the exploration.", size="xs", c="dimmed"),
        join_canvas(linked, p.get("target"), chosen_links, disabled=join_fixed),
        # the canvas writes these two fields; they stay in the form so the decision reads them
        html.Div([
            dmc.Select(id=_field("gate_data", "join_base"), label="Base table",
                       data=[t["name"] for t in linked["tables"]], value=linked["plan"]["base"],
                       allowDeselect=False, disabled=join_fixed),
            dmc.MultiSelect(id=_field("gate_data", "join_links"), label="Links to join on",
                            data=[{"value": lk["id"], "label": _link_label(lk)}
                                  for lk in linked["links"]],
                            value=chosen_links, disabled=join_fixed, clearable=True),
        ], style={"display": "none"}),
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
        row("A single factor is ranked on",
            f"{(ev.get('rank_by') or [{}])[0].get('metric', 'the metric')} "
            f"({_words(fc.get('criterion') or 'validation')})", "factors.criterion"),
        row("Single-factor screen", "; ".join(x for x in screen if x) or "no rule",
            next((k for k in sources if k.startswith("factors.")
                  and k not in ("factors.transforms", "factors.criterion")), "factors")),
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
    """The modelling criteria as a form: every choice shows what is in force and who set it,
    and can be overridden. A field left as it is changes nothing; ``gate_payload`` sends only
    what differs from the plan on the page, and the override becomes part of the run's
    effective configuration (``overrides.plan``), which is what the engine then follows."""
    ev, sel, fc = plan["evaluation"], plan["selection"], plan["factors"]
    opts, sources = plan.get("options") or {}, plan.get("sources") or {}

    def by(field: str) -> str:
        src = sources.get(field)
        return {"engine": "recommended by the engine", "agent": "recommended by the Design Lead",
                "profile": "set in the profile", "human": "set by the developer"}.get(src, "")

    def pick(field: str, label: str, value: Any, choices: list[str], source: str = "", **kw) -> Any:
        return dmc.Select(id=_field("gate_design", f"plan::{field}"), label=label, value=value,
                          data=[{"value": c, "label": _words(c)} for c in choices],
                          description=by(source) or None, allowDeselect=False, size="xs", **kw)

    def num(field: str, label: str, value: Any, **kw) -> Any:
        return dmc.NumberInput(id=_field("gate_design", f"plan::{field}"), label=label,
                               value=value if value is not None else "", size="xs", **kw)

    def head(title: str, note: str) -> Any:
        return dmc.Stack([dmc.Text(title, size="sm", fw=650),
                          dmc.Text(note, size="xs", c="dimmed")], gap=0, mt="sm")

    rank = list(ev.get("rank_by") or []) + [{}]
    metrics, samples = opts.get("metric") or [], opts.get("sample") or []
    return dmc.Stack([
        dmc.Divider(mt="sm"),
        dmc.Text("Your modelling criteria", fw=650),
        dmc.Text("Each choice below is the recommendation for this data and this use. Change "
                 "any of them and press “Apply edits / answers”: the override is recorded on "
                 "the run and the engine builds to it.", size="xs", c="dimmed"),
        head("1 · Primary metric", "How candidate models are ranked, and on which sample."),
        dmc.SimpleGrid([
            pick("rank.0.metric", "Rank models by", rank[0].get("metric"), metrics, "rank_by"),
            pick("rank.0.sample", "measured on", rank[0].get("sample"), samples),
            dmc.Select(id=_field("gate_design", "plan::rank.1.metric"), label="then by (optional)",
                       value=rank[1].get("metric"), size="xs", clearable=True,
                       data=[{"value": m, "label": m} for m in metrics]),
            pick("rank.1.sample", "measured on", rank[1].get("sample") or "in_sample", samples),
            num("evaluation.pool_size", "Candidates the engine proposes", ev["pool_size"],
                min=1, max=20),
        ], cols={"base": 2, "md": 5}),
        head("2 · How the data are split", "Where models are fitted, compared and confirmed."),
        dmc.SimpleGrid([
            pick("evaluation.test", "Test sample", ev["test"], opts.get("test") or [], "test"),
            num("evaluation.test_fraction", "Test share of rows", ev["test_fraction"],
                min=0, max=0.6, step=0.05, decimalScale=2),
            num("evaluation.holdout_fraction", "Sealed holdout share (0 = none)",
                ev["holdout_fraction"], min=0, max=0.4, step=0.05, decimalScale=2),
            pick("evaluation.validation", "Validation", ev["validation"],
                 opts.get("validation") or [], "validation"),
            num("evaluation.folds", "Validation blocks", ev["folds"], min=2, max=20),
            num("evaluation.horizon", "Periods per block (rolling)", ev.get("horizon"), min=1),
        ], cols={"base": 2, "md": 3}),
        head("3 · Single-factor selection",
             "Each factor is fitted alone, then these rules run in this order. Empty = off."),
        dmc.SimpleGrid([
            pick("factors.criterion", "Rank a factor on", fc.get("criterion") or "validation",
                 opts.get("factor_criterion") or ["validation", "in_sample"], "factors.criterion"),
            num("factors.max_p_value", "p-value at most", fc.get("max_p_value"), min=0, max=1,
                step=0.01, decimalScale=3),
            dmc.Select(id=_field("gate_design", "plan::factors.enforce_sign"),
                       label="Drop a wrong sign", size="xs", allowDeselect=False,
                       value="yes" if fc.get("enforce_sign") else "no",
                       data=[{"value": "yes", "label": "yes"}, {"value": "no", "label": "no"}]),
            num("factors.best_per_source", "Best per column", fc.get("best_per_source"), min=1),
            num("factors.max_correlation", "Correlation at most", fc.get("max_correlation"),
                min=0, max=1, step=0.05, decimalScale=2),
            num("factors.top_n", "Shortlist size", fc.get("top_n"), min=1),
            num("factors.max_missing", "Missing at most", fc.get("max_missing"), min=0, max=1,
                step=0.05, decimalScale=2),
        ], cols={"base": 2, "md": 4}),
        dmc.MultiSelect(id=_field("gate_design", "plan::factors.transforms"),
                        label="Transformations tried for each column", size="xs",
                        value=list(fc.get("transforms") or []),
                        data=[{"value": t, "label": _words(t)}
                              for t in (opts.get("transforms") or []) if t != "none"]),
        head("4 · Multi-factor selection",
             "Pick the method that builds models from the shortlist. One is enough; if you "
             "pick several, every model they evaluate is ranked in one list."),
        dmc.SimpleGrid([
            dmc.MultiSelect(id=_field("gate_design", "plan::selection.methods"),
                            label="Method", size="xs", description=by("methods") or None,
                            value=list(sel.get("methods") or []),
                            data=[{"value": m, "label": _words(m)}
                                  for m in opts.get("methods") or []]),
            pick("selection.criterion", "A step is kept when this improves", sel["criterion"],
                 opts.get("criterion") or [], "criterion"),
            num("selection.max_factors", "Factors at most", sel["max_factors"], min=1, max=12),
            num("selection.max_p_value", "Every p-value at most", sel.get("max_p_value"),
                min=0, max=1, step=0.01, decimalScale=3),
        ], cols={"base": 2, "md": 4}),
    ], gap=6)


def _who(c: dict) -> Any:
    """A candidate as it is referred to on every page: its name, with the id beside it."""
    return dmc.Group([dmc.Text(c.get("name") or c.get("label") or c["id"], size="sm", fw=600),
                      dmc.Code(c["id"])], gap=6, wrap="nowrap")


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

    leader_rows = [[c.get("rank") or "", _who(c),
                    dmc.Text(c["family"], ff="monospace", size="sm"),
                    dmc.Text(", ".join(c["features"]), size="xs", ff="monospace"),
                    fmt(c.get("score")), dmc.Text(" + ".join(_words(m) for m in c.get("methods") or []),
                                                  size="xs"),
                    tags(c)] for c in leaders]
    pool_rows = [[_who(m), dmc.Text(m["label"], size="sm"),
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
            dmc.Group([dmc.Text({"grid": "Families without a stepwise path",
                                 "same_factors": "Same factors, another family"}.get(
                                     path["method"], _words(path["method"]).capitalize()),
                                fw=650, size="sm"),
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
    try:
        ranking = json.loads(service.read_artifact(state.run_id, p.get("ranking_path") or "",
                                                   None) or "[]") if p.get("ranking_path") else []
    except ValueError:
        ranking = []
    below = [r for r in ranking if r["id"] not in known]
    walked = [x for x in p.get("paths") or [] if x["method"] not in ("grid", "same_factors")]
    if len(walked) > 1:
        how = (f"{len(walked)} methods were run ("
               + ", ".join(_words(x["method"]) for x in walked) + "). They are not alternatives "
               "to choose between afterwards: every model any of them evaluated is ranked in one "
               "list, and the leaders are the best across all of them. “Found by” says which "
               "method reached each model. To build with one method only, choose it in the plan "
               "at the design gate.")
    elif walked:
        how = (f"Models are built by {_words(walked[0]['method'])}, the method chosen in the "
               "plan (change it, or add another, at the design gate). Each step shows what was "
               "added or dropped and what it did to the criterion.")
    else:
        how = "Each family on the slate is fitted on the shortlisted factors."
    gate = gate_block("gate_candidates", state, res, seat=seat, body=[
        dmc.Text("Candidate pool", size="sm", fw=600),
        dmc.Text(f"Tick every candidate that should go on to outcomes analysis: as many as you "
                 f"want to compare (the engine proposes {len(pool_ids)}). Each is fitted, tested "
                 "and compared on the test sample; the champion is chosen there. A name can be "
                 "changed here and is used on every later page.", size="xs", c="dimmed"),
        dmc.CheckboxGroup(dmc.Stack([
            dmc.Group([
                dmc.Checkbox(value=c["id"], label=None),
                dmc.TextInput(id=_field("gate_candidates", f"name::{c['id']}"),
                              value=c.get("name") or "", size="xs", w=150),
                dmc.Text(f"{c['family']} · {', '.join(c['features'])}", size="xs",
                         ff="monospace", style={"flex": 1}),
                dmc.Text(f"{fmt(c.get('score'))}", size="xs", ff="monospace"),
                dmc.Badge("breaks a constraint", color="red", size="xs", variant="light")
                if c.get("violations") else None,
            ], gap="xs", wrap="nowrap", align="center") for c in leaders], gap=4),
            id=_field("gate_candidates", "pool"), value=[i for i in pool_ids if i in known]),
        dmc.MultiSelect(id=_field("gate_candidates", "extra"), mt="sm", searchable=True,
                        label=f"Add from the rest of the ranking ({len(below)} more models)",
                        description="Search by name, family or factor. Every model the search "
                                    "evaluated can join the pool.",
                        value=[i for i in pool_ids if i not in known],
                        data=[{"value": r["id"],
                               "label": f"#{r['rank']} {r['name']} · {', '.join(r['features'])} "
                                        f"· {fmt(r.get('score'))}"} for r in below],
                        limit=40, maxDropdownHeight=280),
    ], primary=[
        dmc.Button("Accept the pool", id=_act("gate_candidates", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply the ticked pool and names", id=_act("gate_candidates", "edit"),
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
        section("Multi-factor selection", *path_blocks, description=how) if path_blocks else None,
        section("Leaders", table(["#", "Model", "Family", "Factors", score_label, "Found by", ""],
                                 leader_rows, max_height=520),
                description=f"The top {len(leaders)} of {len(ranking) or len(leaders)} ranked "
                            "models, one line per distinct specification. The whole ranking is "
                            "stages/model/ranking.json, and any model in it can be added to the "
                            "pool in the review above."),
        section("Candidate pool", table(["Model", "Specification", "Factors", score_label, "Largest p",
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
    names = {c["id"]: c.get("name") or c["id"] for c in p.get("candidates") or []}
    for cid, values in list(data["forecast"].items())[:7]:
        series.append({"name": names.get(cid, cid)
                       + (" (champion)" if cid == p.get("champion_id") else ""),
                       "y": values})
    return {"kind": "line", "x": data["x"], "series": series, "x_title": data.get("x_label", ""),
            "y_title": "actual and forecast"}


def _champion_proposal(p: dict, cands: list, metric: str, sample: str) -> str | None:
    champ = next((c for c in cands if c["id"] == p.get("champion_id")), None)
    if not champ:
        return None
    return (f"{champ.get('name') or champ['id']} ({champ['label']}): the "
            f"{_words(p.get('champion_source', 'engine'))}'s choice, compared with "
            f"{len(cands) - 1} other candidate(s) on the {sample} sample.")


def _champion_evidence(p: dict, cands: list, metric: str) -> list[tuple[str, str]]:
    """The champion's figures as the engine recorded them, for the decision card."""
    champ = next((c for c in cands if c["id"] == p.get("champion_id")), None)
    if not champ:
        return []
    m = (champ.get("outcomes") or {}) | (champ.get("metrics") or {})
    out = [(k.replace("_", " "), fmt(m.get(k), 4)) for k in dict.fromkeys(
        [metric, "gini", "ks", "psi", "rmse"]) if k and m.get(k) is not None][:4]
    out.append(("flags", fmt(len(champ.get("flags") or []))))
    return out


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
    cand_rows = [[dmc.Group([_who(c),
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
    scen_rows = [[_who(c), fmt(c.get("scenario_baseline")),
                  *[next((f"{fmt(s['mean'])}  ({s['delta']:+.4g})" for s in c.get("scenarios") or []
                          if s["name"] == n), "—") for n in names]]
                 for c in cands if c.get("scenarios")]
    gate = gate_block("gate_champion", state, res, seat=seat, body=[
        dmc.RadioGroup(dmc.Stack([
            dmc.Radio(label=f"{c.get('name') or c['id']} · {c['label']} — {sample} {metric} "
                            f"{fmt((c.get('metrics') or {}).get(metric))}",
                      value=c["id"], disabled=c.get("role") == "challenger" or bool(c.get("error")))
            for c in cands], gap=4),
            id=_field("gate_champion", "champion"), label="Champion",
            description="Chosen on how the candidates forecast, not only on how they fitted.",
            value=champion),
    ], primary=[
        dmc.Button("Accept the champion", id=_act("gate_champion", "accept"),
                   leftSection=icon("tabler:check")),
    ], alternative=[
        dmc.Button("Choose the selected candidate", id=_act("gate_champion", "override"),
                   variant="light", color="orange", leftSection=icon("tabler:switch-horizontal")),
    ], fold="Choose another candidate",
        proposal=_champion_proposal(p, cands, metric, sample),
        evidence=_champion_evidence(p, cands, metric))
    head = [("Champion", str(p.get("champion_name") or champion or "—"),
             f"{champion} · chosen by the {p.get('champion_source', 'engine')}"),
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
                            + (" (trial runs of this model are counted on the Run record)"
                               if state.lineage is not None else "")
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
    gate = gate_block("gate_validation", state, res, seat=seat,
                      proposal=(f"Verdict {verdict}: the validator recommends "
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
                            "(val-*) are the validator's challenge."),
    ]


def trials_section(tried: dict | None) -> Any:
    """The runs of this model as validation read them: every trial run looked at the test
    sample, so the count that matters is the model's."""
    from ..trials import describe

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


# --- comply / document / review ------------------------------------------------------------
def comply_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    p = res.payload or {}
    r = p.get("readiness") or {}
    sr = p.get("sr26_2") or p.get("sr11_7") or {}  # sr11_7: recorded before SR 26-2
    nist = p.get("nist_ai_rmf") or {}
    status_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    return [
        dmc.Alert(r.get("narrative", ""), title=f"Readiness: {str(r.get('level', '—')).replace('_', ' ')}",
                  color="blue", variant="light", icon=icon("tabler:clipboard-check")),
        recommendation_card(_rec(res), extra=dmc.List([dmc.ListItem(a["statement"])
                                                       for a in r.get("priority_actions", [])], size="sm")
                            if r.get("priority_actions") else None),
        section("SR 26-2 evidence", table(["Element", "Status", "Evidence"], [
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
    gate = gate_block("gate_signoff", state, res, seat=seat,
                      proposal=f"Review verdict {res.verdict.value}. Approving seals the package "
                               "of recorded facts; it is not deployment.",
                      evidence=[("code anchors", fmt(len(p.get("code_links", [])))),
                                ("findings", fmt(len(res.findings)))], primary=[
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
                changes: dict | None = None, seat: str = "developer",
                data_info: dict | None = None) -> list:
    """The full panel for a stage, including running/failed/pending placeholders. ``changes`` is
    ``service.step_changes`` for this stage (what its last re-run changed), when there is one.
    ``data_info`` is what explore's page says about the data (:func:`source_panel`)."""
    status = state.status_of(stage)
    before = source_panel(state, seat, data_info) if stage == "explore" else None
    head = dmc.Group([
        dmc.Stack([dmc.Title(LABELS[stage], order=3),
                   dmc.Text(STAGE_BLURB[stage], size="sm", c="dimmed", maw=760)], gap=2),
        dmc.Group([verdict_badge(res.verdict.value) if res is not None and status != "running"
                   else None, step_badge(status, "md")], gap="xs"),
    ], justify="space-between", align="flex-start")
    if status == "failed":
        msg = state.steps[stage].message or "The step failed."
        return [head, before, dmc.Alert(msg, title="This step failed", color="red",
                                        variant="light", icon=icon("tabler:alert-circle")),
                dmc.Group([dmc.Button("Retry", id={"type": "retry", "step": stage},
                                      leftSection=icon("tabler:refresh"))])]
    if res is None or status in ("pending", "running") and res is None:
        body = (dmc.Stack([dmc.Group([dmc.Loader(size="sm", type="dots"),
                                      dmc.Text("Working — the agent's recommendation and the "
                                               "engine's evidence appear here when it finishes.",
                                               size="sm", c="dimmed")]),
                           dmc.Skeleton(h=90, radius="md"), dmc.Skeleton(h=220, radius="md")])
                if status == "running" else empty(
                    "Starts once the data is given." if state.status_of("gate_source") == "awaiting"
                    else "Not started yet."))
        return [head, before, body]
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
    return [head, before, notice, stage_questions(stage, state, seat),
            *[b for b in body if b is not None], *tools_section(res, state, scheme),
            stage_challenges(stage, state)]
