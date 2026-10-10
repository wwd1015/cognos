"""The COGNOS workbench — a Dash + Mantine app for model developers.

Two pages: **Runs** (every run, and a drawer to start a new one from a project profile or a
synthetic demo, as a new model development or a model update, with the business intent document
and the existing model's artifacts uploaded there) and the **run workspace** (stage rail, the selected stage's evidence beside its
agent's recommendation, the human gate, live activity, questions & challenges, and the agent audit).

The engine runs in background threads (``cognos.service``); the browser polls run state once a
second and re-renders only what changed, so a refresh never loses anything — all state is on disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from .. import __version__, service
from ..engine import GateError
from ..engine.graph import GATE_OF_STAGE, GATES, LABELS, STAGE_OF_GATE, STAGES
from ..engine.process import CORE_DESIGN, SEAT_LABEL, SEAT_OF_GATE, journal, next_action, seat_label
from .components import (
    empty,
    fmt,
    icon,
    kpis,
    run_badge,
    section,
    status_icon,
    table,
    verdict_badge,
)
from .panels import compare_page, stage_panel
from .theme import FONT_CSS, MANTINE_THEME

ASSETS = Path(__file__).with_name("assets")


# --- layout ----------------------------------------------------------------------------------
def header() -> dmc.AppShellHeader:
    """The masthead: a serif wordmark over a double rule, and a few quiet controls."""
    return dmc.AppShellHeader(html.Div(html.Div([
        dcc.Link([html.Span("COGNOS", className="masthead-word"),
                  html.Span("Model development workbench", className="masthead-eyebrow")],
                 href="/", className="masthead-brand"),
        html.Div([
            html.Span("agents recommend · you decide · the engine disposes",
                      className="masthead-meta"),
            dcc.Link("All runs", href="/", className="masthead-link"),
            html.Span(f"v{__version__}", className="masthead-meta"),
            dmc.Switch(id="theme-switch", onLabel=icon("tabler:moon", 13),
                       offLabel=icon("tabler:sun", 13), size="sm", color="dark"),
        ], className="masthead-tools"),
    ], className="masthead-row"), className="masthead"))


def build_layout() -> Any:
    return dmc.MantineProvider(
        id="mantine", theme=MANTINE_THEME, forceColorScheme="light",
        children=[
            dcc.Location(id="url"),
            dcc.Store(id="theme-store", storage_type="local"),
            dcc.Interval(id="poll", interval=1000),
            dmc.NotificationContainer(id="notify", position="top-right", autoClose=5000),
            dmc.AppShell([
                header(),
                dmc.AppShellMain(html.Div(id="page", className="cognos-page")),
            ], header={"height": 72}, padding="md"),
        ])


def runs_page() -> Any:
    profiles = service.list_profiles()
    providers = service.provider_list()
    prov_data = [{"value": p["id"], "label": f"{p['label']}" + ("" if p["available"] else " — unavailable"),
                  "disabled": not p["available"]} for p in providers if p["id"] != "replay"]
    default_prov = next((p["id"] for p in providers if p["available"] and p["id"] == "claude_cli"),
                        "heuristic")
    sources = {s["kind"]: s for s in service.plugin_list()["sources"]}
    snow = sources.get("snowflake", {})
    own = [{"value": "data:file", "label": "Upload a data file (.csv, .parquet, .xlsx)"},
           {"value": "data:snowflake", "label": "Snowflake table or query"
            + ("" if snow.get("available") else " — not configured")}]
    source_data = ([{"group": "Your data — the Data Analyst proposes the target", "items": own}]
                   + [{"group": "Synthetic demos", "items": [
        {"value": f"demo:{k}", "label": v} for k, v in service.DEMO_LABELS.items()]}]
        + ([{"group": "Project profiles", "items": [
            {"value": f"profile:{p['path']}", "label": f"{p['name']} ({p['path']})"}
            for p in profiles]}] if profiles else []))
    prior_runs = [{"value": r["run_id"], "label": f"{r['project']} · {r['run_id']}"}
                  for r in service.list_runs() if not r["legacy"] and r["champion"]]
    drawer = dmc.Drawer(id="new-drawer", title=dmc.Text("Start a run", fw=650), position="right",
                        size="lg", padding="lg", children=dmc.Stack([
        dmc.SegmentedControl(id="new-kind", value="new", fullWidth=True, data=[
            {"value": "new", "label": "New model development"},
            {"value": "update", "label": "Model update"}]),
        dmc.Select(id="new-source", label="Data & design", data=source_data, value="demo:commercial",
                   allowDeselect=False, searchable=True),
        html.Div(dmc.Stack([
            dmc.TextInput(id="new-name", label="Model name", placeholder="e.g. commercial_pd"),
            html.Div(dmc.Stack([
                dmc.Text("Data file", size="sm", fw=500),
                upload_box("new-data", "Drop the dataset, or click to choose", multiple=False),
            ], gap=4), id="new-data-file"),
            html.Div(dmc.Stack([
                dmc.TextInput(id="new-sf-table", label="Table",
                              placeholder="DATABASE.SCHEMA.TABLE"),
                dmc.Textarea(id="new-sf-query", label="Or a read-only query", autosize=True,
                             minRows=2, placeholder="SELECT ... FROM ... WHERE ..."),
                dmc.NumberInput(id="new-sf-limit", label="Row limit (optional)", min=1),
                dmc.Text(snow.get("note") or "Connects with the SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER "
                         "and password or key in this server's environment. Nothing secret is "
                         "stored in the run.", size="xs", c="dimmed"),
            ], gap=4), id="new-data-sf", style={"display": "none"}),
        ], gap="xs"), id="new-data-wrap", style={"display": "none"}),
        dmc.Stack([
            dmc.Group([
                dmc.Text("Business intent document", size="sm", fw=500, id="new-intent-label"),
                dmc.Button("Download the template", id="new-template", variant="subtle",
                           size="compact-xs", leftSection=icon("tabler:download", 14), n_clicks=0),
            ], justify="space-between"),
            upload_box("new-intent", "Drop the filled-in template, or click to choose "
                                     "(.md, .txt, .docx)", multiple=False),
            dmc.Text("What the model is for, in the sponsor's words. The Intake Analyst reads it "
                     "and asks about anything left unclear. A synthetic demo brings its own.",
                     size="xs", c="dimmed", id="new-intent-help"),
        ], gap=4),
        dmc.Stack([
            dmc.Text("Background documents (optional)", size="sm", fw=500),
            upload_box("new-support", "Policies, data dictionaries, earlier analyses",
                       multiple=True),
        ], gap=4),
        html.Div(dmc.Stack([
            dmc.Text("The existing model", size="sm", fw=500),
            upload_box("new-prior", "White paper, development and deployment code, validation "
                                    "and monitoring reports", multiple=True),
            dmc.Select(id="new-prior-run", label="Or an earlier COGNOS run of it",
                       data=prior_runs, clearable=True, searchable=True,
                       placeholder="none"),
        ], gap=4), id="new-prior-wrap", style={"display": "none"}),
        dmc.SegmentedControl(id="new-mode", value="interactive", fullWidth=True, data=[
            {"value": "interactive", "label": "Interactive — I review each gate"},
            {"value": "autonomous", "label": "Autonomous — prototype"}]),
        dmc.Select(id="new-provider", label="Who makes the recommendations", data=prov_data,
                   value=default_prov, allowDeselect=False,
                   description="LLM agents recommend; the deterministic agents run offline."),
        dmc.Alert("The engine computes every number and guards the sealed holdout; agents only "
                  "recommend, and you decide at the gates.", color="oxblood", variant="light",
                  icon=icon("tabler:shield-lock")),
        dmc.Button("Start run", id="new-create", leftSection=icon("tabler:player-play"), fullWidth=True),
        html.Div(id="new-feedback"),
    ], gap="md"))
    return dmc.Container([
        dmc.Group([
            html.Div([html.Div("Runs", className="eyebrow"),
                      html.Div("Model development", className="page-title"),
                      html.Div("A run starts from the sponsor's business intent (a new model, or "
                               "an update of an existing one) and takes it through nine stages "
                               "and six review gates.", className="page-lede")]),
            dmc.Button("New run", id="new-open", leftSection=icon("tabler:plus")),
        ], justify="space-between", align="flex-end", mt="lg", mb="xl", wrap="nowrap"),
        html.Div(id="runs-figures", children=runs_figures()),
        dmc.Card([
            dmc.Group([html.Div("All runs", className="ledger-title"), compare_picker()],
                      justify="space-between", align="center", mb="xs", wrap="nowrap"),
            html.Div(id="runs-table", children=runs_table()),
        ], mt="xl"),
        html.Div([html.Span("Agents recommend, people decide, the engine computes every number."),
                  html.Span("Internal use")], className="page-foot"),
        drawer,
        dcc.Download(id="template-download"),
    ], size="xl", py="md")


def runs_figures() -> Any:
    """The figures above the run list: how much work is open and whose move it is."""
    rows = [r for r in service.list_runs() if not r["legacy"]]
    if not rows:
        return None
    awaiting = [r for r in rows if r["status"] == "awaiting"]
    spend = sum(r["spend_usd"] or 0 for r in rows)
    return kpis([
        ("Runs", fmt(len(rows)), None),
        ("Awaiting review", fmt(len(awaiting)),
         LABELS.get(awaiting[0]["waiting_on"], "") if awaiting else "nothing waiting"),
        ("Completed or approved", fmt(sum(r["status"] in ("completed", "approved") for r in rows)),
         f"{sum(r['status'] == 'approved' for r in rows)} signed off"),
        ("Blocked or failed", fmt(sum(r["status"] in ("blocked", "failed", "rejected")
                                      for r in rows)), None),
        ("Agent spend", f"${spend:,.2f}", "this workspace"),
    ])


def upload_box(id_: str, hint: str, *, multiple: bool) -> Any:
    """A document drop zone and, under it, the names of what was chosen."""
    return html.Div([
        dcc.Upload(id=id_, multiple=multiple, className="cognos-upload",
                   children=dmc.Group([icon("tabler:upload", 16), dmc.Text(hint, size="xs")],
                                      gap="xs", justify="center")),
        html.Div(id=f"{id_}-names"),
    ])


def upload_names(names: Any) -> Any:
    names = [names] if isinstance(names, str) else list(names or [])
    if not names:
        return None
    return dmc.Group([dmc.Badge(n, variant="light", color="oxblood", size="sm", tt="none",
                                leftSection=icon("tabler:file-text", 12)) for n in names],
                     gap=4, mt=4)


def saved_uploads(contents: Any, names: Any) -> list[str]:
    """Save what an Upload holds and return the paths (one or many files)."""
    if not contents:
        return []
    if isinstance(contents, str):
        contents, names = [contents], [names]
    return [service.save_upload(n, c) for c, n in zip(contents, names or [], strict=False)]


def compare_picker() -> Any:
    """Pick two runs to compare. A control of its own (not checkboxes in the table) because the
    table re-renders as runs progress and would drop a half-made selection."""
    runs = [r for r in service.list_runs() if not r["legacy"]]
    if len(runs) < 2:
        return None
    data = [{"value": r["run_id"], "label": f"{r['run_id']} · {r['project']}" + (f" · {r['champion']}" if r["champion"] else "")}
            for r in runs]
    return dmc.Group([
        dmc.MultiSelect(id="cmp-pick", data=data, maxValues=2, searchable=True, clearable=True,
                        placeholder="Pick two runs to compare", leftSection=icon("tabler:git-compare"),
                        w=420, size="xs", comboboxProps={"withinPortal": True}),
        dmc.Button("Compare", id="cmp-go", variant="default", size="xs", disabled=True, n_clicks=0),
    ], gap="xs", wrap="nowrap")


def runs_table() -> Any:
    rows = service.list_runs()
    if not rows:
        return html.Div([html.Div("Nothing here yet.", className="ledger-title"),
                         html.Div("Start a run with “New run”: upload a business intent document, "
                                  "or try a synthetic demo.", className="ledger-desc")],
                        style={"padding": "28px 0"})
    body = []
    for r in rows:
        link = (dcc.Link(r["project"], href=f"/run/{r['run_id']}",
                         style={"fontWeight": 600, "color": "var(--ink)"}) if not r["legacy"]
                else dmc.Text(r["project"], c="dimmed", size="sm"))
        body.append([
            link, dmc.Text(r["run_id"], size="xs", ff="monospace", c="dimmed"),
            run_badge(r["status"], "sm"),
            dmc.Text(LABELS.get(r["waiting_on"], "") if r["waiting_on"] else "", size="sm"),
            dmc.Text(r["mode"] + (" · update" if r.get("kind") == "update" else ""), size="sm"),
            dmc.Text(r["provider"], size="xs", ff="monospace"),
            dmc.Text(r["champion"] or "—", size="xs", ff="monospace"),
            dmc.Text(f"${r['spend_usd']:.2f}" if r["spend_usd"] else "—", size="xs", ff="monospace"),
            dmc.Text((r["updated_at"] or "")[:16].replace("T", " "), size="xs", ff="monospace",
                     c="dimmed"),
        ])
    return table(["Model", "Run", "Status", "Waiting on", "Mode", "Agents", "Champion", "Spend",
                  "Updated"], body)


def workspace_page(run_id: str) -> Any:
    return html.Div([
        dcc.Store(id="ws-run", data=run_id),
        dcc.Store(id="ws-step", data=None),
        dcc.Store(id="ws-seat", data="developer"),
        dcc.Store(id="ws-key", data=None),
        dmc.Group([
            html.Span("Acting as", className="eyebrow"),
            dmc.SegmentedControl(
                id="seat-switch", value="developer", size="xs",
                data=[{"value": k, "label": v.title()} for k, v in SEAT_LABEL.items()
                      if k != "express"],
            ),
        ], gap="sm", mt="xs", mb="lg"),
        dmc.Grid([
            dmc.GridCol([html.Div(id="ws-head"), html.Div(id="ws-rail", className="cognos-rail")],
                        span={"base": 24, "md": 6}),
            dmc.GridCol([
                dmc.Tabs([dmc.TabsList([
                    dmc.TabsTab("Workflow", value="workflow", leftSection=icon("tabler:route")),
                    dmc.TabsTab("Questions & challenges", value="questions",
                                leftSection=icon("tabler:messages")),
                    dmc.TabsTab("Agent audit", value="audit", leftSection=icon("tabler:list-search")),
                ])], id="ws-tab", value="workflow", mb="md"),
                html.Div(id="ws-content", className="cognos-content"),
            ], span={"base": 24, "md": 13}),
            dmc.GridCol(html.Div(id="ws-activity"), span={"base": 24, "md": 5}),
        ], gutter="lg", columns=24),
        dmc.Modal(id="io-modal", size="80%", title="Agent call", children=html.Div(id="io-body")),
        dcc.Download(id="export-download"),
    ])


# --- workspace fragments ---------------------------------------------------------------------
def run_header(st) -> Any:
    res = service.results(st.run_id)
    model = res.get("model")
    facts = [("Mode", st.mode), ("Agents", st.provider)]
    if model is not None:
        facts += [("Champion", str(model.metrics.get("champion", ""))),
                  ("CV", fmt(model.metrics.get("cv_mean")))]
    if st.spend_usd:
        facts.append(("Spend", f"${st.spend_usd:.2f}"))
    return dmc.Card([
        dmc.Group([html.Span("Run", className="eyebrow"), run_badge(st.status, "sm")],
                  justify="space-between"),
        html.Div(st.project, className="run-hero-name"),
        html.Div(st.run_id, className="run-hero-id"),
        html.Dl([x for k, v in facts for x in (html.Dt(k), html.Dd(v))], className="run-facts"),
        html.Div(next_action(st)["text"], className="run-hero-next"),
        dmc.Alert(st.halted_reason, color="red", variant="light", mt="xs")
        if st.halted_reason and st.status in ("blocked", "rejected", "failed") else None,
        run_actions(st.run_id),
    ], className="run-hero cognos-plain")


def run_actions(run_id: str) -> Any:
    prev = service.previous_run(run_id)
    return dmc.Group([
        dmc.Button("Export", id="export-btn", variant="default", size="compact-sm", n_clicks=0,
                   leftSection=icon("tabler:package-export", 14)),
        dcc.Link(dmc.Button("Compare with previous run", variant="subtle", size="compact-sm",
                            leftSection=icon("tabler:git-compare", 14)),
                 href=f"/compare/{prev}/{run_id}") if prev else None,
    ], gap="xs", mt="sm")


def rail(st, selected: str) -> Any:
    items = []
    for i, stage in enumerate(STAGES, 1):
        status = st.status_of(stage)
        gate = GATE_OF_STAGE.get(stage)
        gstatus = st.status_of(gate) if gate else None
        right = None
        if gstatus == "awaiting":
            who = seat_label(SEAT_OF_GATE.get(gate, ""))
            right = dmc.Badge(who, color="oxblood", size="xs", variant="filled")
        elif st.steps[stage].verdict and status in ("done", "blocked"):
            right = verdict_badge(st.steps[stage].verdict, "xs")
        desc = LABELS[gate] + " — " + gstatus.replace("_", " ") if gstatus and gstatus not in (
            "pending", "skipped") else step_badge_text(status)
        if status == "stale" and st.steps[stage].rerun_reason:  # say what made it out of date
            why = st.steps[stage].rerun_reason
            desc = "Out of date — " + (why if len(why) <= 70 else why[:69].rstrip() + "…")
        items.append(dmc.NavLink(
            id={"type": "rail", "step": stage},
            label=dmc.Text([html.Span(f"{i:02d}", className="rail-no"), LABELS[stage]],
                           size="sm", fw=600 if stage == selected else 500),
            description=desc, leftSection=status_icon("awaiting" if gstatus == "awaiting" else status),
            rightSection=right, active=stage == selected, variant="subtle", color="oxblood",
            disabled=status == "skipped", n_clicks=0))
    return dmc.Card([html.Div("Stages", className="eyebrow", style={"marginBottom": 6}),
                     *items])


def step_badge_text(status: str) -> str:
    from .theme import STEP

    return STEP.get(status, ("", "", status))[2]


def activity(run_id: str) -> Any:
    evs = service.run_events(run_id, limit=60)[::-1][:40]
    tones = {"step_failed": "bad", "step_blocked": "bad", "agent_error": "bad",
             "gate_waiting": "accent", "gate_decision": "accent", "challenge": "accent",
             "agent_start": "quiet", "progress": "quiet"}
    items = [html.Div([
        html.Div(time.strftime("%H:%M:%S", time.localtime(e["ts"])), className="activity-time"),
        html.Div(e["message"], className="activity-text"),
    ], className="activity-item", **{"data-tone": tones.get(e["type"], "plain")}) for e in evs]
    return dmc.Card([html.Div("Activity", className="eyebrow", style={"marginBottom": 4}),
                     dmc.ScrollArea(items if items else empty("Nothing yet."), h=640, type="auto",
                                    scrollbars="y")])


def questions_tab(st, seat: str = "developer") -> Any:
    gaps = [g for g in st.gaps]
    q_rows = []
    for g in gaps:
        if g.status == "open":
            ctrl = dmc.Group([
                dmc.TextInput(id={"type": "gap-text", "gap": g.id}, placeholder="Your answer",
                              style={"flex": 1}),
                dmc.Button("Answer", id={"type": "gap-act", "gap": g.id, "action": "answer"},
                           size="xs", disabled=seat != "developer"),
                dmc.Button("Accept as assumption", id={"type": "gap-act", "gap": g.id,
                                                       "action": "assume"},
                           size="xs", variant="subtle",
                           disabled=seat != "developer" or g.design_field in CORE_DESIGN)
                if g.design_field not in CORE_DESIGN else dmc.Text(
                    "A sponsor decision — answer it. It cannot be assumed.", size="xs", c="dimmed"),
            ], gap="xs", wrap="nowrap")
        else:
            ctrl = dmc.Text([dmc.Badge(g.status, size="xs", variant="light"), " ", g.answer or ""],
                            size="sm")
        q_rows.append([dmc.Code(g.id), dmc.Badge(g.category, size="xs", variant="outline"),
                       dmc.Stack([dmc.Text(g.question, size="sm"), ctrl], gap=4)])
    c_rows = [[dmc.Code(c.id), c.source, LABELS.get(c.target_stage, c.target_stage),
               dmc.Badge(c.status, size="xs", variant="light",
                         color="oxblood" if c.status == "open" else "gray"),
               dmc.Stack([dmc.Text(c.message, size="sm"),
                          dmc.Text(f"↳ {c.response}", size="xs", c="dimmed") if c.response else None],
                         gap=2)] for c in st.challenges]
    d_rows = [[d.gate, d.action.replace("_", " "), d.seat or d.actor, d.reason or "—",
               (d.at or "")[:19].replace("T", " ")] for d in st.decisions]
    record = journal(st, service.results(st.run_id))
    r_rows = [[(row["at"] or "")[:19].replace("T", " "), row["who"], row["name"],
               dmc.Text(row["what"], size="sm")] for row in record]
    return dmc.Stack([
        section("The record", table(["When", "Who", "Name", "What"], r_rows)
                if r_rows else empty("Nothing stamped yet."),
                description="People, agents, and the engine, in order. Computed from the stamps "
                            "already on the run."),
        section("Questions for the sponsor", table(["Id", "Kind", "Question"], q_rows)
                if q_rows else empty("No open design or data questions."),
                description="Answering a design question fills the design brief and re-runs the "
                            "stage that raised it."),
        section("Challenges", table(["Id", "From", "Sent to", "Status", "Challenge / response"], c_rows)
                if c_rows else empty("No challenges yet."),
                description="Send-backs from you and findings routed back by the independent "
                            "validator; each agent must answer every challenge."),
        section("Decision log", table(["Gate", "Action", "Seat", "Reason", "At"], d_rows)
                if d_rows else empty("No decisions yet.")),
    ], gap="md")


def audit_tab(run_id: str) -> Any:
    rows = []
    for a in service.audit(run_id)[::-1]:
        color = {"ok": "green", "invalid": "orange", "error": "red"}.get(a["status"], "gray")
        rows.append([
            dmc.Button(a["call_id"][-14:], id={"type": "audit-row", "call": a["call_id"]},
                       variant="subtle", size="compact-xs", ff="monospace", n_clicks=0,
                       leftSection=icon("tabler:file-search", 12)),
            a["agent"], a["attempt"], dmc.Badge(a["status"], color=color, size="xs", variant="light"),
            dmc.Code(backend_label(a.get("provider"), a.get("model"))),
            fmt(a.get("duration_s"), 2), f"${a['cost_usd']:.4f}" if a.get("cost_usd") else "—",
            dmc.Text((a.get("error") or "")[:140], size="xs", c="dimmed"),
        ])
    return section("Agent calls", table(["Call", "Agent", "Try", "Status", "Backend", "Seconds",
                                         "Cost", "Engine check"], rows, max_height=640)
                   if rows else empty("No agent calls yet."),
                   description="Every attempt is kept with its full input (prompt + context slice) "
                               "and raw output. Click a call to inspect it.")


def auto_step(st) -> str:
    for g in GATES:
        if st.status_of(g) == "awaiting":
            return STAGE_OF_GATE[g]
    for s in STAGES:
        if st.status_of(s) in ("running", "failed", "blocked"):
            return s
    done = [s for s in STAGES if st.status_of(s) == "done"]
    return done[-1] if done else STAGES[0]


# --- app ---------------------------------------------------------------------------------------
def create_app(runs_dir: str | None = None) -> Dash:
    if runs_dir:
        os.environ["COGNOS_RUNS_DIR"] = str(runs_dir)
    app = Dash(__name__, title="COGNOS", suppress_callback_exceptions=True,
               assets_folder=str(ASSETS), update_title=None, external_stylesheets=[FONT_CSS])
    app.layout = build_layout()
    register_callbacks(app)
    return app


def register_callbacks(app: Dash) -> None:
    @app.callback(Output("mantine", "forceColorScheme"), Output("theme-store", "data"),
                  Input("theme-switch", "checked"), prevent_initial_call=True)
    def set_theme(dark):
        scheme = "dark" if dark else "light"
        return scheme, {"scheme": scheme}

    @app.callback(Output("theme-switch", "checked"), Input("theme-store", "modified_timestamp"),
                  State("theme-store", "data"))
    def load_theme(_, data):
        return bool(data and data.get("scheme") == "dark")

    @app.callback(Output("page", "children"), Input("url", "pathname"))
    def route(path):
        path = path or "/"
        if path.startswith("/run/"):
            run_id = path.split("/")[2]
            try:
                service.state(run_id)
            except FileNotFoundError:
                return dmc.Container(dmc.Alert(f"No run {run_id}.", color="red"), py="xl")
            return workspace_page(run_id)
        if path.startswith("/compare/"):
            parts = path.strip("/").split("/")
            try:
                return dmc.Container(dmc.Stack(compare_page(service.compare(parts[1], parts[2])), gap="md"),
                                     size="xl", py="md")
            except (IndexError, FileNotFoundError):
                return dmc.Container(dmc.Alert("Pick two existing runs to compare (from the Runs page).",
                                               color="red"), py="xl")
        return runs_page()

    @app.callback(Output("cmp-go", "disabled"), Input("cmp-pick", "value"))
    def can_compare(picked):
        return len(picked or []) != 2

    @app.callback(Output("url", "pathname", allow_duplicate=True), Input("cmp-go", "n_clicks"),
                  State("cmp-pick", "value"), prevent_initial_call=True)
    def go_compare(clicks, picked):
        if not clicks or len(picked or []) != 2:
            return no_update
        a, b = sorted(picked)  # run ids start with their timestamp: the older run is A
        return f"/compare/{a}/{b}"

    @app.callback(Output("export-download", "data"), Output("notify", "sendNotifications", allow_duplicate=True),
                  Input("export-btn", "n_clicks"), State("ws-run", "data"), prevent_initial_call=True)
    def export(clicks, run_id):
        if not clicks:  # the button was just rendered, not clicked
            return no_update, no_update
        try:
            path = service.export_run(run_id)
        except OSError as exc:
            return no_update, [notice("Export failed", str(exc), "red")]
        return dcc.send_file(str(path)), [notice(
            "Exported", "Documents, results, decisions and the audit log. The data is never included.", "oxblood")]

    @app.callback(Output("runs-table", "children"), Input("poll", "n_intervals"),
                  State("url", "pathname"), prevent_initial_call=True)
    def refresh_runs(n, path):
        if (path or "/") != "/" or (n or 0) % 3:
            return no_update
        return runs_table()

    @app.callback(Output("new-drawer", "opened"), Input("new-open", "n_clicks"),
                  prevent_initial_call=True)
    def open_drawer(clicks):
        return bool(clicks) or no_update

    @app.callback(Output("new-prior-wrap", "style"), Output("new-intent-label", "children"),
                  Input("new-kind", "value"))
    def show_kind(kind):
        update = kind == "update"
        return ({"display": "block" if update else "none"},
                "Model update request" if update else "Business intent document")

    @app.callback(Output("template-download", "data"), Input("new-template", "n_clicks"),
                  State("new-kind", "value"), prevent_initial_call=True)
    def download_template(clicks, kind):
        if not clicks:
            return no_update
        name = "model_update_request.md" if kind == "update" else "business_intent.md"
        return {"content": service.intent_template(kind or "new"), "filename": name,
                "type": "text/markdown"}

    @app.callback(Output("new-data-wrap", "style"), Output("new-data-file", "style"),
                  Output("new-data-sf", "style"), Input("new-source", "value"))
    def show_data(source):
        show, hide = {"display": "block"}, {"display": "none"}
        own = str(source or "").startswith("data:")
        return (show if own else hide, show if source == "data:file" else hide,
                show if source == "data:snowflake" else hide)

    for _box in ("new-intent", "new-support", "new-prior", "new-data"):
        app.callback(Output(f"{_box}-names", "children"), Input(_box, "filename"))(upload_names)

    @app.callback(Output("url", "pathname"), Output("new-feedback", "children"),
                  Input("new-create", "n_clicks"),
                  State("new-source", "value"), State("new-mode", "value"),
                  State("new-provider", "value"), State("new-kind", "value"),
                  State("new-intent", "contents"), State("new-intent", "filename"),
                  State("new-support", "contents"), State("new-support", "filename"),
                  State("new-prior", "contents"), State("new-prior", "filename"),
                  State("new-prior-run", "value"), State("new-name", "value"),
                  State("new-data", "contents"), State("new-data", "filename"),
                  State("new-sf-table", "value"), State("new-sf-query", "value"),
                  State("new-sf-limit", "value"), prevent_initial_call=True)
    def create(clicks, source, mode, provider, kind, intent, intent_name, support,
               support_names, prior, prior_names, prior_run, name=None, data=None,
               data_name=None, sf_table=None, sf_query=None, sf_limit=None):
        if not clicks:  # the button was just rendered, not clicked
            return no_update, no_update
        try:
            if source.startswith("data:"):
                spec = data_source_spec(source[5:], saved_uploads(data, data_name), sf_table,
                                        sf_query, sf_limit)
                if isinstance(spec, str):
                    return no_update, dmc.Alert(spec, color="yellow", variant="light")
                cfg = service.config_from_data(
                    name or (data_name or sf_table or "model").rsplit(".", 1)[0], spec)
            elif source.startswith("demo:"):
                cfg = service.demo_config(source[5:])
            else:
                cfg = service.load_config(source[8:])
            problem = engagement_problem(kind, cfg, bool(intent), bool(prior or prior_run))
            if problem:
                return no_update, dmc.Alert(problem, color="yellow", variant="light")
            update = kind == "update"
            engagement = {
                "kind": kind or "new",
                "intent": next(iter(saved_uploads(intent, intent_name)), None),
                "supporting": saved_uploads(support, support_names),
                "prior_artifacts": saved_uploads(prior, prior_names) if update else [],
                "prior_run": prior_run if update else None,
            }
            run_id = service.create_run(cfg, mode=mode, provider=provider, engagement=engagement)
            service.start(run_id)
        except Exception as exc:  # show, don't crash the page
            return no_update, dmc.Alert(str(exc), color="red", variant="light")
        return f"/run/{run_id}", None

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
            content_key = (f"wf|{sel}|{st.status_of(sel)}|{st.steps[sel].runs}|"
                           f"{st.status_of(gate) if gate else ''}|{len(st.decisions)}|"
                           f"{scheme}|{seat}|{pkg}")
        elif tab == "questions":
            content_key = "q|" + "|".join(f"{g.id}:{g.status}" for g in st.gaps) + "|" + "|".join(
                f"{c.id}:{c.status}" for c in st.challenges) + f"|{len(st.decisions)}|{scheme}|{seat}|{pkg}"
        else:
            content_key = f"a|{len(service.audit(run_id))}|{scheme}"
        full_key = f"{st.version}|{content_key}"
        if key == full_key:
            return (no_update,) * 5
        content = no_update
        if not key or key.split("|", 1)[1] != content_key:
            if tab == "workflow":
                res = service.results(run_id).get(sel)
                content = dmc.Stack(stage_panel(sel, res, st, scheme,
                                                service.step_changes(run_id, sel), seat),
                                    gap="md")
            elif tab == "questions":
                content = questions_tab(st, seat)
            else:
                content = audit_tab(run_id)
        return run_header(st), rail(st, sel), activity(run_id), content, full_key

    @app.callback(Output("ws-step", "data"), Output("ws-tab", "value"),
                  Input({"type": "rail", "step": ALL}, "n_clicks"), prevent_initial_call=True)
    def pick_step(clicks):
        if not ctx.triggered_id or not any(clicks or []):
            return no_update, no_update
        return ctx.triggered_id["step"], "workflow"

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
        payload, reason = gate_payload(gate, action, fields, run_id)
        try:
            service.submit_gate(run_id, gate, action, payload, reason, seat=seat)
        except GateError as exc:
            return [notice("Decision refused", str(exc), "red")], no_update
        return [notice("Decision recorded", f"{LABELS[gate]}: {action.replace('_', ' ')}", "oxblood")], None

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
                service.reopen(run_id, trig["gate"], seat=seat)
                return [notice("Gate re-opened", "Revise the decision below.", "violet")], None
            service.retry(run_id, trig["step"])
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
        assume = trig["action"] == "assume"
        if not assume and not text.strip():
            return [notice("Answer needed", "Type an answer, or accept it as an assumption.",
                           "yellow")], no_update
        try:
            service.answer_gap(run_id, trig["gap"], text, assume=assume, seat=seat)
        except GateError as exc:
            return [notice("Not possible", str(exc), "red")], no_update
        return [notice("Recorded", "The stage that raised it will re-run." if not assume else
                       "Recorded as an assumption.", "oxblood")], None

    @app.callback(Output("io-modal", "opened"), Output("io-body", "children"),
                  Input({"type": "audit-row", "call": ALL}, "n_clicks"), State("ws-run", "data"),
                  prevent_initial_call=True)
    def show_io(clicks, run_id):
        trig = ctx.triggered_id
        if not trig or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        io = service.agent_io(run_id, trig["call"])
        inp, out = io.get("input") or {}, io.get("output") or {}
        return True, dmc.Tabs([
            dmc.TabsList([dmc.TabsTab("Output", value="out"), dmc.TabsTab("Context slice", value="ctx"),
                          dmc.TabsTab("Prompt", value="prompt"), dmc.TabsTab("System", value="sys")]),
            dmc.TabsPanel(dmc.Code(json.dumps(out, indent=1, default=str), block=True), value="out"),
            dmc.TabsPanel(dmc.Code(json.dumps(inp.get("context"), indent=1, default=str), block=True),
                          value="ctx"),
            dmc.TabsPanel(dmc.Code(inp.get("prompt", ""), block=True), value="prompt"),
            dmc.TabsPanel(dmc.Code(inp.get("system", ""), block=True), value="sys"),
        ], value="out")


def backend_label(provider: str | None, model: str | None) -> str:
    provider = provider or "?"
    return provider if not model or model in (provider, "heuristic", "replay") else f"{provider}:{model}"


def notice(title: str, message: str, color: str) -> dict:
    return {"action": "show", "title": title, "message": message, "color": color,
            "id": f"n{time.time_ns()}"}


def data_source_spec(kind: str, uploaded: list[str], table: str | None, query: str | None,
                     limit: Any) -> dict | str:
    """The ``data.source`` the form describes, or (a string) why it is not complete yet."""
    if kind == "file":
        if not uploaded:
            return "Upload the data file to model (.csv, .parquet or .xlsx)."
        return {"kind": "file", "path": uploaded[0]}
    table, query = (table or "").strip(), (query or "").strip()
    if not table and not query:
        return "Name the Snowflake table, or give a read-only query."
    spec: dict = {"kind": kind, "limit": int(limit) if limit else None}
    spec.update({"query": query} if query else {"table": table})
    return spec


def engagement_problem(kind: str | None, cfg, has_intent: bool, has_prior: bool) -> str | None:
    """Why a run cannot start yet, in the words of the form. A development starts from the
    sponsor's intent; an update also from the model it changes."""
    eng = cfg.engagement
    if kind == "update":
        if not (has_intent or (eng.kind == "update" and eng.intent)):
            return ("A model update starts from the update request. Download the template, "
                    "fill it in, and upload it.")
        if not (has_prior or eng.prior_artifacts or eng.prior_run):
            return ("A model update needs the existing model: upload its white paper, code or "
                    "reports, or pick an earlier COGNOS run of it.")
        return None
    if not (has_intent or eng.intent):
        return ("A new model development starts from the business intent document. Download "
                "the template, fill it in, and upload it.")
    return None


def gate_payload(gate: str, action: str, fields: dict, run_id: str) -> tuple[dict, str]:
    """Translate form fields into the engine's gate payload."""
    reason = (fields.get("reason") or "").strip()
    payload: dict = {}
    if action == "send_back":
        payload["message"] = (fields.get("message") or reason or "").strip()
        if fields.get("target"):
            payload["target"] = fields["target"]
        return payload, reason or payload["message"]
    if gate == "gate_data" and action == "edit":
        payload["exclude_columns"] = list(fields.get("exclude") or [])
        if fields.get("target"):
            payload["target"] = fields["target"]
    if gate in ("gate_intent", "gate_design"):
        answers = {k.split("::", 1)[1]: v for k, v in fields.items()
                   if k.startswith("answer::") and v and str(v).strip()}
        st = service.state(run_id)
        answers = {k: v for k, v in answers.items() if (g := st.gap(k)) and g.status == "open"}
        if answers:
            payload["answers"] = answers
    if gate == "gate_design":
        keep = fields.get("keep")
        hyps = (service.results(run_id).get("ideate").payload or {}).get("hypotheses", [])
        if action == "edit" and keep is not None and set(keep) != {h["id"] for h in hyps}:
            payload["slate"] = [{k: h[k] for k in ("family", "feature_strategy", "role", "priority",
                                                   "rationale")} for h in hyps if h["id"] in keep]
    if gate == "gate_champion" and action == "override":
        payload["champion"] = fields.get("champion")
    return payload, reason


def run_server(host: str = "127.0.0.1", port: int = 8050, runs_dir: str | None = None,
               debug: bool = False) -> None:
    app = create_app(runs_dir)
    print(f"COGNOS workbench on http://{host}:{port}  (runs: {service.runs_root()})")
    app.run(host=host, port=port, debug=debug)
