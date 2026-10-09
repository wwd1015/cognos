"""The COGNOS workbench — a Dash + Mantine app for model developers.

Two pages: **Runs** (every run, and a drawer to start a new one from a project profile or a
synthetic demo) and the **run workspace** (stage rail, the selected stage's evidence beside its
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
    run_badge,
    section,
    status_icon,
    table,
    verdict_badge,
)
from .panels import compare_page, stage_panel
from .theme import MANTINE_THEME

ASSETS = Path(__file__).with_name("assets")


# --- layout ----------------------------------------------------------------------------------
def header() -> dmc.AppShellHeader:
    return dmc.AppShellHeader(dmc.Group([
        dmc.Group([
            dcc.Link(dmc.Group([
                dmc.ThemeIcon(icon("tabler:topology-star-3", 18), radius="md", size="lg",
                              variant="gradient", gradient={"from": "indigo", "to": "cyan"}),
                dmc.Stack([dmc.Text("COGNOS", fw=750, size="lg", lh=1),
                           dmc.Text("agents recommend · you decide · the engine disposes",
                                    size="xs", c="dimmed", lh=1.2)], gap=0),
            ], gap="xs"), href="/", style={"textDecoration": "none", "color": "inherit"}),
        ]),
        dmc.Group([
            dmc.Badge(f"v{__version__}", variant="outline", color="gray"),
            dcc.Link(dmc.Button("Runs", variant="subtle", leftSection=icon("tabler:list")), href="/"),
            dmc.Switch(id="theme-switch", onLabel=icon("tabler:moon", 14),
                       offLabel=icon("tabler:sun", 14), size="md", color="gray"),
        ], gap="sm"),
    ], justify="space-between", h="100%", px="md"))


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
            ], header={"height": 60}, padding="md"),
        ])


def runs_page() -> Any:
    profiles = service.list_profiles()
    providers = service.provider_list()
    prov_data = [{"value": p["id"], "label": f"{p['label']}" + ("" if p["available"] else " — unavailable"),
                  "disabled": not p["available"]} for p in providers if p["id"] != "replay"]
    default_prov = next((p["id"] for p in providers if p["available"] and p["id"] == "claude_cli"),
                        "heuristic")
    source_data = ([{"group": "Synthetic demos", "items": [
        {"value": f"demo:{k}", "label": v} for k, v in service.DEMO_LABELS.items()]}]
        + ([{"group": "Project profiles", "items": [
            {"value": f"profile:{p['path']}", "label": f"{p['name']} ({p['path']})"}
            for p in profiles]}] if profiles else []))
    drawer = dmc.Drawer(id="new-drawer", title=dmc.Text("Start a run", fw=650), position="right",
                        size="md", padding="lg", children=dmc.Stack([
        dmc.Select(id="new-source", label="Data & design", data=source_data, value="demo:commercial",
                   allowDeselect=False, searchable=True),
        dmc.SegmentedControl(id="new-mode", value="interactive", fullWidth=True, data=[
            {"value": "interactive", "label": "Interactive — I review each gate"},
            {"value": "autonomous", "label": "Autonomous — prototype"}]),
        dmc.Select(id="new-provider", label="Who makes the recommendations", data=prov_data,
                   value=default_prov, allowDeselect=False,
                   description="LLM agents recommend; the deterministic agents run offline."),
        dmc.Alert("The engine computes every number and guards the sealed holdout; agents only "
                  "recommend, and you decide at the gates.", color="indigo", variant="light",
                  icon=icon("tabler:shield-lock")),
        dmc.Button("Start run", id="new-create", leftSection=icon("tabler:player-play"), fullWidth=True),
        html.Div(id="new-feedback"),
    ], gap="md"))
    return dmc.Container([
        dmc.Group([
            dmc.Stack([dmc.Title("Model development runs", order=2),
                       dmc.Text("Each run takes a dataset and a design brief through eight stages "
                                "and five review gates.", c="dimmed", size="sm")], gap=2),
            dmc.Button("New run", id="new-open", leftSection=icon("tabler:plus"), size="md"),
        ], justify="space-between", mb="lg"),
        compare_picker(),
        dmc.Card(html.Div(id="runs-table", children=runs_table()), p=0),
        drawer,
    ], size="xl", py="md")


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
                        flex=1, maw=720, comboboxProps={"withinPortal": True}),
        dmc.Button("Compare", id="cmp-go", variant="light", disabled=True, n_clicks=0),
    ], mb="md", align="flex-end")


def runs_table() -> Any:
    rows = service.list_runs()
    if not rows:
        return dmc.Center(dmc.Stack([icon("tabler:folder-open", 36, color="gray"),
                                     dmc.Text("No runs yet — start one with “New run”.", c="dimmed")],
                                    align="center"), h=180)
    body = []
    for r in rows:
        link = (dcc.Link(r["run_id"], href=f"/run/{r['run_id']}") if not r["legacy"]
                else dmc.Text(r["run_id"], c="dimmed", size="sm"))
        body.append([
            link, dmc.Text(r["project"], fw=500, size="sm"), run_badge(r["status"], "sm"),
            dmc.Text(LABELS.get(r["waiting_on"], "") if r["waiting_on"] else "", size="sm"),
            dmc.Text(r["mode"], size="sm"), dmc.Code(r["provider"]),
            dmc.Text(r["champion"] or "", size="sm", ff="monospace"),
            dmc.Text(f"${r['spend_usd']:.2f}" if r["spend_usd"] else "—", size="sm"),
            dmc.Text((r["updated_at"] or "")[:19].replace("T", " "), size="xs", c="dimmed"),
        ])
    return table(["Run", "Project", "Status", "Waiting on", "Mode", "Agents", "Champion", "Spend",
                  "Updated"], body)


def workspace_page(run_id: str) -> Any:
    return html.Div([
        dcc.Store(id="ws-run", data=run_id),
        dcc.Store(id="ws-step", data=None),
        dcc.Store(id="ws-seat", data="developer"),
        dcc.Store(id="ws-key", data=None),
        dmc.Group([
            dmc.Text("Acting as", size="sm", c="dimmed"),
            dmc.SegmentedControl(
                id="seat-switch", value="developer",
                data=[{"value": k, "label": v.title()} for k, v in SEAT_LABEL.items()
                      if k != "express"],
            ),
        ], gap="sm", mb="sm"),
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
    return dmc.Card([
        dmc.Group([dmc.Text(st.project, fw=700, size="lg"), run_badge(st.status)],
                  justify="space-between"),
        dmc.Text(st.run_id, size="xs", c="dimmed", ff="monospace"),
        dmc.Group([dmc.Badge(st.mode, variant="light", color="gray", size="sm"),
                   dmc.Badge(f"agents: {st.provider}", variant="light", color="indigo", size="sm"),
                   dmc.Badge(f"${st.spend_usd:.2f}", variant="light", color="gray", size="sm")
                   if st.spend_usd else None], gap=6, mt="xs"),
        dmc.Text(f"Champion {model.metrics.get('champion', '')} · CV "
                 f"{fmt(model.metrics.get('cv_mean'))}", size="sm", mt="xs")
        if model is not None else None,
        dmc.Text(next_action(st)["text"], size="sm", mt="xs"),
        dmc.Alert(st.halted_reason, color="red", variant="light", mt="xs", p="xs")
        if st.halted_reason and st.status in ("blocked", "rejected", "failed") else None,
        run_actions(st.run_id),
    ], mb="md", p="md")


def run_actions(run_id: str) -> Any:
    prev = service.previous_run(run_id)
    return dmc.Group([
        dmc.Button("Export", id="export-btn", variant="light", size="compact-sm", n_clicks=0,
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
            right = dmc.Badge(who, color="violet", size="sm", variant="filled")
        elif st.steps[stage].verdict and status in ("done", "blocked"):
            right = verdict_badge(st.steps[stage].verdict, "xs")
        desc = LABELS[gate] + " — " + gstatus.replace("_", " ") if gstatus and gstatus not in (
            "pending", "skipped") else step_badge_text(status)
        if status == "stale" and st.steps[stage].rerun_reason:  # say what made it out of date
            why = st.steps[stage].rerun_reason
            desc = "Out of date — " + (why if len(why) <= 70 else why[:69].rstrip() + "…")
        items.append(dmc.NavLink(
            id={"type": "rail", "step": stage},
            label=dmc.Text(f"{i}. {LABELS[stage]}", size="sm", fw=600 if stage == selected else 500),
            description=desc, leftSection=status_icon("awaiting" if gstatus == "awaiting" else status),
            rightSection=right, active=stage == selected, variant="light", color="indigo",
            disabled=status == "skipped", n_clicks=0))
    return dmc.Card([dmc.Text("Stages", size="xs", c="dimmed", fw=600, tt="uppercase", mb=4),
                     *items], p="xs")


def step_badge_text(status: str) -> str:
    from .theme import STEP

    return STEP.get(status, ("", "", status))[2]


def activity(run_id: str) -> Any:
    evs = service.run_events(run_id, limit=60)[::-1][:40]
    colors = {"step_done": "green", "step_failed": "red", "step_blocked": "red",
              "gate_waiting": "violet", "gate_decision": "indigo", "agent_invalid": "orange",
              "agent_error": "red", "challenge": "violet", "agent_start": "blue"}
    items = [dmc.TimelineItem(
        title=dmc.Text(time.strftime("%H:%M:%S", time.localtime(e["ts"])), size="xs", c="dimmed"),
        children=dmc.Text(e["message"], size="xs"),
        bullet=dmc.ThemeIcon(icon("tabler:point-filled", 10), size=16, radius="xl",
                             color=colors.get(e["type"], "gray"), variant="light")) for e in evs]
    return dmc.Card([dmc.Text("Activity", size="xs", c="dimmed", fw=600, tt="uppercase", mb="xs"),
                     dmc.ScrollArea(dmc.Timeline(items, bulletSize=16, lineWidth=2) if items
                                    else empty("Nothing yet."), h=620, type="auto")], p="sm")


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
                         color="violet" if c.status == "open" else "gray"),
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
               assets_folder=str(ASSETS), update_title=None)
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
            "Exported", "Documents, results, decisions and the audit log. The data is never included.", "indigo")]

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

    @app.callback(Output("url", "pathname"), Output("new-feedback", "children"),
                  Input("new-create", "n_clicks"),
                  State("new-source", "value"), State("new-mode", "value"),
                  State("new-provider", "value"), prevent_initial_call=True)
    def create(clicks, source, mode, provider):
        if not clicks:  # the button was just rendered, not clicked
            return no_update, no_update
        try:
            if source.startswith("demo:"):
                cfg = service.demo_config(source[5:])
            else:
                cfg = service.load_config(source[8:])
            run_id = service.create_run(cfg, mode=mode, provider=provider)
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
        return [notice("Decision recorded", f"{LABELS[gate]}: {action.replace('_', ' ')}", "indigo")], None

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
                       "Recorded as an assumption.", "indigo")], None

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
    if gate == "gate_design":
        answers = {k.split("::", 1)[1]: v for k, v in fields.items()
                   if k.startswith("answer::") and v and str(v).strip()}
        st = service.state(run_id)
        answers = {k: v for k, v in answers.items() if (g := st.gap(k)) and g.status == "open"}
        if answers:
            payload["answers"] = answers
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
