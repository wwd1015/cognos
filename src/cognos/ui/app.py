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
from ..engine.process import SEAT_LABEL, SEAT_OF_GATE, journal, next_action, seat_label
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
from .panels import (
    SPONSOR_NOTE,
    challenge_rows,
    compare_page,
    gap_cell,
    gap_need,
    gap_outcomes,
    gap_required,
    questions_summary,
    stage_panel,
)
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
            dcc.Link("All runs", href="/", className="masthead-link"),
            html.Span(["as ", html.B(service.whoami())], className="masthead-meta",
                      title="Recorded on your decisions and answers (COGNOS_USER, else your "
                            "login name)."),
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
            + ("" if snow.get("available") else " — not configured")},
           {"value": "data:linked", "label": "Several sources (files, Snowflake tables): the "
                                             "Data Analyst proposes the join"}]
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
        dmc.Select(id="new-source", label="Data & design", data=source_data, value="demo:cni",
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
            html.Div(dmc.Stack([
                dmc.Text("Data files", size="sm", fw=500),
                upload_box("new-data-multi", "Drop the files, or click to choose (.csv, "
                                             ".parquet, .xlsx)", multiple=True),
                dmc.Textarea(id="new-linked-sf", label="Snowflake tables or queries (optional)",
                             autosize=True, minRows=2,
                             placeholder="one per line:\nloans = RISK.CREDIT.LOANS\n"
                                         "ratings = SELECT obligor_id, rating FROM RISK.CREDIT.RATINGS"),
                dmc.Text("Give two or more sources in total. The engine measures how they "
                         "connect, the Data Analyst proposes the base table and the joins, and "
                         "you confirm or change the join at the data gate. If you already have "
                         "the join, use one query instead.", size="xs", c="dimmed"),
            ], gap=4), id="new-data-linked", style={"display": "none"}),
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
                               "and seven review gates.", className="page-lede")]),
            dmc.Button("New run", id="new-open", leftSection=icon("tabler:plus")),
        ], justify="space-between", align="flex-end", mt="lg", mb="xl", wrap="nowrap"),
        html.Div(id="runs-figures", children=runs_figures()),
        dmc.Card([
            dmc.Group([html.Div("All runs", className="ledger-title"),
                       html.Div(id="runs-tools", children=runs_tools())],
                      justify="space-between", align="center", mb="xs", wrap="nowrap"),
            html.Div("A run is one development of one model, and it stands at one place in the "
                     "workflow at a time. A model has one main run; trial runs that try a "
                     "decision differently are listed under it.",
                     className="ledger-desc"),
            html.Div(id="runs-table", children=runs_table()),
        ], mt="xl"),
        html.Div([html.Span("Agents recommend, people decide, the engine computes every number."),
                  html.Span("Internal use")], className="page-foot"),
        drawer,
        delete_modal(),
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


def runs_tools() -> Any:
    """The controls beside the run list: compare two runs, delete old ones."""
    tools = [dmc.Button("Delete runs", id="del-open", variant="subtle", color="red", size="xs",
                        leftSection=icon("tabler:trash", 14), n_clicks=0)]
    picker = compare_picker()
    if picker is not None:
        tools.insert(0, picker)
    return dmc.Group(tools, gap="xs", wrap="nowrap")


def delete_options() -> list[dict[str, str]]:
    return [{"value": r["run_id"], "label": f"{r['project']} · {r['run_id']} · {r['status']}"}
            for r in service.list_runs()]


def delete_modal() -> Any:
    """Pick runs, read what will be lost, tick the box, delete. Nothing is removed before the
    box is ticked and the button pressed."""
    return dmc.Modal(id="del-modal", title=dmc.Text("Delete runs", fw=650), size="lg",
                     children=dmc.Stack([
        dmc.MultiSelect(id="del-pick", label="Runs to delete", data=delete_options(),
                        searchable=True, clearable=True, placeholder="Pick one or more runs",
                        comboboxProps={"withinPortal": True}),
        html.Div(id="del-warning"),
        dmc.Checkbox(id="del-sure", checked=False,
                     label="I understand these runs are removed for good and cannot be restored."),
        dmc.Group([
            dmc.Button("Cancel", id="del-cancel", variant="default", n_clicks=0),
            dmc.Button("Delete for good", id="del-go", color="red", disabled=True, n_clicks=0,
                       leftSection=icon("tabler:trash", 16)),
        ], justify="flex-end"),
    ], gap="md"))


def deletion_warning(run_ids: list[str] | None) -> Any:
    """What the picked runs hold, so the person confirms knowing what goes."""
    if not run_ids:
        return dmc.Text("Deleting a run removes its results, documents, decisions and audit log "
                        "from this workspace. Export a run first to keep a copy.",
                        size="sm", c="dimmed")
    try:
        rows = service.deletion_preview(list(run_ids))
    except (KeyError, ValueError) as exc:
        return dmc.Alert(str(exc.args[0] if exc.args else exc), color="red", variant="light")
    size = sum(r["size_bytes"] for r in rows) / 1e6
    lines = [f"{len(rows)} run{'s' if len(rows) != 1 else ''}, {size:,.1f} MB on disk. Results, "
             "documents, decisions and the audit log go with them. Export first to keep a copy."]
    signed = [r for r in rows if r["packages"]]
    if signed:
        lines.append("Signed-off packages will be lost: "
                     + ", ".join(f"{r['run_id']} ({r['packages']})" for r in signed) + ".")
    elsewhere = [r for r in rows if r["status"] == "running" and not r["blocked"]]
    if elsewhere:
        lines.append("Marked as running, another process may still be working on: "
                     + ", ".join(r["run_id"] for r in elsewhere) + ".")
    kept = [r for r in rows if r.get("keeps_line")]
    if kept:
        lines.append("One line stays in the model's record for each run of a model with trial "
                     "runs (" + ", ".join(r["run_id"] for r in kept) + "): what it tried, how "
                     "far it went and how often it used the test sample.")
    blocked = [r for r in rows if r["blocked"]]
    if blocked:
        lines.append("Cannot be deleted now: "
                     + "; ".join(f"{r['run_id']} ({r['blocked']})" for r in blocked)
                     + ". Remove them from the list.")
    return dmc.Alert(dmc.Stack([dmc.Text(t, size="sm") for t in lines], gap=4),
                     title="This cannot be undone", color="red" if signed or blocked else "yellow",
                     variant="light", icon=icon("tabler:alert-triangle"))


def deletion_blocked(run_ids: list[str] | None) -> bool:
    """True when the picked runs cannot all be deleted now (nothing picked counts)."""
    if not run_ids:
        return True
    try:
        return any(r["blocked"] for r in service.deletion_preview(list(run_ids)))
    except (KeyError, ValueError):
        return True


def runs_table() -> Any:
    rows = service.list_runs()
    if not rows:
        return html.Div([html.Div("Nothing here yet.", className="ledger-title"),
                         html.Div("Start a run with “New run”: upload a business intent document, "
                                  "or try a synthetic demo.", className="ledger-desc")],
                        style={"padding": "28px 0"})
    # A model has one main run; a trial run tries one of its decisions differently and is
    # listed under it. Runs are named for that role, not numbered: a number says nothing
    # about which run counts. Two runs started separately from the same profile are two
    # developments, each with its own main run.
    projects: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        projects.setdefault(r["project"], {}).setdefault(r["model"], []).append(r)
    body = []
    for project, models in projects.items():
        for m, (model_id, runs) in enumerate(models.items()):
            # the main run first, then its trial runs in the order they were made
            runs = sorted(runs, key=lambda r: (r.get("role") == "trial", r["run_id"]))
            trials_n = sum(1 for r in runs if r.get("role") == "trial")
            for i, r in enumerate(runs):
                label = "Trial run" if r.get("role") == "trial" else "Main run"
                link = (dcc.Link(label, href=f"/run/{r['run_id']}",
                                 style={"fontWeight": 600, "color": "var(--ink)"})
                        if not r["legacy"] else dmc.Text(label, c="dimmed", size="sm"))
                if i == 0 and m == 0:
                    note = (f"{len(models)} separate developments" if len(models) > 1 else
                            f"{trials_n} trial run{'s' if trials_n != 1 else ''}" if trials_n
                            else "")
                    model = dmc.Stack([dmc.Text(project, size="sm", fw=650),
                                       dmc.Text(note, size="xs", c="dimmed") if note else None],
                                      gap=0)
                elif i == 0:
                    model = dmc.Text("same profile, started separately", size="xs", c="dimmed")
                else:
                    model = dmc.Text("same model", size="xs", c="dimmed")
                role = run_role(r)
                body.append([
                    model,
                    dmc.Stack([link,
                               dmc.Text(r["run_id"], size="xs", ff="monospace", c="dimmed"),
                               dmc.Text(role, size="xs", c="dimmed") if role else None], gap=0),
                    dmc.Group([run_badge(r["status"], "sm"),
                               dmc.Text(where_it_stands(r), size="sm")], gap="xs", wrap="nowrap"),
                    dmc.Text(r["mode"] + (" · update" if r.get("kind") == "update" else ""),
                             size="sm"),
                    dmc.Text(r["provider"], size="xs", ff="monospace"),
                    dmc.Text(r["champion"] or "—", size="xs", ff="monospace"),
                    dmc.Text(f"${r['spend_usd']:.2f}" if r["spend_usd"] else "—", size="xs",
                             ff="monospace"),
                    dmc.Text((r["updated_at"] or "")[:16].replace("T", " "), size="xs",
                             ff="monospace", c="dimmed"),
                ])
            if not any(r.get("role") for r in runs):
                continue
            # A deleted trial run keeps its line: its files are gone, the fact that it ran is not.
            for gone in service.model_record(model_id)["lines"]:
                if gone.get("deleted_at"):
                    body.append([
                        dmc.Text("same model", size="xs", c="dimmed"),
                        dmc.Stack([dmc.Text("Deleted trial run", size="sm", c="dimmed"),
                                   dmc.Text(gone["run_id"], size="xs", ff="monospace", c="dimmed"),
                                   dmc.Text(gone.get("note") or "", size="xs", c="dimmed")], gap=0),
                        dmc.Text(f"files deleted; it had {gone['stands']} and used the test "
                                 f"sample {gone.get('test_evaluations') or 0} time(s)",
                                 size="sm", c="dimmed"),
                        "", "", dmc.Text(gone.get("champion") or "—", size="xs", ff="monospace",
                                         c="dimmed"), "",
                        dmc.Text((gone["deleted_at"] or "")[:16].replace("T", " "), size="xs",
                                 ff="monospace", c="dimmed"),
                    ])
    return table(["Model", "Run", "Where it stands", "Mode", "Agents", "Champion", "Spend",
                  "Updated"], body)


def run_role(r: dict) -> str:
    """What a trial run is for, under its name (a main run needs no more than its name)."""
    if r.get("role") != "trial":
        return ""
    gate = r.get("trial_gate")
    if not gate:
        return "was the main run before another run replaced it"
    return (f"from “{LABELS.get(gate, gate)}”"
            + (f": {r['trial_note']}" if r.get("trial_note") else ""))


def where_it_stands(r: dict) -> str:
    """One run is at one place: the gate it is waiting at, or how it ended."""
    if r.get("trial_ended"):
        return "trial finished at the champion; it does not go to validation"
    if r.get("waiting_on"):
        return f"waiting at “{LABELS.get(r['waiting_on'], r['waiting_on'])}”"
    return {"completed": "finished, not signed off", "approved": "signed off",
            "rejected": "rejected", "blocked": "blocked", "failed": "a step failed",
            "running": "working", "created": "not started"}.get(r.get("status") or "", "")


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
        ], gap="sm", mt="xs", mb="sm"),
        # The run's views. "Stages" is the work, one stage at a time; the other two cover the
        # whole run, so they sit above the stage index, not inside a stage.
        dmc.Tabs([dmc.TabsList([
            dmc.TabsTab("Stages", value="workflow", leftSection=icon("tabler:route")),
            dmc.TabsTab("Run record", value="questions", leftSection=icon("tabler:notebook")),
            dmc.TabsTab("Agent calls", value="audit", leftSection=icon("tabler:list-search")),
        ])], id="ws-tab", value="workflow", mb="lg"),
        dmc.Grid([
            dmc.GridCol([html.Div(id="ws-head"), html.Div(id="ws-rail", className="cognos-rail")],
                        span={"base": 24, "md": 6}),
            dmc.GridCol(html.Div(id="ws-content", className="cognos-content"),
                        id="ws-main-col", span={"base": 24, "md": 13}),
            dmc.GridCol(html.Div(id="ws-activity"), id="ws-activity-col",
                        span={"base": 24, "md": 5}),
        ], gutter="lg", columns=24),
        dmc.Modal(id="io-modal", size="80%", title="Agent call", children=html.Div(id="io-body")),
        trial_modal(), promote_modal(),
        dcc.Download(id="export-download"),
    ])


# --- workspace fragments ---------------------------------------------------------------------
def trial_modal() -> Any:
    """Copy this run at a decision and take the decision differently. The run stays as it is."""
    return dmc.Modal(id="trial-modal", title=dmc.Text("Try a different decision", fw=650),
                     size="lg", children=dmc.Stack([
        dmc.Text("This makes a trial run: a copy of this run that waits at the decision you "
                 "pick, so you can decide it differently and compare. This run is not changed "
                 "and nothing is lost.", size="sm"),
        dmc.List([
            dmc.ListItem("A trial run stops after the champion is chosen. It does not go to "
                         "validation, so it cannot open the sealed holdout."),
            dmc.ListItem("Every use of the test sample is counted for the model, across all of "
                         "its runs. Validation reports the total."),
            dmc.ListItem("A trial run can be deleted. One line stays in the model's record."),
        ], size="sm", spacing=4),
        dmc.Select(id="trial-gate", label="Decision to take differently", data=[],
                   allowDeselect=False, comboboxProps={"withinPortal": True}),
        dmc.Textarea(id="trial-note", label="What do you want to try?", autosize=True, minRows=2,
                     placeholder="For example: best subset instead of forward selection",
                     description="Kept as this trial run's line in the model's record."),
        html.Div(id="trial-feedback"),
        dmc.Group([
            dmc.Button("Cancel", id="trial-cancel", variant="default", n_clicks=0),
            dmc.Button("Make the trial run", id="trial-go", n_clicks=0,
                       leftSection=icon("tabler:git-branch", 16)),
        ], justify="flex-end"),
    ], gap="md"))


def promote_modal() -> Any:
    return dmc.Modal(id="promote-modal", title=dmc.Text("Make this the main run", fw=650),
                     size="lg", children=dmc.Stack([
        dmc.Text("This trial run becomes the main run of the model and goes on to validation. "
                 "The run it replaces is kept on record as a trial run, with everything it "
                 "found; a package it had signed off is no longer in force.", size="sm"),
        dmc.Textarea(id="promote-note", label="Why this run?", autosize=True, minRows=2,
                     placeholder="For example: lower test error with one factor fewer"),
        html.Div(id="promote-feedback"),
        dmc.Group([
            dmc.Button("Cancel", id="promote-cancel", variant="default", n_clicks=0),
            dmc.Button("Make it the main run", id="promote-go", n_clicks=0,
                       leftSection=icon("tabler:arrow-up-circle", 16)),
        ], justify="flex-end"),
    ], gap="md"))


def trial_gate_options(run_id: str) -> list[dict[str, str]]:
    return [{"value": g, "label": LABELS[g]} for g in service.trial_gates(run_id)]


def role_fact(st) -> tuple[str, str] | None:
    """The run's place in its model, for the run header (None for a model with one run)."""
    lin = st.lineage
    if lin is None:
        return None
    if lin.trial and lin.parent:
        return ("Trial run", f"of {lin.parent}, from “{LABELS.get(lin.gate or '', lin.gate)}”"
                             + (f": {lin.note}" if lin.note else ""))
    if lin.trial:
        return ("Trial run", f"was the main run until {lin.replaced_by} replaced it")
    others = service.trial_record(st.run_id)["n_trials"]
    return ("Main run", f"{others} trial run{'s' if others != 1 else ''} on record")


def run_header(st) -> Any:
    res = service.results(st.run_id)
    model = res.get("model")
    facts = [("Mode", st.mode), ("Agents", st.provider)]
    role = role_fact(st)
    if role is not None:
        facts.insert(0, role)
    if st.created_by:
        facts.append(("Started by", st.created_by))
    backtest = res.get("backtest")
    if model is not None:
        facts.append(("Pool", f"{fmt(model.metrics.get('n_pool'))} candidate(s)"))
    if backtest is not None and (backtest.payload or {}).get("champion_id"):
        # The champion exists once outcomes analysis has compared the pool.
        facts += [("Champion", f"{backtest.payload['champion_id']} · "
                               f"{backtest.metrics.get('champion', '')}"),
                  (str(backtest.payload.get("evaluation_sample", "test")).replace("_", " "),
                   fmt(backtest.metrics.get("oos_metric")))]
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
    st = service.state(run_id)
    lin = st.lineage
    # A trial run is read against the run it was made from; any other run against the one before.
    against, label = service.previous_run(run_id), "Compare with previous run"
    if lin is not None and lin.parent:
        against, label = lin.parent, "Compare with the run it was made from"
        try:
            service.state(against)
        except (FileNotFoundError, ValueError):
            against = None  # its files were deleted; its line is in the run record
    return dmc.Group([
        dmc.Button("Try a different decision", id="trial-open", variant="default",
                   size="compact-sm", n_clicks=0, leftSection=icon("tabler:git-branch", 14))
        if service.trial_gates(run_id) else None,
        dmc.Button("Make this the main run", id="promote-open", size="compact-sm", n_clicks=0,
                   leftSection=icon("tabler:arrow-up-circle", 14)) if st.is_trial else None,
        dmc.Button("Export", id="export-btn", variant="default", size="compact-sm", n_clicks=0,
                   leftSection=icon("tabler:package-export", 14)),
        dcc.Link(dmc.Button(label, variant="subtle", size="compact-sm",
                            leftSection=icon("tabler:git-compare", 14)),
                 href=f"/compare/{against}/{run_id}") if against else None,
    ], gap="xs", mt="sm")


def rail(st, selected: str | None) -> Any:
    """The stage index. ``selected`` is None while a run-wide view is open: no stage is the
    current page then, and a click on one returns to it."""
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
        if status == "running" and st.steps[stage].by:  # whose machine is doing the work
            desc = f"Running — {st.steps[stage].by}"
        asked = [g for g in st.gaps if g.stage == stage and g.status == "open"]
        if asked:  # questions are answered on the stage's page: say they are there
            must = sum(1 for g in asked if gap_required(g))
            desc += (f" · {len(asked)} open question{'s' if len(asked) != 1 else ''}"
                     + (f", {must} required" if must else ""))
        items.append(dmc.NavLink(
            id={"type": "rail", "step": stage},
            label=dmc.Text([html.Span(f"{i:02d}", className="rail-no"), LABELS[stage]],
                           size="sm", fw=600 if stage == selected else 500),
            description=desc, leftSection=status_icon("awaiting" if gstatus == "awaiting" else status),
            rightSection=right, active=stage == selected, variant="subtle", color="oxblood",
            # (Mantine turns the right section a quarter when a link is clicked, as if it were
            # the chevron of a submenu; ours is a status label)
            disableRightSectionRotation=True, disabled=status == "skipped", n_clicks=0))
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
    """The run record: everything stamped on the run, across every stage."""
    q_rows = [[LABELS.get(g.stage, g.stage), gap_need(g), gap_cell(g, seat)]
              for g in sorted(st.gaps, key=lambda g: (g.status != "open", not gap_required(g)))]
    d_rows = [[LABELS.get(d.gate, d.gate), d.action.replace("_", " "), d.by or "—",
               d.seat or d.actor, d.reason or "—", stamp(d.at)] for d in st.decisions]
    record = journal(st, service.results(st.run_id))
    r_rows = [[stamp(row["at"]), row["who"], row["name"],
               dmc.Text(row["what"], size="sm")] for row in record]
    trials_section = model_runs_section(st)
    return dmc.Stack([
        view_lede("Run record", "The whole run, every stage: who did what and when, every "
                  "question asked of the sponsor, every challenge and every decision. A "
                  "stage's own questions and challenges are also on that stage's page."),
        section("Timeline", table(["When", "Who", "Name", "What"], r_rows, max_height=520)
                if r_rows else empty("Nothing stamped yet."),
                description="People, agents and the engine, in order. Computed from the stamps "
                            "already on the run."),
        trials_section,
        section(f"Questions for the business sponsor ({len(st.gaps)})",
                dmc.Text(questions_summary(list(st.gaps)), size="sm", mb="xs") if q_rows else None,
                gap_outcomes() if q_rows else None,
                table(["Raised at", "", "Question"], q_rows)
                if q_rows else empty("No questions were raised."),
                description=SPONSOR_NOTE + " An answer re-runs the stage that raised it."),
        section(f"Challenges ({len(st.challenges)})",
                table(["Id", "From", "Sent to", "Status", "Challenge / response"],
                      challenge_rows(st.challenges))
                if st.challenges else empty("No challenges yet."),
                description="Send-backs from a person at a gate and findings routed back by the "
                            "independent validator; each agent must answer every challenge."),
        section(f"Decision log ({len(d_rows)})",
                table(["Gate", "Action", "By", "Seat", "Reason", "At"], d_rows)
                if d_rows else empty("No decisions yet."),
                description="What a person (or express preparation) decided at each gate."),
    ], gap="md")


def model_runs_section(st) -> Any:
    """Every run of this model on one line each, deleted trial runs included, with how often
    the test sample was used across them. The same list validation and the white paper read."""
    from ..trials import CHANGE_LABELS

    rec = service.trial_record(st.run_id)
    if rec["n_runs"] < 2:
        return section("Runs of this model",
                       empty("This is the only run of this model. Use “Try a different "
                             "decision” to take a decision another way without losing this "
                             "result."),
                       description="A trial run is a copy of a run made at a decision. It stops "
                                   "before validation and it stays on record.")
    rows = []
    for r in rec["lines"]:
        gone = bool(r.get("deleted_at"))
        name = (dmc.Text(r["run_id"], size="xs", ff="monospace", c="dimmed") if gone or
                r["run_id"] == st.run_id else
                dcc.Link(dmc.Text(r["run_id"], size="xs", ff="monospace"),
                         href=f"/run/{r['run_id']}"))
        what = ("Main run" if r["role"] == "main" else "Trial run") + (
            " (this run)" if r["run_id"] == st.run_id else "") + (
            ", the main run before" if r.get("replaced_at") and r["role"] == "trial" else "") + (
            ", files deleted" if gone else "")
        tried = "; ".join(x for x in (
            f"from “{LABELS.get(r['gate'], r['gate'])}”" if r.get("gate") else "",
            r.get("note") or "",
            "changed " + ", ".join(CHANGE_LABELS.get(k, k) for k in r["changed"])
            if r.get("changed") else "") if x) or "—"
        found = "—"
        if r.get("champion"):
            found = r["champion"] + (f" · {r['metric']} {fmt(r['test_metric'])}"
                                     if r.get("test_metric") is not None else "")
        rows.append([dmc.Stack([dmc.Text(what, size="sm", fw=600), name], gap=0),
                     dmc.Text(tried, size="sm"), dmc.Text(r["stands"], size="sm"),
                     dmc.Text(found, size="sm"),
                     dmc.Text(fmt(r.get("test_evaluations")), size="sm", ff="monospace"),
                     dmc.Text(r.get("by") or "—", size="sm"), stamp(r.get("at"))])
    opened = len(rec["holdout_scored"])
    return section(
        f"Runs of this model ({rec['n_runs']})",
        kpis([("Runs", fmt(rec["n_runs"]), f"{rec['n_trials']} besides the main run"),
              ("Test sample used", fmt(rec["test_evaluations"]), "times, across all runs"),
              ("Sealed holdout", "not opened" if not opened else f"opened for {opened}",
               "only the main run can open it" if not opened else "champion(s) of this model")]),
        table(["Run", "What it tried", "Where it stands", "Champion (test sample)",
               "Test uses", "By", "Made"], rows),
        description="One line for each run of this model. A trial run copies a run at a "
                    "decision so the decision can be taken differently; it stops before "
                    "validation. Deleting a trial run removes its files and keeps its line.")


def view_lede(title: str, text: str) -> Any:
    return html.Div([html.Div(title, className="page-title", style={"fontSize": 24}),
                     html.Div(text, className="page-lede")])


def stamp(at: str | None) -> Any:
    """A recorded time, on one line."""
    return html.Span((at or "")[:19].replace("T", " "), className="stamp")


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
    return dmc.Stack([
        view_lede("Agent calls", "The audit trail of the agents, for the whole run: each call "
                  "an agent made, what it was shown and what it answered. Use it to check an "
                  "agent's work; the run record says what was decided."),
        section(f"Calls ({len(rows)})",
                table(["Call", "Agent", "Try", "Status", "Backend", "Sec.", "Cost", "Check"],
                      rows, max_height=640)
                if rows else empty("No agent calls yet."),
                description="Every attempt is kept with its full input (prompt + context "
                            "slice) and raw output. Click a call to inspect it."),
    ], gap="md")


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

    @app.callback(Output("del-modal", "opened"), Output("del-pick", "data"),
                  Output("del-pick", "value"), Output("del-sure", "checked"),
                  Input("del-open", "n_clicks"), Input("del-cancel", "n_clicks"),
                  prevent_initial_call=True)
    def open_delete(opened, cancelled):
        if ctx.triggered_id == "del-open" and opened:
            return True, delete_options(), [], False
        if ctx.triggered_id == "del-cancel" and cancelled:
            return False, no_update, [], False
        return no_update, no_update, no_update, no_update

    @app.callback(Output("del-warning", "children"), Output("del-go", "disabled"),
                  Input("del-pick", "value"), Input("del-sure", "checked"))
    def warn_delete(picked, sure):
        return deletion_warning(picked), not sure or deletion_blocked(picked)

    @app.callback(Output("del-modal", "opened", allow_duplicate=True),
                  Output("runs-table", "children", allow_duplicate=True),
                  Output("runs-figures", "children"), Output("runs-tools", "children"),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Input("del-go", "n_clicks"), State("del-pick", "value"),
                  State("del-sure", "checked"), prevent_initial_call=True)
    def delete(clicks, picked, sure):
        if not clicks or not picked or not sure:  # never without the ticked box
            return (no_update,) * 5
        try:
            gone = service.delete_runs(list(picked))
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            return (no_update,) * 4 + ([notice("Nothing deleted", str(exc), "red")],)
        return (False, runs_table(), runs_figures(), runs_tools(),
                [notice("Deleted", f"{len(gone)} run{'s' if len(gone) != 1 else ''} removed.",
                        "oxblood")])

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
                  Output("new-data-sf", "style"), Output("new-data-linked", "style"),
                  Input("new-source", "value"))
    def show_data(source):
        show, hide = {"display": "block"}, {"display": "none"}
        own = str(source or "").startswith("data:")
        return (show if own else hide, show if source == "data:file" else hide,
                show if source == "data:snowflake" else hide,
                show if source == "data:linked" else hide)

    for _box in ("new-intent", "new-support", "new-prior", "new-data", "new-data-multi"):
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
                  State("new-sf-limit", "value"),
                  State("new-data-multi", "contents"), State("new-data-multi", "filename"),
                  State("new-linked-sf", "value"), prevent_initial_call=True)
    def create(clicks, source, mode, provider, kind, intent, intent_name, support,
               support_names, prior, prior_names, prior_run, name=None, data=None,
               data_name=None, sf_table=None, sf_query=None, sf_limit=None, multi=None,
               multi_names=None, linked_sf=None):
        if not clicks:  # the button was just rendered, not clicked
            return no_update, no_update
        try:
            if source == "data:linked":
                spec = linked_sources_spec(saved_uploads(multi, multi_names), linked_sf)
                if isinstance(spec, str):
                    return no_update, dmc.Alert(spec, color="yellow", variant="light")
                cfg = service.config_from_data(name or "linked_model", spec)
            elif source.startswith("data:"):
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
            asked = ",".join(f"{g.id}:{g.status}" for g in st.gaps if g.stage == sel)
            sent = ",".join(f"{c.id}:{c.status}" for c in st.challenges if c.target_stage == sel)
            content_key = (f"wf|{sel}|{st.status_of(sel)}|{st.steps[sel].runs}|"
                           f"{st.status_of(gate) if gate else ''}|{len(st.decisions)}|"
                           f"{scheme}|{seat}|{pkg}|{asked}|{sent}")
        elif tab == "questions":
            content_key = "q|" + "|".join(f"{g.id}:{g.status}" for g in st.gaps) + "|" + "|".join(
                f"{c.id}:{c.status}" for c in st.challenges) + f"|{len(st.decisions)}|{scheme}|{seat}|{pkg}|{service.trial_stamp(run_id)}"
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
        return (run_header(st), rail(st, sel if tab == "workflow" else None), activity(run_id),
                content, full_key)

    @app.callback(Output("ws-main-col", "span"), Output("ws-activity-col", "style"),
                  Input("ws-tab", "value"))
    def widen(tab):
        """A run-wide view takes the room of the live activity feed."""
        if tab == "workflow":
            return {"base": 24, "md": 13}, {"display": "block"}
        return {"base": 24, "md": 18}, {"display": "none"}

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
        assume, limitation = trig["action"] == "assume", trig["action"] == "limitation"
        if not (assume or limitation) and not text.strip():
            return [notice("Answer needed", "Type an answer, or record it as an assumption or "
                           "a limitation.", "yellow")], no_update
        try:
            service.answer_gap(run_id, trig["gap"], text, assume=assume, limitation=limitation,
                               seat=seat)
        except GateError as exc:
            return [notice("Not possible", str(exc), "red")], no_update
        said = ("Recorded as a limitation. Validation will ask how it is mitigated."
                if limitation else
                "Recorded as an assumption. Validation will ask how sensitive the result is "
                "to it." if assume else "The stage that raised it will re-run.")
        return [notice("Recorded", said, "oxblood")], None

    @app.callback(Output("trial-modal", "opened"), Output("trial-gate", "data"),
                  Output("trial-gate", "value"), Output("trial-note", "value"),
                  Output("trial-feedback", "children"),
                  Input("trial-open", "n_clicks"), Input("trial-cancel", "n_clicks"),
                  State("ws-run", "data"), State("ws-step", "data"), prevent_initial_call=True)
    def open_trial(opened, cancelled, run_id, step):
        if ctx.triggered_id == "trial-open" and opened:
            options = trial_gate_options(run_id)
            values = [o["value"] for o in options]
            # Start on the decision of the page being read, else the latest one reached.
            here = GATE_OF_STAGE.get(step or "")
            return (True, options, here if here in values else (values[-1] if values else None),
                    "", None)
        if ctx.triggered_id == "trial-cancel" and cancelled:
            return False, no_update, no_update, no_update, None
        return (no_update,) * 5

    @app.callback(Output("url", "pathname", allow_duplicate=True),
                  Output("trial-feedback", "children", allow_duplicate=True),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Input("trial-go", "n_clicks"), State("trial-gate", "value"),
                  State("trial-note", "value"), State("ws-run", "data"), State("ws-seat", "data"),
                  prevent_initial_call=True)
    def make_trial(clicks, gate, note, run_id, seat):
        if not clicks:
            return no_update, no_update, no_update
        try:
            new_id = service.trial_run(run_id, gate or "", note or "", seat=seat)
        except (GateError, OSError) as exc:
            return no_update, dmc.Alert(str(exc), color="red", variant="light"), no_update
        return f"/run/{new_id}", None, [notice(
            "Trial run made", f"Decide “{LABELS[gate]}” differently here. The run you came "
            "from is unchanged.", "oxblood")]

    @app.callback(Output("promote-modal", "opened"), Output("promote-note", "value"),
                  Output("promote-feedback", "children"),
                  Input("promote-open", "n_clicks"), Input("promote-cancel", "n_clicks"),
                  prevent_initial_call=True)
    def open_promote(opened, cancelled):
        if ctx.triggered_id == "promote-open" and opened:
            return True, "", None
        if ctx.triggered_id == "promote-cancel" and cancelled:
            return False, no_update, None
        return no_update, no_update, no_update

    @app.callback(Output("promote-modal", "opened", allow_duplicate=True),
                  Output("promote-feedback", "children", allow_duplicate=True),
                  Output("notify", "sendNotifications", allow_duplicate=True),
                  Output("ws-key", "data", allow_duplicate=True),
                  Input("promote-go", "n_clicks"), State("promote-note", "value"),
                  State("ws-run", "data"), State("ws-seat", "data"), prevent_initial_call=True)
    def promote(clicks, note, run_id, seat):
        if not clicks:
            return (no_update,) * 4
        try:
            service.promote_trial(run_id, note or "", seat=seat)
        except (GateError, OSError) as exc:
            return no_update, dmc.Alert(str(exc), color="red", variant="light"), no_update, no_update
        return False, None, [notice("This is now the main run",
                                    "It goes on to independent validation.", "oxblood")], None

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


def linked_sources_spec(uploaded: list[str], snowflake_lines: str | None) -> dict | str:
    """``{"sources": [...]}`` for the several-sources form, or (a string) why it is not
    complete yet."""
    from .. import linking

    try:
        return linking.named_sources(uploaded, snowflake_lines or "")
    except linking.LinkError as exc:
        return str(exc)


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
        if fields.get("join_base"):  # several sources: the join as the form now states it
            payload["join"] = {"base": fields["join_base"],
                               "links": list(fields.get("join_links") or []),
                               "many": fields.get("join_many") or "aggregate",
                               "unmatched": fields.get("join_unmatched") or "plan"}
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
        plan = (service.results(run_id).get("ideate").payload or {}).get("plan")
        if action == "edit" and plan:
            patch = plan_patch(plan, fields)
            if patch:
                payload["plan"] = patch
    if gate == "gate_candidates" and action == "edit":
        extra = fields.get("extra") or []
        if isinstance(extra, str):  # ids typed by hand, comma separated
            extra = [x.strip() for x in extra.replace(";", ",").split(",")]
        payload["pool"] = list(dict.fromkeys([*(fields.get("pool") or []), *(x for x in extra if x)]))
        model = (service.results(run_id).get("model").payload or {})
        shown = {a["id"]: a.get("name") for a in model.get("leaders") or []}
        names = {k.split("::", 1)[1]: " ".join(str(v).split()) for k, v in fields.items()
                 if k.startswith("name::") and v and str(v).strip()}
        names = {k: v for k, v in names.items() if shown.get(k) != v}
        if names:
            payload["names"] = names
    if gate == "gate_champion" and action == "override":
        payload["champion"] = fields.get("champion")
    return payload, reason


def plan_patch(plan: dict, fields: dict) -> dict:
    """What the plan form changed: only the fields that differ from the plan on the page."""
    def blank(v: Any) -> bool:
        return v is None or v == ""

    patch: dict = {}
    for key, value in fields.items():
        if not key.startswith("plan::") or key.startswith("plan::rank."):
            continue
        section_name, _, name = key[len("plan::"):].partition(".")
        if section_name not in plan or name not in plan[section_name]:
            continue
        current = plan[section_name][name]
        value = None if blank(value) else value
        if isinstance(current, bool):
            value = value == "yes" if value in ("yes", "no") else bool(value)
        if isinstance(current, list) and value is None:
            continue  # an untouched multi-select
        if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, float):
            value = int(value)
        if value != current and not (blank(value) and blank(current)):
            patch.setdefault(section_name, {})[name] = value
    rank = []
    for i in (0, 1):
        metric = fields.get(f"plan::rank.{i}.metric")
        if not blank(metric):
            rank.append({"metric": metric,
                         "sample": fields.get(f"plan::rank.{i}.sample") or "validation"})
    if rank and rank != list(plan["evaluation"].get("rank_by") or []):
        patch.setdefault("evaluation", {})["rank_by"] = rank
    return patch


def run_server(host: str = "127.0.0.1", port: int = 8050, runs_dir: str | None = None,
               debug: bool = False) -> None:
    app = create_app(runs_dir)
    print(f"COGNOS workbench on http://{host}:{port}  (runs: {service.runs_root()})")
    app.run(host=host, port=port, debug=debug)
