"""The Runs page: the run list, its figures and tools, and the drawer that starts a run."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ... import service
from ...engine.graph import (
    LABELS,
)
from ..components import (
    fmt,
    icon,
    kpis,
    run_badge,
    table,
)
from ..panels import (
    guide_text,
)


def start_choices(templates: list[dict]) -> list[dict]:
    """What a run can start from: a built-in template, one the team saved, or nothing."""
    groups = [("built-in", "Built-in templates"), ("team", "Team templates")]
    out = [{"group": label, "items": [{"value": t["ref"], "label": t["title"]}
                                      for t in templates if t["origin"] == origin]}
           for origin, label in groups]
    return [g for g in out if g["items"]] + [{"group": "No template", "items": [
        {"value": "scratch", "label": "From scratch — configure the whole process yourself"}]}]


def start_note(ref: str | None, templates: list[dict]) -> Any:
    """What the chosen template is for, what it sets up, and the data it is usually built on."""
    t = next((t for t in templates if t["ref"] == ref), None)
    if t is None:
        return dmc.Text("Every choice is made in the run: the Design agent proposes the plan from "
                        "the intent and the data, and you confirm it at the gates.",
                        size="xs", c="dimmed")
    saved = (f"Saved by {t['saved_by']} on {t['saved_at'][:10]} from run {t['from_run']}. "
             if t["origin"] == "team" else "")
    rows = [dmc.Text([dmc.Text(label, fw=650, span=True), "  ", text], size="xs")
            for label, text in t["setup"]]
    body = [dmc.Text(saved + t["description"], size="sm"), *rows]
    if t["guide"]:
        body.append(dmc.Accordion([dmc.AccordionItem([
            dmc.AccordionControl("The data this kind of model is built on",
                                 icon=icon("tabler:database")),
            dmc.AccordionPanel(guide_text(t["guide"])),
        ], value="data")], variant="default"))
    return dmc.Stack(body, gap=4, mt=4)


def runs_page() -> Any:
    templates = service.list_templates()
    providers = service.provider_list()
    prov_data = [{"value": p["id"], "label": f"{p['label']}" + ("" if p["available"] else " — unavailable"),
                  "disabled": not p["available"]} for p in providers if p["id"] != "replay"]
    default_prov = next((p["id"] for p in providers if p["available"] and p["id"] == "claude_cli"),
                        "heuristic")
    first = templates[0]["ref"] if templates else "scratch"
    prior_runs = [{"value": r["run_id"], "label": f"{r['project']} · {r['run_id']}"}
                  for r in service.list_runs() if not r["legacy"] and r["champion"]]
    drawer = dmc.Drawer(id="new-drawer", title=dmc.Text("Start a run", fw=650), position="right",
                        size="lg", padding="lg", children=dmc.Stack([
        dmc.SegmentedControl(id="new-kind", value="new", fullWidth=True, data=[
            {"value": "new", "label": "New model development"},
            {"value": "update", "label": "Model update"}]),
        dmc.TextInput(id="new-name", label="Model name", placeholder="e.g. cni_pd_2026"),
        dmc.Stack([
            dmc.Select(id="new-start", label="Start from", data=start_choices(templates),
                       value=first, allowDeselect=False, searchable=True,
                       description="A template is a standard setup for one kind of model. "
                                   "Every choice in it can be changed at the gates."),
            html.Div(start_note(first, templates), id="new-start-note"),
        ], gap=4),
        dmc.Stack([
            dmc.Group([
                dmc.Text("Business intent document", size="sm", fw=500, id="new-intent-label"),
                dmc.Button("Download the template", id="new-template", variant="subtle",
                           size="compact-xs", leftSection=icon("tabler:download", 14), n_clicks=0),
            ], justify="space-between"),
            upload_box("new-intent", "Drop the filled-in template, or click to choose "
                                     "(.md, .txt, .docx)", multiple=False),
            dmc.Text("What the model is for, in the sponsor's words. The Intake agent reads it "
                     "and asks about anything left unclear. You give the data once the intent "
                     "is confirmed.", size="xs", c="dimmed", id="new-intent-help"),
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
        model_select(providers, default_prov),
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
                               "and eight review gates.", className="page-lede")]),
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
        html.Div([html.Span("The engine computes every number, agents recommend, people decide."),
                  html.Span("Internal use")], className="page-foot"),
        drawer,
        delete_modal(),
        dcc.Download(id="template-download"),
    ], size="xl", py="md")


def model_choices(providers: list[dict], provider: str | None) -> tuple[list[dict], str | None]:
    """The models a run may give its agents under ``provider``, default first; none for the
    deterministic agents."""
    p = next((p for p in providers if p["id"] == provider), None)
    if p is None or p["kind"] in ("heuristic", "replay") or not p.get("models"):
        return [], None
    data = [{"value": m, "label": m + (" (default)" if i == 0 else "")}
            for i, m in enumerate(p["models"])]
    return data, p["models"][0]


def model_note(providers: list[dict], provider: str | None) -> str:
    p = next((p for p in providers if p["id"] == provider), None)
    if p is not None and p["kind"] == "claude_cli":
        return ("Spend is the Claude CLI's own figure for the model that ran, at list prices "
                "(a subscription login is not billed per call).")
    return "Spend is estimated from the model's tokens and list prices."


def model_select(providers: list[dict], provider: str | None) -> Any:
    data, value = model_choices(providers, provider)
    return html.Div(dmc.Select(id="new-model", label="Model", data=data, value=value,
                               allowDeselect=False, searchable=True,
                               description=model_note(providers, provider)),
                    id="new-model-wrap", style={} if data else {"display": "none"})


def runs_figures() -> Any:
    """The figures above the run list: how much work is open and whose move it is."""
    rows = [r for r in service.list_runs() if not r["legacy"]]
    if not rows:
        return None
    awaiting = [r for r in rows if r["status"] == "awaiting" and r["waiting_on"]]
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
    # about which run counts.
    projects: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        projects.setdefault(r["project"], {}).setdefault(r["model"], []).append(r)
    body = []
    for project, models in projects.items():
        for model_id, runs in models.items():
            # the main run first, then its trial runs in the order they were made
            runs = sorted(runs, key=lambda r: (r.get("role") == "trial", r["run_id"]))
            trials_n = sum(1 for r in runs if r.get("role") == "trial")
            for i, r in enumerate(runs):
                label = "Trial run" if r.get("role") == "trial" else "Main run"
                link = (dcc.Link(label, href=f"/run/{r['run_id']}",
                                 style={"fontWeight": 600, "color": "var(--ink)"})
                        if not r["legacy"] else dmc.Text(label, c="dimmed", size="sm"))
                if i == 0:
                    # One model, one main run. Two runs started separately from the same
                    # profile are two models: each gets its own heading, told apart by when
                    # it was started.
                    notes = [f"started {started(model_id)}"] if len(models) > 1 else []
                    if trials_n:
                        notes.append(f"{trials_n} trial run{'s' if trials_n != 1 else ''}")
                    model = dmc.Stack([dmc.Text(project, size="sm", fw=650),
                                       dmc.Text(" · ".join(notes), size="xs", c="dimmed")
                                       if notes else None], gap=0)
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


def started(run_id: str) -> str:
    """When a run was started, read from its id (ids begin with a UTC timestamp)."""
    import re

    m = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", run_id)
    return "{}-{}-{} {}:{}".format(*m.groups()) if m else run_id


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
    if r.get("replaced"):
        return r["replaced"]
    if r.get("trial_ended"):
        return "trial finished at the champion; it does not go to validation"
    if r.get("waiting_on"):
        return f"waiting at “{LABELS.get(r['waiting_on'], r['waiting_on'])}”"
    return {"completed": "finished, not signed off", "approved": "signed off",
            "rejected": "rejected", "blocked": "blocked", "failed": "a step failed",
            "running": "working", "created": "not started"}.get(r.get("status") or "", "")


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
