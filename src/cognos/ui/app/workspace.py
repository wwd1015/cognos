"""The run workspace: the page, its modals, the run header, the stage rail, the activity, the
run record and the agent calls."""

from __future__ import annotations

import time
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ... import service
from ...agents.contracts import FRIENDLY
from ...engine.graph import (
    GATE_BEFORE_STAGE,
    GATE_OF_STAGE,
    GATES,
    LABELS,
    STAGE_AFTER_GATE,
    STAGE_OF_GATE,
    STAGES,
)
from ...engine.process import SEAT_LABEL, SEAT_OF_GATE, journal, seat_label
from .. import ask as ask_ui
from ..components import (
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
from ..panels import (
    SPONSOR_NOTE,
    challenge_rows,
    gap_cell,
    gap_need,
    gap_outcomes,
    gap_required,
    questions_summary,
)


def workspace_page(run_id: str) -> Any:
    return html.Div([
        dcc.Store(id="ws-run", data=run_id),
        dcc.Store(id="ws-step", data=None),
        dcc.Store(id="ws-seat", data="developer"),
        dcc.Store(id="ws-key", data=None),
        # Whether the stage list is shown; kept in the browser so the choice carries across runs.
        dcc.Store(id="ws-focus", data=False, storage_type="local"),
        dmc.Group([
            dmc.Group([
                html.Span("Acting as", className="eyebrow"),
                dmc.SegmentedControl(
                    id="seat-switch", value="developer", size="xs",
                    data=[{"value": k, "label": v.title()} for k, v in SEAT_LABEL.items()
                          if k != "express"],
                ),
            ], gap="sm"),
            dmc.Group([
                # The live log is one click away, not a column: the page keeps the width.
                dmc.Button("Activity", id="ws-activity-open", variant="subtle", color="gray",
                           size="xs", n_clicks=0, leftSection=icon("tabler:activity", 16)),
                dmc.Tooltip(dmc.ActionIcon(icon("tabler:arrows-maximize", 18),
                                           id="ws-focus-toggle", variant="subtle", color="gray",
                                           n_clicks=0, **{"aria-label": "Focus on the page"}),
                            label="Focus on the page: hide the stage list", id="ws-focus-tip"),
            ], gap="xs"),
        ], justify="space-between", mt="xs", mb="sm"),
        # The run's views. "Stages" is the work, one stage at a time; the other two cover the
        # whole run, so they sit above the stage index, not inside a stage.
        dmc.Tabs([dmc.TabsList([
            dmc.TabsTab("Stages", value="workflow", leftSection=icon("tabler:route")),
            dmc.TabsTab("Run record", value="questions", leftSection=icon("tabler:notebook")),
            dmc.TabsTab("Agent calls", value="audit", leftSection=icon("tabler:list-search")),
        ])], id="ws-tab", value="workflow", mb="lg"),
        dmc.Grid([
            dmc.GridCol([html.Div(id="ws-head"), html.Div(id="ws-rail", className="cognos-rail")],
                        id="ws-left-col", span={"base": 24, "md": 6}),
            dmc.GridCol(html.Div(id="ws-content", className="cognos-content"),
                        id="ws-main-col", span={"base": 24, "md": 18}),
        ], gutter="lg", columns=24),
        dmc.Drawer(html.Div(id="ws-activity"), id="ws-activity-drawer", position="right",
                   size="md", title=dmc.Text("Activity", fw=650), keepMounted=True,
                   zIndex=400),  # above the Ask button
        dmc.Modal(id="io-modal", size="80%", title="Agent call", children=html.Div(id="io-body")),
        trial_modal(), promote_modal(), template_modal(),
        dcc.Download(id="export-download"),
        ask_ui.dock(),
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


def template_modal() -> Any:
    return dmc.Modal(id="tpl-modal", title=dmc.Text("Save as a team template", fw=650),
                     size="lg", children=dmc.Stack([
        dmc.Text("The setup this run finished with becomes a starting point for the next "
                 "model of its kind: the template it started from, with the design answers, "
                 "the development plan as you edited it and the model families laid over it. "
                 "The data stays with this run; the template only says what it was built on.",
                 size="sm"),
        dmc.TextInput(id="tpl-title", label="Title",
                      placeholder="For example: C&I PD — middle market"),
        dmc.Textarea(id="tpl-description", label="What it is for (optional)", autosize=True,
                     minRows=2),
        dmc.Text("Everyone on this runs folder can start from it.", size="xs", c="dimmed"),
        html.Div(id="tpl-feedback"),
        dmc.Group([
            dmc.Button("Cancel", id="tpl-cancel", variant="default", n_clicks=0),
            dmc.Button("Save the template", id="tpl-go", n_clicks=0,
                       leftSection=icon("tabler:template", 16)),
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
    facts = [("Mode", st.mode), ("Agents", backend_label(st.provider,
                                                         service.agent_model(st.run_id)))]
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
        facts += [("Champion", str(backtest.payload.get("champion_name")
                                   or backtest.payload["champion_id"])),
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
        html.Div(service.next_action(st)["text"], className="run-hero-next"),
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
        dmc.Button("Save as template", id="tpl-open", variant="default", size="compact-sm",
                   n_clicks=0, leftSection=icon("tabler:template", 14))
        if st.status in ("completed", "approved") and not st.is_trial else None,
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
        if st.status_of(STAGE_AFTER_GATE.get(stage, "")) == "awaiting":
            gate = STAGE_AFTER_GATE[stage]  # the data is asked for on explore's page
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
    from ..theme import STEP

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
    return html.Div(items if items else empty("Nothing yet."), className="activity-list")


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
    from ...trials import CHANGE_LABELS

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


def call_label(call_id: str) -> str:
    """A call as a person reads it: the time it was made and its short code. The agent has its
    own column, so the id's agent part is left out instead of cut through."""
    stamp_, _, rest = call_id.partition("_")
    code = rest.rsplit("_", 1)[-1] if rest else ""
    if len(stamp_) == 15 and stamp_[8] == "-":
        t = stamp_[9:]
        return f"{t[:2]}:{t[2:4]}:{t[4:]} · {code}" if code else f"{t[:2]}:{t[2:4]}:{t[4:]}"
    return call_id


def audit_tab(run_id: str) -> Any:
    rows = []
    for a in service.audit(run_id)[::-1]:
        color = {"ok": "green", "invalid": "orange", "error": "red"}.get(a["status"], "gray")
        rows.append([
            dmc.Button(call_label(a["call_id"]), id={"type": "audit-row", "call": a["call_id"]},
                       variant="subtle", size="compact-xs", ff="monospace", n_clicks=0,
                       leftSection=icon("tabler:file-search", 12)),
            FRIENDLY.get(a["agent"], a["agent"]), a["attempt"], dmc.Badge(a["status"], color=color, size="xs", variant="light"),
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
            return STAGE_OF_GATE.get(g) or GATE_BEFORE_STAGE[g]
    for s in STAGES:
        if st.status_of(s) in ("running", "failed", "blocked"):
            return s
    done = [s for s in STAGES if st.status_of(s) == "done"]
    return done[-1] if done else STAGES[0]


def backend_label(provider: str | None, model: str | None) -> str:
    provider = provider or "?"
    return provider if not model or model in (provider, "heuristic", "replay") else f"{provider}:{model}"
