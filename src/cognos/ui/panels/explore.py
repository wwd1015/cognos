"""The explore page: analyses, the sources and the join, and the data gate."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import html

from ... import service
from ...artifacts import StageResult
from ...engine.process import SEAT_OF_GATE
from ...engine.state import RunState
from .. import charts, transform_review
from ..components import (
    empty,
    findings_table,
    fmt,
    graph,
    icon,
    kpis,
    pct,
    recommendation_card,
    section,
    table,
)
from ._shared import _act, _field, _findings, _rec, _summary_line
from .gate import gate_block

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
    code exactly as the Data agent wrote it."""
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
    between them. Click a link to open its settings (in the join or not, several rows per
    key, rows with no match); drag from a column of one table to a column of another to draw
    a link, which the engine measures before it can be used; a tile's button makes it the
    base table. The page keeps all of it in the `join_spec` field the decision reads; the
    drawing, the dragging and the settings are `assets/join_canvas.js`."""
    plan = linked["plan"]
    steps = {s["link"]: s for s in plan["steps"]}
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
        place[t["name"]] = (12 + col * 224, 400 + row * 270)
    height = 400 + ((len(others) + 2) // 3 or 1) * 270
    tiles = []
    for t in [base, *others]:
        mine = keys.get(t["name"], set()) | set(t.get("unique_columns") or [])
        names = sorted(t["columns"], key=lambda c: (c not in mine, c != target))
        shown = [html.Li(c, className="jc-key" if c in mine else
                         "jc-target" if c == target else None, **{"data-col": c})
                 for c in names]
        x, y = place[t["name"]]
        tiles.append(html.Div([
            html.Header([html.B(t["name"]), html.Span(f"{fmt(t['n_rows'])} rows")]),
            html.Ul(shown),
            html.Button("base table" if t["name"] == plan["base"] else "Make base",
                        className="jc-make-base", disabled=disabled),
        ], className="jc-tile" + (" base" if t["name"] == plan["base"] else ""),
            style={"left": f"{x}px", "top": f"{y}px"}, title=t.get("description") or None,
            **{"data-table": t["name"]}))
    chips = []
    for lk in linked["links"]:
        step = steps.get(lk["id"])
        attrs = {"data-a": lk["a"], "data-b": lk["b"], "data-link": lk["id"],
                 "data-on": ", ".join(a if a == b else f"{a} = {b}"
                                      for a, b in zip(lk["a_on"], lk["b_on"], strict=True)),
                 "data-match": f"{max(lk['a_in_b'], lk['b_in_a']):.0%}",
                 "data-rel": _RELATION.get(lk["relation"], lk["relation"]),
                 "data-many": (step or {}).get("many", "") if (step or {}).get("many") in
                 ("aggregate", "first") else "",
                 "data-unmatched": (step or {}).get("unmatched", ""),
                 "data-by": lk.get("by", "") if lk.get("drawn") else ""}
        chips.append(html.Button(
            _link_chip_text(lk),
            className=chip_class(lk["id"] in chosen, lk["id"] in steps)
            + (" drawn" if lk.get("drawn") else ""),
            disabled=disabled, title=_link_label(lk) + (f" · drawn by {lk['by']}"
                                                        if lk.get("drawn") else ""),
            **attrs))
    spec = {"base": plan["base"], "links": list(chosen), "rules": {}, "drawn": []}
    n_drawn = sum(1 for lk in linked["links"] if lk.get("drawn"))
    return html.Div([
        html.Div(html.Div([html.Div(className="jc-svg-host"), *tiles, *chips],
                          className="jc-canvas" + (" locked" if disabled else ""),
                          style={"height": f"{height}px", "minWidth": f"{width}px"},
                          **{"data-drawn": str(n_drawn)}),
                 className="jc-wrap"),
        html.Div([html.Span([html.I(className="m"), "Measured by the engine"]),
                  html.Span([html.I(className="p"), "Suggested by the Data agent"]),
                  html.Span([html.I(className="c"), "In your join"]),
                  html.Span([html.I(className="d"), "Drawn by a person"])],
                 className="jc-legend"),
        html.Div("Click a link to see it and set its rules. To join on columns the engine did "
                 "not pair, drag from a column in one table to a column in another. Drag a "
                 "table's name to move it. Changes count when you press Apply my changes."
                 if not disabled else "Stated in the project profile: the join cannot be "
                 "changed here.", className="jc-help"),
        dmc.TextInput(id=_field("gate_data", "join_spec"), value=json.dumps(spec),
                      style={"display": "none"}),
    ], className="jc")


def chip_class(chosen: bool, proposed: bool) -> str:
    """How a link reads on the canvas: in the join (and whether the agent suggested it), or
    only measured, or suggested and left out."""
    if chosen:
        return "jc-chip chosen" + (" suggested" if proposed else "")
    return "jc-chip " + ("proposed" if proposed else "measured")


def linking_section(linked: dict | None, *, title: str = "Sources and join",
                    lead: list | None = None) -> Any:
    """Several sources: what they are, the join as the engine ran it, and the links it measured
    but did not use."""
    if not linked:
        return None
    plan, report = linked["plan"], linked["report"]
    why = (linked.get("rationale") or {})
    reasons = why.get("joins") or {}
    unmatched = why.get("unmatched") or {}
    how = {"profile": "stated in the project profile", "decision": "set at the data gate",
           "agent": "proposed by the Data agent"}.get(plan["source"], "")
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
        title, *(lead or []), sources,
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


def _step(n: int, title: str, *children, description: str | None = None,
          status: Any = None) -> Any:
    """One numbered step of a page that follows the work in order."""
    return section(f"{n} · {title}", *children, right=status, description=description)


def explore_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    """The Explore page, in the order the work is done: the data, the join, the outcome and the
    variables, how each variable relates to the outcome, the transformations, then the summary
    and the decision. The data gate's fields sit in the step they belong to; the gate reads
    them all when it is decided."""
    p = res.payload or {}
    ts = p.get("target_summary") or {}
    suspects = set(p.get("leakage_suspects", []))
    corr = {c["feature"]: c["corr"] for c in p.get("top_correlations", [])}
    rec = _rec(res) or {}
    out = rec.get("output") or {}
    decisions = {d["column"]: d for d in out.get("column_decisions", [])}
    editable = (state.status_of("gate_data") == "awaiting"
                and seat == SEAT_OF_GATE.get("gate_data"))
    revised = state.last_decision("gate_data") is not None
    excluded = list(state.overrides.exclude_columns if revised
                    else p.get("recommended_exclusions", []))
    n = iter(range(1, 10))
    steps: list[Any] = []

    # 1 · the data
    src = p.get("source") or {}
    if src:
        steps.append(_step(
            next(n), "The data",
            table(["Connector", "Location", "Rows × columns", "Fetched", "Snapshot SHA-256"],
                  [[src.get("connector", "—"),
                    dmc.Text(src.get("query") or src.get("location") or "—", size="xs",
                             ff="monospace"),
                    f"{fmt(src.get('n_rows'))} × {fmt(src.get('n_cols'))}",
                    str(src.get("fetched_at", ""))[:19].replace("T", " "),
                    dmc.Code(str(src.get("snapshot_sha256", ""))[:12])]]),
            description="Fetched once; every stage reads this snapshot."))

    # 2 · the join
    linked = p.get("linking") or None
    if linked:
        join_fixed = linked["plan"]["source"] == "profile"
        chosen_links = [s["link"] for s in linked["plan"]["steps"]]
        canvas = [
            dmc.Text("Stated in the project profile." if join_fixed else
                     "The tables, the base table framed in ink, and every link the engine "
                     "measured with how well it matches. Changing the join rebuilds the "
                     "dataset and re-runs the exploration.", size="xs", c="dimmed"),
            join_canvas(linked, p.get("target"), chosen_links,
                        disabled=join_fixed or not editable),
        ]
        steps.append(linking_section(linked, title=f"{next(n)} · The join", lead=canvas))

    # 3 · the outcome and the variables
    fixed = p.get("target_source") == "profile"
    how = {"profile": "fixed in the project profile", "decision": "set at the data gate",
           "agent": "proposed by the Data agent"}.get(p.get("target_source", ""), "")
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
    decision_rows = [[dmc.Text(d["column"], ff="monospace", size="sm"),
                      dmc.Badge(d["decision"], color="red" if d["decision"] == "exclude" else "gray",
                                variant="light", size="sm"), d["reason"]]
                     for d in out.get("column_decisions", [])]
    steps.append(_step(
        next(n), "Outcome and variables",
        dmc.Select(id=_field("gate_data", "target"), label="Dependent variable",
                   data=target_options, value=p.get("target"), allowDeselect=False,
                   disabled=fixed or not editable, searchable=True, maw=520,
                   description=("Fixed in the project profile." if fixed else
                                f"The outcome as {how}. Changing it re-runs the "
                                "exploration.")),
        dmc.Group([dmc.Badge(p.get("task", ""), variant="light", size="sm"),
                   dmc.Text(p.get("target_rationale") or "", size="sm")], gap="xs", mt=6)
        if p.get("target") else None,
        dmc.MultiSelect(id=_field("gate_data", "exclude"), label="Columns to exclude from modeling",
                        data=[{"value": c, "label": c} for c in p.get("features", [])],
                        value=excluded, searchable=True, clearable=True, mt="md",
                        disabled=not editable,
                        description="Pre-filled with the Data agent's recommendation."),
        html.Details([html.Summary(f"The Data agent's candidates ({len(cand_rows)}) and column "
                                   "decisions", className="gate-fold"),
                      table(["Feature", "Expected", "Corr.", "Why"], cand_rows)
                      if cand_rows else None,
                      table(["Column", "Decision", "Reason"], decision_rows)
                      if decision_rows else None,
                      dmc.List([dmc.ListItem(c["statement"]) for c in out.get("data_quality", [])],
                               size="sm", mt="xs") if out.get("data_quality") else None],
                     className="gate-more") if cand_rows or decision_rows else None,
        description="What the model predicts and the columns it may use: hypotheses for the "
                    "design stage, not a selection."))

    # 4 · how each variable relates to the outcome
    analyses = p.get("analyses") or []
    scripts = [a for a in analyses if a["kind"] == "code"]
    profile_rows = []
    for f in p.get("features", []):
        d = decisions.get(f)
        profile_rows.append([
            dmc.Group([dmc.Text(f, size="sm", ff="monospace"),
                       dmc.Badge("leakage suspect", color="red", size="xs", variant="light")
                       if f in suspects else None], gap=6),
            p.get("dtypes", {}).get(f, ""),
            pct(p.get("missing", {}).get(f, 0.0)),
            fmt(corr.get(f), 3),
            (dmc.Badge(d["decision"], color="red" if d["decision"] == "exclude" else "gray",
                       variant="light", size="sm") if d else ""),
        ])
    steps.append(_step(
        next(n), "How each variable relates to the outcome",
        table(["Feature", "Type", "Missing", "Corr. with target", "Agent"], profile_rows,
              max_height=300),
        dmc.Alert(f"{len(scripts)} analysis script(s) written by the Data agent are part of "
                  "what you are accepting: the code goes to validation and into the white "
                  "paper as it stands.", color="grape", variant="light",
                  icon=icon("tabler:code"), mt="sm") if scripts else None,
        html.Details([
            html.Summary(f"Analyses ({len(analyses)}), requested by the Data agent and run by "
                         "the engine", className="gate-fold"),
            source_counts(analyses),
            dmc.SimpleGrid([analysis_card(state.run_id, a, scheme) for a in analyses],
                           cols={"base": 1, "lg": 2}, spacing="md"),
        ], className="gate-more", open=bool(scripts)) if analyses else None,
        description="Each feature's type, missing share and correlation with the outcome, "
                    "and the analyses behind them. A built-in or plugin tool is reviewed code; "
                    "a script is code the agent wrote for this run, kept with its hash and "
                    "re-run at validation."))

    # 5 · the transformations
    review = p.get("transform_review") or {}
    tx_cols = review.get("columns") or []
    if tx_cols:
        from ...modeling import column_transforms as ct

        kinds = review.get("kinds") or {}
        decided = ({c: {"forms": ct.forms_of(s)} for c, s in state.overrides.transforms.items()}
                   if revised else {})
        data = {"kinds": kinds, "hints": review.get("hints") or {},
                "grid": review.get("grid") or {},
                "columns": {c["column"]: c for c in tx_cols},
                "order": [c["column"] for c in tx_cols]}
        plain = review.get("as_they_are") or []
        steps.append(_step(
            next(n), "Transformations",
            transform_review.block(
                "txd", data, decided, editable=editable, excluded=excluded,
                lead=f"This model needs {review['label']}. {review['need']} Decide the "
                     "forms each variable enters in: as it is, transformed, or several at once "
                     "(each form is a variable of its own in the single-factor screen)."),
            dmc.Text(f"{len(plain)} flag(s) and categories enter as they are: "
                     + ", ".join(plain), size="xs", c="dimmed", mt="sm") if plain else None,
            description="Every form is decided here. Logs, lags and changes are used as "
                        "decided; a fitted shape or an expression gets its parameters at the "
                        "design gate, on the development sample."))

    # 6 · summary and decision
    in_force = state.overrides.transforms if revised else {}
    tx_data = {"order": [c["column"] for c in tx_cols]}
    summary = html.Div([
        _summary_line("Dependent variable",
                      f"{p.get('target', '—')} ({p.get('task', '')})"),
        _summary_line("Variables", f"{len(p.get('features', [])) - len(excluded)} in, "
                                   f"{len(excluded)} excluded"),
        _summary_line("Join", f"{linked['plan']['base']} + {len(linked['plan']['steps'])} "
                              "table(s)") if linked else None,
        _summary_line("Transformations", html.Span(
            transform_review.summary(tx_data, in_force, excluded) if not editable else "",
            id="txd-summary")) if tx_cols else None,
        _summary_line("Analysis scripts", f"{len(scripts)} to accept as code") if scripts
        else None,
    ], className="sum-block")
    gate = gate_block("gate_data", state, res, seat=seat, body=[summary], primary=[
        dmc.Button("Accept recommendation", id=_act("gate_data", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply my decisions", id=_act("gate_data", "edit"), variant="light",
                   leftSection=icon("tabler:edit")),
    ])
    steps.append(_step(
        next(n), "Summary and decision",
        recommendation_card(rec, title="The Data agent's recommendation"),
        gate,
        description="“Accept recommendation” takes the Data agent's exclusions and the "
                    "engine's kind for every variable. “Apply my decisions” takes the steps "
                    "above as they stand, and needs every variable decided."))
    return [
        kpis([("Rows", fmt(p.get("n_rows")), None),
              ("Features", fmt(len(p.get("features", []))), None),
              ("Event rate", pct(ts.get("positive_rate")) if "positive_rate" in ts else "—",
               "positive share of the target"),
              ("Leakage suspects", fmt(len(suspects)), "|corr| ≥ 0.98 with the target")]),
        *steps,
        section("Engine findings", findings_table(_findings(res))),
    ]
