"""The explore page: analyses, the sources and the join, and the data gate."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from ... import service
from ...artifacts import StageResult
from ...engine.gates import carried_forward
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
                    lead: list | None = None, tail: list | None = None) -> Any:
    """Several sources: what they are, the join as the engine ran it, and the links it measured
    but did not use. ``tail`` follows the join (the Explore page puts the outcome there)."""
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
        *(tail or []),
        description=f"{len(linked['tables'])} sources joined into one modelling table of "
                    f"{fmt(report['n_rows'])} rows (the base table has "
                    f"{fmt(report.get('base_rows', report['n_rows']))}). A join never adds or "
                    "repeats a row.")


def _step(n: int, title: str, *children, description: str | None = None,
          status: Any = None) -> Any:
    """One numbered step of a page that follows the work in order."""
    return section(f"{n} · {title}", *children, right=status, description=description)


def analyses_by_column(analyses: list[dict], features: list[str]) -> tuple[dict, list]:
    """Each variable's own analyses (a parameter names it), and those across variables."""
    by: dict[str, list[dict]] = {f: [] for f in features}
    general: list[dict] = []
    for a in analyses:
        named = {v.strip() for val in (a.get("params") or {}).values()
                 for v in str(val).split(",")}
        hit = [f for f in features if f in named]
        for f in hit:
            by[f].append(a)
        if not hit:
            general.append(a)
    return by, general


# Run-wide analyses that hold one row per variable: their numbers go on the variable's line.
PER_VARIABLE = {"predictive_power": "feature", "stability": "feature", "collinearity": "feature",
                "outliers": "column"}


def per_variable_figures(run_id: str, analyses: list[dict]) -> dict[str, dict[str, dict]]:
    """``{tool: {variable: row}}`` from the run-wide analyses that rank every variable. The
    numbers are the engine's, read from the analyses' own tables; nothing is computed here."""
    out: dict[str, dict[str, dict]] = {}
    for a in analyses:
        key = PER_VARIABLE.get(a.get("tool") or "")
        if not key or a.get("status") != "ok" or not a.get("artifact"):
            continue
        try:
            result = json.loads(service.read_artifact(run_id, a["artifact"]) or "{}")
        except ValueError:
            continue
        for t in result.get("tables", [])[:1]:
            cols = t.get("columns") or []
            if key not in cols:
                continue
            out[a["tool"]] = {str(r[cols.index(key)]): dict(zip(cols, r, strict=False))
                              for r in t.get("rows") or []}
    return out


def _power(row: dict | None) -> Any:
    """Predictive power on its own: information value for a binary outcome, else the
    strength the tool measured."""
    if not row:
        return None
    return row.get("information_value", row.get("strength"))


EXPECTED = {"+": "rises", "-": "falls", "nonlinear": "non-linear", "unknown": "unknown"}


def var_row_class(col: str, inside: bool, advised_out: set[str]) -> str:
    """A variable's row: dimmed when it is left out, marked where it departs from the advice."""
    return ("var-row" + ("" if inside else " out")
            + (" differs" if inside == (col in advised_out) else ""))


def variables_table(p: dict, decisions: dict, corr: dict, suspects: set, excluded: list[str], *,
                    editable: bool, cards: dict[str, list], carried: bool = False,
                    figures: dict[str, dict[str, dict]] | None = None) -> Any:
    """One row per variable: its type, missing share and correlation with the outcome, what
    the Data agent expects of it and advises, and a switch that takes it in or leaves it out.
    A row opens on the agent's reason and that variable's analyses. The switches fill the
    data gate's exclusions (the hidden field); the decision is still the gate's."""
    features = list(p.get("features", []))
    cand = {c["column"]: c for c in p.get("feature_candidates", [])}
    advised_out = set(p.get("recommended_exclusions", [])) | {
        c for c, d in decisions.items() if d["decision"] == "exclude"}
    figures = figures or {}
    power, psi, vif, outl = (figures.get(k, {}) for k in PER_VARIABLE)
    measure = next((r.get("measure") for r in power.values() if r.get("measure")), None)
    titles = {"IV": "Information value: predictive power on its own (above 0.3 strong, below "
                    "0.02 weak, above 0.5 check for leakage)",
              "PSI": "Population stability index, recent periods against the earliest "
                     "(above 0.25: the variable has shifted)",
              "VIF": "Variance inflation factor: how far the other variables explain it "
                     "(above 5: collinear)"}
    power_label = "IV" if not measure else ("|ρ|" if "spearman" in measure else "η²")
    head = html.Div([html.Div([html.Span(h, title=titles.get(h)) for h in (
                                  "Variable", "Type", "Missing", "Corr.", power_label, "PSI",
                                  "VIF", "Data agent")],
                              className="var-line"),
                     html.Span("In", className="var-in-head")], className="var-head")
    rows = []
    for f in features:
        d, c = decisions.get(f), cand.get(f)
        advice = ("exclude" if f in advised_out else d["decision"] if d else "keep")
        detail = [x for x in (
            dmc.Text([html.B("Data agent: "), d["reason"]], size="sm") if d else None,
            dmc.Text([html.B("Why it may matter: "), c["rationale"]]
                     + ([f" Expected to {EXPECTED.get(c['relationship'], 'move')} with the "
                         "outcome." if c.get("relationship") not in (None, "unknown") else ""]),
                     size="sm", mt=4) if c and c.get("rationale") else None,
            dmc.Text([html.B("Stability: "), f"mean {fmt(psi[f].get('reference_mean'), 4)} in "
                      f"the earliest periods, {fmt(psi[f].get('recent_mean'), 4)} in the recent "
                      f"ones (PSI {fmt(psi[f].get('psi'), 3)})."], size="sm", mt=4)
            if f in psi else None,
            dmc.Text([html.B("Spread: "), f"1st percentile {fmt(outl[f].get('p1'), 4)}, median "
                      f"{fmt(outl[f].get('median'), 4)}, 99th {fmt(outl[f].get('p99'), 4)}; "
                      f"{pct(outl[f].get('outside_3_iqr') or 0)} beyond 3 × IQR, skew "
                      f"{fmt(outl[f].get('skew'), 2)}."], size="sm", mt=4)
            if f in outl else None,
            dmc.Text("Its correlation with the outcome is high enough to suspect leakage "
                     "(|corr| ≥ 0.98).", size="sm", c="red", mt=4) if f in suspects else None,
            dmc.SimpleGrid(cards[f], cols={"base": 1, "lg": 2}, spacing="md", mt="sm")
            if cards.get(f) else None,
        ) if x is not None] or [dmc.Text("No analysis of this variable on its own.",
                                         size="sm", c="dimmed")]
        inside = f not in excluded
        rows.append(html.Div([
            html.Details([
                html.Summary([
                    html.Span([html.Span(f, className="var-name"),
                               dmc.Badge("leakage suspect", color="red", size="xs",
                                         variant="light") if f in suspects else None],
                              className="var-cell-name"),
                    html.Span(p.get("dtypes", {}).get(f, ""), className="var-num"),
                    html.Span(pct(p.get("missing", {}).get(f, 0.0)), className="var-num"),
                    html.Span(fmt(corr.get(f), 3), className="var-num"),
                    html.Span(fmt(_power(power.get(f)), 3) if f in power else "—",
                              className="var-num"),
                    html.Span(fmt(psi[f].get("psi"), 2) if f in psi else "—",
                              className="var-num" + (" var-warn" if f in psi and (
                                  psi[f].get("psi") or 0) > 0.25 else "")),
                    html.Span(fmt(vif[f].get("vif"), 1) if f in vif else "—",
                              className="var-num" + (" var-warn" if f in vif and (
                                  vif[f].get("vif") or 0) > 5 else "")),
                    html.Span(dmc.Badge(advice, variant="light", size="sm",
                                        color="red" if advice == "exclude" else "gray")),
                ], className="var-line"),
                html.Div(detail, className="var-detail"),
            ]),
            dmc.Switch(id={"type": "var-in", "col": f}, checked=inside, disabled=not editable,
                       size="sm", color="oxblood", className="var-switch",
                       **{"aria-label": f"Use {f} in the model"}),
        ], id={"type": "var-row", "col": f}, className=var_row_class(f, inside, advised_out)))
    n_out = len([f for f in features if f in excluded])
    return html.Div([
        dmc.Group([
            dmc.Text(f"{len(features) - n_out} in · {n_out} out", id="var-count", size="sm",
                     fw=600, ff="monospace"),
            dmc.Text("Switches start from your earlier choices, where they still apply."
                     if editable and carried else
                     "Switches start from the Data agent's advice." if editable else "",
                     size="xs", c="dimmed"),
        ], justify="space-between", mb="xs"),
        # The gate reads the exclusions from this field; the switches keep it in step.
        dmc.MultiSelect(id=_field("gate_data", "exclude"),
                        data=[{"value": f, "label": f} for f in features], value=excluded,
                        style={"display": "none"}),
        dcc.Store(id="var-advised", data=sorted(advised_out)),
        html.Div([head, *rows], className="var-table"),
        dmc.Text(f"{power_label}, PSI and VIF come from the analyses below (hover a heading for "
                 "what it means; — where the analysis did not cover the variable). Values past "
                 "a usual threshold are marked.", size="xs", c="dimmed", mt=6)
        if figures else None,
    ])


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
    # New data set the data gate's earlier choices aside: offered again where they still
    # apply, never in force until the gate is decided (gates.carried_forward).
    carried = carried_forward(state) if editable else None
    features = list(p.get("features", []))
    if carried:
        excluded = [c for c in carried.get("exclude_columns") or [] if c in features]
    else:
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

    # 2 · the join and the outcome
    fixed = p.get("target_source") == "profile"
    how = {"profile": "fixed in the project profile", "decision": "set at the data gate",
           "agent": "proposed by the Data agent"}.get(p.get("target_source", ""), "")
    target_options = ([{"value": c["column"], "label": f"{c['column']} — {c['why']}"}
                       for c in p.get("target_candidates", [])]
                      or [{"value": p.get("target", ""), "label": p.get("target", "")}])
    if p.get("target") and p["target"] not in [o["value"] for o in target_options]:
        target_options.insert(0, {"value": p["target"], "label": p["target"]})
    before = (carried or {}).get("target")
    outcome = [
        dmc.Select(id=_field("gate_data", "target"), label="Dependent variable",
                   data=target_options, value=p.get("target"), allowDeselect=False,
                   disabled=fixed or not editable, searchable=True, maw=520, mt="md",
                   description=("Fixed in the project profile." if fixed else
                                f"The outcome as {how}. Changing it re-runs the "
                                "exploration.")),
        dmc.Group([dmc.Badge(p.get("task", ""), variant="light", size="sm"),
                   dmc.Text(p.get("target_rationale") or "", size="sm")], gap="xs", mt=6)
        if p.get("target") else None,
        dmc.Text(f"Before the data changed, the dependent variable was {before}.", size="xs",
                 c="dimmed", mt=4) if before and before != p.get("target") else None,
    ]
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
        steps.append(linking_section(linked, title=f"{next(n)} · Join and outcome",
                                     lead=canvas, tail=outcome))
    else:
        steps.append(_step(next(n), "The outcome", *outcome,
                           description="What the model predicts."))

    # 3 · the variables: what each one shows, and whether it goes in
    analyses = p.get("analyses") or []
    scripts = [a for a in analyses if a["kind"] == "code"]
    by_column, general = analyses_by_column(analyses, features)
    steps.append(_step(
        next(n), "Variables",
        variables_table(p, decisions, corr, suspects, excluded, editable=editable,
                        cards={f: [analysis_card(state.run_id, a, scheme) for a in by_column[f]]
                               for f in features if by_column[f]},
                        carried=bool(carried),
                        figures=per_variable_figures(state.run_id, general)),
        dmc.Alert(f"{len(scripts)} analysis script(s) written by the Data agent are part of "
                  "what you are accepting: the code goes to validation and into the white "
                  "paper as it stands.", color="grape", variant="light",
                  icon=icon("tabler:code"), mt="sm") if scripts else None,
        html.Details([
            html.Summary(f"{len(general)} whole-dataset analyses, requested by the Data agent "
                         "and run by the engine", className="gate-fold"),
            source_counts(general),
            dmc.SimpleGrid([analysis_card(state.run_id, a, scheme) for a in general],
                           cols={"base": 1, "lg": 2}, spacing="md"),
        ], className="gate-more",
            open=any(a["kind"] == "code" for a in general)) if general else None,
        html.Details([html.Summary("Data quality, as the Data agent reads it",
                                   className="gate-fold"),
                      dmc.List([dmc.ListItem(c["statement"]) for c in out["data_quality"]],
                               size="sm", mt="xs")],
                     className="gate-more") if out.get("data_quality") else None,
        description="Each variable's numbers and the Data agent's advice, with a switch to "
                    "take it in or leave it out. Open a row for the agent's reason and the "
                    "analyses of that variable. A built-in or plugin tool is reviewed code; a "
                    "script is code the agent wrote for this run, kept with its hash and re-run "
                    "at validation."))

    # 4 · the transformations
    review = p.get("transform_review") or {}
    tx_cols = review.get("columns") or []
    if tx_cols:
        from ...modeling import column_transforms as ct

        kinds = review.get("kinds") or {}
        reviewed = {c["column"] for c in tx_cols}
        decided = ({c: {"forms": ct.forms_of(s)}
                    for c, s in (carried.get("transforms") or {}).items() if c in reviewed}
                   if carried else
                   {c: {"forms": ct.forms_of(s)} for c, s in state.overrides.transforms.items()}
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
    elif "transform_review" not in p:  # explored before transformations existed (ADR-0024)
        steps.append(_step(next(n), "Transformations", empty(
            "This exploration was recorded before COGNOS reviewed transformations, so there "
            "are none to decide here. A new run from the same template has this step.")))

    # 5 · summary and decision
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
