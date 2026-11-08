"""The transformation picker (ADR-0024): one variable at a time, on two pages.

On the Explore page (``prefix="txd"``, the data gate) each reviewed variable is decided as a
kind: transformed, and how, or kept as it is. On the design page (``prefix="txp"``, the design
gate) each transformed variable gets its parameters, previewed on the development sample.

Both share one layout: a short list on the left (status, the check that failed, the engine's
recommendation, the decision), the selected variable on the right, and a progress line. The
decisions live in a hidden gate field as JSON until the gate is decided, so choosing and
previewing never redraws the page. Pure functions of their inputs; the callbacks are in
``ui/app.py`` (``register_transform_callbacks``).
"""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from . import charts
from .components import empty, graph, icon

FIELD = {"txd": ("gate_data", "transforms"), "txp": ("gate_design", "transform_params")}
FILTERS = {"open": "Open", "transform": "Transform", "decided": "Decided", "all": "All"}


def opt(name: str) -> dict:
    """The id of a control that is not always on the page (a read-only pane has no buttons,
    the expression box shows only for a custom kind). Callbacks read it with ``ALL``, so a
    missing one is an empty list rather than a callback Dash refuses to run."""
    return {"type": name, "i": 0}


def field_id(prefix: str) -> dict:
    gate, name = FIELD[prefix]
    return {"type": "gate-field", "gate": gate, "field": name}


def decisions(value: str | None) -> dict[str, Any]:
    """The decisions in the hidden field."""
    try:
        out = json.loads(value or "{}")
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def kind_of(decision: Any) -> str | None:
    return decision.get("kind") if isinstance(decision, dict) else decision


def wanted(data: dict, excluded: list[str] | None) -> list[str]:
    """The variables that need a decision: every one listed, less the ones excluded."""
    gone = set(excluded or [])
    return [c for c in data.get("order") or [] if c not in gone]


def next_open(data: dict, decided: dict, current: str | None,
              excluded: list[str] | None = None) -> str | None:
    """The next variable without a decision after ``current`` (wrapping), else ``current``."""
    order = wanted(data, excluded)
    if not order:
        return current
    start = order.index(current) + 1 if current in order else 0
    for c in order[start:] + order[:start]:
        if c not in decided:
            return c
    return current


def _flag(entry: dict) -> tuple[str, str] | None:
    """The first failed check, as a short word and a colour, for the list."""
    for c in entry.get("checks") or []:
        if not c["ok"]:
            return {"one_way": ("turns", "yellow"), "straight": ("bends", "yellow"),
                    "tails": ("long tail", "yellow"), "skew": ("skewed", "yellow"),
                    "drift": ("drifts", "yellow"), "lead": ("lags", "yellow"),
                    "flat": ("weak", "gray")}.get(c["id"], (c["label"], "yellow"))
    return None


def progress(prefix: str, data: dict, decided: dict, excluded: list[str] | None) -> Any:
    order = wanted(data, excluded)
    done = sum(1 for c in order if c in decided)
    left = len(order) - done
    verb = "decided" if prefix == "txd" else "set"
    return dmc.Group([
        dmc.Progress(value=100 * done / len(order) if order else 100, w=140, size="sm",
                     color="green" if not left else "oxblood"),
        dmc.Text(f"{done} of {len(order)} {verb}" + (f" · {left} open" if left else ""),
                 size="xs", c="dimmed" if left else "green", fw=600 if not left else None),
    ], gap="xs")


def rows(prefix: str, data: dict, decided: dict, *, selected: str | None, filt: str,
         search: str, excluded: list[str] | None) -> list:
    """The list on the left: one compact row per variable."""
    kinds = data.get("kinds") or {}
    out = []
    for col in wanted(data, excluded):
        entry = data["columns"][col]
        rec = (entry.get("recommended") or {}).get("kind") if prefix == "txd" else entry.get(
            "kind")
        mine = kind_of(decided.get(col))
        shown = mine or rec
        if search and search.lower() not in col.lower():
            continue
        if filt == "open" and col in decided:
            continue
        if filt == "decided" and col not in decided:
            continue
        if filt == "transform" and (shown or "as_is") == "as_is":
            continue
        flag = _flag(entry) if prefix == "txd" else None
        out.append(html.Div([
            html.Span("✓" if col in decided else "○",
                      className="tx-dot" + (" done" if col in decided else "")),
            html.Span(col, className="tx-col", title=col),
            html.Span(flag[0], className=f"tx-flag {flag[1]}") if flag else html.Span(),
            html.Span(kinds.get(shown, shown or ""),
                      className="tx-kind" + (" mine" if mine else "")),
        ], id={"type": f"{prefix}-row", "col": col}, n_clicks=0,
            className="tx-row" + (" sel" if col == selected else "")))
    return out or [empty("Nothing matches.")]


def block(prefix: str, data: dict, decided: dict, *, editable: bool, excluded: list[str],
          lead: str) -> Any:
    """The picker: stores, the list with its filter and the pane for the selected variable."""
    first = next_open(data, decided, None, excluded) or (wanted(data, excluded) or [None])[0]
    return html.Div([
        dcc.Store(id=f"{prefix}-data", data=data),
        dcc.Store(id=f"{prefix}-sel", data=first),
        dmc.TextInput(id=field_id(prefix), value=json.dumps(decided), style={"display": "none"}),
        dmc.Text(lead, size="sm", mb="sm", maw=820),
        dmc.Grid([
            dmc.GridCol(dmc.Stack([
                dmc.Group([
                    dmc.SegmentedControl(id=f"{prefix}-filter", size="xs",
                                         value="open" if editable else "all",
                                         data=[{"value": k, "label": v}
                                               for k, v in FILTERS.items()]),
                    dmc.TextInput(id=f"{prefix}-search", placeholder="Find a variable",
                                  size="xs", w=150, leftSection=icon("tabler:search", 14)),
                ], gap="xs", justify="space-between"),
                html.Div(id=f"{prefix}-progress"),
                html.Div(id=f"{prefix}-list", className="tx-list"),
            ], gap="xs"), span={"base": 12, "md": 5}),
            dmc.GridCol(dcc.Loading(  # the pane stays in place, dimmed, while it redraws
                html.Div(id=f"{prefix}-detail", className="tx-detail"), delay_show=250,
                overlay_style={"visibility": "visible", "opacity": 0.55}, type="dot",
                color="var(--mantine-color-oxblood-6)"), span={"base": 12, "md": 7}),
        ], gutter="lg"),
        dcc.Store(id=f"{prefix}-editable", data=editable),
        dcc.Store(id=f"{prefix}-previews", data={}),
    ], className="tx-picker")


# --- the pane on the right -------------------------------------------------------------------
def _checks(entry: dict) -> Any:
    return dmc.Stack([
        dmc.Group([
            dmc.ThemeIcon(icon("tabler:check" if c["ok"] else "tabler:alert-triangle", 13),
                          size="sm", radius="xl", variant="light",
                          color="green" if c["ok"] else "yellow"),
            dmc.Text([dmc.Text(c["label"], span=True, fw=600), f": {c['text']}"], size="xs"),
        ], gap=6, wrap="nowrap", align="flex-start")
        for c in entry.get("checks") or []], gap=4)


def explore_detail(col: str | None, data: dict, decided: dict, scheme: str, *,
                   editable: bool, pick: str | None = None, sketch: dict | None = None,
                   expr: str = "", error: str | None = None) -> Any:
    """Step one, one variable: how it relates to the outcome, the checks and the engine's
    recommendation; then a preview of the kind being tried (``pick``, drawn from ``sketch``
    over the curve as explored) and a button that makes it the decision."""
    if not col or col not in (data.get("columns") or {}):
        return empty("Choose a variable on the left.")
    entry = data["columns"][col]
    kinds, hints = data.get("kinds") or {}, data.get("hints") or {}
    rec = entry["recommended"]
    mine = kind_of(decided.get(col))
    pick = pick or mine or rec["kind"]
    if pick == "custom" and not expr and isinstance(decided.get(col), dict):
        expr = decided[col].get("expr", "")
    way = "rises" if entry["direction"] == "+" else "falls"
    if sketch and sketch.get("curve"):
        shown = {**sketch, "points": entry.get("points") if sketch.get("axis") == col else None}
        fig = charts.shape(shown, scheme, column=sketch.get("axis") or col,
                           rate=entry.get("rate", False), height=300,
                           line=f"{kinds.get(pick, pick)} (preview)",
                           baseline=entry["curve"] if sketch.get("axis") == col else None)
        note = sketch.get("caption") or ""
    else:
        fig = charts.shape({"observed": entry["observed"], "curve": entry["curve"],
                            "points": entry.get("points")}, scheme, column=col,
                           rate=entry.get("rate", False), height=300,
                           line="as explored (smoothed)")
        note = (hints.get(pick, "") if pick == "library" else
                f"Write an expression of x ({col}) and preview it." if pick == "custom"
                else f"{col} cannot be {kinds.get(pick, pick)}; try another kind")
    tries = [dmc.Tooltip(dmc.Button(
        kinds[k], id={"type": "txd-try", "kind": k}, size="xs",
        variant="filled" if k == pick else "default",
        color="oxblood" if k == pick else "gray",
        rightSection=(dmc.Badge("recommended", size="xs", variant="light", color="oxblood")
                      if k == rec["kind"] else None),
        leftSection=icon("tabler:check", 14) if k == mine else None),
        label=hints.get(k, ""), multiline=True, w=240, withArrow=True) for k in kinds]
    chosen = {"kind": "custom", "expr": sketch.get("expr", expr)} if (
        pick == "custom" and sketch) else pick
    if pick == "custom" and isinstance(decided.get(col), dict) and mine == "custom":
        mine_same = decided[col].get("expr") == (sketch or {}).get("expr", expr)
    else:
        mine_same = pick == mine
    confirm = (dmc.Group([
        dmc.Button(f"Use {kinds.get(pick, pick)}", size="sm",
                   id={"type": "txd-confirm", "i": 0, "pick": json.dumps(chosen)},
                   leftSection=icon("tabler:check", 16),
                   disabled=not (sketch or pick == "library")),
        dmc.Text("decided" if mine_same else
                 f"decided: {kinds.get(mine, mine)}" if mine else "not decided yet",
                 size="xs", c="green" if mine_same else "dimmed"),
    ], gap="sm") if editable else dmc.Text(["Decided: ", html.B(kinds.get(mine, "not decided"))],
                                           size="sm"))
    custom = dmc.Group([
        dmc.TextInput(id=opt("txd-expr"), value=expr, size="xs", w=300, error=error,
                      label=f"Expression of x (x is {col})",
                      placeholder="e.g. np.minimum(x, 3) or np.log1p(x)"),
        dmc.Button("Preview", id=opt("txd-expr-go"), size="xs", variant="light",
                   leftSection=icon("tabler:eye", 14)),
    ], gap="xs", align="flex-end") if editable and pick == "custom" else None
    return dmc.Stack([
        dmc.Group([dmc.Text(col, ff="monospace", fw=700, size="lg"),
                   dmc.Badge(f"outcome {way} with it ({entry['direction_from']})", size="xs",
                             variant="outline", color="gray", tt="none")], gap="xs"),
        _checks(entry),
        html.Div([html.Span("Engine recommends ", className="tx-rec-k"),
                  html.B(kinds.get(rec["kind"], rec["kind"])), html.Span(f": {rec['why']}.")],
                 className="tx-rec"),
        dmc.Group(tries, gap=6) if editable else None,
        custom,
        graph(fig),
        dmc.Text(note, size="xs", c="dimmed"),
        confirm,
        dmc.Text("Try a kind to see what it does; the dotted line is the variable as explored. "
                 "The preview reads every row explore has, so nothing it places is kept: the "
                 "parameters are fitted at the design gate, on the development sample.",
                 size="xs", c="dimmed") if editable else None,
    ], gap="sm")


def _params(spec: dict, kind: str, editable: bool, fitted: list[float] | None = None) -> Any:
    """Every parameter control, each kind's shown only for that kind (``visible``). Knots a
    person gave fill the box; knots the fit placed are only its placeholder, so keeping the
    settings does not pin them."""
    knots = ", ".join(f"{k:.4g}" for k in spec.get("knots") or [])
    hint = ("placed at " + ", ".join(f"{k:.4g}" for k in fitted)) if fitted else "e.g. 2.5, 6"
    groups = {
        "knee": dmc.Group([
            dmc.NumberInput(id="txp-knees", label="Knees at most", min=1, max=2, step=1,
                            value=spec.get("knees", 2), w=120, size="xs", disabled=not editable),
            dmc.TextInput(id="txp-knots", label="Knots (blank: placed by the fit)", value=knots,
                          placeholder=hint, w=220, size="xs", disabled=not editable),
        ], gap="sm", align="flex-end"),
        "smooth": dmc.Stack([
            dmc.Text("Share of rows each local fit reads", size="xs", fw=500),
            dmc.Slider(id="txp-span", min=0.1, max=0.9, step=0.05, value=spec.get("span", 0.5),
                       w=280, size="sm", disabled=not editable,
                       marks=[{"value": v, "label": f"{v:.0%}"} for v in (0.2, 0.5, 0.8)]),
        ], gap=4, mb="md"),
        "cap": dmc.Group([
            dmc.NumberInput(id="txp-lower", label="Lower percentile", min=0, max=20, step=0.5,
                            value=spec.get("lower", 1.0), w=130, size="xs",
                            disabled=not editable),
            dmc.NumberInput(id="txp-upper", label="Upper percentile", min=80, max=100, step=0.5,
                            value=spec.get("upper", 99.0), w=130, size="xs",
                            disabled=not editable),
        ], gap="sm"),
        "change": dmc.Select(id="txp-op", label="Which change", size="xs", w=220,
                             value=spec.get("op", "diff"), disabled=not editable,
                             data=[{"value": "diff", "label": "on the period before"},
                                   {"value": "pct", "label": "growth on the period before"},
                                   {"value": "yoy_diff", "label": "on a year before"},
                                   {"value": "yoy_pct", "label": "growth on a year before"}]),
        "custom": dmc.TextInput(id="txp-expr", label="Expression of x", size="xs", w=300,
                                value=spec.get("expr", ""), disabled=not editable,
                                placeholder="e.g. np.minimum(x, 3)"),
        "lag": dmc.NumberInput(id="txp-lag", label="Periods behind", min=0, max=8, step=1,
                               value=spec.get("lag", 0), w=120, size="xs",
                               disabled=not editable),
    }
    shown = visible(kind)
    return html.Div([html.Div(g, id=f"txp-p-{name}", style=shown[name])
                     for name, g in groups.items()])


def visible(kind: str | None) -> dict[str, dict]:
    """Which parameter groups a kind shows."""
    show = {"knee": ("knee",), "smooth": ("smooth",), "cap": ("cap",),
            "change": ("change", "lag"), "lag": ("lag",), "custom": ("custom",)}.get(kind or "", ())
    return {g: ({} if g in show else {"display": "none"}) for g in PARAM_GROUPS}


PARAM_GROUPS = ("knee", "smooth", "cap", "change", "lag", "custom")


def design_detail(col: str | None, data: dict, decided: dict, scheme: str, *,
                  editable: bool, preview: dict | None = None) -> Any:
    """Step two, one variable: the transformation fitted on the development sample, its
    parameters and a refit that redraws only the chart."""
    if not col or col not in (data.get("columns") or {}):
        return empty("Choose a variable on the left.")
    entry = data["columns"][col]
    kinds = data.get("kinds") or {}
    spec = {**(entry.get("spec") or {}), **(decided.get(col) or {})}
    if preview is not None:  # the last refit of this variable, so chart and controls agree
        entry = preview
        spec = {**(preview.get("spec") or {}), **(decided.get(col) or {})}
    kind = data["columns"][col].get("kind") or spec.get("kind")  # decided at the data gate
    return dmc.Stack([
        dmc.Group([dmc.Text(col, ff="monospace", fw=700, size="lg"),
                   dmc.Badge("your settings" if col in decided else "engine's settings",
                             size="xs", variant="light",
                             color="oxblood" if col in decided else "gray")], gap="xs"),
        dmc.Group([
            dmc.Text("Transformation", size="xs", c="dimmed"),
            dmc.Text(kinds.get(kind, kind), size="sm", fw=600),
            dmc.Badge("decided at the data gate", size="xs", variant="outline", color="gray",
                      leftSection=icon("tabler:lock", 10)),
        ], gap="xs"),
        dcc.Loading(html.Div(fit_view(entry, scheme), id="txp-view"), type="dot",
                    color="var(--mantine-color-oxblood-6)", delay_show=250),
        _params(spec, kind, editable,
                fitted=entry.get("knots") if entry.get("origin") == "fitted" else None),
        dmc.Group([
            dmc.Button("Refit", id=opt("txp-refit"), size="xs", variant="light",
                       leftSection=icon("tabler:refresh", 14)),
            dmc.Button("Keep these settings", id=opt("txp-keep"), size="xs",
                       leftSection=icon("tabler:check", 14)),
        ], gap="xs") if editable else None,
        dmc.Text("Fitted on the development sample only, cut as the model stage will cut it; "
                 "the test sample and the holdout are not read. Refit redraws this chart; "
                 "nothing is recorded until the design gate is decided. To use another "
                 "transformation, reopen the data gate.", size="xs", c="dimmed"),
    ], gap="sm")


def fit_view(entry: dict | None, scheme: str, error: str | None = None) -> list:
    """The chart and its caption for one fitted transformation (or why it could not fit)."""
    if error or not entry or entry.get("failed"):
        return [dmc.Alert(error or (entry or {}).get("caption") or "Could not be fitted.",
                          color="red", variant="light", icon=icon("tabler:alert-circle"))]
    fig = charts.shape(entry, scheme, column=entry.get("axis") or "",
                       rate=_is_rate(entry), height=300)
    return [graph(fig), dmc.Text(entry.get("caption") or "", size="xs", c="dimmed")]


def _is_rate(entry: dict) -> bool:
    values = [o[1] for o in entry.get("observed") or []] + [c[1] for c in entry.get("curve") or []]
    return bool(values) and all(0 <= v <= 1 for v in values)


def spec_from(kind: str, knees: Any, knots: str | None, span: Any, lower: Any, upper: Any,
              op: str | None, lag: Any, expr: str | None = None) -> dict[str, Any]:
    """The parameters as the controls state them, for the kind chosen (knots parsed)."""
    spec: dict[str, Any] = {"kind": kind}
    if kind == "knee":
        spec["knees"] = int(knees or 2)
        spec["knots"] = sorted(float(k) for k in str(knots or "").replace(";", ",").split(",")
                               if k.strip())
    elif kind == "smooth":
        spec["span"] = float(span or 0.5)
    elif kind == "cap":
        spec["lower"], spec["upper"] = float(lower or 1.0), float(upper or 99.0)
    elif kind in ("change", "lag"):
        spec["lag"] = int(lag or 0)
        if kind == "change":
            spec["op"] = op or "diff"
    elif kind == "custom":
        spec["expr"] = " ".join(str(expr or "").split())
    return spec
