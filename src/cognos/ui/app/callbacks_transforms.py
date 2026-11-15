"""Callbacks of the transformation picker (ADR-0024)."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ... import service
from ...modeling import column_transforms as ct
from .. import transform_review


def register_transform_callbacks(app: Dash) -> None:
    """The transformation picker (ui/transform_review.py, ADR-0024) on the Explore page
    (``txd``) and the design page (``txp``). Choosing, deciding and refitting change only the
    list, the pane and the chart: the page is not redrawn and the variable stays in view."""
    tr = transform_review

    def _excluded(prefix: str, value: Any) -> list[str]:
        return list(value or []) if prefix == "txd" else []

    for prefix in ("txd", "txp"):
        field = tr.field_id(prefix)
        exclude = ({"type": "gate-field", "gate": "gate_data", "field": "exclude"}
                   if prefix == "txd" else f"{prefix}-data")
        exclude_prop = "value" if prefix == "txd" else "modified_timestamp"

        @app.callback(Output(f"{prefix}-list", "children"), Output(f"{prefix}-progress", "children"),
                      *([Output("txd-summary", "children")] if prefix == "txd" else []),
                      Input(f"{prefix}-filter", "value"), Input(f"{prefix}-search", "value"),
                      Input(field, "value"), Input(f"{prefix}-sel", "data"),
                      Input(exclude, exclude_prop), State(f"{prefix}-data", "data"),
                      State(f"{prefix}-editable", "data"))
        def draw_list(filt, search, value, sel, excl, data, editable, _p=prefix):
            if not data:
                return (no_update,) * (3 if _p == "txd" else 2)
            decided = tr.decisions(value)
            excluded = _excluded(_p, excl)
            out = (tr.rows(_p, data, decided, selected=sel, filt=filt or "all",
                           search=search or "", excluded=excluded),
                   tr.progress(_p, data, decided, excluded))
            if _p != "txd":
                return out
            if not editable:
                return (*out, no_update)
            return (*out, tr.summary(data, decided, excluded))

        @app.callback(Output(f"{prefix}-sel", "data", allow_duplicate=True),
                      Input({"type": f"{prefix}-row", "col": ALL}, "n_clicks"),
                      prevent_initial_call=True)
        def choose(clicks, _p=prefix):
            trig = ctx.triggered_id
            if not trig or not any(c for c in (clicks or []) if c):
                return no_update
            return trig["col"]

    def _sketch(run_id, col, key, expr):
        """One form previewed on the rows explore has: (sketch, error)."""
        spec = ct.spec_of_key(key, expr)
        if spec["kind"] == "custom" and not (expr or "").strip():
            return None, None
        try:
            return service.sketch_transform(run_id, col, spec["kind"], expr=expr or "",
                                            op=spec["op"], lag=spec["lag"]), None
        except ValueError as exc:
            return None, str(exc)
        except (KeyError, FileNotFoundError):
            return None, None

    def _shown(run_id, col, keys, expr):
        """Each form in ``keys`` with its preview sketch."""
        return [(k, _sketch(run_id, col, k, expr)[0]) for k in keys]

    @app.callback(Output("txd-detail", "children"), Input("txd-sel", "data"),
                  Input(tr.field_id("txd"), "value"), State("txd-data", "data"),
                  State("txd-editable", "data"), State("ws-run", "data"),
                  State("theme-store", "data"))
    def draw_explore(sel, value, data, editable, run_id, theme):
        # The pane for the selected variable, its ticked forms drawn. Ticking and previewing
        # change only the count and the chart, so the ticks stay as they are.
        if not data:
            return no_update
        decided = tr.decisions(value)
        entry = (data.get("columns") or {}).get(sel or "")
        shown, expr = [], ""
        if entry and run_id:
            forms = tr.forms_shown(entry, decided.get(sel))
            expr = next((f.get("expr", "") for f in forms if f["kind"] == "custom"), "")
            keys = tr.drawn_keys([ct.form_key(f) for f in forms]) or [ct.form_key(forms[0])]
            shown = _shown(run_id, sel, keys, expr)
        return tr.explore_detail(sel, data, decided, (theme or {}).get("scheme", "light"),
                                 editable=bool(editable), sketch=shown, expr=expr)

    @app.callback(Output({"type": "txd-preview", "i": ALL}, "children"),
                  Input({"type": "txd-try", "key": ALL}, "n_clicks"),
                  Input({"type": "txd-expr-go", "i": ALL}, "n_clicks"),
                  Input({"type": "txd-form", "key": ALL}, "checked"),
                  State({"type": "txd-expr", "i": ALL}, "value"), State("txd-sel", "data"),
                  State("txd-data", "data"), State("ws-run", "data"),
                  State("theme-store", "data"), prevent_initial_call=True)
    def preview_form(tries, go, checked, exprs, sel, data, run_id, theme):
        # The ticked forms of the column's own axis are drawn together; a lagged or changed
        # form, whose axis is its own, is drawn alone when its name is clicked.
        trig = ctx.triggered_id
        none = [no_update] * len(exprs or [None])
        if not isinstance(trig, dict) or not run_id or not sel:
            return none
        expr = (exprs or [""])[0] or ""
        on = [st["id"]["key"] for st, c in zip(ctx.inputs_list[2], checked or [], strict=False)
              if c]
        if trig["type"] == "txd-try":
            if not any(c for c in (tries or []) if c):
                return none
            key = trig["key"]
            keys = [key] if key not in tr.SAME_AXIS else (
                tr.drawn_keys(on) if key in on else [*tr.drawn_keys(on), key])
        else:
            if trig["type"] == "txd-expr-go" and not any(c for c in (go or []) if c):
                return none
            keys = tr.drawn_keys(on + (["custom"] if trig["type"] == "txd-expr-go" else []))
        entry = (data.get("columns") or {}).get(sel)
        shown, error = [], None
        for k in keys:
            sk, err = _sketch(run_id, sel, k, expr)
            shown.append((k, sk))
            error = error or err
        view = tr.preview_view(sel, entry, data, (theme or {}).get("scheme", "light"), shown,
                               expr)
        if error:
            view = [dmc.Alert(error, color="red", variant="light"), *view]
        return [view]

    @app.callback(Output({"type": "txd-count", "i": ALL}, "children"),
                  Input({"type": "txd-form", "key": ALL}, "checked"),
                  State("txd-sel", "data"), State(tr.field_id("txd"), "value"),
                  prevent_initial_call=True)
    def count_forms(checked, sel, value):
        n = sum(1 for c in checked or [] if c)
        return [tr.count_words(n, sel in tr.decisions(value))]

    @app.callback(Output(tr.field_id("txd"), "value"),
                  Output("txd-sel", "data", allow_duplicate=True),
                  Input({"type": "txd-confirm", "i": ALL}, "n_clicks"),
                  State({"type": "txd-form", "key": ALL}, "checked"),
                  State({"type": "txd-expr", "i": ALL}, "value"),
                  State("txd-sel", "data"), State(tr.field_id("txd"), "value"),
                  State("txd-data", "data"),
                  State({"type": "gate-field", "gate": "gate_data", "field": "exclude"}, "value"),
                  prevent_initial_call=True)
    def decide(clicks, checked, exprs, sel, value, data, excl):
        # The ticked forms become the decision; the next open variable comes up.
        if not any(c for c in (clicks or []) if c) or not sel:
            return no_update, no_update
        expr = " ".join(str((exprs or [""])[0] or "").split())
        keys = [s["id"]["key"] for s, on in zip(ctx.states_list[0], checked or [], strict=False)
                if on]
        forms = [ct.spec_of_key(k, expr) for k in keys if k != "custom" or expr]
        if not forms:
            return no_update, no_update
        decided = {**tr.decisions(value), sel: {"forms": forms}}
        return json.dumps(decided), tr.next_open(data, decided, sel, list(excl or []))

    @app.callback(Output("txp-detail", "children"), Input("txp-sel", "data"),
                  Input(tr.field_id("txp"), "value"), State("txp-data", "data"),
                  State("txp-editable", "data"), State("txp-previews", "data"),
                  State("theme-store", "data"))
    def draw_design(sel, value, data, editable, previews, theme):
        if not data:
            return no_update
        return tr.design_detail(sel, data, tr.decisions(value),
                                (theme or {}).get("scheme", "light"), editable=bool(editable),
                                preview=(previews or {}).get(sel or ""))

    # The kind was decided at the data gate (ADR-0024): read from the fitted entry, not a control.
    _params = [State(f"txp-{n}", "value") for n in ("knees", "knots", "span", "lower",
                                                    "upper", "op", "lag", "expr")]

    def _kind(data, sel):
        return ((data or {}).get("columns") or {}).get(sel or "", {}).get("kind")

    @app.callback(Output("txp-view", "children"), Output("txp-previews", "data"),
                  Input({"type": "txp-refit", "i": ALL}, "n_clicks"), *_params,
                  State("txp-sel", "data"), State("txp-data", "data"),
                  State("txp-previews", "data"),
                  State("ws-run", "data"), State("theme-store", "data"),
                  prevent_initial_call=True)
    def refit(clicks, knees, knots, span, lower, upper, op, lag, expr, sel, data,
              previews, run_id, theme):
        # The Refit button: the chart alone is redrawn.
        if not sel or not run_id or not any(c for c in (clicks or []) if c):
            return no_update, no_update
        kind = _kind(data, sel)
        scheme = (theme or {}).get("scheme", "light")
        try:
            spec = tr.spec_from(kind, knees, knots, span, lower, upper, op, lag, expr)
            entry = service.preview_transform(run_id, sel, spec)
        except ValueError as exc:
            return tr.fit_view(None, scheme, error=str(exc)), no_update
        return tr.fit_view(entry, scheme), {**(previews or {}), sel: entry}

    @app.callback(Output(tr.field_id("txp"), "value"),
                  Output("txp-sel", "data", allow_duplicate=True),
                  Input({"type": "txp-keep", "i": ALL}, "n_clicks"), *_params,
                  State("txp-sel", "data"),
                  State(tr.field_id("txp"), "value"), State("txp-data", "data"),
                  prevent_initial_call=True)
    def keep(clicks, knees, knots, span, lower, upper, op, lag, expr, sel, value, data):
        if not any(c for c in (clicks or []) if c) or not sel:
            return no_update, no_update
        kind = _kind(data, sel)
        try:
            spec = tr.spec_from(kind, knees, knots, span, lower, upper, op, lag, expr)
        except ValueError:
            return no_update, no_update  # unreadable knots: the refit says why
        decided = {**tr.decisions(value), sel: spec}
        return json.dumps(decided), tr.next_open(data, decided, sel)
