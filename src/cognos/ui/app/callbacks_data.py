"""Callbacks of "Provide the data" and the join canvas."""

from __future__ import annotations

import json

import dash_mantine_components as dmc
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, no_update

from ... import service
from ..panels import (
    _RELATION,
    _link_chip_text,
    catalog_results,
    catalog_rows,
)
from .runs import saved_uploads, upload_names


def register(app: Dash) -> None:
    # The join canvas (panels.join_canvas, assets/join_canvas.js) keeps the join in the data
    # gate's join_spec field. A link drawn on it is measured here, on the run's sources,
    # before it can be used; the decision measures it again when it is sent.
    @app.server.route("/api/measure-link", methods=["POST"])
    def measure_link_route():
        from flask import jsonify, request

        q = request.get_json(silent=True) or {}
        try:
            lk = service.measure_link(str(q.get("run") or ""), str(q.get("a") or ""),
                                      list(q.get("a_on") or []), str(q.get("b") or ""),
                                      list(q.get("b_on") or []))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"ok": False, "error": str(exc)})
        return jsonify({"ok": True, "chip": _link_chip_text({**lk, "id": ""}),
                        "match": f"{max(lk['a_in_b'], lk['b_in_a']):.0%}",
                        "rel": _RELATION.get(lk["relation"], lk["relation"]),
                        "low": max(lk["a_in_b"], lk["b_in_a"]) < 0.5})

    # The catalog picker on "Provide the data" (panels.source_form): search, add, remove and
    # make base change the chosen list; the list is drawn again with each table's fit.
    @app.callback(Output("src-results", "children"),
                  Input("src-search", "value"), Input("src-browse", "n_clicks"),
                  Input("src-pick", "data"), State("ws-run", "data"))
    def src_results(text, browse, pick, run_id):
        text = (text or "").strip()
        if not text and not (browse or 0) % 2:
            return []
        try:
            found = service.catalog_search(text, run_id, limit=12 if text else 200)
        except Exception as exc:  # the catalog source failed: say so in the form
            return [dmc.Alert(f"The catalog cannot be searched: {exc}", color="yellow",
                              variant="light")]
        return catalog_results(found, list(pick or []), text=text)

    @app.callback(Output("src-pick", "data"),
                  Input({"type": "src-add", "table": ALL}, "n_clicks"),
                  Input({"type": "src-remove", "table": ALL}, "n_clicks"),
                  Input({"type": "src-base", "table": ALL}, "n_clicks"),
                  State("src-pick", "data"), prevent_initial_call=True)
    def src_pick(_add, _remove, _base, pick):
        trig = ctx.triggered_id
        if not trig or not (ctx.triggered and ctx.triggered[0].get("value")):
            return no_update
        pick, tid = list(pick or []), trig["table"]
        if trig["type"] == "src-add" and tid not in pick:
            return [*pick, tid]
        if trig["type"] == "src-remove":
            return [t for t in pick if t != tid]
        if trig["type"] == "src-base" and tid in pick:
            return [tid, *(t for t in pick if t != tid)]
        return no_update

    @app.callback(Output("src-chosen", "children"),
                  Output({"type": "gate-field", "gate": "gate_source", "field": "catalog"},
                         "value", allow_duplicate=True),
                  Input("src-pick", "data"), State("ws-run", "data"),
                  prevent_initial_call=True)
    def src_chosen(pick, run_id):
        pick = list(pick or [])
        try:
            rows = catalog_rows(service.catalog_cards(pick), service.catalog_fit(run_id, pick))
        except Exception as exc:
            rows = [dmc.Alert(f"The catalog cannot be read: {exc}", color="yellow",
                              variant="light")]
        return rows, json.dumps(pick)

    @app.callback(Output({"type": "gate-field", "gate": "gate_source", "field": MATCH}, "value"),
                  Output({"type": "src-names", "field": MATCH}, "children"),
                  Input({"type": "src-upload", "field": MATCH}, "contents"),
                  State({"type": "src-upload", "field": MATCH}, "filename"),
                  prevent_initial_call=True)
    def keep_source_upload(contents, names):
        """Save dropped data files at once; the decision carries their paths."""
        if not contents:
            return no_update, no_update
        return json.dumps(saved_uploads(contents, names)), upload_names(names)
