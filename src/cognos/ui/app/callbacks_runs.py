"""Callbacks of the Runs page: compare, export, the run list, deleting runs and starting one."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, ctx, dcc, no_update

from ... import service
from .layout import notice
from .runs import (
    delete_options,
    deletion_blocked,
    deletion_warning,
    engagement_problem,
    model_choices,
    model_note,
    runs_figures,
    runs_table,
    runs_tools,
    saved_uploads,
    start_note,
    upload_names,
)


def register(app: Dash) -> None:
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

    @app.callback(Output("new-start-note", "children"), Input("new-start", "value"),
                  prevent_initial_call=True)
    def show_start(ref):
        return start_note(ref, service.list_templates())

    for _box in ("new-intent", "new-support", "new-prior"):
        app.callback(Output(f"{_box}-names", "children"), Input(_box, "filename"))(upload_names)

    @app.callback(Output("new-model", "data"), Output("new-model", "value"),
                  Output("new-model", "description"), Output("new-model-wrap", "style"),
                  Input("new-provider", "value"), prevent_initial_call=True)
    def pick_provider(provider):
        providers = service.provider_list()
        data, value = model_choices(providers, provider)
        return data, value, model_note(providers, provider), {} if data else {"display": "none"}

    @app.callback(Output("url", "pathname"), Output("new-feedback", "children"),
                  Input("new-create", "n_clicks"),
                  State("new-start", "value"), State("new-mode", "value"),
                  State("new-provider", "value"), State("new-kind", "value"),
                  State("new-intent", "contents"), State("new-intent", "filename"),
                  State("new-support", "contents"), State("new-support", "filename"),
                  State("new-prior", "contents"), State("new-prior", "filename"),
                  State("new-prior-run", "value"), State("new-name", "value"),
                  State("new-model", "value"), prevent_initial_call=True)
    def create(clicks, start, mode, provider, kind, intent, intent_name, support,
               support_names, prior, prior_names, prior_run, name=None, model=None):
        if not clicks:  # the button was just rendered, not clicked
            return no_update, no_update
        try:
            if not (name or "").strip():
                return no_update, dmc.Alert("Name the model: it is how its runs are found.",
                                            color="yellow", variant="light")
            cfg = service.template_config(None if start in (None, "scratch") else start, name)
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
            choices, _ = model_choices(service.provider_list(), provider)
            run_id = service.create_run(cfg, mode=mode, provider=provider,
                                        model=model if choices else None, engagement=engagement)
            service.start(run_id)
        except Exception as exc:  # show, don't crash the page
            return no_update, dmc.Alert(str(exc), color="red", variant="light")
        return f"/run/{run_id}", None
