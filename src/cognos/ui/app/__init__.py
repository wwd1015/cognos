"""The COGNOS workbench — a Dash + Mantine app for model developers.

Two pages: **Runs** (every run, and a drawer to start a new one from a project profile or a
synthetic demo, as a new model development or a model update, with the business intent document
and the existing model's artifacts uploaded there) and the **run workspace** (stage rail, the selected stage's evidence beside its
agent's recommendation, the human gate, live activity, questions & challenges, and the agent audit).

The engine runs in background threads (``cognos.service``); the browser polls run state once a
second and re-renders only what changed, so a refresh never loses anything — all state is on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from dash import Dash

from ... import service
from ...engine import GateError
from .. import design_steps
from ..panels import catalog_rows
from ..theme import FONT_CSS
from . import (
    callbacks_actions,
    callbacks_ask,
    callbacks_data,
    callbacks_gates,
    callbacks_runs,
    callbacks_workspace,
    layout,
    routing,
)
from .callbacks_transforms import register_transform_callbacks
from .layout import build_layout, column_layout, header, notice
from .payloads import gate_payload, plan_patch, source_payload
from .runs import (
    compare_picker,
    delete_modal,
    delete_options,
    deletion_blocked,
    deletion_warning,
    engagement_problem,
    model_choices,
    model_note,
    model_select,
    run_role,
    runs_figures,
    runs_page,
    runs_table,
    runs_tools,
    saved_uploads,
    start_choices,
    start_note,
    started,
    upload_box,
    upload_names,
    where_it_stands,
)
from .workspace import (
    activity,
    audit_tab,
    auto_step,
    backend_label,
    call_label,
    model_runs_section,
    promote_modal,
    questions_tab,
    rail,
    role_fact,
    run_actions,
    run_header,
    stamp,
    step_badge_text,
    template_modal,
    trial_gate_options,
    trial_modal,
    view_lede,
    workspace_page,
)

__all__ = [
    "ASSETS", "GateError", "activity", "audit_tab", "auto_step", "backend_label", "build_layout",
    "call_label", "catalog_rows", "column_layout", "compare_picker", "create_app", "delete_modal",
    "delete_options", "deletion_blocked", "deletion_warning", "engagement_problem",
    "gate_payload", "header", "model_choices", "model_note", "model_runs_section",
    "model_select", "notice", "plan_patch", "promote_modal", "questions_tab", "rail",
    "register_callbacks", "register_transform_callbacks", "role_fact", "run_actions",
    "run_header", "run_role", "run_server", "runs_figures", "runs_page", "runs_table",
    "runs_tools", "saved_uploads", "source_payload", "stamp", "start_choices", "start_note",
    "started", "step_badge_text", "template_modal", "trial_gate_options", "trial_modal",
    "upload_box", "upload_names", "view_lede", "where_it_stands", "workspace_page",
]

ASSETS = Path(__file__).parent.with_name("assets")


# --- app ---------------------------------------------------------------------------------------
def create_app(runs_dir: str | None = None) -> Dash:
    if runs_dir:
        os.environ["COGNOS_RUNS_DIR"] = str(runs_dir)
    app = Dash(__name__, title="COGNOS", suppress_callback_exceptions=True,
               assets_folder=str(ASSETS), update_title=None, external_stylesheets=[FONT_CSS])
    # the tab icon is the logo's mark (docs/brand/cognos-mark.svg)
    app.index_string = app.index_string.replace(
        "{%favicon%}", f'<link rel="icon" type="image/svg+xml" href="{app.get_asset_url("favicon.svg")}">')
    app.layout = build_layout()
    register_callbacks(app)
    return app


def register_callbacks(app: Dash) -> None:
    register_transform_callbacks(app)
    design_steps.register_design_callbacks(app, plan_patch)
    layout.register(app)
    routing.register(app)
    callbacks_runs.register(app)
    callbacks_workspace.register(app)
    callbacks_ask.register(app)
    callbacks_gates.register(app)
    callbacks_data.register(app)
    callbacks_gates.register_answers(app)
    callbacks_actions.register(app)


def run_server(host: str = "127.0.0.1", port: int = 8050, runs_dir: str | None = None,
               debug: bool = False) -> None:
    app = create_app(runs_dir)
    print(f"COGNOS workbench on http://{host}:{port}  (runs: {service.runs_root()})")
    app.run(host=host, port=port, debug=debug)
