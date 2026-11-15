"""One panel per stage: the engine's evidence beside the agent's recommendation, and the human gate.

Every panel is a pure function of (stage result, run state, color scheme, run id) so it can be
rendered and tested without a browser. Gate controls carry pattern-matching ids
(``{"type": "gate-field" | "gate-act", "gate": ..., ...}``) that one callback in ``app.py`` reads.
"""

# ruff: noqa: F401 (the imports from the page modules are re-exports)

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc

from ...artifacts import StageResult
from ...engine.graph import LABELS
from ...engine.process import shown_status
from ...engine.state import RunState
from ..components import empty, icon, step_badge, verdict_badge

# Every name below was reachable as ``cognos.ui.panels.<name>`` when the panels were one module,
# and still is: the stage pages import from here.
from ._shared import (
    _act,
    _field,
    _findings,
    _rec,
    _summary_line,
    _who,
    _words,
)
from .backtest import (
    _champion_evidence,
    _champion_proposal,
    _forecast_spec,
    _unmoved,
    _unmoved_note,
    backtest_panel,
)
from .changes import (
    _change_badge,
    _delta,
    _value,
    compare_page,
    rerun_note,
)
from .closing import (
    _strip_frontmatter,
    comply_panel,
    document_panel,
    review_panel,
)
from .explore import (
    _RELATION,
    SOURCES,
    _link_chip_text,
    _link_label,
    _step,
    analysis_card,
    analysis_source,
    chip_class,
    explore_panel,
    join_canvas,
    linking_section,
    source_badge,
    source_counts,
    tools_section,
)
from .gate import (
    _send_back,
    gate_block,
)
from .ideate import (
    design_transforms,
    ideate_panel,
)
from .intake import (
    _BASIS,
    intake_panel,
)
from .model import (
    _sign_badge,
    model_funnel_section,
    model_panel,
    transforms_section,
)
from .questions import (
    GAP_OUTCOMES,
    GAP_STATUS,
    SPONSOR_NOTE,
    _open_first,
    challenge_rows,
    gap_cell,
    gap_need,
    gap_outcomes,
    gap_required,
    questions_summary,
    stage_challenges,
    stage_questions,
)
from .source import (
    FIT,
    _fit_badge,
    _fit_reasons,
    _grain,
    _src_upload,
    catalog_origin,
    catalog_results,
    catalog_rows,
    guide_text,
    source_form,
    source_panel,
)
from .validate import (
    trials_section,
    validate_panel,
)

STAGE_BLURB = {
    "intake": "The engine reads the business intent and, for an update, the existing model's "
              "artifacts; the Intake agent fills the brief and interviews the sponsor where the "
              "goal is not clear.",
    "explore": "The engine fetches and profiles the data and runs the analyses the Data agent "
               "asks for; the agent names the dependent variable, the features worth "
               "considering and the columns that may not be inputs.",
    "ideate": "The engine assesses structure and frameworks; the Design agent ranks the slate and "
              "raises what only the sponsor can decide.",
    "model": "The engine follows the development plan: every factor fitted alone, the library "
             "cut to a shortlist, multi-factor models built by the plan's methods and ranked. The "
             "Modeling agent recommends the pool of candidates that goes on; no champion is chosen here.",
    "backtest": "The engine scores every pool member on the test sample and under each scenario; "
                "the Backtest agent reads the comparison and recommends the champion.",
    "validate": "The engine's rubric re-derives risk from artifacts; the Validation agent adds "
                "an effective challenge — it never sees the Modeling agent's reasoning.",
    "comply": "A non-gating readiness report: evidence organized, human-only steps listed.",
    "document": "The white paper as an OKF bundle. The Writing agent drafts prose; every number "
                "is rendered from a recorded fact.",
    "review": "The engine verifies every docs↔code anchor; stale references BLOCK.",
}


PANELS = {"intake": intake_panel, "explore": explore_panel, "ideate": ideate_panel, "model": model_panel,
          "backtest": backtest_panel, "validate": validate_panel, "comply": comply_panel,
          "document": document_panel, "review": review_panel}


# Pages that end on a choice among things already measured: the decision is read first and
# the evidence sits behind tabs, one open at a time. A page whose decision is a form filled
# step by step (intake, explore, ideate) keeps its own order.
DECISION_FIRST = frozenset({"model", "backtest", "validate", "comply", "document", "review"})
NOTE_TAB = "The agent's note"


def _has_class(component: Any, name: str) -> bool:
    return name in (getattr(component, "className", None) or "").split()


def decision_first(body: list) -> list:
    """Lay a stage page out with the decision on top: the gate, the figure strip, anything
    that is neither, then the agent's note and every section as tabs. Nothing is dropped and
    nothing is added; a page with fewer than two things to file keeps its order."""
    gate = [b for b in body if _has_class(b, "cognos-gate")]
    figures = [b for b in body if _has_class(b, "ledger-kpis")]
    filed = [(NOTE_TAB, b) for b in body if _has_class(b, "cognos-rec")] + [
        (str(getattr(b, "data-title", "") or "More"), b) for b in body
        if _has_class(b, "ledger-section")]
    if len(filed) < 2:
        return body
    placed = {id(b) for b in (*gate, *figures, *(b for _, b in filed))}
    loose = [b for b in body if id(b) not in placed]
    values = [f"t{i}" for i in range(len(filed))]
    tabs = dmc.Tabs([
        dmc.TabsList([dmc.TabsTab(title, value=v) for v, (title, _) in zip(values, filed, strict=True)]),
        *[dmc.TabsPanel(b, value=v, pt="md") for v, (_, b) in zip(values, filed, strict=True)],
    ], value=values[0], className="stage-tabs")
    return [*gate, *figures, *loose, tabs]


def stage_panel(stage: str, res: StageResult | None, state: RunState, scheme: str,
                changes: dict | None = None, seat: str = "developer",
                data_info: dict | None = None) -> list:
    """The full panel for a stage, including running/failed/pending placeholders. ``changes`` is
    ``service.step_changes`` for this stage (what its last re-run changed), when there is one.
    ``data_info`` is what explore's page says about the data (:func:`source_panel`)."""
    status = state.status_of(stage)
    before = source_panel(state, seat, data_info) if stage == "explore" else None
    head = dmc.Group([
        dmc.Stack([dmc.Title(LABELS[stage], order=3),
                   dmc.Text(STAGE_BLURB[stage], size="sm", c="dimmed", maw=760)], gap=2),
        dmc.Group([verdict_badge(res.verdict.value) if res is not None and status != "running"
                   else None, step_badge(shown_status(state, stage), "md")], gap="xs"),
    ], justify="space-between", align="flex-start")
    if status == "failed":
        msg = state.steps[stage].message or "The step failed."
        return [head, before, dmc.Alert(msg, title="This step failed", color="red",
                                        variant="light", icon=icon("tabler:alert-circle")),
                dmc.Group([dmc.Button("Retry", id={"type": "retry", "step": stage},
                                      leftSection=icon("tabler:refresh"))])]
    if res is None or status in ("pending", "running") and res is None:
        body = (dmc.Stack([dmc.Group([dmc.Loader(size="sm", type="dots"),
                                      dmc.Text("Working — the agent's recommendation and the "
                                               "engine's evidence appear here when it finishes.",
                                               size="sm", c="dimmed")]),
                           dmc.Skeleton(h=90, radius="md"), dmc.Skeleton(h=220, radius="md")])
                if status == "running" else empty(
                    "Starts once the data is given." if state.status_of("gate_source") == "awaiting"
                    else "Not started yet."))
        return [head, before, body]
    notice = None
    if status == "running":
        notice = dmc.Alert("Re-running with new inputs; the evidence below is from the previous run.",
                           color="blue", variant="light", icon=icon("tabler:loader-2"))
    elif status == "stale":
        why = state.steps[stage].rerun_reason
        notice = dmc.Alert(["This stage will re-run because — ", dmc.Text(why, span=True, fw=600)] if why
                           else "An earlier input changed; this stage will re-run.",
                           title="Out of date: the evidence below predates that decision" if why else None,
                           color="yellow", variant="light", icon=icon("tabler:refresh-alert"))
    else:
        notice = rerun_note(stage, state, changes)
    body = [b for b in [*PANELS[stage](res, state, scheme, seat),
                        *tools_section(res, state, scheme), stage_challenges(stage, state)]
            if b is not None]
    if stage in DECISION_FIRST:
        body = decision_first(body)
    return [head, before, notice, stage_questions(stage, state, seat), *body]
