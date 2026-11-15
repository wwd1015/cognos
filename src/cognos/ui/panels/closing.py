"""The comply, document and review pages."""

from __future__ import annotations

import dash_mantine_components as dmc

from ... import service
from ...artifacts import StageResult
from ...engine.process import preparation_of
from ...engine.state import RunState
from ..components import (
    findings_table,
    fmt,
    icon,
    kpis,
    markdown,
    recommendation_card,
    section,
    table,
    verdict_badge,
)
from ._shared import _act, _findings, _rec
from .gate import gate_block


def comply_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    p = res.payload or {}
    r = p.get("readiness") or {}
    sr = p.get("sr26_2") or p.get("sr11_7") or {}  # sr11_7: recorded before SR 26-2
    nist = p.get("nist_ai_rmf") or {}
    status_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    return [
        dmc.Alert(r.get("narrative", ""), title=f"Readiness: {str(r.get('level', '—')).replace('_', ' ')}",
                  color="blue", variant="light", icon=icon("tabler:clipboard-check")),
        recommendation_card(_rec(res), extra=dmc.List([dmc.ListItem(a["statement"])
                                                       for a in r.get("priority_actions", [])], size="sm")
                            if r.get("priority_actions") else None),
        section("SR 26-2 evidence", table(["Element", "Status", "Evidence"], [
            [k.replace("_", " "), dmc.Badge(e["status"], color=status_color.get(e["status"], "gray"),
                                            variant="light", size="sm"),
             dmc.Text(e["evidence"], size="xs")] for k, e in sr.items()])),
        section("NIST AI RMF", table(["Function", "Status", "Evidence"], [
            [k.upper(), dmc.Badge(e["status"], color=status_color.get(e["status"], "gray"),
                                  variant="light", size="sm"), dmc.Text(e["evidence"], size="xs")]
            for k, e in nist.items()])),
        section("Outstanding human steps", dmc.List([dmc.ListItem(s)
                                                    for s in p.get("outstanding_human_steps", [])],
                                                   size="sm")),
    ]


def document_panel(res: StageResult, state: RunState, scheme: str, _seat: str = "developer") -> list:
    root = None  # the app points service.runs_root at its runs dir (COGNOS_RUNS_DIR)
    tabs = [("narrative", "Narrative", "docs/narrative.md"),
            ("decisions", "Decision log", "docs/decisions.md"),
            ("card", "Model card", "docs/model_card.md"),
            ("paper", "White paper", "stages/document/whitepaper.md")]
    p = res.payload or {}
    return [
        kpis([("Concepts", fmt(p.get("n_concepts")), "OKF bundle"),
              ("Code anchors", fmt(len(p.get("code_links", []))), "verified by review"),
              ("Bundle", "docs/", None)]),
        recommendation_card(_rec(res)),
        dmc.Tabs([
            dmc.TabsList([dmc.TabsTab(label, value=key) for key, label, _ in tabs]),
            *[dmc.TabsPanel(dmc.Paper(markdown(_strip_frontmatter(
                service.read_artifact(state.run_id, path, root) or "_Not written yet._")),
                p="lg", withBorder=True, mt="sm"), value=key) for key, _, path in tabs],
        ], value="narrative", variant="outline"),
    ]


def review_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    gate = gate_block("gate_signoff", state, res, seat=seat,
                      proposal=f"Review verdict {res.verdict.value}. Approving seals the package "
                               "of recorded facts; it is not deployment.",
                      evidence=[("code anchors", fmt(len(p.get("code_links", [])))),
                                ("findings", fmt(len(res.findings)))], primary=[
        dmc.Button("Approve & sign off", id=_act("gate_signoff", "approve"), color="green",
                   leftSection=icon("tabler:signature"), disabled=res.verdict.value == "BLOCK"),
        dmc.Button("Reject", id=_act("gate_signoff", "reject"), color="red", variant="light",
                   leftSection=icon("tabler:x")),
    ], send_back=False)
    sealed = state.package
    package_note = None
    if sealed is not None and sealed.status == "sealed":
        prep = " Express prepared the analysis." if sealed.preparation == "express" else ""
        package_note = dmc.Alert(
            f"Package v{sealed.version} is sealed ({sealed.digest[:12]}).{prep} "
            "A later edit supersedes it; the file is not rewritten. Approval is not deployment.",
            color="green", variant="light", icon=icon("tabler:rosette-discount-check"))
    elif state.status == "completed" and sealed is None:
        express = state.mode == "autonomous" or preparation_of(state) == "express"
        package_note = dmc.Alert(
            ("Express finished this analysis and accepted the gates. That is not a signature. "
             if express else
             "This analysis finished without a sealed package. ")
            + "An approver seals a package from here, once the design brief is answered.",
            color="yellow", variant="light", icon=icon("tabler:signature"))
    return [
        dmc.Group([verdict_badge(res.verdict.value, "lg"),
                   dmc.Text(res.summary, size="sm")], gap="sm"),
        package_note,
        kpis([(k.replace("_", " ").title(), fmt(v), None) for k, v in list(res.metrics.items())[:4]])
        if res.metrics else None,
        gate,
        section("Consistency findings", findings_table(_findings(res))),
        section("Checked anchors", dmc.Code("\n".join(p.get("checked_links", [])[:40]) or "—",
                                            block=True)) if p.get("checked_links") else None,
    ]


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            return text[end + 4:].lstrip()
    return text
