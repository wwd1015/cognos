"""What a re-run changed, and run against run comparison."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import dcc

from ...engine.graph import LABELS
from ...engine.state import RunState
from ..components import (
    empty,
    fmt,
    icon,
    run_badge,
    section,
    table,
    verdict_badge,
)


def _change_badge(better: str | None) -> Any:
    """How a number moved from A (or the previous run of a step) to B. Icon and word, never color
    alone; no badge where no direction is claimed."""
    if better == "b":
        return dmc.Badge("Improved", color="green", variant="light", size="sm",
                         leftSection=icon("tabler:trending-up", 12))
    if better == "a":
        return dmc.Badge("Worse", color="orange", variant="light", size="sm",
                         leftSection=icon("tabler:trending-down", 12))
    return None


def _delta(v: Any) -> str:
    return "—" if v is None else "0" if v == 0 else f"{v:+,.4f}".rstrip("0").rstrip(".")


def _value(v: Any) -> Any:
    if isinstance(v, list | tuple):
        return ", ".join(str(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return dmc.Code(json.dumps(v, default=str))
    return fmt(v)


def rerun_note(stage: str, state: RunState, changes: dict | None) -> Any:
    """Why this stage ran again and what that changed, once it has. None on a first run."""
    step = state.steps[stage]
    if not step.rerun_reason or step.status in ("stale", "running", "pending") or step.runs < 2:
        return None
    head = dmc.Text(["Ran again because — ", dmc.Text(step.rerun_reason, span=True, fw=600)], size="sm")
    if changes is None:
        return dmc.Alert(head, color="gray", variant="light", icon=icon("tabler:history"))
    if changes["unchanged"]:
        body = dmc.Text("The re-run changed nothing here: same verdict, same metrics.", size="sm", c="dimmed")
    else:
        moved = [r for r in changes["metrics"] if r["delta"] != 0]
        lines = []
        if changes["champion"]:
            lines.append(dmc.Text(["Champion: ", dmc.Code(str(changes["champion"]["a"])), " → ",
                                   dmc.Code(str(changes["champion"]["b"]))], size="sm"))
        if changes["verdict"]["a"] != changes["verdict"]["b"]:
            lines.append(dmc.Group([dmc.Text("Verdict:", size="sm"), verdict_badge(changes["verdict"]["a"], "sm"),
                                    dmc.Text("→", size="sm"), verdict_badge(changes["verdict"]["b"], "sm")], gap=6))
        f = changes["findings"]
        if f["new"] or f["gone"]:
            lines.append(dmc.Text(f"Findings: {len(f['new'])} new, {len(f['gone'])} no longer raised.", size="sm"))
        body = dmc.Stack([*lines, table(["Metric", "Before", "After", "Change", ""], [
            [r["label"], _value(r["a"]), _value(r["b"]), _delta(r["delta"]), _change_badge(r["better"])]
            for r in moved]) if moved else None], gap="xs")
    return dmc.Alert(dmc.Stack([head, body], gap="xs"), color="gray", variant="light",
                     icon=icon("tabler:history"), title="What the re-run changed")


def compare_page(c: dict) -> list:
    """Run B against run A: what was decided differently, and what it did to the results. A pure
    function of ``service.compare``'s dictionary."""
    def card(tag: str, r: dict) -> Any:
        return dmc.Card([
            dmc.Group([dmc.Badge(tag, variant="filled", color="dark", size="lg", radius="sm"),
                       dmc.Text(r["project"], fw=650), run_badge(r["status"], "sm")], gap="xs"),
            dcc.Link(dmc.Text(r["run_id"], size="xs", ff="monospace"), href=f"/run/{r['run_id']}"),
            dmc.Text(["Champion ", dmc.Code(str(r["champion"] or "—"))], size="sm", mt="xs"),
            dmc.Text(f"{r['mode']} · agents: {r['provider']}" + (f" · metric: {r['metric']}" if r["metric"] else ""),
                     size="xs", c="dimmed", mt=4),
        ], p="md")

    a, b = c["a"], c["b"]
    head = dmc.Group([
        dmc.Stack([dmc.Title("Compare runs", order=2),
                   dmc.Text("B relative to A. Every number is read from the two runs' recorded results; "
                            "“improved” is only claimed where the metric has a direction.",
                            size="sm", c="dimmed")], gap=2),
        dcc.Link(dmc.Button("Swap A and B", variant="subtle", leftSection=icon("tabler:arrows-exchange")),
                 href=f"/compare/{b['run_id']}/{a['run_id']}"),
    ], justify="space-between", align="flex-start")

    decided = [[d["label"], *[
        "—" if x is None else dmc.Stack([dmc.Text(f"{x['action'].replace('_', ' ')} · {'you' if x['actor'] == 'human' else 'auto'}", size="sm"),
                                         dmc.Text(x["reason"], size="xs", c="dimmed") if x["reason"] else None], gap=0)
        for x in (d["a"], d["b"])]] for d in c["decisions"] if d["different"]]
    decided += [[dmc.Stack([dmc.Text(o["what"], size="sm"), dmc.Text(o["note"], size="xs", c="dimmed") if o["note"] else None], gap=0),
                 _value(o["a"]), _value(o["b"])] for o in c["overrides"]]
    decided += [[dmc.Code(k["key"]), _value(k["a"]), _value(k["b"])] for k in c["config"]]

    moved = [r for r in c["metrics"] if r["delta"] != 0]
    same = len(c["metrics"]) - len(moved)
    verdicts = [[v["label"], verdict_badge(v["a"], "sm") if v["a"] else "—", verdict_badge(v["b"], "sm") if v["b"] else "—"]
                for v in c["verdicts"] if v["a"] != v["b"]]
    f = c["findings"]

    def finding_rows(rows: list[dict]) -> list:
        return [[dmc.Badge(r["severity"].title(), variant="light", color="gray", size="sm"), LABELS[r["stage"]],
                 dmc.Text(r["message"], size="sm")] for r in rows[:12]]

    return [
        head,
        *[dmc.Alert(text, color="yellow", variant="light", icon=icon("tabler:alert-triangle")) for text in c["caveats"]],
        dmc.SimpleGrid([card("A", a), card("B", b)], cols={"base": 1, "sm": 2}, spacing="md"),
        section("Decided differently", table(["What", "A", "B"], decided) if decided
                else empty("Nothing: the same decisions, overrides and configuration."),
                description="Gate decisions, the overrides they produced, and the project configuration."),
        section("What it did to the results",
                table(["Stage", "Metric", "A", "B", "Change", ""], [
                    [LABELS[r["stage"]], r["label"], _value(r["a"]), _value(r["b"]), _delta(r["delta"]),
                     _change_badge(r["better"])] for r in moved]) if moved
                else empty("No metric differs between the two runs."),
                dmc.Text(f"{same} metric(s) are identical in both runs.", size="xs", c="dimmed", mt="xs") if same else None,
                description="Only the metrics that differ. Change is B minus A."),
        section("Verdicts", table(["Stage", "A", "B"], verdicts) if verdicts
                else empty("Every stage reached the same verdict in both runs.")),
        section("Findings", dmc.Stack([
            dmc.Text(f"Only in B ({len(f['new'])})", size="sm", fw=600),
            table(["Severity", "Stage", "Finding"], finding_rows(f["new"])) if f["new"] else empty("None."),
            dmc.Text(f"Only in A ({len(f['gone'])})", size="sm", fw=600, mt="sm"),
            table(["Severity", "Stage", "Finding"], finding_rows(f["gone"])) if f["gone"] else empty("None."),
        ], gap=4), description="Matched by stage and finding id."),
    ]
