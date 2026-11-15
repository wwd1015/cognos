"""The model page: the pool of candidates."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import html

from ... import service
from ...artifacts import StageResult
from ...engine.state import RunState
from .. import charts, transform_review
from ..components import (
    chips,
    findings_table,
    fmt,
    graph,
    icon,
    kpis,
    recommendation_card,
    section,
    table,
)
from ._shared import _act, _field, _findings, _rec, _who, _words
from .gate import gate_block


def _sign_badge(sign: str, expected: str | None) -> Any:
    if sign not in ("+", "-"):
        return dmc.Text("—", size="sm", c="dimmed")
    if expected in ("+", "-") and expected != sign:
        return dmc.Badge(f"{sign}  (expected {expected})", color="red", variant="light", size="sm",
                         leftSection=icon("tabler:alert-triangle", 12))
    return dmc.Text(sign, size="sm", ff="monospace")


def model_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    plan = p.get("plan") or {}
    first = (p.get("rank_by") or [{}])[0]
    score_label = f"{first.get('metric', 'score')} ({_words(first.get('sample', 'validation'))})"
    ledger = json.loads(service.read_artifact(state.run_id, "stages/model/ledger.json",
                                              None) or "[]")
    rec = _rec(res) or {}
    out = rec.get("output") or {}
    recommended = set(out.get("pool") or [])
    pool_ids = list(p.get("pool_ids") or [])
    leaders = p.get("leaders") or []
    expected = p.get("expected_signs") or {}

    def tags(c: dict) -> Any:
        return dmc.Group([
            dmc.Badge("in the pool", color="oxblood", size="xs") if c["id"] in pool_ids else None,
            dmc.Badge("Modeling agent", color="blue", size="xs", variant="light")
            if c["id"] in recommended else None,
            dmc.Badge("within 1 SE", color="gray", size="xs", variant="outline")
            if c.get("within_one_se") else None,
            dmc.Badge("challenger only", color="grape", size="xs", variant="light")
            if c.get("role") == "challenger" else None,
            dmc.Badge(f"{len(c['violations'])} constraint(s) broken", color="red", size="xs",
                      variant="light", leftSection=icon("tabler:alert-triangle", 12))
            if c.get("violations") else None], gap=4)

    leader_rows = [[c.get("rank") or "", _who(c),
                    dmc.Text(c["family"], ff="monospace", size="sm"),
                    chips(c["features"]),
                    fmt(c.get("score")), dmc.Text(" + ".join(_words(m) for m in c.get("methods") or []),
                                                  size="xs"),
                    tags(c)] for c in leaders]
    pool_rows = [[_who(m), dmc.Text(m["label"], size="sm"),
                  chips(m["features"]),
                  fmt(m.get("score")), fmt(m.get("max_p_value"), 3),
                  f"{m['diagnostics_passed']}/{m['diagnostics_run']}",
                  dmc.Text("; ".join(m.get("violations") or []) or "none", size="xs")]
                 for m in p.get("pool") or []]
    signs = [[dmc.Text(s["feature"], ff="monospace", size="sm"), s["expected"], s["observed"],
              dmc.Badge(s["assessment"].replace("_", " "),
                        color={"consistent": "green", "wrong_sign": "red"}.get(s["assessment"], "gray"),
                        variant="light", size="sm")]
             for s in out.get("sign_checks", [])]
    factor_rows = [[dmc.Text(r["name"], ff="monospace", size="sm"),
                    dmc.Text(_words(r.get("transform", "level"))
                             + (f", lag {r['lag']}" if r.get("lag") else ""), size="xs"),
                    _sign_badge(r.get("sign", "n/a"), expected.get(r["name"])
                                or expected.get(r["source"])),
                    fmt(r.get("p_value"), 4), fmt(r.get("in_sample")), fmt(r.get("validation")),
                    dmc.Text(r.get("reason") or "", size="xs")]
                   for r in p.get("single_factor") or []]
    screen_rows = [[_words(r["rule"]), fmt(r["setting"]) if not isinstance(r["setting"], list)
                    else ", ".join(r["setting"]), fmt(r["before"]), fmt(r["after"]),
                    fmt(r["before"] - r["after"])] for r in p.get("screen") or []]
    path_blocks = []
    for path in p.get("paths") or []:
        steps = [[s["step"], s["action"], dmc.Text(s.get("factor") or "", ff="monospace", size="sm"),
                  chips(s["factors"]), fmt(s.get("criterion")),
                  f"{s['change']:+.4g}" if s.get("change") is not None else "",
                  f"{s['tried']} tried, {s['rejected_by_constraints']} rejected"]
                 for s in path.get("steps") or []]
        path_blocks.append(dmc.Stack([
            dmc.Group([dmc.Text({"grid": "Families without a stepwise path",
                                 "same_factors": "Same factors, another family"}.get(
                                     path["method"], _words(path["method"]).capitalize()),
                                fw=650, size="sm"),
                       dmc.Badge(path["family"], variant="light", color="gray", size="xs"),
                       dmc.Text(f"criterion: {_words(path['criterion'])}", size="xs", c="dimmed")],
                      gap="xs"),
            table(["Step", "Action", "Factor", "Factors in the model", "Criterion", "Change",
                   "Considered"], steps) if steps else None,
            dmc.Text(path.get("note") or "", size="xs", c="dimmed"),
        ], gap=4, mb="sm"))
    samples = p.get("samples") or {}
    lib = p.get("library") or {}
    known = {c["id"] for c in leaders}
    try:
        ranking = json.loads(service.read_artifact(state.run_id, p.get("ranking_path") or "",
                                                   None) or "[]") if p.get("ranking_path") else []
    except ValueError:
        ranking = []
    below = [r for r in ranking if r["id"] not in known]
    walked = [x for x in p.get("paths") or [] if x["method"] not in ("grid", "same_factors")]
    if len(walked) > 1:
        how = (f"{len(walked)} methods were run ("
               + ", ".join(_words(x["method"]) for x in walked) + "). They are not alternatives "
               "to choose between afterwards: every model any of them evaluated is ranked in one "
               "list, and the leaders are the best across all of them. “Found by” says which "
               "method reached each model. To build with one method only, choose it in the plan "
               "at the design gate.")
    elif walked:
        how = (f"Models are built by {_words(walked[0]['method'])}, the method chosen in the "
               "plan (change it, or add another, at the design gate). Each step shows what was "
               "added or dropped and what it did to the criterion.")
    else:
        how = "Each family on the slate is fitted on the shortlisted factors."
    gate = gate_block("gate_candidates", state, res, seat=seat, body=[
        dmc.Text("Candidate pool", size="sm", fw=600),
        dmc.Text(f"Tick every candidate that should go on to outcomes analysis: as many as you "
                 f"want to compare (the engine proposes {len(pool_ids)}). Each is fitted, tested "
                 "and compared on the test sample; the champion is chosen there. A name can be "
                 "changed here and is used on every later page.", size="xs", c="dimmed"),
        dmc.CheckboxGroup(dmc.Stack([
            dmc.Group([
                dmc.Checkbox(value=c["id"], label=None),
                dmc.TextInput(id=_field("gate_candidates", f"name::{c['id']}"),
                              value=c.get("name") or "", size="xs", w=150),
                dmc.Group([dmc.Text(c["family"], size="xs", ff="monospace"),
                           chips(c["features"])], gap=6, style={"flex": 1}),
                dmc.Text(f"{fmt(c.get('score'))}", size="xs", ff="monospace"),
                dmc.Badge("breaks a constraint", color="red", size="xs", variant="light")
                if c.get("violations") else None,
            ], gap="xs", wrap="nowrap", align="center") for c in leaders], gap=4),
            id=_field("gate_candidates", "pool"), value=[i for i in pool_ids if i in known]),
        dmc.MultiSelect(id=_field("gate_candidates", "extra"), mt="sm", searchable=True,
                        label=f"Add from the rest of the ranking ({len(below)} more models)",
                        description="Search by name, family or factor. Every model the search "
                                    "evaluated can join the pool.",
                        value=[i for i in pool_ids if i not in known],
                        data=[{"value": r["id"],
                               "label": f"#{r['rank']} {r['name']} · {', '.join(r['features'])} "
                                        f"· {fmt(r.get('score'))}"} for r in below],
                        limit=40, maxDropdownHeight=280),
    ], primary=[
        dmc.Button("Accept the pool", id=_act("gate_candidates", "accept"),
                   leftSection=icon("tabler:check")),
        dmc.Button("Apply the ticked pool and names", id=_act("gate_candidates", "edit"),
                   variant="light", leftSection=icon("tabler:edit")),
    ])
    bench = []
    for key, label in (("challenger_benchmark", "Ensemble ceiling"), ("structural", "Merton structural"),
                       ("migration", "Rating migration")):
        b = p.get(key) or {}
        if key == "challenger_benchmark" and b:
            bench.append([label, fmt(b.get("benchmark_score")), "not deployed"])
        elif (b.get("benchmark") or {}):
            bench.append([label, f"AUC {fmt(b['benchmark'].get('test_roc_auc'))} (test sample)",
                          "not deployed"])
    return [
        kpis([("Factors", fmt(lib.get("n_factors")), f"from {fmt(lib.get('n_sources'))} column(s)"),
              ("Shortlisted", fmt(len(p.get("shortlist") or [])), "after the single-factor screen"),
              ("Models evaluated", fmt(p.get("n_candidates_tried")),
               " + ".join(_words(m) for m in (plan.get("selection") or {}).get("methods") or [])),
              ("Pool", fmt(len(pool_ids)), f"set by the {p.get('pool_source', 'engine')}")]),
        recommendation_card(rec, extra=table(["Factor", "Expected", "Observed", "Assessment"], signs)
                            if signs else None),
        gate,
        section("Samples", kpis([
            ("Development", fmt(samples.get("development")), "models are fitted here"),
            ("Validation", f"{len(samples.get('validation_blocks') or [])} block(s)",
             _words((plan.get("evaluation") or {}).get("validation", "—"))),
            ("Test", fmt(samples.get("test")), "the pool is compared here"),
            ("Sealed holdout", fmt(samples.get("holdout")) if samples.get("holdout") else "none",
             "scored once, after the choice")]),
            *[dmc.Text(n, size="xs", c="dimmed") for n in [samples.get("validation_note"),
                                                             *(lib.get("notes") or [])] if n],
            description=p.get("plan_summary")),
        transforms_section(state.run_id, p, scheme),
        section("Single-factor screen",
                table(["Rule", "Setting", "Factors in", "Factors out", "Dropped"], screen_rows),
                table(["Exception", "Factor", "The developer's reason", "Rule overruled"],
                      [[_words("kept anyway" if e["exception"] == "keep" else "left out"),
                        dmc.Text(e["factor"], ff="monospace", size="sm"),
                        dmc.Text(e["reason"], size="xs"),
                        dmc.Text("; ".join(e.get("overruled") or []) or "—", size="xs")]
                       for e in p.get("exceptions") or []]) if p.get("exceptions") else None,
                dmc.Text("Shortlist", size="xs", fw=650, c="dimmed", mt="sm"),
                table(["Factor", "Transformation", "Sign", "p-value", "In sample", "Validation", ""],
                      factor_rows, max_height=360),
                description="Every factor fitted alone on the development sample. The rule and "
                            "the reason for each dropped factor are in "
                            "stages/model/single_factor.json."),
        model_funnel_section(state.run_id, p),
        section("Multi-factor selection", *path_blocks, description=how) if path_blocks else None,
        section("Leaders", table(["#", "Model", "Family", "Factors", score_label, "Found by", ""],
                                 leader_rows, max_height=520),
                description=f"The top {len(leaders)} of {len(ranking) or len(leaders)} ranked "
                            "models, one line per distinct specification. The whole ranking is "
                            "stages/model/ranking.json, and any model in it can be added to the "
                            "pool in the review above."),
        section("Candidate pool", table(["Model", "Specification", "Factors", score_label, "Largest p",
                                         "Diagnostics", "Constraints broken"], pool_rows),
                description="Each member is fitted on the development sample, tested and saved. "
                            "None is the champion yet."),
        section("Experiment ledger", graph(charts.ledger(ledger, first.get("metric", "score"),
                                                         leaders[0]["label"] if leaders else None,
                                                         scheme), "No ledger.")),
        section("Challenger benchmarks", table(["Benchmark", "Score", "Status"], bench))
        if bench else None,
        section("Findings", findings_table(_findings(res))),
    ]


def model_funnel_section(run_id: str, p: dict) -> Any:
    """Every distinct model through the plan's rules, best ranked first (ADR-0025): the funnel,
    then who each rule set aside and why. None for a run recorded before the funnel."""
    from ...modeling import plan as plan_mod
    from .. import design_steps as dz

    mf = p.get("model_funnel")
    if not mf:
        return None
    try:
        full = json.loads(service.read_artifact(run_id, "stages/model/model_funnel.json", None)
                          or "{}")
    except ValueError:
        full = {}
    lines = full.get("lines") or mf["lines"]
    total = lines[0]["before"] if lines else 0
    kept = len(full.get("kept") or mf.get("kept") or [])
    blocks = [html.Details([
        html.Summary(f"{line['label'].capitalize()} — {line['before'] - line['after']} set "
                     "aside", className="gate-fold"),
        table(["Model", "Why"], [[d["name"], dmc.Text(d["reason"], size="xs")]
                                 for d in line.get("dropped") or []], max_height=240),
    ], className="gate-more") for line in lines[1:] if line["before"] - line["after"]]
    ev = (p.get("plan") or {}).get("evaluation") or {}
    return section(
        "Models through the plan's rules",
        dz.funnel([{**line, "rule": line["rule"]} for line in lines[1:]], total=total,
                  selected="kept", prefix="mf", extra=[("kept", "kept", kept)]),
        *blocks,
        dmc.Text(f"What is left is ranked by {plan_mod.rank_words(ev)}." if ev.get("rank_by")
                 else "", size="sm", mt="xs"),
        description=f"The best {mf.get('n_models', total)} of {mf.get('n_ranked', total)} "
                    "ranked models, one per distinct specification. Each leaves at the first "
                    "rule it breaks; the full list is stages/model/model_funnel.json.")


def transforms_section(run_id: str, p: dict, scheme: str) -> Any:
    """Each transformed column as the model stage fitted it on the development sample
    (ADR-0024): the observed outcome per band and the curve the model sees. None when no
    column is transformed."""
    if not p.get("transforms"):
        return None
    try:
        report = json.loads(service.read_artifact(run_id, "stages/model/transforms.json", None)
                            or "{}")
    except ValueError:
        report = {}
    cells = []
    for col, entry in (report or p["transforms"]).items():
        cells.append(dmc.Stack([
            dmc.Group([dmc.Text(col, size="sm", fw=650, ff="monospace"),
                       dmc.Code((entry.get("factor") or {}).get("name", "")
                                if isinstance(entry.get("factor"), dict)
                                else entry.get("factor", ""))], gap=6),
            *(transform_review.fit_view(entry, scheme) if entry.get("curve")
              else [dmc.Text(entry.get("caption") or "", size="xs", c="dimmed")]),
        ], gap=2))
    return section(
        "Transformed variables", dmc.SimpleGrid(cells, cols={"base": 1, "md": 2}, spacing="lg"),
        description="Fitted on the development sample only, after the split, then frozen: the "
                    "test sample and the sealed holdout get the same points. Dots are the "
                    "observed outcome per band; the line is what the model sees.")
