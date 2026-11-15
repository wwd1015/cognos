"""The backtest page: the pool compared on the test sample, and the champion."""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc

from ... import service
from ...artifacts import StageResult
from ...engine.state import RunState
from .. import charts
from ..components import (
    chips,
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
from ._shared import _act, _field, _findings, _rec, _who, _words
from .gate import gate_block


def _forecast_spec(run_id: str, p: dict) -> dict | None:
    """Actual and forecast per pool member on the comparison sample, as a chart spec."""
    if not p.get("forecasts_path") or p.get("outcomes_analysis"):
        return None
    try:
        data = json.loads(service.read_artifact(run_id, p["forecasts_path"], None) or "null")
    except ValueError:
        data = None
    if not data or not data.get("actual"):
        return None
    series = [{"name": "actual", "y": data["actual"]}]
    names = {c["id"]: c.get("name") or c["id"] for c in p.get("candidates") or []}
    for cid, values in list(data["forecast"].items())[:7]:
        series.append({"name": names.get(cid, cid)
                       + (" (champion)" if cid == p.get("champion_id") else ""),
                       "y": values})
    return {"kind": "line", "x": data["x"], "series": series, "x_title": data.get("x_label", ""),
            "y_title": "actual and forecast"}


def _champion_proposal(p: dict, cands: list, metric: str, sample: str) -> str | None:
    champ = next((c for c in cands if c["id"] == p.get("champion_id")), None)
    if not champ:
        return None
    return (f"{champ.get('name') or champ['id']}: the "
            f"{_words(p.get('champion_source', 'engine'))}'s choice, compared with "
            f"{len(cands) - 1} other candidate(s) on the {sample} sample.")


def _champion_evidence(p: dict, cands: list, metric: str) -> list[tuple[str, str]]:
    """The champion's figures as the engine recorded them, for the decision card."""
    champ = next((c for c in cands if c["id"] == p.get("champion_id")), None)
    if not champ:
        return []
    m = (champ.get("outcomes") or {}) | (champ.get("metrics") or {})
    out = [(k.replace("_", " "), fmt(m.get(k), 4)) for k in dict.fromkeys(
        [metric, "gini", "ks", "psi", "rmse"]) if k and m.get(k) is not None][:4]
    out.append(("flags", fmt(len(champ.get("flags") or []))))
    return out


def _unmoved(scenarios: list[dict], key: str) -> bool:
    """Every scenario left the forecast where it was (a model with no macro factors)."""
    return all(abs(float(s.get(key) or 0.0)) < 1e-12 for s in scenarios)


def _unmoved_note(name: str) -> Any:
    return dmc.Text(f"{name} has no macro factors, so the scenarios do not move its forecast.",
                    size="sm", c="dimmed", mt="xs")


def backtest_panel(res: StageResult, state: RunState, scheme: str, seat: str = "developer") -> list:
    p = res.payload or {}
    oa = p.get("outcomes_analysis") or {}
    fa = p.get("forecast_analysis") or {}
    interp = p.get("interpretation") or {}
    port = p.get("portfolio_analysis") or {}
    stress = (p.get("stress_testing") or {}).get("scenarios") or []
    cands = p.get("candidates") or []
    metric = p.get("oos_metric_name", "")
    sample = _words(p.get("evaluation_sample", "test"))
    champion = p.get("champion_id")
    keys = [k for k in (metric, "rmse", "mae", "r2", "bias", "gini", "ks",
                        "expected_calibration_error", "psi")
            if any(k in ((c.get("outcomes") or {}) | (c.get("metrics") or {})) for c in cands)]
    keys = list(dict.fromkeys(keys))
    labels = {"expected_calibration_error": "calibration error", "r2": "R²"}
    cand_rows = [[dmc.Group([_who(c),
                             dmc.Badge("champion", color="oxblood", size="xs")
                             if c["id"] == champion else None,
                             dmc.Badge("analyst pick", color="blue", size="xs", variant="light")
                             if c["id"] == p.get("recommended_id") and c["id"] != champion
                             else None], gap=4),
                  chips(c.get("features")),
                  fmt(c.get("development_score")),
                  *[fmt(((c.get("outcomes") or {}) | (c.get("metrics") or {})).get(k)) for k in keys],
                  dmc.Text("; ".join(c.get("flags") or []) or (c.get("error") or ""), size="xs")]
                 for c in cands]
    names = list(dict.fromkeys(s["name"] for c in cands for s in c.get("scenarios") or []))
    moved = [c for c in cands if c.get("scenarios") and not _unmoved(c["scenarios"], "delta")]
    still = [c for c in cands if c.get("scenarios") and _unmoved(c["scenarios"], "delta")]
    scen_rows = [[_who(c), fmt(c.get("scenario_baseline")),
                  *[next((f"{fmt(s['mean'])}  ({s['delta']:+.4g})" for s in c.get("scenarios") or []
                          if s["name"] == n), "—") for n in names]]
                 for c in moved]
    champ_name = next((c.get("name") for c in cands if c["id"] == champion), None) \
        or p.get("champion_name") or "The champion"
    stress_still = bool(stress) and _unmoved(stress, "delta_pd")
    gate = gate_block("gate_champion", state, res, seat=seat, body=[
        dmc.RadioGroup(dmc.Stack([
            dmc.Radio(label=f"{c.get('name') or c['id']} · {c['label']} — {sample} {metric} "
                            f"{fmt((c.get('metrics') or {}).get(metric))}",
                      value=c["id"], disabled=c.get("role") == "challenger" or bool(c.get("error")))
            for c in cands], gap=4),
            id=_field("gate_champion", "champion"), label="Champion",
            description="Chosen on how the candidates forecast, not only on how they fitted.",
            value=champion),
    ], primary=[
        dmc.Button("Accept the champion", id=_act("gate_champion", "accept"),
                   leftSection=icon("tabler:check")),
    ], alternative=[
        dmc.Button("Choose the selected candidate", id=_act("gate_champion", "override"),
                   variant="light", color="orange", leftSection=icon("tabler:switch-horizontal")),
    ], fold="Choose another candidate",
        proposal=_champion_proposal(p, cands, metric, sample),
        evidence=_champion_evidence(p, cands, metric))
    head = [("Champion", str(p.get("champion_name") or champion or "—"),
             f"chosen by the {p.get('champion_source', 'engine')}"),
            ("Candidates compared", fmt(len(cands)), f"on the {sample} sample")]
    if oa:
        head += [("Gini", fmt(oa.get("gini"), 3), "discrimination"),
                 ("Calibration error", fmt(oa.get("expected_calibration_error"), 3),
                  "mean |obs − pred|")]
    else:
        head += [(f"{sample.capitalize()} {metric}", fmt(p.get("oos_metric")), None),
                 ("Bias", f"{fa['bias']:+.4g}" if fa.get("bias") is not None else "—",
                  "mean forecast − mean outcome")]
    return [
        kpis(head),
        recommendation_card(_rec(res), extra=dmc.Stack([
            dmc.SimpleGrid([
                dmc.Stack([dmc.Text(k.title(), size="xs", c="dimmed", fw=600), dmc.Text(v, size="sm")],
                          gap=2) for k, v in interp.items()], cols={"base": 1, "md": 3})
            if interp else None], gap="xs")),
        gate,
        section("Candidates side by side",
                table(["Candidate", "Factors", "Development", *[labels.get(k, k) for k in keys], ""],
                      cand_rows),
                description=f"Every pool member scored on the {sample} sample, best first on "
                            f"{metric}. The test sample has been used "
                            f"{fmt(p.get('test_evaluations'))} time(s) in this run"
                            + (" (trial runs of this model are counted on the Run record)"
                               if state.lineage is not None else "")
                            + ("; a sealed holdout is scored at validation."
                               if p.get("sealed_holdout") else
                               ", and no sealed holdout was reserved: the champion's result here "
                               "is also what it was chosen on.")),
        section("Forecasts against outcomes",
                graph(charts.from_spec(_forecast_spec(state.run_id, p), scheme)),
                description="Candidates that rank together on the development sample can part "
                            "ways here.") if _forecast_spec(state.run_id, p) else None,
        section("Scenarios, candidate by candidate",
                table(["Candidate", "Baseline", *names], scen_rows) if scen_rows else None,
                *[_unmoved_note(c.get("name") or c.get("label") or c["id"]) for c in still],
                description="Mean forecast under each scenario (change on the baseline).")
        if moved or still else None,
        section("Champion: calibration by score band",
                graph(charts.calibration(oa.get("calibration_table") or [], scheme), "n/a"),
                description=f"Evaluation sample: {sample}"
                            + (" (IMPACT)" if p.get("used_impact") else " (built-in scorer)"))
        if oa else None,
        section("Champion: portfolio loss simulation", kpis([
            ("Expected loss", pct(port.get("expected_loss"), 2), "share of EAD"),
            (f"VaR {fmt(port.get('confidence'), 3)}", pct(port.get("var"), 2), None),
            ("Expected shortfall", pct(port.get("es"), 2), None)])) if port else None,
        section("Champion: stress scenarios",
                _unmoved_note(champ_name) if stress_still else
                table(["Scenario", "Mean PD", "Δ PD"],
                      [[s["name"], pct(s.get("mean_pd"), 2), f"{s.get('delta_pd', 0):+.2%}"]
                       for s in stress]))
        if stress and oa else None,
        section("Findings", findings_table(_findings(res))),
    ]
