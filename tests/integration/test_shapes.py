"""Shaped factors (ADR-0024): a cap or a knee is chosen with the outcome in view, so it is
fitted on the development sample alone, after the split, and frozen for every other sample."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cognos import service
from cognos.config import ShapeSpec
from cognos.modeling import factors


def _bent(n: int = 6000, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Risk falls with coverage until 3, then stops falling."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(1.0, 0.6, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.0 - 1.2 * np.minimum(x, 3.0))))).astype(float)
    return pd.DataFrame({"coverage": x}), y


def test_a_knee_is_found_where_the_risk_stops_falling():
    df, y = _bent()
    fit = factors.fit_shape(df["coverage"], y, knees=2, direction="-")
    assert fit["knees_used"] == 1 and abs(fit["knots"][0] - 3.0) < 0.6
    assert fit["monotonic"] and fit["origin"] == "fitted"
    values = [p[1] for p in fit["points"]]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_knots_a_person_gives_are_kept():
    df, y = _bent()
    fit = factors.fit_shape(df["coverage"], y, knees=0, knots=[2.0], direction="-")
    assert fit["origin"] == "given" and fit["knots"] == [2.0]


def test_a_shaped_factor_reads_its_points_never_an_outcome():
    df, y = _bent()
    shaped, report, notes = factors.shape_library(
        df, y, ["coverage"], {"transforms": ["knee", "cap"], "max_knees": 2},
        expected={"coverage": "-"}, classification=True)
    assert {f.transform for f in shaped} == {"knee", "cap"}
    assert report["coverage"]["set_by"] == "plan" and notes
    other = pd.DataFrame({"coverage": [0.1, 2.0, 50.0, np.nan]})  # no outcome column at all
    out = factors.apply_library(other, [f.to_dict() for f in shaped])
    knee = next(f for f in shaped if f.transform == "knee")
    cap = next(f for f in shaped if f.transform == "cap")
    assert out[cap.name].iloc[0] == cap.points[0][0] and out[cap.name].iloc[2] == cap.points[-1][0]
    assert out[knee.name].isna().iloc[3]
    assert factors.expected_sign({"name": knee.name, "transform": "knee", "source": "coverage"},
                                 {"coverage": "-"}) == "+"


def test_a_column_s_own_form_wins_over_the_plan():
    df, y = _bent()
    req = factors.shape_requests(df, ["coverage"], {"transforms": ["knee"], "max_knees": 2,
                                                    "shapes": {"coverage": ShapeSpec(knees=0)}})
    assert req["coverage"]["knees"] == 0 and req["coverage"]["plain_cap"]


def test_a_run_shapes_on_the_development_sample_and_freezes_the_points(runs_dir):
    cfg = service.template_config("built-in:pd_logistic_scorecard", "pd", root=runs_dir)
    cfg.factors.shapes = {"interest_coverage": ShapeSpec(knees=2)}
    run_id = service.create_run(cfg, mode="autonomous", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    info = service.data_info(run_id, runs_dir, with_catalog=True)
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(info["preselected"], root=runs_dir),
                        root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    res = service.results(run_id, runs_dir)
    survey = {s["column"]: s for s in res["explore"].payload["shape_survey"]}
    assert "interest_coverage" in survey and survey["interest_coverage"]["observed"]
    shapes = res["model"].payload["shapes"]
    assert shapes["interest_coverage"]["set_by"] == "column"
    name = shapes["interest_coverage"]["factors"][0]
    run = service.runs_root(runs_dir) / run_id
    train = pd.read_parquet(run / "data/train.parquet")
    test = pd.read_parquet(run / "data/test.parquet")
    assert name in train.columns and name in test.columns
    report = service.read_artifact(run_id, "stages/model/shapes.json", runs_dir)
    assert report and name in report
    points = json.loads(report)["interest_coverage"]["points"]
    knee = factors.Factor(name=name, source="interest_coverage", transform="knee", points=points)
    again = factors.apply_library(test.drop(columns=[name]), [knee])
    assert np.allclose(again[name], test[name], equal_nan=True)
