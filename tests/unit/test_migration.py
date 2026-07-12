"""Rating-migration layer: estimator properties, term structure, loss forecast, pipeline wiring.

Covers the migration module's contracts (row-stochastic NR-adjusted cohort matrix, PAVA
rank-ordering, absorbing-default term structure, EL arithmetic, serve-time recompute) and the model
stage's hybrid integration (rating swapped for migration_pd, train-only estimation, holdout
challenger benchmark, scorer round-trip on rating-only rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cognos import synth
from cognos.config import CognosConfig
from cognos.modeling.migration import (
    AGENCY_SCALE,
    MigrationSpec,
    _pava_increasing,
    augment_frame,
    cumulative_default_curve,
    estimate_transition_matrix,
    expected_loss_forecast,
    fit_migration,
    infer_scale,
)
from cognos.orchestrator import Orchestrator
from cognos.runtime.score import load_scorer


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synth.make_rating_migration_dataset(n_obligors=250, start_year=1995, end_year=2023)


@pytest.fixture(scope="module")
def spec() -> MigrationSpec:
    return MigrationSpec(rating_col="rating", next_rating_col="next_rating",
                         scale=list(AGENCY_SCALE), horizon_periods=3)


# --- estimator ---------------------------------------------------------------------


def test_matrix_is_row_stochastic_and_rank_orders(panel, spec):
    est = estimate_transition_matrix(panel["rating"], panel["next_rating"], spec)
    m = np.asarray(est["matrix"])
    assert m.shape == (7, 8)  # 7 live grades -> 7 live + D
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-12)
    assert (m >= 0).all()
    pds = [est["one_period_pd"][s] for s in AGENCY_SCALE]
    assert all(a <= b + 1e-12 for a, b in zip(pds, pds[1:], strict=False)), \
        "one-period PDs must be monotone across the scale (PAVA on by default)"
    assert est["diagnostics"]["diagonally_dominant"], "ratings should be sticky"
    # speculative-grade defaults dominate: CCC PD far above BBB PD
    assert est["one_period_pd"]["CCC"] > 10 * est["one_period_pd"]["BBB"]


def test_nr_adjustment_removes_withdrawn_from_denominator(spec):
    r = pd.Series(["BB"] * 10)
    nxt = pd.Series(["BB"] * 5 + ["NR"] * 4 + ["D"])
    est = estimate_transition_matrix(r, nxt, MigrationSpec(
        rating_col="rating", next_rating_col="next_rating", scale=["BB"], smoothing=0.0,
        monotone_pd=False))
    assert est["n_withdrawn"] == 4
    assert est["row_support"]["BB"] == 6  # 10 obs - 4 withdrawn
    assert est["one_period_pd"]["BB"] == pytest.approx(1 / 6)


def test_pava_monotonization_is_reported(spec):
    vals = _pava_increasing(np.array([0.05, 0.01, 0.02, 0.10]), np.ones(4))
    assert (np.diff(vals) >= 0).all()
    # the violation cascades: 0.05 pools with 0.01, then with 0.02 -> one block of three
    assert vals[0] == pytest.approx(vals[1]) == pytest.approx(vals[2])
    assert vals[0] == pytest.approx((0.05 + 0.01 + 0.02) / 3)  # block keeps the weighted mean


def test_infer_scale_orders_by_default_rate(panel):
    scale = infer_scale(panel["rating"], panel["next_rating"])
    assert scale[-1] == "CCC" and scale[-2] == "B"
    assert set(scale) == set(AGENCY_SCALE)


# --- term structure + loss forecast ------------------------------------------------


def test_cumulative_curve_monotone_with_absorbing_default(panel, spec):
    est = estimate_transition_matrix(panel["rating"], panel["next_rating"], spec)
    curve = cumulative_default_curve(est["matrix"], est["scale"], periods=5)
    for s in AGENCY_SCALE:
        c = curve[s]
        assert len(c) == 5
        assert all(a <= b + 1e-12 for a, b in zip(c, c[1:], strict=False)), \
            f"cumulative PD must be non-decreasing for {s}"
        assert 0.0 <= c[0] <= c[-1] <= 1.0
    # horizon-1 cumulative PD equals the matrix default column
    assert curve["B"][0] == pytest.approx(est["one_period_pd"]["B"])


def test_expected_loss_arithmetic(panel, spec):
    model = fit_migration(panel, spec)
    ratings = pd.Series(["BBB", "B", "B"])
    ead = np.array([100.0, 50.0, 50.0])
    fc = expected_loss_forecast(ratings, model, lgd=0.4, ead=ead)
    pooled = fc["pooled"]
    expect = 0.4 * (100 * model["pd_map"]["BBB"] + 100 * model["pd_map"]["B"])
    assert pooled["expected_loss"] == pytest.approx(expect)
    assert pooled["expected_loss_rate"] == pytest.approx(expect / 200.0)
    b_band = next(b for b in pooled["bands"] if b["rating"] == "B")
    assert b_band["n_obligors"] == 2 and b_band["ead"] == pytest.approx(100.0)


def test_conditional_matrices_capture_the_regime(panel, spec):
    model = fit_migration(panel, spec, condition_col="regime")
    cond = model["conditional"]
    assert set(cond) == {"expansion", "recession"}
    # recession-conditioned speculative-grade PDs must exceed expansion-conditioned ones
    assert cond["recession"]["cumulative_pd"]["B"] > cond["expansion"]["cumulative_pd"]["B"]


def test_augment_maps_ratings_and_fills_unseen(panel, spec):
    model = fit_migration(panel, spec)
    df = pd.DataFrame({"rating": ["AA", "CCC", "XX"]})
    out, info = augment_frame(df, model)
    assert out["migration_pd"].iloc[0] == pytest.approx(model["pd_map"]["AA"])
    assert out["migration_pd"].iloc[1] == pytest.approx(model["pd_map"]["CCC"])
    assert out["migration_pd"].iloc[2] == pytest.approx(model["fill"])
    assert info["n_unmapped"] == 1
    assert out["migration_pd"].iloc[1] > out["migration_pd"].iloc[0]


def test_spec_roundtrip(spec):
    assert MigrationSpec.from_dict(spec.to_dict()) == spec


# --- model-stage integration --------------------------------------------------------


@pytest.fixture(scope="module")
def migration_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("migrun")
    csv = d / "agency.csv"
    synth.make_rating_migration_dataset(n_obligors=150, start_year=2005,
                                        end_year=2023).to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "mig", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "asof",
                 "drop_columns": ["obligor_id", "regime", "ead"]},
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 4, "cv_folds": 3, "model_families": ["logit"]},
        "migration": {"enabled": True, "rating_col": "rating",
                      "next_rating_col": "next_rating",
                      "rating_scale": list(AGENCY_SCALE), "condition_col": "regime",
                      "lgd": 0.4, "ead_column": "ead"},
        "stages": {"enabled": ["explore", "ideate", "model", "backtest"]},
    })
    orch = Orchestrator(cfg, runs_root=str(d / "runs"))
    for s in cfg.stages.enabled:
        orch.run_stage(s)
    return orch


def test_hybrid_feature_replaces_the_raw_rating(migration_run):
    p = migration_run.ctx.require("model").payload
    assert "migration_pd" in p["raw_features"]
    assert "rating" not in p["raw_features"], \
        "raw rating must be swapped out (exact function of migration_pd)"
    assert "next_rating" not in p["raw_features"], "outcome data must never be a feature"
    mig = p["migration"]
    assert mig["estimate"]["n_obs"] > 0
    assert mig["benchmark"]["deployed"] is False
    assert 0.5 < mig["benchmark"]["holdout_roc_auc"] <= 1.0


def test_loss_forecast_reported_on_oot_book(migration_run):
    mig = migration_run.ctx.require("model").payload["migration"]
    fc = mig["loss_forecast"]
    assert fc["book"] == "out_of_time_holdout"
    assert 0.0 < fc["pooled"]["expected_loss_rate"] < 0.25
    assert fc["conditional"]["recession"]["expected_loss_rate"] > \
        fc["conditional"]["expansion"]["expected_loss_rate"]


def test_ideate_marks_migration_candidate_and_model_stage_fits_train_only(migration_run):
    ideate = migration_run.ctx.require("ideate").payload
    fw = next(f for f in ideate["framework_assessment"] if f["framework"] == "transition_matrix")
    assert fw["applicable"] and fw["role"] == "candidate"
    model = migration_run.ctx.require("model").payload
    # train-only estimation: matrix support cannot exceed the training partition
    assert model["migration"]["estimate"]["n_obs"] <= model["n_train"]


def test_scorer_recomputes_migration_pd_from_rating(migration_run):
    p = migration_run.ctx.require("model").payload
    bundle = load_scorer(p["scorer_path"])
    assert bundle.migration is not None
    assert "rating" in (bundle.base_features or [])
    fresh = pd.DataFrame({
        "rating": ["AAA", "CCC"],
        "debt_to_ebitda": [1.5, 8.0], "interest_coverage": [12.0, 1.0],
        "operating_margin": [0.15, -0.05], "log_total_assets": [21.0, 18.0],
        "sector": ["services", "retail_trade"], "unemployment_rate": [5.0, 5.0],
    })
    scores = bundle.score_frame(fresh)
    assert scores[1] > scores[0], "CCC obligor must score riskier than AAA"


def test_backtest_scores_the_migration_holdout(migration_run):
    bt = migration_run.ctx.require("backtest").payload
    assert bt["evaluation_sample"] == "out_of_time"
    assert bt["outcomes_analysis"]["gini"] > 0.2
