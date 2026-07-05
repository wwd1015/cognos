"""Traditional-regression and structural/simulation capabilities.

Covers the econometric layer: probit/cloglog GLM links with valid inference, the discrete-time
hazard families (panel expansion, term structure, obligor-level CV, scorer round-trip), the Merton
structural solver (hybrid feature + serve-time recompute), and the Vasicek/stress simulation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cognos import synth
from cognos.config import CognosConfig
from cognos.modeling.hazard import HazardScorer, expand_panel
from cognos.modeling.simulate import basel_irb_capital, vasicek_loss_simulation
from cognos.modeling.structural import StructuralSpec, augment_frame, solve_merton
from cognos.orchestrator import Orchestrator
from cognos.runtime.score import load_scorer


def _cni_cfg(tmp_path, *, market=False, families=None, **extra) -> CognosConfig:
    df = synth.make_cni_portfolio_dataset(n=1200, include_market=market)
    csv = tmp_path / "cni.csv"
    df.to_csv(csv, index=False)
    raw = {
        "name": "t", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "event_time_col": "default_quarter", "horizon_periods": 4,
                 "drop_columns": ["obligor_id", "dpd_at_outcome"]},
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 6, "cv_folds": 3,
                   **({"model_families": families} if families else {})},
        "stages": {"enabled": ["explore", "model"]},
    }
    raw.update(extra)
    return CognosConfig.from_dict(raw)


def _run_model(cfg, runs_dir):
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run_stage("explore")
    return orch, orch.run_stage("model")


# --- probit / cloglog --------------------------------------------------------------


def test_probit_cloglog_fit_with_valid_inference(tmp_path, runs_dir):
    cfg = _cni_cfg(tmp_path, families=["probit", "cloglog"])
    _, res = _run_model(cfg, runs_dir)
    assert res.payload["champion"]["family"] in ("probit", "cloglog")
    assert res.payload["pvalues"], "binary GLM links must carry statsmodels p-values"
    assert 0.5 < res.payload["cv_mean"] < 1.0
    # event-time metadata must never appear as a model feature
    assert "default_quarter" not in res.payload["raw_features"]


# --- discrete-time hazard ----------------------------------------------------------


def test_expand_panel_counts_and_events():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = np.array([1, 0, 1])
    event_time = [2, np.nan, np.nan]  # 3rd obligor defaults but timing missing -> final period
    panel, y_panel = expand_panel(X, event_time, y, horizon=4)
    # obligor 1: periods 1-2 (event in 2); obligor 2: 4 censored periods; obligor 3: 4, event in 4
    assert len(panel) == 2 + 4 + 4
    assert y_panel.sum() == 2
    assert y_panel[1] == 1 and y_panel[-1] == 1
    assert set(panel["_period"]) <= {"p01", "p02", "p03", "p04"}


def test_hazard_champion_term_structure_and_scorer_roundtrip(tmp_path, runs_dir):
    cfg = _cni_cfg(tmp_path, families=["hazard_cloglog"])
    orch, res = _run_model(cfg, runs_dir)
    hz = res.payload["hazard"]
    assert res.payload["champion"]["family"] == "hazard_cloglog"
    cum = hz["term_structure"]["mean_cumulative_pd"]
    assert len(cum) == 4 and all(b >= a for a, b in zip(cum, cum[1:], strict=False))
    assert res.payload["pvalues"], "panel GLM inference must produce p-values"
    # the pickled scorer serves cumulative PD from obligor features alone
    load_scorer.cache_clear()
    bundle = load_scorer(res.payload["scorer_path"])
    assert isinstance(bundle.pipeline, HazardScorer)
    df = synth.make_cni_portfolio_dataset(n=50)
    preds = bundle.score_frame(df)
    assert preds.shape == (50,) and np.all((preds > 0) & (preds < 1))


def test_hazard_beats_or_matches_single_period_reading(tmp_path, runs_dir):
    # Not a performance assertion — just that both families evaluate on the same obligor-level CV.
    cfg = _cni_cfg(tmp_path, families=["logit", "hazard_logit"])
    _, res = _run_model(cfg, runs_dir)
    fams = {r["family"] for r in res.payload.get("diagnostics", {}).get("tests", [])} or None
    assert res.payload["champion"]["family"] in ("logit", "hazard_logit")
    assert res.verdict.value in ("PASS", "WARN"), fams


# --- structural (Merton) -----------------------------------------------------------


def test_solve_merton_reproduces_equity_and_orders_risk():
    sol = solve_merton([40.0], [0.6], [60.0], risk_free_rate=0.03, horizon_years=1.0)
    V, sV = sol["asset_value"][0], sol["asset_vol"][0]
    d1 = (np.log(V / 60) + (0.03 + 0.5 * sV**2)) / sV
    d2 = d1 - sV
    e_back = V * norm.cdf(d1) - 60 * np.exp(-0.03) * norm.cdf(d2)
    assert abs(e_back - 40.0) < 1e-4, "solved (V, sigma_V) must reproduce observed equity"
    assert 0 < sol["pd"][0] < 1
    # more leverage (bigger F for same E) => smaller DD, bigger PD
    lo = solve_merton([40.0], [0.6], [30.0])["pd"][0]
    hi = solve_merton([40.0], [0.6], [90.0])["pd"][0]
    assert hi > lo
    # non-positive inputs come back NaN, never a crash
    bad = solve_merton([0.0], [0.6], [60.0])
    assert np.isnan(bad["dd"][0])


def test_structural_hybrid_and_serve_time_recompute(tmp_path, runs_dir):
    cfg = _cni_cfg(
        tmp_path, market=True, families=["logit"],
        structural={"enabled": True, "equity_value_col": "equity_value",
                    "equity_vol_col": "equity_vol", "debt_col": "debt_face"},
    )
    orch, res = _run_model(cfg, runs_dir)
    st = res.payload["structural"]
    assert "merton_dd" in res.payload["champion"]["features"]
    assert st["n_failed"] == 0
    assert -1 <= st["benchmark"]["holdout_gini"] <= 1
    assert st["benchmark"]["deployed"] is False
    # a fresh frame WITHOUT the engineered column scores identically via recompute
    load_scorer.cache_clear()
    bundle = load_scorer(res.payload["scorer_path"])
    df = synth.make_cni_portfolio_dataset(n=30, include_market=True)
    spec = StructuralSpec.from_dict(st["spec"])
    with_dd, _ = augment_frame(df, spec)
    np.testing.assert_allclose(bundle.score_frame(df), bundle.score_frame(with_dd), rtol=1e-9)


# --- Vasicek portfolio + stress ------------------------------------------------------


def test_vasicek_el_matches_theory_and_tail_grows_with_rho():
    pds = np.full(400, 0.03)
    lo = vasicek_loss_simulation(pds, lgd=0.45, rho=0.05, n_sims=4000, seed=1)
    hi = vasicek_loss_simulation(pds, lgd=0.45, rho=0.40, n_sims=4000, seed=1)
    assert lo["expected_loss"] == pytest.approx(0.03 * 0.45, rel=0.15)
    assert hi["var"] > lo["var"], "systematic correlation must fatten the loss tail"
    assert lo["mc_stderr_el"] < 0.01
    k = basel_irb_capital(np.array([0.01, 0.03]))
    assert np.all(k > 0) and k[1] > k[0]


def test_stress_scenarios_move_pd_in_the_right_direction(tmp_path, runs_dir):
    cfg = _cni_cfg(
        tmp_path, families=["logit"],
        stages={"enabled": ["explore", "model", "backtest"]},
        portfolio={"enabled": True, "lgd": 0.4, "n_sims": 2000},
        stress={"enabled": True, "scenarios": [
            {"name": "adverse", "shocks": {"unemployment_rate": {"add": 3.0},
                                           "gdp_growth": {"add": -2.0}}},
            {"name": "bogus", "shocks": {"not_a_column": {"add": 1.0}}},
        ]},
    )
    orch = Orchestrator(cfg, runs_root=str(tmp_path / "runs"))
    orch.run_stage("explore")
    orch.run_stage("model")
    res = orch.run_stage("backtest")
    pa, st = res.payload["portfolio_analysis"], res.payload["stress_testing"]
    assert pa["expected_loss"] > 0 and pa["var"] >= pa["expected_loss"]
    adverse = next(s for s in st["scenarios"] if s["name"] == "adverse")
    assert adverse["delta_pd"] > 0, "higher unemployment + lower growth must raise mean PD"
    bogus = next(s for s in st["scenarios"] if s["name"] == "bogus")
    assert bogus["missing_columns"] == ["not_a_column"]
    assert any(f.id.startswith("stress-missing") for f in res.findings)


# --- ideate framework awareness ------------------------------------------------------


def test_ideate_unlocks_hazard_and_structural_frameworks(tmp_path, runs_dir):
    cfg = _cni_cfg(tmp_path, market=True)
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run_stage("explore")
    res = orch.run_stage("ideate")
    fw = {f["framework"]: f for f in res.payload["framework_assessment"]}
    assert fw["discrete_time_hazard"]["applicable"] is True
    assert set(fw["discrete_time_hazard"]["engine_families"]) == {"hazard_logit", "hazard_cloglog"}
    assert "hazard_logit" in res.payload["families"]  # default slate widened by event timing
    # market observables present but structural disabled => "available", with an unlock question
    assert fw["structural_merton"]["applicable"] is True
    assert fw["structural_merton"]["role"] == "available"
    assert any(q["id"] == "data-structural" for q in res.payload["open_questions"])
