"""Modeling core: metrics, fitters, ratchet search, ensemble, backtest statistics."""

from __future__ import annotations

import numpy as np

from cognos import synth
from cognos.modeling import (
    Candidate,
    build_search_space,
    cv_score,
    fit_full,
    greedy_ensemble,
    holdout_split,
    is_better,
    make_fit_predict,
    ratchet_search,
    score,
)
from cognos.modeling.backtest_stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    sharpe_ratio,
)


def test_metric_score_and_direction():
    y = np.array([1.0, 2.0, 3.0])
    assert score("rmse", y, y) == 0.0
    assert score("r2", y, y) == 1.0
    assert is_better("rmse", 0.1, 0.2)  # lower better
    assert is_better("roc_auc", 0.9, 0.8)  # higher better
    assert is_better("rmse", 1.0, None)  # first candidate always accepted


def test_holdout_split_time_ordered():
    df = synth.make_timeseries_dataset(n=100)
    train, hold = holdout_split(df, holdout_fraction=0.2, datetime_col="date", random_state=1)
    assert len(hold) == 20 and len(train) == 80
    # time-ordered: every holdout date is after every training date
    assert train["date"].max() <= hold["date"].min()


def test_cv_score_oof_shape_and_leakage_safe():
    df = synth.make_regression_dataset(n=120)
    X = df[["x1", "x2", "x3"]]
    y = coerce_target_reg(df)
    cand = Candidate(family="ols", features=["x1", "x2", "x3"])
    cv = cv_score(make_fit_predict(cand, is_classification=False), X, y,
                  metric="rmse", is_classification=False, folds=4)
    assert cv.oof_pred.shape == (120,)
    assert cv.n_folds == 4 and cv.mean > 0


def test_ratchet_search_regression_finds_signal():
    df = synth.make_regression_dataset(n=200)
    X = df[["x1", "x2", "x3", "region"]]
    y = coerce_target_reg(df)
    sr = ratchet_search(X, y, task="regression", metric="rmse", is_classification=False,
                        max_candidates=8, folds=3)
    assert sr.champion is not None
    assert sr.champion_cv.mean < 1.5  # real signal recovered
    assert any(r.status == "keep" for r in sr.ledger)
    assert "idx\tlabel" in sr.ledger_tsv()


def test_fit_full_ols_has_coefficients():
    df = synth.make_regression_dataset(n=200)
    X = df[["x1", "x2", "x3"]]
    y = coerce_target_reg(df)
    fitted = fit_full(Candidate(family="ols", features=["x1", "x2", "x3"]), X, y,
                      task="regression", is_classification=False)
    coefs = fitted.coefficients()
    assert coefs is not None and len(coefs) >= 3
    assert fitted.sm_result is not None  # statsmodels inference present
    preds = fitted.predict(X)
    assert preds.shape == (200,)


def test_build_search_space_orders_linear_first():
    df = synth.make_regression_dataset(n=80)
    X = df[["x1", "x2", "x3"]]
    y = coerce_target_reg(df)
    space = build_search_space("regression", ["x1", "x2", "x3"], X, y)
    assert len(space) > 0
    assert space[0].is_linear  # cheap/simple first


def test_greedy_ensemble_improves_or_matches():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    # two noisy estimators of y; ensemble should not be worse than best single
    p1 = y + rng.normal(0, 0.5, 100)
    p2 = y + rng.normal(0, 0.5, 100)
    ens = greedy_ensemble([p1, p2], y, metric="rmse", is_classification=False)
    assert ens is not None
    assert ens.ensemble_score <= ens.best_single_score + 1e-9


def test_sharpe_and_dsr():
    rng = np.random.default_rng(1)
    r = rng.normal(0.05, 1.0, 250)
    assert isinstance(sharpe_ratio(r), float)
    d = deflated_sharpe_ratio(r, n_trials=50)
    assert 0.0 <= d["deflated_sharpe"] <= 1.0
    # more trials => higher benchmark => not-higher DSR
    d2 = deflated_sharpe_ratio(r, n_trials=500)
    assert d2["sr0_expected_max"] >= d["sr0_expected_max"]


def test_expected_max_sharpe_monotonic():
    assert expected_max_sharpe(0.04, 100) > expected_max_sharpe(0.04, 10) >= 0


def test_pbo_detects_overfitting():
    rng = np.random.default_rng(2)
    T, N = 200, 10
    # Overfit library: pure noise -> in-sample best is random -> PBO should be substantial.
    noise = rng.normal(0, 1, (T, N))
    pbo_noise = pbo_cscv(noise, n_splits=8)["pbo"]
    # Genuinely-good library: one strategy dominates IS and OOS -> low PBO.
    signal = rng.normal(0, 0.1, (T, N))
    signal[:, 0] += 5.0  # strategy 0 is reliably best everywhere
    pbo_signal = pbo_cscv(signal, n_splits=8)["pbo"]
    assert pbo_signal < pbo_noise
    assert pbo_signal < 0.2


def coerce_target_reg(df):
    return df["target"].astype(float).to_numpy()
