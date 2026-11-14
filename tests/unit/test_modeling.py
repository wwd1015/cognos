"""Modeling core: metrics, fitters, the candidate space, ensemble, backtest statistics."""

from __future__ import annotations

import numpy as np

from cognos import synth
from cognos.modeling import (
    Candidate,
    build_search_space,
    cv_score,
    fit_full,
    greedy_ensemble,
    is_better,
    make_fit_predict,
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


def test_cv_score_oof_shape_and_leakage_safe():
    df = synth.make_regression_dataset(n=120)
    X = df[["x1", "x2", "x3"]]
    y = coerce_target_reg(df)
    cand = Candidate(family="ols", features=["x1", "x2", "x3"])
    cv = cv_score(make_fit_predict(cand, is_classification=False), X, y,
                  metric="rmse", is_classification=False, folds=4)
    assert cv.oof_pred.shape == (120,)
    assert cv.n_folds == 4 and cv.mean > 0


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


def test_inference_design_is_full_rank():
    # ADR-0006/Q6: the statsmodels inference design must be full-rank (K-1 dummies + intercept),
    # not the all-K prediction encoding that creates the dummy-variable trap.
    from cognos.modeling.fit import build_inference_design

    df = synth.make_regression_dataset(n=120)
    design = build_inference_design(df[["x1", "x2", "x3", "region"]])
    assert "const" in design.columns
    region_cols = [c for c in design.columns if c.startswith("region_")]
    assert len(region_cols) == 3  # 4 levels -> K-1 = 3 dummies, no trap
    M = design.to_numpy(dtype=float)
    assert np.linalg.matrix_rank(M) == M.shape[1]  # full rank
    assert np.linalg.cond(M) < 1e3  # well-conditioned (the all-K design was ~1e15)


def coerce_target_reg(df):
    return df["target"].astype(float).to_numpy()


def test_the_direct_fit_gives_the_pipelines_numbers_to_the_last_bit(monkeypatch):
    """A linear specification on numeric columns is scaled and fitted without the pipeline
    (validation makes thousands of such fits). It may only be faster, never different."""
    import pandas as pd
    import pytest

    from cognos.modeling import fit

    rng = np.random.default_rng(7)
    n = 240
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.gamma(2.0, size=n),
                      "flag": rng.random(n) > 0.6, "k": rng.integers(0, 9, size=n)})
    signal = X["a"] - 0.5 * X["b"] + X["flag"]
    targets = {True: (signal + rng.normal(size=n) > 0).astype(int).to_numpy(),
               False: np.exp(0.2 * signal + 0.1 * rng.normal(size=n)).to_numpy()}
    train, held = X.iloc[:180], X.iloc[180:]
    families = sorted(fit.LINEAR_FAMILIES - set(fit.HAZARD_FAMILIES))
    checked = 0
    for family in families:
        is_clf = family in {"logit", "ridge_logit", "lasso_logit", *fit.BINARY_GLM_LINKS}
        y = targets[is_clf][:180]
        for features in (["a"], ["a", "b", "flag", "k"]):
            cand = fit.Candidate(family=family, features=features)
            direct = fit.make_fit_predict(cand, is_classification=is_clf)(train, y)(held)
            with monkeypatch.context() as m:
                m.setattr(fit, "_all_numeric", lambda X: False)  # the pipeline, as before
                piped = fit.make_fit_predict(cand, is_classification=is_clf)(train, y)(held)
            for got, want in zip(direct, piped, strict=True):
                assert (got is None and want is None) or np.array_equal(got, want), (family, features)
            checked += 1
    assert checked == 2 * len(families) and checked >= 20

    # a text column still goes through the encoder, so it keeps the pipeline
    mixed = train.assign(seg=np.where(train["a"] > 0, "x", "y"))
    assert not fit._all_numeric(mixed[["a", "seg"]]) and fit._all_numeric(mixed[["a", "flag"]])
    with pytest.raises(KeyError):
        fit.make_fit_predict(fit.Candidate(family="ols", features=["a"]),
                             is_classification=False)(train, targets[False][:180])(held[["b"]])
