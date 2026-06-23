"""Statistical diagnostic battery."""

from __future__ import annotations

from cognos import synth
from cognos.modeling import Candidate, fit_full
from cognos.stages import stat_tests


def test_battery_on_ols_runs_residual_tests():
    df = synth.make_regression_dataset(n=200)
    feats = ["x1", "x2", "x3"]
    y = df["target"].astype(float).to_numpy()
    fitted = fit_full(Candidate(family="ols", features=feats), df[feats], y,
                      task="regression", is_classification=False)
    report = stat_tests.run_battery(fitted, df[feats], y, is_classification=False)
    assert report["n_run"] > 0
    names = {t["name"] for t in report["tests"]}
    # residual + collinearity tests should be present for a linear model
    assert {"breusch_pagan", "durbin_watson", "jarque_bera"} & names
    assert "max_failed_severity" in report
    for t in report["tests"]:
        assert set(t) >= {"name", "category", "passed", "severity", "interpretation"}


def test_battery_classification_no_crash():
    df = synth.make_classification_dataset(n=200)
    feats = ["x1", "x2", "x3"]
    y = df["target"].astype(int).to_numpy()
    fitted = fit_full(Candidate(family="logit", features=feats), df[feats], y,
                      task="classification", is_classification=True)
    report = stat_tests.run_battery(fitted, df[feats], y, is_classification=True)
    assert "tests" in report and report["n_failed"] >= 0


def test_battery_ml_model_skips_residual_tests():
    df = synth.make_classification_dataset(n=160)
    feats = ["x1", "x2", "x3"]
    y = df["target"].astype(int).to_numpy()
    fitted = fit_full(Candidate(family="random_forest", features=feats, hyperparams={"n_estimators": 30}),
                      df[feats], y, task="ml_classification", is_classification=True)
    report = stat_tests.run_battery(fitted, df[feats], y, is_classification=True)
    # no statsmodels result => residual tests skipped, but call must succeed
    assert isinstance(report["tests"], list)
