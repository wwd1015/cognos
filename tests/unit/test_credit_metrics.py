"""Credit-risk validation metrics: discrimination, calibration, stability."""

from __future__ import annotations

import numpy as np

from cognos.modeling.credit_metrics import (
    calibration_table,
    credit_outcomes,
    expected_calibration_error,
    gini,
    ks_statistic,
    psi,
    psi_label,
)


def _separable():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    # scores well-separated by class -> high discrimination, decent calibration
    scores = np.concatenate([rng.uniform(0.0, 0.4, 200), rng.uniform(0.6, 1.0, 200)])
    return y, scores


def test_gini_and_ks_high_for_separable():
    y, scores = _separable()
    assert gini(y, scores) > 0.8
    assert ks_statistic(y, scores) > 0.7


def test_gini_zero_for_random():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    scores = rng.uniform(0, 1, 500)
    assert abs(gini(y, scores)) < 0.2


def test_calibration_table_and_ece():
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)  # perfectly calibrated by construction
    table = calibration_table(y, p, n_bands=10)
    assert len(table) == 10
    assert all({"band", "n", "predicted", "observed"} <= set(r) for r in table)
    assert expected_calibration_error(y, p, 10) < 0.06  # well calibrated


def test_psi_detects_shift():
    rng = np.random.default_rng(3)
    dev = rng.normal(0.3, 0.1, 1000)
    same = rng.normal(0.3, 0.1, 1000)
    shifted = rng.normal(0.6, 0.1, 1000)
    assert psi(dev, same) < 0.1  # stable
    assert psi(dev, shifted) > 0.25  # significant shift
    assert psi_label(psi(dev, shifted)) == "significant shift"


def test_credit_outcomes_aggregate():
    y, scores = _separable()
    dev = np.random.default_rng(4).uniform(0, 1, 400)
    out = credit_outcomes(y, scores, dev_scores=dev)
    assert set(out) >= {"gini", "auc", "ks", "expected_calibration_error", "calibration_table",
                        "psi", "psi_label", "n_scored"}
    assert out["gini"] > 0.8 and 0.0 <= out["auc"] <= 1.0
