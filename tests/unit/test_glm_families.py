"""GLM/econometric families (poisson, gamma, tweedie) for commercial credit work.

These are opt-in linear/GLM families: scaled + sklearn-predicted, but with no statsmodels
inference (they report sklearn coef_ only) and excluded from the general regression default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognos.modeling.fit import DEFAULT_FAMILIES, Candidate, fit_full
from cognos.modeling.search import _hp_grid


def _make_count_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    rate = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)  # positive mean for a Poisson target
    y = rng.poisson(rate).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def test_poisson_fits_and_predicts_nonnegative():
    X, y = _make_count_dataset()
    fitted = fit_full(Candidate(family="poisson", features=["x1", "x2"]), X, y,
                      task="glm_regression", is_classification=False)
    preds = fitted.predict(X)
    assert preds.shape == (len(y),)
    assert (preds >= -1e-6).all()  # Poisson mean is non-negative
    coefs = fitted.coefficients()
    assert isinstance(coefs, dict) and len(coefs) >= 2
    assert fitted.sm_result is None  # GLMs skip statsmodels inference


def test_tweedie_grid_nonempty_and_glm_default_reachable():
    assert _hp_grid("tweedie")  # non-empty grid
    assert _hp_grid("poisson")
    assert "poisson" in DEFAULT_FAMILIES["glm_regression"]
