"""Integration adapters: IMPACT adapter, runtime scorer."""

from __future__ import annotations

import os

import numpy as np

from cognos import synth
from cognos.integrations.impact_adapter import (
    build_entity_config,
    impact_available,
    score_with_impact,
)
from cognos.modeling import Candidate, fit_full
from cognos.runtime.score import load_scorer, save_scorer, score_frame, score_row


def _make_scorer(tmp_path):
    df = synth.make_regression_dataset(n=120)
    feats = ["x1", "x2", "x3", "region"]
    y = df["target"].astype(float).to_numpy()
    fitted = fit_full(Candidate(family="ols", features=feats), df[feats], y,
                      task="regression", is_classification=False)
    path = str(tmp_path / "scorer.joblib")
    save_scorer(path, fitted)
    return df, feats, path


def test_runtime_scorer_roundtrip(tmp_path):
    df, feats, path = _make_scorer(tmp_path)
    load_scorer.cache_clear()
    preds = score_frame(df, path)
    assert preds.shape == (len(df),)
    one = score_row(df.iloc[0], path)
    assert abs(one - preds[0]) < 1e-6


def test_a_scorer_saved_again_at_the_same_path_is_read_again(tmp_path):
    """A pool member fitted again keeps its id, so its file keeps its path: the scores must
    come from the fit on disk, never from the one a long-lived process read before."""
    df, feats, path = _make_scorer(tmp_path)
    first = score_frame(df, path)
    assert load_scorer(path) is load_scorer(path)  # an unchanged file is read once

    half = df.iloc[: len(df) // 2]  # the same specification on another development sample
    refit = fit_full(Candidate(family="ols", features=feats), half[feats],
                     half["target"].astype(float).to_numpy(), task="regression",
                     is_classification=False)
    save_scorer(path, refit)
    second = score_frame(df, path)
    assert np.allclose(second, refit.pipeline.predict(df[feats]))
    assert not np.allclose(second, first)

    # ... and a file another process replaced (no save in this one) is noticed as well
    other = str(tmp_path / "other.joblib")
    save_scorer(other, fit_full(Candidate(family="ols", features=feats[:2]), df[feats[:2]],
                                df["target"].astype(float).to_numpy(), task="regression",
                                is_classification=False))
    load_scorer(path)
    os.replace(other, path)
    assert load_scorer(path).raw_features == feats[:2]


def test_build_entity_config_has_derived_score(tmp_path):
    df, feats, path = _make_scorer(tmp_path)
    cfg = build_entity_config(entity_name="E", primary_key="row_id", df=df, raw_features=feats,
                              source_path="x.parquet", model_path=path)
    names = [f["name"] for f in cfg["fields"]]
    assert "row_id" in names and "cognos_score" in names
    score_field = next(f for f in cfg["fields"] if f["name"] == "cognos_score")
    assert score_field["derived"]["function"] == "cognos.runtime.score.score_row"
    assert score_field["derived"]["kwargs"]["model_path"] == path
    assert cfg["sources"][0]["primary"] is True


def test_score_with_impact_fallback(tmp_path):
    df, feats, path = _make_scorer(tmp_path)
    load_scorer.cache_clear()
    res = score_with_impact(df, model_path=path, raw_features=feats,
                            work_dir=tmp_path / "wf", prefer_impact=False)
    assert res.used_impact is False
    assert "score" in res.scored_df.columns and len(res.scored_df) == len(df)


def test_score_with_impact_real_path_if_available(tmp_path):
    df, feats, path = _make_scorer(tmp_path)
    load_scorer.cache_clear()
    res = score_with_impact(df, model_path=path, raw_features=feats,
                            work_dir=tmp_path / "wi", prefer_impact=True)
    assert "score" in res.scored_df.columns and len(res.scored_df) == len(df)
    if impact_available():
        assert res.used_impact is True  # genuinely ran through IMPACT EntityPipeline
        assert res.config_path is not None
