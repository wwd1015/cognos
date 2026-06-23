"""Integration adapters: autoforge protocol, IMPACT adapter, runtime scorer."""

from __future__ import annotations

from cognos import synth
from cognos.integrations.autoforge_loop import (
    ExperimentLedger,
    LedgerRow,
    format_metrics,
    parse_metrics,
    run_ratchet,
)
from cognos.integrations.impact_adapter import (
    build_entity_config,
    impact_available,
    score_with_impact,
)
from cognos.modeling import Candidate, fit_full
from cognos.runtime.score import load_scorer, save_scorer, score_frame, score_row


def test_parse_and_format_metrics_roundtrip():
    text = "noise line\nrmse:    0.123456\nr2:    0.870000\nnot a metric: hello\n"
    m = parse_metrics(text)
    assert m == {"rmse": 0.123456, "r2": 0.87}
    again = parse_metrics(format_metrics({"rmse": 0.5}))
    assert abs(again["rmse"] - 0.5) < 1e-9


def test_run_ratchet_keeps_best_and_logs_crash():
    exps = [lambda: {"rmse": 1.0}, lambda: {"rmse": 0.5}, lambda: (_ for _ in ()).throw(ValueError("boom")),
            lambda: {"rmse": 0.7}]
    champ, ledger = run_ratchet(exps, metric="rmse", labels=["a", "b", "c", "d"])
    assert champ == 0.5
    statuses = [r.status for r in ledger.rows]
    assert statuses == ["keep", "keep", "crash", "discard"]
    assert "commit\tmetric_value\tstatus\tdescription" in ledger.to_tsv()


def test_experiment_ledger_extra_columns():
    led = ExperimentLedger(metric="rmse", extra_columns=["cv_std"])
    led.append(LedgerRow(commit="c1", metric_value=0.5, status="keep", description="d", extras={"cv_std": 0.1}))
    tsv = led.to_tsv()
    assert "cv_std" in tsv.splitlines()[0]
    assert len(led.kept) == 1


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
