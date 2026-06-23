"""Adapter to the IMPACT library (impact.entity.pipeline.EntityPipeline).

IMPACT is a Python library, not a CLI/backtester: it is a declarative, YAML-config-driven engine
that builds a typed entity feature table. COGNOS integrates by *encoding the trained model as an
IMPACT derived field* — IMPACT imports ``cognos.runtime.score.score_row`` and applies it row-wise
(``df.apply(functools.partial(score_row, model_path=...), axis=1)``), producing a scored feature
table. COGNOS then computes backtest analytics on that table.

When IMPACT is not installed the adapter transparently falls back to the built-in scorer, so the
pipeline always runs; ``ImpactRunResult.used_impact`` records which path executed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..runtime.score import score_frame

SCORE_FIELD = "cognos_score"
SCORE_FUNCTION = "cognos.runtime.score.score_row"


def impact_available() -> bool:
    return importlib.util.find_spec("impact") is not None


def _impact_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "str"


def build_entity_config(
    *,
    entity_name: str,
    primary_key: str,
    df: pd.DataFrame,
    raw_features: list[str],
    source_path: str,
    model_path: str,
    extra_fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct an IMPACT EntityConfig dict embedding the model as a derived score field."""
    fields: list[dict[str, Any]] = [
        {"name": primary_key, "source": primary_key, "dtype": "str", "primary_key": True}
    ]
    for feat in raw_features:
        fields.append({"name": feat, "source": feat, "dtype": _impact_dtype(df[feat])})
    fields.append(
        {
            "name": SCORE_FIELD,
            "derived": {"function": SCORE_FUNCTION, "kwargs": {"model_path": str(model_path)}},
            "dtype": "float64",
            "description": "COGNOS model score (probability or point prediction)",
        }
    )
    for extra in extra_fields or []:
        fields.append(extra)

    return {
        "entity": {"name": entity_name, "description": "COGNOS scored entity", "version": "1.0"},
        "expression_packages": {"pd": "pandas", "np": "numpy"},
        "sources": [
            {
                "name": "scored_input",
                "type": "parquet" if source_path.endswith(".parquet") else "csv",
                "primary": True,
                "path": str(source_path),
            }
        ],
        "fields": fields,
    }


@dataclass
class ImpactRunResult:
    scored_df: pd.DataFrame
    used_impact: bool
    config_path: str | None = None
    source_path: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def score_with_impact(
    df: pd.DataFrame,
    *,
    model_path: str,
    raw_features: list[str],
    work_dir: str | Path,
    entity_name: str = "ScoredEntity",
    primary_key: str | None = None,
    prefer_impact: bool = True,
) -> ImpactRunResult:
    """Score ``df`` through IMPACT (preferred) or the built-in fallback. Always returns a scored df."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    scoring_df = df.copy().reset_index(drop=True)
    pk = primary_key or "row_id"
    if pk not in scoring_df.columns:
        scoring_df[pk] = [f"r{i}" for i in range(len(scoring_df))]

    if prefer_impact and impact_available():
        try:
            return _run_via_impact(scoring_df, model_path, raw_features, work_dir, entity_name, pk)
        except Exception as exc:  # robust fallback — IMPACT problems must never break the pipeline
            res = _run_fallback(scoring_df, model_path, raw_features, pk)
            res.note = f"IMPACT path failed ({type(exc).__name__}: {exc}); used built-in fallback."
            return res
    res = _run_fallback(scoring_df, model_path, raw_features, pk)
    res.note = "IMPACT not installed; used built-in fallback scorer." if not impact_available() else \
        "IMPACT available but fallback requested."
    return res


def _run_via_impact(df, model_path, raw_features, work_dir, entity_name, pk) -> ImpactRunResult:
    from impact.entity.pipeline import EntityPipeline  # type: ignore

    source_path = str(work_dir / "impact_source.parquet")
    keep = [pk, *raw_features]
    df[keep].to_parquet(source_path, index=False)

    cfg = build_entity_config(
        entity_name=entity_name, primary_key=pk, df=df, raw_features=raw_features,
        source_path=source_path, model_path=str(model_path),
    )
    config_path = str(work_dir / "impact_entity.yaml")
    with open(config_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    result = EntityPipeline(config_path).run(mode="dataframe")
    scored = result.dataframe.rename(columns={SCORE_FIELD: "score"})
    vr = getattr(result, "validation_report", None)
    validation = {
        "error_count": getattr(vr, "error_count", 0) if vr else 0,
        "warning_count": getattr(vr, "warning_count", 0) if vr else 0,
    }
    return ImpactRunResult(scored_df=scored, used_impact=True, config_path=config_path,
                           source_path=source_path, validation=validation,
                           note="Scored via IMPACT EntityPipeline (derived-field model embedding).")


def _run_fallback(df, model_path, raw_features, pk) -> ImpactRunResult:
    scored = df.copy()
    scored["score"] = score_frame(scored, model_path)
    cols = [pk, *raw_features, "score"]
    return ImpactRunResult(scored_df=scored[cols], used_impact=False)
