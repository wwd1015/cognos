"""An example COGNOS plugin for stages other than data exploration.

Every stage's agent can be given tools. A tool names the stages it serves and the inputs it
needs; the engine hands it exactly those, runs it when the stage's agent asks, and records the
result. Two tools here:

- ``score_band_monotonicity`` (validation and outcomes analysis): a custom model test on the
  champion and the sealed holdout, with a pass/fail check. A failed check becomes a finding of
  the stage; at validation a ``high`` one fails the validation.
- ``events_per_feature`` (design): a sample-size screen the Design Lead can run before it
  proposes a slate.

Load it like any plugin: ``plugins: [validation_tools]`` in a profile,
``COGNOS_PLUGINS=validation_tools``, or the ``cognos.plugins`` entry-point group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognos.plugins import Tool


def score_band_monotonicity(df, params: dict[str, str], env: dict) -> dict:
    """Does the observed outcome rise with the score, band by band, on data the model never saw?"""
    holdout, target = env["holdout"], env["target"]
    bands = int(params.get("bands") or 5)
    score = np.asarray(env["score"](holdout), dtype=float)
    y = pd.to_numeric(holdout[target], errors="coerce").to_numpy(dtype=float)
    band = pd.qcut(pd.Series(score).rank(method="first"), q=bands, labels=False)
    tab = (pd.DataFrame({"band": band + 1, "score": score, "y": y}).groupby("band")
           .agg(rows=("y", "size"), mean_score=("score", "mean"), observed=("y", "mean"))
           .reset_index())
    steps = np.diff(tab["observed"].to_numpy())
    reversals = int((steps < 0).sum())
    return {
        "title": "Observed outcome by score band (sealed holdout)",
        "summary": {"n_bands": int(len(tab)), "reversals": reversals,
                    "top_to_bottom_ratio": float(tab["observed"].iloc[-1]
                                                 / max(tab["observed"].iloc[0], 1e-9))},
        "table": {"columns": list(tab.columns), "rows": tab.round(4).to_numpy().tolist()},
        "chart": {"kind": "bar", "title": "Observed outcome by score band",
                  "x": [str(b) for b in tab["band"]], "x_title": "score band (low to high)",
                  "y_title": "observed outcome",
                  "series": [{"name": "observed", "y": list(tab["observed"])}]},
        "checks": [{"name": "monotonic", "passed": reversals <= int(params.get("tolerance") or 0),
                    "severity": "high",
                    "detail": f"{reversals} band(s) where the observed outcome falls as the "
                              "score rises"}],
    }


def events_per_feature(df: pd.DataFrame, params: dict[str, str], env: dict) -> dict:
    """Events per candidate feature: the usual floor for a stable logistic specification."""
    floor = float(params.get("floor") or 10)
    y = pd.to_numeric(df[env["target"]], errors="coerce")
    events = float(min(y.sum(), len(y) - y.sum())) if env.get("task") == "classification" \
        else float(len(y))
    n_features = max(len(env["features"]), 1)
    epv = events / n_features
    return {
        "title": "Events per candidate feature",
        "summary": {"events": events, "n_features": n_features, "events_per_feature": epv},
        "checks": [{"name": "enough_events", "passed": epv >= floor, "severity": "medium",
                    "detail": f"{epv:.1f} events per candidate feature against a floor of "
                              f"{floor:g}"}],
    }


def register(registry) -> None:
    registry.add_tool(Tool(
        "score_band_monotonicity",
        "Custom model test: does the observed outcome rise with the score, band by band, on the "
        "sealed holdout?", score_band_monotonicity,
        {"bands": "number of score bands (default 5)",
         "tolerance": "reversals allowed before the check fails (default 0)"},
        needs_target=True, stages=("backtest", "validate"), needs=("model", "holdout"),
        version="0.1"))
    registry.add_tool(Tool(
        "events_per_feature",
        "Sample-size screen: events per candidate feature against a floor.", events_per_feature,
        {"floor": "minimum events per feature (default 10)"},
        needs_target=True, stages=("ideate",), needs=("data",), version="0.1"))
