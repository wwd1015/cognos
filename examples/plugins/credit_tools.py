"""An example COGNOS plugin: one credit-specific analysis tool and one data source.

Load it by naming the module in a profile (``plugins: [credit_tools]``, with this directory on
``PYTHONPATH``), in ``COGNOS_PLUGINS=credit_tools``, or from an installed package through the
``cognos.plugins`` entry-point group::

    [project.entry-points."cognos.plugins"]
    credit_tools = "credit_tools"

``cognos plugins`` then lists ``information_value`` beside the built-in tools, and the Data
Analyst can ask for it by name. A tool is ordinary code you review once; what it reports is
recorded as the engine's numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognos.datasources import DataSource
from cognos.plugins import AnalysisTool


def information_value(df: pd.DataFrame, params: dict[str, str], env: dict) -> dict:
    """Weight of evidence per band and the information value of one feature against a binary
    target (the scorecard developer's first screen of a candidate driver)."""
    col, target = params["column"], env["target"]
    if env.get("task") != "classification":
        raise ValueError("information value needs a binary target")
    y = pd.to_numeric(df[target], errors="coerce")
    s = df[col]
    bins = int(params.get("bins") or 8)
    band = (pd.qcut(s, q=bins, duplicates="drop").astype(str)
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins else s.astype(str))
    g = pd.DataFrame({"band": band, "y": y}).groupby("band", observed=True)["y"]
    bad, n = g.sum(), g.size()
    good = n - bad
    # half a count in each cell keeps an empty cell from producing an infinite weight
    dist_bad = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))
    dist_good = (good + 0.5) / (good.sum() + 0.5 * len(good))
    woe = np.log(dist_good / dist_bad)
    iv = float(((dist_good - dist_bad) * woe).sum())
    tab = pd.DataFrame({"band": woe.index, "rows": n.to_numpy(), "event_rate": (bad / n).to_numpy(),
                        "woe": woe.to_numpy()})
    strength = ("unpredictive" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
                else "strong" if iv < 0.5 else "suspiciously strong")
    return {
        "title": f"Weight of evidence: {col}",
        "summary": {"information_value": iv, "strength": strength, "n_bands": int(len(tab))},
        "table": {"columns": list(tab.columns), "rows": tab.round(4).to_numpy().tolist(),
                  "n_rows": int(len(tab))},
        "chart": {"kind": "bar", "title": f"Weight of evidence by band of {col}",
                  "x": list(tab["band"]), "x_title": col, "y_title": "weight of evidence",
                  "series": [{"name": "WoE", "y": list(tab["woe"])}]},
    }


def _load_fixed_width(source) -> pd.DataFrame:
    """A connector for a format COGNOS does not read itself (here: fixed-width text)."""
    return pd.read_fwf(source.path)


def register(registry) -> None:
    registry.add_tool(AnalysisTool(
        "information_value",
        "Weight of evidence per band and information value of one feature against a binary "
        "target: how much a candidate driver separates events from non-events.",
        information_value, {"column": "the feature", "bins": "quantile bands (default 8)"},
        needs_target=True))
    registry.add_source(DataSource("fixed_width", "Fixed-width text file", _load_fixed_width))
