"""Built-in analysis tools. Each is a pure function of the dataset and its parameters: the
numbers it reports are the engine's. Plugins add more through ``Registry.add_tool``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .charts import cells, table


def _column(df: pd.DataFrame, params: dict[str, str], key: str = "column") -> str:
    col = params.get(key, "")
    if col not in df.columns:
        raise ValueError(f"unknown column {col!r}")
    return col


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def distribution(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    col = _column(df, params)
    s = df[col]
    summary: dict[str, Any] = {"n": int(s.notna().sum()), "missing_share": float(s.isna().mean()),
                               "n_unique": int(s.nunique())}
    if _numeric(s) and s.nunique() > 12:
        x = s.dropna().astype(float)
        counts, edges = np.histogram(x, bins=min(20, max(5, int(np.sqrt(len(x))))))
        summary.update(mean=float(x.mean()), std=float(x.std()), min=float(x.min()),
                       p50=float(x.median()), max=float(x.max()), skew=float(x.skew()))
        mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)]
        chart = {"kind": "histogram", "x": cells(mids),
                 "series": [{"name": "rows", "y": cells(counts)}], "x_title": col,
                 "y_title": "rows"}
        tab = pd.DataFrame({"bin_from": edges[:-1], "bin_to": edges[1:], "rows": counts})
    else:
        vc = s.astype("object").where(s.notna(), "(missing)").value_counts().head(15)
        summary["top_share"] = float(vc.iloc[0] / len(s)) if len(vc) else 0.0
        chart = {"kind": "bar", "x": [str(v) for v in vc.index],  # levels are labels, not numbers
                 "series": [{"name": "rows", "y": cells(vc)}],
                 "x_title": col, "y_title": "rows"}
        tab = vc.rename_axis(col).reset_index(name="rows")
    return {"title": f"Distribution of {col}", "summary": summary, "table": table(tab),
            "chart": {**chart, "title": f"Distribution of {col}"}}


def target_distribution(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    out = distribution(df, {"column": env["target"]}, env)
    y = pd.to_numeric(df[env["target"]], errors="coerce")
    if env.get("task") == "classification":
        out["summary"]["event_rate"] = float(y.mean())
        out["summary"]["n_events"] = int(y.sum())
    out["title"] = out["chart"]["title"] = f"Target: {env['target']}"
    return out


def target_relationship(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Mean of the target across the range of one feature: bands for a number, levels for a
    category. For a binary target the mean is the event rate."""
    col, target = _column(df, params), env["target"]
    y = pd.to_numeric(df[target], errors="coerce")
    s = df[col]
    rate = "event rate" if env.get("task") == "classification" else f"mean {target}"
    summary: dict[str, Any] = {}
    if _numeric(s) and s.nunique() > 12:
        bins = max(3, min(20, int(params.get("bins") or 10)))
        bands = pd.qcut(s, q=bins, duplicates="drop")
        g = pd.DataFrame({"band": bands, "y": y, "x": s}).groupby("band", observed=True)
        tab = g.agg(rows=("y", "size"), feature_mean=("x", "mean"), target_mean=("y", "mean")
                    ).reset_index()
        tab["band"] = tab["band"].astype(str)
        rho = float(pd.Series(s).corr(y, method="spearman"))
        steps = np.sign(np.diff(tab["target_mean"].to_numpy()))
        summary = {"spearman": rho, "n_bands": int(len(tab)),
                   "monotonic": bool(len(steps) > 0 and (np.all(steps >= 0) or np.all(steps <= 0))),
                   "lowest_band_mean": float(tab["target_mean"].iloc[0]),
                   "highest_band_mean": float(tab["target_mean"].iloc[-1])}
        x = cells(tab["feature_mean"].round(4))
        kind = "line"
    else:
        level = s.astype("object").where(s.notna(), "(missing)")
        g = pd.DataFrame({"level": level, "y": y}).groupby("level", observed=True)
        tab = g.agg(rows=("y", "size"), target_mean=("y", "mean")).reset_index()
        tab = tab.sort_values("rows", ascending=False).head(15)
        summary = {"n_levels": int(level.nunique()),
                   "spread": float(tab["target_mean"].max() - tab["target_mean"].min())}
        x = cells(tab["level"])
        kind = "bar"
    chart = {"kind": kind, "title": f"{rate.capitalize()} by {col}", "x": x, "x_title": col,
             "y_title": rate, "y_format": "%" if env.get("task") == "classification" else "",
             "series": [{"name": rate, "y": cells(tab["target_mean"])}]}
    return {"title": f"{rate.capitalize()} by {col}", "summary": summary, "table": table(tab),
            "chart": chart}


def correlation_matrix(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Pairwise Pearson correlation of the numeric features most related to the target."""
    wanted = [c.strip() for c in (params.get("columns") or "").split(",") if c.strip()]
    target = env.get("target")
    numeric = [c for c in (wanted or env.get("features") or list(df.columns))
               if c in df.columns and _numeric(df[c]) and c != target]
    num = df[numeric].apply(pd.to_numeric, errors="coerce")
    if target and target in df.columns and not wanted:
        order = num.corrwith(pd.to_numeric(df[target], errors="coerce")).abs().sort_values(
            ascending=False)
        numeric = list(order.index[:10])
    numeric = numeric[:12]
    if len(numeric) < 2:
        raise ValueError("fewer than two numeric columns to correlate")
    cols = numeric + ([target] if target and target in df.columns else [])
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(3)
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(numeric) for b in numeric[i + 1:]]
    pairs.sort(key=lambda p: -abs(p[2]))
    top = pairs[0] if pairs else None
    return {
        "title": "Correlation matrix",
        "summary": {"n_columns": len(cols),
                    "strongest_pair": f"{top[0]} / {top[1]}" if top else "none",
                    "strongest_pair_corr": top[2] if top else 0.0,
                    "n_pairs_above_0_8": sum(1 for p in pairs if abs(p[2]) >= 0.8)},
        "table": table(pd.DataFrame(pairs[:20], columns=["a", "b", "corr"])),
        "chart": {"kind": "heatmap", "title": "Correlation matrix", "x": cols, "y": cols,
                  "z": corr.to_numpy().tolist()},
    }


def missingness(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    share = df.isna().mean().sort_values(ascending=False)
    share = share[share > 0].head(25)
    tab = share.rename_axis("column").reset_index(name="missing_share")
    out: dict[str, Any] = {
        "title": "Missing values by column",
        "summary": {"n_columns_with_missing": int((df.isna().mean() > 0).sum()),
                    "worst_share": float(share.iloc[0]) if len(share) else 0.0},
        "table": table(tab)}
    if len(share):
        out["chart"] = {"kind": "bar", "title": "Missing values by column", "x": cells(share.index),
                        "series": [{"name": "missing", "y": cells(share)}], "y_title": "share of rows",
                        "y_format": "%"}
    return out


def time_trend(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Rows and the target's mean per period of the date column (or of ``column``)."""
    col = params.get("column") or env.get("datetime_col") or ""
    if col not in df.columns:
        raise ValueError("no date column: pass column=, or set data.datetime_col")
    target = env["target"]
    period = df[col].astype(str)
    if period.nunique() > 60:  # too fine: fall back to the month or year prefix
        period = period.str.slice(0, 7 if period.str.slice(0, 7).nunique() <= 60 else 4)
    g = pd.DataFrame({"period": period, "y": pd.to_numeric(df[target], errors="coerce")}
                     ).groupby("period")
    tab = g.agg(rows=("y", "size"), target_mean=("y", "mean")).reset_index().sort_values("period")
    rate = "event rate" if env.get("task") == "classification" else f"mean {target}"
    return {
        "title": f"{rate.capitalize()} over {col}",
        "summary": {"n_periods": int(len(tab)), "first_period": str(tab["period"].iloc[0]),
                    "last_period": str(tab["period"].iloc[-1]),
                    "min_period_mean": float(tab["target_mean"].min()),
                    "max_period_mean": float(tab["target_mean"].max()),
                    "last_period_rows": int(tab["rows"].iloc[-1])},
        "table": table(tab),
        "chart": {"kind": "line", "title": f"{rate.capitalize()} over {col}", "x": cells(tab["period"]),
                  "x_title": col, "y_title": rate,
                  "y_format": "%" if env.get("task") == "classification" else "",
                  "series": [{"name": rate, "y": cells(tab["target_mean"])}]},
    }


def register(registry) -> None:
    from ..plugins import AnalysisTool

    for tool in (
        AnalysisTool("distribution", "Histogram of a numeric column, or the most frequent levels "
                     "of a categorical one.", distribution, {"column": "the column to describe"}),
        AnalysisTool("target_distribution", "The dependent variable's distribution; the event "
                     "rate and event count for a binary target.", target_distribution,
                     needs_target=True),
        AnalysisTool("target_relationship", "How the target's mean (the event rate) moves across "
                     "one feature: quantile bands for a number, levels for a category. Shows "
                     "direction, monotonicity and non-linearity.", target_relationship,
                     {"column": "the feature", "bins": "number of quantile bands (default 10)"},
                     needs_target=True),
        AnalysisTool("correlation_matrix", "Pairwise correlation among the numeric features most "
                     "related to the target: finds collinear candidates.", correlation_matrix,
                     {"columns": "optional comma-separated columns (default: the top ten)"}),
        AnalysisTool("missingness", "Share of missing values per column.", missingness),
        AnalysisTool("time_trend", "Row count and the target's mean per period: drift, "
                     "seasonality, thin or right-censored recent periods.", time_trend,
                     {"column": "the date column (default: data.datetime_col)"}, needs_target=True),
    ):
        registry.add_tool(tool)
