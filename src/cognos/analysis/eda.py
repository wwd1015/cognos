"""The standard data-exploration battery: the looks every dataset gets before it is modelled.

Like the tools in ``tools.py``, each is a pure function of the dataset and its parameters, so
the numbers are the engine's. They are written for real data, not only the synthetic demos:
missing values, text columns, constant columns and thin bands are expected, and a question that
cannot be answered on this dataset raises ``ValueError`` with the reason (recorded on the
analysis, never a failed stage).

    data_overview       one row per column: type, gaps, distinct values, range, what looks odd
    outliers            share of rows beyond the IQR fences, per numeric column
    predictive_power    univariate ranking: information value and Gini (binary target),
                        |Spearman| or the variance share explained (continuous target)
    woe_bins            weight of evidence across one feature's bands (binary target)
    stability           population stability index of each feature, early periods vs late
    collinearity        variance inflation factors among the numeric candidates
    scatter             two numeric columns against each other
    missing_vs_target   is a gap informative: the target where a column is missing vs present
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .charts import cells, table

MISSING = "(missing)"
OTHER = "(other)"
# Conventional reading of information value and PSI (Siddiqi); used for counts, never a verdict.
IV_WEAK, IV_STRONG, IV_SUSPICIOUS = 0.02, 0.3, 0.5
PSI_WATCH, PSI_SHIFT = 0.1, 0.25


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _continuous(s: pd.Series) -> bool:
    return _numeric(s) and s.nunique() > 12


def _features(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> list[str]:
    """The columns a ranking covers: ``columns=`` when given, else the run's features, else
    every column but the target."""
    wanted = [c.strip() for c in (params.get("columns") or "").split(",") if c.strip()]
    unknown = [c for c in wanted if c not in df.columns]
    if unknown:
        raise ValueError(f"unknown column(s) {unknown}")
    cols = wanted or env.get("features") or list(df.columns)
    return [c for c in cols if c in df.columns and c != env.get("target")]


def _int(params: dict[str, str], key: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(float(params.get(key) or default))))
    except ValueError:
        raise ValueError(f"{key} must be a number") from None


def _bands(s: pd.Series, bins: int = 10, edges: np.ndarray | None = None) -> pd.Series:
    """Each row's band as text: quantile bands for a number, levels for a category (the rarest
    folded into one), and a band of its own for missing values."""
    if _continuous(s):
        x = pd.to_numeric(s, errors="coerce")
        if edges is None:
            cut = pd.qcut(x, q=bins, duplicates="drop")
        else:
            cut = pd.cut(x, bins=edges, include_lowest=True)
        return cut.astype(str).where(x.notna(), MISSING)
    level = s.astype("object").where(s.notna(), MISSING).astype(str)
    keep = level.value_counts().index[:20]
    return level.where(level.isin(keep), OTHER)


def _edges(s: pd.Series, bins: int) -> np.ndarray | None:
    """Quantile edges of a numeric column, open at both ends so later data always falls in."""
    if not _continuous(s):
        return None
    x = pd.to_numeric(s, errors="coerce").dropna()
    inner = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1))[1:-1])
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _binary(df: pd.DataFrame, env: dict[str, Any]) -> pd.Series:
    y = pd.to_numeric(df[env["target"]], errors="coerce")
    if env.get("task") != "classification" or not set(pd.unique(y.dropna())) <= {0, 1}:
        raise ValueError("this tool needs a binary target coded 0/1")
    return y


def _woe(band: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Rows, events and weight of evidence per band. WoE = ln(share of non-events / share of
    events), with half a row added to each cell so an empty cell stays finite."""
    ok = y.notna()
    g = pd.DataFrame({"band": band[ok], "y": y[ok]}).groupby("band", observed=True, sort=False)["y"]
    tab = g.agg(rows="size", events="sum").reset_index()
    good, bad = tab["rows"] - tab["events"], tab["events"]
    dist_good = (good + 0.5) / (good.sum() + 0.5 * len(tab))
    dist_bad = (bad + 0.5) / (bad.sum() + 0.5 * len(tab))
    tab["event_rate"] = tab["events"] / tab["rows"]
    tab["woe"] = np.log(dist_good / dist_bad)
    tab["iv"] = (dist_good - dist_bad) * tab["woe"]
    return tab


def _gini(s: pd.Series, y: pd.Series) -> float | None:
    """|2·AUC − 1| of a numeric column used alone as a score (rank-based)."""
    x = pd.to_numeric(s, errors="coerce")
    ok = x.notna() & y.notna()
    n1, n0 = int((y[ok] == 1).sum()), int((y[ok] == 0).sum())
    if not n1 or not n0:
        return None
    ranks = x[ok].rank()
    auc = (ranks[y[ok] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(abs(2 * auc - 1))


def data_overview(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    n = len(df)
    rows, kinds = [], {"number": 0, "category": 0, "flag": 0, "date": 0}
    constant, id_like = [], []
    for c in df.columns:
        s = df[c]
        distinct = int(s.nunique())
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "date"
        elif pd.api.types.is_bool_dtype(s) or distinct <= 2:
            kind = "flag"
        elif _numeric(s):
            kind = "number"
        else:
            kind = "category"
        kinds[kind] += 1
        notes = []
        if distinct <= 1:
            notes.append("constant")
            constant.append(c)
        elif distinct == n and not pd.api.types.is_float_dtype(s):
            notes.append("one value per row (an identifier?)")
            id_like.append(c)
        elif kind == "category" and distinct > 50:
            notes.append("many levels")
        if c == env.get("target"):
            notes.append("the target")
        row = {"column": c, "type": kind, "missing_share": float(s.isna().mean()),
               "distinct": distinct, "min": None, "median": None, "max": None, "most_frequent": None,
               "note": "; ".join(notes)}
        if kind == "number":
            x = pd.to_numeric(s, errors="coerce")
            row.update(min=float(x.min()), median=float(x.median()), max=float(x.max()))
        elif distinct:
            row["most_frequent"] = str(s.mode(dropna=True).iloc[0])
        rows.append(row)
    shown = {k: v for k, v in kinds.items() if v}
    return {
        "title": "Data overview",
        "summary": {"n_rows": n, "n_columns": int(df.shape[1]), "n_numeric": kinds["number"],
                    "n_categorical": kinds["category"], "n_flags": kinds["flag"],
                    "n_duplicate_rows": int(df.duplicated().sum()),
                    "n_columns_with_missing": int(df.isna().any().sum()),
                    "n_constant": len(constant), "n_identifier_like": len(id_like)},
        "table": table(pd.DataFrame(rows), max_rows=200),
        "chart": {"kind": "bar", "title": "Columns by type", "x": list(shown),
                  "series": [{"name": "columns", "y": cells(shown.values())}],
                  "y_title": "columns"},
    }


def outliers(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for c in _features(df, params, env):
        if not _continuous(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not len(x) or iqr == 0:
            continue
        rows.append({"column": c,
                     "outside_1_5_iqr": float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()),
                     "outside_3_iqr": float(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean()),
                     "p1": float(x.quantile(0.01)), "median": float(x.median()),
                     "p99": float(x.quantile(0.99)), "min": float(x.min()), "max": float(x.max()),
                     "skew": float(x.skew())})
    if not rows:
        raise ValueError("no numeric column with a spread to check")
    tab = pd.DataFrame(rows).sort_values(["outside_3_iqr", "outside_1_5_iqr"], ascending=False)
    top = tab.head(15)
    worst = tab.iloc[0]
    return {
        "title": "Outliers by column",
        "summary": {"n_columns_checked": int(len(tab)),
                    "n_columns_with_extremes": int((tab["outside_3_iqr"] > 0).sum()),
                    "most_extreme_column": str(worst["column"]),
                    "most_extreme_share": float(worst["outside_3_iqr"]),
                    "most_skewed_column": str(tab.loc[tab["skew"].abs().idxmax(), "column"]),
                    "largest_abs_skew": float(tab["skew"].abs().max())},
        "table": table(tab, max_rows=100),
        "chart": {"kind": "bar", "title": "Rows beyond the IQR fences", "x": cells(top["column"]),
                  "y_title": "share of rows", "y_format": "%",
                  "series": [{"name": "beyond 1.5 × IQR", "y": cells(top["outside_1_5_iqr"])},
                             {"name": "beyond 3 × IQR", "y": cells(top["outside_3_iqr"])}]},
    }


def predictive_power(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Each feature on its own against the target. A ranking for screening: it says nothing
    about what a feature adds beside the others."""
    cols = _features(df, params, env)
    y = pd.to_numeric(df[env["target"]], errors="coerce")
    binary = env.get("task") == "classification" and set(pd.unique(y.dropna())) <= {0, 1}
    bins = _int(params, "bins", 10, 3, 20)
    rows = []
    for c in cols:
        s = df[c]
        if s.nunique() <= 1:
            continue
        row: dict[str, Any] = {"feature": c, "type": "number" if _numeric(s) else "category",
                               "missing_share": float(s.isna().mean())}
        if binary:
            row["information_value"] = float(_woe(_bands(s, bins), y)["iv"].sum())
            row["gini"] = _gini(s, y) if _numeric(s) else None
        elif _numeric(s):
            row["strength"] = abs(float(pd.to_numeric(s, errors="coerce").corr(y, method="spearman")))
            row["measure"] = "|spearman|"
        else:  # share of the target's variance between the levels (eta squared)
            ok = y.notna()
            means = y[ok].groupby(_bands(s)[ok]).transform("mean")
            total = float(((y[ok] - y[ok].mean()) ** 2).sum())
            row["strength"] = float(((means - y[ok].mean()) ** 2).sum() / total) if total else 0.0
            row["measure"] = "variance share"
        rows.append(row)
    if not rows:
        raise ValueError("no feature varies")
    key = "information_value" if binary else "strength"
    tab = pd.DataFrame(rows).sort_values(key, ascending=False, na_position="last")
    top = tab.head(15)
    best = tab.iloc[0]
    summary: dict[str, Any] = {"n_features": int(len(tab)), "strongest": str(best["feature"]),
                               f"strongest_{key}": float(best[key])}
    if binary:
        iv = tab["information_value"]
        summary.update(n_iv_above_0_3=int((iv >= IV_STRONG).sum()),
                       n_iv_below_0_02=int((iv < IV_WEAK).sum()),
                       n_iv_above_0_5_check_leakage=int((iv > IV_SUSPICIOUS).sum()))
    label = "information value" if binary else "strength"
    return {"title": "Univariate predictive power", "summary": summary,
            "table": table(tab, max_rows=100),
            "chart": {"kind": "bar", "title": f"Univariate predictive power ({label})",
                      "x": cells(top["feature"]), "y_title": label,
                      "series": [{"name": label, "y": cells(top[key])}]}}


def woe_bins(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    col = params.get("column", "")
    if col not in df.columns:
        raise ValueError(f"unknown column {col!r}")
    y = _binary(df, env)
    s = df[col]
    band = _bands(s, _int(params, "bins", 10, 3, 20))
    tab = _woe(band, y)
    if _continuous(s):  # bands in the feature's order, the missing band last
        low = pd.to_numeric(s, errors="coerce").groupby(band).min()
        tab = tab.assign(_low=tab["band"].map(low)).sort_values("_low", na_position="last"
                                                                 ).drop(columns="_low")
    else:
        tab = tab.sort_values("woe")
    ordered = tab[tab["band"] != MISSING]["woe"].to_numpy()
    steps = np.sign(np.diff(ordered))
    return {
        "title": f"Weight of evidence by {col}",
        "summary": {"information_value": float(tab["iv"].sum()), "n_bands": int(len(tab)),
                    "monotonic": bool(len(steps) > 0 and (np.all(steps >= 0) or np.all(steps <= 0))),
                    "lowest_event_rate": float(tab["event_rate"].min()),
                    "highest_event_rate": float(tab["event_rate"].max()),
                    "smallest_band_rows": int(tab["rows"].min())},
        "table": table(tab),
        "chart": {"kind": "bar", "title": f"Weight of evidence by {col}", "x": cells(tab["band"]),
                  "x_title": col, "y_title": "weight of evidence (higher = fewer events)",
                  "series": [{"name": "WoE", "y": cells(tab["woe"])}]},
    }


def _psi(ref: pd.Series, new: pd.Series) -> float:
    levels = sorted(set(ref.unique()) | set(new.unique()))
    a = ref.value_counts().reindex(levels, fill_value=0)
    b = new.value_counts().reindex(levels, fill_value=0)
    pa = (a + 0.5) / (a.sum() + 0.5 * len(levels))
    pb = (b + 0.5) / (b.sum() + 0.5 * len(levels))
    return float(((pb - pa) * np.log(pb / pa)).sum())


def stability(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Has the population moved? PSI of each feature between the earlier periods (the reference)
    and the later ones. ``split`` names the first period of the later part; without it the
    periods are cut in half."""
    col = params.get("column") or env.get("datetime_col") or ""
    if col not in df.columns:
        raise ValueError("no date column: pass column=, or set data.datetime_col")
    period = df[col].astype(str)
    periods = sorted(period.dropna().unique())
    if len(periods) < 2:
        raise ValueError(f"{col} has one period: nothing to compare")
    split = params.get("split") or periods[len(periods) // 2]
    late = period >= split
    if not late.any() or late.all():
        raise ValueError(f"split {split!r} leaves one side empty")
    bins = _int(params, "bins", 10, 3, 20)
    rows = []
    for c in _features(df, params, env):
        s = df[c]
        if c == col or s.nunique() <= 1:
            continue
        edges = _edges(s[~late], bins)
        psi = _psi(_bands(s[~late], bins, edges), _bands(s[late], bins, edges))
        row = {"feature": c, "psi": psi, "reference_mean": None, "recent_mean": None}
        if _numeric(s):
            x = pd.to_numeric(s, errors="coerce")
            row.update(reference_mean=float(x[~late].mean()), recent_mean=float(x[late].mean()))
        rows.append(row)
    if not rows:
        raise ValueError("no feature varies")
    tab = pd.DataFrame(rows).sort_values("psi", ascending=False)
    top = tab.head(15)
    summary: dict[str, Any] = {
        "reference_periods": f"{periods[0]} to {max(p for p in periods if p < split)}",
        "recent_periods": f"{split} to {periods[-1]}",
        "reference_rows": int((~late).sum()), "recent_rows": int(late.sum()),
        "most_shifted": str(tab.iloc[0]["feature"]), "largest_psi": float(tab.iloc[0]["psi"]),
        "n_psi_above_0_25": int((tab["psi"] >= PSI_SHIFT).sum()),
        "n_psi_0_1_to_0_25": int(((tab["psi"] >= PSI_WATCH) & (tab["psi"] < PSI_SHIFT)).sum())}
    if env.get("target") in df.columns:
        y = pd.to_numeric(df[env["target"]], errors="coerce")
        summary.update(reference_target_mean=float(y[~late].mean()),
                       recent_target_mean=float(y[late].mean()))
    return {"title": f"Population stability over {col}", "summary": summary,
            "table": table(tab, max_rows=100),
            "chart": {"kind": "bar", "title": "Population stability index, early vs late periods",
                      "x": cells(top["feature"]), "y_title": "PSI",
                      "series": [{"name": "PSI", "y": cells(top["psi"])}]}}


def collinearity(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    """Variance inflation factor of each numeric candidate: how well the others predict it."""
    numeric = [c for c in _features(df, params, env) if _numeric(df[c]) and df[c].nunique() > 1]
    target = env.get("target")
    num = df[numeric].apply(pd.to_numeric, errors="coerce")
    if not params.get("columns") and target in df.columns and len(numeric) > 15:
        order = num.corrwith(pd.to_numeric(df[target], errors="coerce")).abs().sort_values(
            ascending=False)
        numeric = list(order.index[:15])
    if len(numeric) < 2:
        raise ValueError("fewer than two numeric columns to compare")
    x = num[numeric].fillna(num[numeric].median())
    vif = np.diag(np.linalg.pinv(x.corr().to_numpy()))
    tab = pd.DataFrame({"feature": numeric, "vif": vif}).sort_values("vif", ascending=False)
    return {
        "title": "Collinearity (variance inflation factors)",
        "summary": {"n_columns": int(len(tab)), "highest": str(tab.iloc[0]["feature"]),
                    "highest_vif": float(tab.iloc[0]["vif"]),
                    "n_vif_above_5": int((tab["vif"] > 5).sum()),
                    "n_vif_above_10": int((tab["vif"] > 10).sum())},
        "table": table(tab),
        "chart": {"kind": "bar", "title": "Variance inflation factor", "x": cells(tab["feature"]),
                  "y_title": "VIF", "series": [{"name": "VIF", "y": cells(tab["vif"])}]},
    }


def scatter(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    a, b = params.get("column", ""), params.get("against") or env.get("target") or ""
    for c in (a, b):
        if c not in df.columns:
            raise ValueError(f"unknown column {c!r}")
        if not _numeric(df[c]):
            raise ValueError(f"{c} is not numeric")
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("fewer than three rows have both values")
    shown = pair.sample(n=min(400, len(pair)), random_state=0).sort_values(a)
    return {
        "title": f"{b} against {a}",
        "summary": {"n": int(len(pair)), "pearson": float(pair[a].corr(pair[b])),
                    "spearman": float(pair[a].corr(pair[b], method="spearman")),
                    "points_shown": int(len(shown))},
        "table": table(pair.describe().T.rename_axis("column").reset_index()),
        "chart": {"kind": "scatter", "title": f"{b} against {a}", "x": cells(shown[a]),
                  "x_title": a, "y_title": b, "series": [{"name": b, "y": cells(shown[b])}]},
    }


def missing_vs_target(df: pd.DataFrame, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    y = pd.to_numeric(df[env["target"]], errors="coerce")
    least = _int(params, "min_rows", 20, 1, 10 ** 9)
    rows = []
    for c in _features(df, params, env):
        gap = df[c].isna()
        if gap.sum() < least or (~gap).sum() < least:
            continue
        rows.append({"column": c, "missing_share": float(gap.mean()), "missing_rows": int(gap.sum()),
                     "target_mean_when_missing": float(y[gap].mean()),
                     "target_mean_when_present": float(y[~gap].mean())})
    rate = "event rate" if env.get("task") == "classification" else f"mean {env['target']}"
    out: dict[str, Any] = {"title": f"{rate.capitalize()} where a value is missing",
                           "summary": {"n_columns_compared": len(rows), "min_rows_per_side": least}}
    if not rows:
        return out
    tab = pd.DataFrame(rows)
    tab["difference"] = tab["target_mean_when_missing"] - tab["target_mean_when_present"]
    tab = tab.reindex(tab["difference"].abs().sort_values(ascending=False).index)
    top = tab.head(15)
    out["summary"].update(largest_gap_column=str(tab.iloc[0]["column"]),
                          largest_gap=float(tab.iloc[0]["difference"]))
    out["table"] = table(tab)
    out["chart"] = {"kind": "bar", "title": out["title"], "x": cells(top["column"]),
                    "y_title": rate, "y_format": "%" if env.get("task") == "classification" else "",
                    "series": [{"name": "missing", "y": cells(top["target_mean_when_missing"])},
                               {"name": "present", "y": cells(top["target_mean_when_present"])}]}
    return out


def register(registry) -> None:
    from ..plugins import AnalysisTool

    columns = "optional comma-separated columns (default: the run's features)"
    bins = "number of quantile bands for a numeric column (default 10)"
    for tool in (
        AnalysisTool("data_overview", "One row per column: type, share missing, distinct values, "
                     "range or most frequent level, and what looks odd (constant, identifier-"
                     "like, many levels). Also counts duplicate rows. Start here.", data_overview),
        AnalysisTool("outliers", "Share of rows beyond the 1.5 × and 3 × IQR fences for each "
                     "numeric column, with its tails and skew: candidates for capping or a "
                     "transform.", outliers, {"columns": columns}),
        AnalysisTool("predictive_power", "Ranks every feature on its own against the target: "
                     "information value and Gini for a binary target, |Spearman| or the "
                     "variance share explained otherwise. A very high value is a leakage "
                     "suspect.", predictive_power, {"columns": columns, "bins": bins},
                     needs_target=True),
        AnalysisTool("woe_bins", "Weight of evidence and event rate across one feature's bands "
                     "(binary target): the shape a scorecard or a logit would have to follow.",
                     woe_bins, {"column": "the feature", "bins": bins}, needs_target=True),
        AnalysisTool("stability", "Population stability index of each feature between the "
                     "earlier and the later periods of the date column: which inputs have "
                     "drifted.", stability,
                     {"column": "the date column (default: data.datetime_col)",
                      "split": "first period of the later part (default: the middle period)",
                      "columns": columns, "bins": bins}),
        AnalysisTool("collinearity", "Variance inflation factor of each numeric candidate: which "
                     "ones the others already explain.", collinearity, {"columns": columns}),
        AnalysisTool("scatter", "Two numeric columns against each other (a sample of at most "
                     "400 points), with their Pearson and Spearman correlation.", scatter,
                     {"column": "the column on the x axis",
                      "against": "the column on the y axis (default: the target)"}),
        AnalysisTool("missing_vs_target", "Is a gap informative? The target's mean where each "
                     "column is missing against where it is present.", missing_vs_target,
                     {"columns": columns, "min_rows": "rows needed on each side (default 20)"},
                     needs_target=True),
    ):
        registry.add_tool(tool)
