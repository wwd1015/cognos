"""The factor library, single-factor analysis and the screen that shortens the library.

Explore hands over columns. A model is rarely built on them as they are: a macro series enters
as a lag, a change or a growth rate, an amount as its log. ``build_library`` widens the columns
into every transformation the plan allows, ``single_factor`` fits each factor alone, and
``screen`` applies the developer's rules, in the order written, until a shortlist is left.

A transformation is a function of one feature column and, for the time operations, of the rows
before it. The target is never in scope, so a factor can be computed on any sample, the sealed
holdout included, without seeing an outcome. Every number here is the engine's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TIME_OPS = ("diff", "pct", "yoy_diff", "yoy_pct", "ma")  # need rows in time order
POINT_OPS = ("log", "sqrt", "square")  # row by row
ALL_OPS = ("lag", *TIME_OPS, *POINT_OPS)
SAME_DIRECTION = ("level", "diff", "pct", "yoy_diff", "yoy_pct", "ma", "log", "sqrt")

OP_LABELS = {
    "level": "as explored", "lag": "lagged", "diff": "change on the period before",
    "pct": "growth on the period before", "yoy_diff": "change on a year before",
    "yoy_pct": "growth on a year before", "ma": "moving average", "log": "logarithm",
    "sqrt": "square root", "square": "square",
}


@dataclass
class Factor:
    """One column of the library: ``transform`` of ``source``, then lagged ``lag`` periods."""

    name: str
    source: str
    transform: str = "level"
    param: int = 0  # periods: 1 for diff / pct, the year for yoy, the window for ma
    lag: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def needs_order(self) -> bool:
        return self.lag > 0 or self.transform in TIME_OPS

    def describe(self) -> str:
        text = OP_LABELS.get(self.transform, self.transform)
        if self.transform == "ma":
            text = f"{self.param}-period moving average"
        return text + (f", lagged {self.lag}" if self.lag else "")


def factor_name(source: str, transform: str, param: int, lag: int) -> str:
    name = source
    if transform != "level":
        name += f"__{transform}" + (str(param) if transform == "ma" else "")
    return name + (f"__lag{lag}" if lag else "")


def infer_period(dates: pd.Series) -> int | None:
    """Periods per year, read from the spacing of the dates (None when it is not regular)."""
    stamps = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(stamps) < 3:
        return None
    days = float(np.median(np.diff(stamps.to_numpy()).astype("timedelta64[D]").astype(float)))
    for lo, hi, period in ((0.5, 1.5, 252), (6, 8, 52), (27, 32, 12), (85, 95, 4), (360, 370, 1)):
        if lo <= days <= hi:
            return period
    return None


def resolve_transforms(asked: list[str], shape: str) -> list[str]:
    """``auto``: time operations for one row per period, otherwise logs of skewed amounts."""
    asked = [a for a in asked or ["auto"]]
    if "none" in asked:
        return []
    out: list[str] = []
    for a in asked:
        if a == "auto":
            out += (["lag", "diff", "yoy_diff", "yoy_pct", "ma"] if shape == "series" else ["log"])
        elif a in ALL_OPS:
            out.append(a)
    if shape != "series":  # a lag needs to know which row came before
        out = [o for o in out if o in POINT_OPS]
    return list(dict.fromkeys(out))


def _base(x: pd.Series, transform: str, param: int) -> pd.Series | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform == "level":
            return x
        if transform == "diff":
            return x - x.shift(param)
        if transform == "yoy_diff":
            return x - x.shift(param)
        if transform in ("pct", "yoy_pct"):
            if not (x.dropna() > 0).all():
                return None  # a growth rate of a series that crosses zero means nothing
            return x / x.shift(param) - 1.0
        if transform == "ma":
            return x.rolling(param).mean()
        if transform == "log":
            xs = x.dropna()
            if (xs > 0).all():
                return np.log(x)
            return np.log1p(x) if (xs >= 0).all() else None
        if transform == "sqrt":
            return np.sqrt(x) if (x.dropna() >= 0).all() else None
        if transform == "square":
            return x * x
    return None


def _compute(frame: pd.DataFrame, f: Factor) -> pd.Series | None:
    x = pd.to_numeric(frame[f.source], errors="coerce").astype(float)
    out = _base(x, f.transform, f.param)
    if out is None:
        return None
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.shift(f.lag) if f.lag else out


def build_library(df: pd.DataFrame, columns: list[str], fc: dict[str, Any], *, shape: str,
                  datetime_col: str | None, protected: tuple[str, ...] = ()
                  ) -> tuple[pd.DataFrame, list[Factor], list[str]]:
    """Every factor the plan allows, as columns beside the explored ones.

    Returns the widened frame (time order when a time operation ran; leading rows a lag
    cannot fill are dropped and counted), the factors, and notes for the record."""
    notes: list[str] = []
    ops = resolve_transforms(list(fc.get("transforms") or ["auto"]), shape)
    ordered = shape == "series" and bool(datetime_col) and datetime_col in df.columns
    frame = (df.sort_values(datetime_col, kind="stable").reset_index(drop=True) if ordered
             else df.reset_index(drop=True))
    period = fc.get("period") or (infer_period(frame[datetime_col]) if ordered else None)
    lags = [0, *sorted({int(k) for k in fc.get("lags") or [] if int(k) > 0})] if "lag" in ops else [0]
    windows = sorted({int(w) for w in fc.get("windows") or [] if int(w) > 1})

    factors: list[Factor] = [Factor(name=c, source=c) for c in columns]
    new: dict[str, pd.Series] = {}
    skipped = 0
    for col in columns:
        if col in protected or not pd.api.types.is_numeric_dtype(frame[col]) \
                or pd.api.types.is_bool_dtype(frame[col]) or frame[col].nunique(dropna=True) <= 2:
            continue  # a category or a flag enters as it is
        bases: list[tuple[str, int]] = [("level", 0)]
        for op in ops:
            if op == "lag":
                continue
            if op in ("diff", "pct"):
                bases.append((op, 1))
            elif op in ("yoy_diff", "yoy_pct"):
                if period and period > 1:
                    bases.append((op, int(period)))
            elif op == "ma":
                bases += [("ma", w) for w in windows]
            elif op == "log":
                # a logarithm earns its place on a long right tail
                if shape == "series" or abs(float(frame[col].skew(skipna=True) or 0.0)) >= 1.0:
                    bases.append(("log", 0))
            else:
                bases.append((op, 0))
        for transform, param in bases:
            for lag in lags:
                if transform == "level" and lag == 0:
                    continue
                f = Factor(name=factor_name(col, transform, param, lag), source=col,
                           transform=transform, param=param, lag=lag)
                if f.name in frame.columns or f.name in new:
                    continue
                values = _compute(frame, f)
                if values is None or values.nunique(dropna=True) < 2:
                    skipped += 1
                    continue
                new[f.name] = values
                factors.append(f)
    if new:
        frame = pd.concat([frame, pd.DataFrame(new)], axis=1)
    if skipped:
        notes.append(f"{skipped} transformation(s) were not computable (a growth rate or a "
                     "logarithm of a series that is not positive, or a constant result)")

    # Leading rows that a lag or a year-on-year change cannot fill.
    derived = [f.name for f in factors if f.needs_order]
    if derived:
        filled = frame[derived].notna().all(axis=1).to_numpy()
        burn = int(np.argmax(filled)) if filled.any() else len(frame)
        if burn:
            frame = frame.iloc[burn:].reset_index(drop=True)
            notes.append(f"the first {burn} period(s) were dropped: the lags and year-on-year "
                         "changes have no history there")
    # A derived factor that is still missing somewhere cannot be fitted on every row.
    broken = [f.name for f in factors
              if f.transform != "level" or f.lag
              if not np.isfinite(pd.to_numeric(frame[f.name], errors="coerce")).all()]
    if broken:
        frame = frame.drop(columns=broken)
        factors = [f for f in factors if f.name not in set(broken)]
        notes.append(f"{len(broken)} transformation(s) were left out: missing on some rows")
    return frame, factors, notes


def apply_library(df: pd.DataFrame, factors: list[Factor | dict], *,
                  datetime_col: str | None = None, replace: bool = False) -> pd.DataFrame:
    """Compute the factors a frame does not carry yet (serving, scenarios). ``replace``
    recomputes the ones it does carry, after their source columns were changed."""
    specs = [f if isinstance(f, Factor) else Factor(**f) for f in factors]
    todo = [f for f in specs if (f.transform != "level" or f.lag)
            and (replace or f.name not in df.columns) and f.source in df.columns]
    if not todo:
        return df
    frame = df
    if any(f.needs_order for f in todo) and datetime_col and datetime_col in df.columns:
        frame = df.sort_values(datetime_col, kind="stable")
    out = df.copy()
    for f in todo:
        values = _compute(frame, f)
        if values is not None:
            out[f.name] = values.reindex(out.index)
    return out


# --- single-factor analysis -----------------------------------------------------------------
def single_factor(dev: pd.DataFrame, y: np.ndarray, factors: list[Factor], *, metric: str,
                  is_classification: bool, hac_lags: int | None = None,
                  validation: Callable[[str], float | None] | None = None) -> list[dict]:
    """Fit every factor alone on the development sample: sign, significance, strength.

    ``hac_lags`` asks for autocorrelation-robust standard errors (one row per period).
    ``validation`` returns the factor's score on the plan's validation sample."""
    import statsmodels.api as sm

    from .metrics import score

    yv = np.asarray(y, dtype=float)
    rows: list[dict] = []
    for f in factors:
        col = dev[f.name]
        numeric = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
        row: dict[str, Any] = {**f.to_dict(), "numeric": bool(numeric),
                               "missing": float(col.isna().mean()), "sign": "n/a",
                               "coefficient": None, "p_value": None, "in_sample": None,
                               "validation": None, "correlation": None, "note": ""}
        ok = col.notna().to_numpy()
        try:
            if numeric:
                x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                ok = ok & np.isfinite(x)
                sd = float(np.std(x[ok])) if ok.any() else 0.0
                if ok.sum() < 8 or sd == 0.0:
                    raise ValueError("too few usable rows, or no variation")
                z = (x[ok] - float(np.mean(x[ok]))) / sd
                design = sm.add_constant(pd.DataFrame({"x": z}), has_constant="add")
                row["correlation"] = float(np.corrcoef(z, yv[ok])[0, 1])
            else:
                dummies = pd.get_dummies(col[ok].astype("object"), drop_first=True, dtype=float)
                if dummies.shape[1] == 0:
                    raise ValueError("one level only")
                design = sm.add_constant(dummies.reset_index(drop=True), has_constant="add")
            if is_classification:
                res = sm.Logit(yv[ok], design).fit(disp=0, maxiter=200)
                pred = np.asarray(res.predict(design), dtype=float)
                row["in_sample"] = score(metric, yv[ok], (pred >= 0.5).astype(int), y_proba=pred)
                overall_p = float(res.llr_pvalue)
            else:
                kw = ({"cov_type": "HAC", "cov_kwds": {"maxlags": int(hac_lags)}}
                      if hac_lags else {})
                res = sm.OLS(yv[ok], design).fit(**kw)
                row["in_sample"] = score(metric, yv[ok], np.asarray(res.predict(design)))
                overall_p = float(res.f_pvalue)
            if numeric:
                coef = float(res.params["x"])
                row["coefficient"] = coef  # per standard deviation of the factor
                row["p_value"] = float(res.pvalues["x"])
                row["sign"] = "0" if abs(coef) < 1e-12 else ("+" if coef > 0 else "-")
            else:
                row["p_value"] = overall_p
        except Exception as exc:  # a factor that cannot be fitted alone is reported, not fatal
            row["note"] = f"not fitted: {type(exc).__name__}: {exc}"[:160]
        if validation is not None and not row["note"]:
            row["validation"] = validation(f.name)
        rows.append(row)
    return rows


def strength(row: dict) -> float | None:
    """What a factor is ranked on: the sample the plan names (``row["ranked_on"]``), else its
    validation score when there is one, else in sample."""
    if row.get("ranked_on") == "in_sample":
        return row.get("in_sample")
    v = row.get("validation")
    return v if v is not None else row.get("in_sample")


def expected_sign(row: dict, expected: dict[str, str]) -> str | None:
    if row["name"] in expected:
        return expected[row["name"]]
    if row.get("transform", "level") in SAME_DIRECTION:
        return expected.get(row["source"])
    return None


def screen(rows: list[dict], rules: dict[str, Any], *, corr: pd.DataFrame | None,
           expected: dict[str, str], maximize: bool,
           protected: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    """Apply the screen. Returns the shortlist (strongest first) and one log line per rule:
    how many factors went in, how many came out, and which were dropped and why. Each row is
    stamped with its fate (``status``, ``dropped_by``, ``reason``)."""
    by_name = {r["name"]: r for r in rows}
    worst = float("-inf") if maximize else float("inf")

    def key(name: str) -> float:
        s = strength(by_name[name])
        s = worst if s is None or not np.isfinite(s) else s
        return -s if maximize else s

    force_in = [n for n in dict.fromkeys([*(rules.get("force_in") or []), *protected])
                if n in by_name]
    alive = sorted(by_name, key=key)
    log: list[dict] = []

    def apply(rule: str, setting: Any, verdict: Callable[[dict], str | None]) -> None:
        nonlocal alive
        before, dropped, keep = len(alive), [], []
        for name in alive:
            why = None if name in force_in else verdict(by_name[name])
            if why:
                dropped.append({"factor": name, "reason": why})
                by_name[name].update(status="dropped", dropped_by=rule, reason=why)
            else:
                keep.append(name)
        alive = keep
        log.append({"rule": rule, "setting": setting, "before": before, "after": len(alive),
                    "dropped": dropped})

    for r in rows:
        r.update(status="kept", dropped_by=None, reason="")
    out = set(rules.get("force_out") or [])
    if out:
        apply("force_out", sorted(out),
              lambda r: "left out by the developer" if r["name"] in out else None)
    apply("not_fitted", None, lambda r: r["note"] or None)
    if rules.get("max_missing") is not None:
        lim = float(rules["max_missing"])
        apply("max_missing", lim,
              lambda r: f"{r['missing']:.0%} missing" if r["missing"] > lim else None)
    if rules.get("max_p_value") is not None:
        lim = float(rules["max_p_value"])
        apply("max_p_value", lim, lambda r: (
            f"p-value {r['p_value']:.3f}" if r["p_value"] is not None and r["p_value"] > lim
            else None))
    if rules.get("enforce_sign"):
        def wrong(r: dict) -> str | None:
            exp = expected_sign(r, expected)
            if exp and r["sign"] in ("+", "-") and r["sign"] != exp:
                return f"sign {r['sign']}, expected {exp}"
            return None
        apply("enforce_sign", True, wrong)
    if rules.get("min_strength") is not None:
        lim = float(rules["min_strength"])

        def weak(r: dict) -> str | None:
            s = strength(r)
            if s is None:
                return None
            return f"alone it scores {s:.4g}" if (s < lim if maximize else s > lim) else None
        apply("min_strength", lim, weak)
    if rules.get("best_per_source"):
        k = int(rules["best_per_source"])
        seen: dict[str, int] = {}

        def crowded(r: dict) -> str | None:
            seen[r["source"]] = seen.get(r["source"], 0) + 1
            return (f"{k} stronger transformation(s) of {r['source']} are kept"
                    if seen[r["source"]] > k else None)
        apply("best_per_source", k, crowded)
    if rules.get("max_correlation") is not None and corr is not None:
        lim = float(rules["max_correlation"])
        kept: list[str] = []

        def twin(r: dict) -> str | None:
            name = r["name"]
            if name in corr.index:
                for other in kept:
                    if other in corr.columns and abs(float(corr.at[name, other])) > lim:
                        return (f"correlation {float(corr.at[name, other]):+.2f} with the "
                                f"stronger {other}")
            kept.append(name)
            return None
        # forced factors are seen first, so nothing is dropped for resembling a weaker one
        alive = [*[n for n in alive if n in force_in], *[n for n in alive if n not in force_in]]
        apply("max_correlation", lim, twin)
        alive = sorted(alive, key=key)
    if rules.get("top_n"):
        n = int(rules["top_n"])
        rank = {name: i for i, name in enumerate(alive)}
        apply("top_n", n, lambda r: (f"rank {rank[r['name']] + 1}, the shortlist keeps {n}"
                                     if rank[r["name"]] >= n else None))
    for name in force_in:
        by_name[name]["reason"] = "kept by rule" if name in protected else "kept by the developer"
    return alive, log
