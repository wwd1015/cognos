"""The factor library, single-factor analysis and the screen that shortens the library.

Explore hands over columns. A model is rarely built on them as they are: a macro series enters
as a lag, a change or a growth rate, an amount as its log. ``build_library`` widens the columns
into every transformation the plan allows, ``single_factor`` fits each factor alone, and
``screen`` applies the developer's rules, in the order written, until a shortlist is left.

A transformation is a function of one feature column and, for the time operations, of the rows
before it. The target is never in scope, so a factor can be computed on any sample, the sealed
holdout included, without seeing an outcome. Every number here is the engine's.

A *shaped* factor (``cap``, ``knee``, ``smooth``; ADR-0024) is the one exception, and it is
kept narrow: its points are fitted by ``transforms.library`` on the development sample alone,
after the split, and then frozen. Computing it on any other sample reads the points, never an
outcome. A column with its own transformation (``fc["columns"]``) enters only in that form.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TIME_OPS = ("diff", "pct", "yoy_diff", "yoy_pct", "ma")  # need rows in time order
POINT_OPS = ("log", "sqrt", "square")  # row by row
SHAPE_OPS = ("cap", "knee", "smooth")  # fitted on the development sample (transforms)
ALL_OPS = ("lag", *TIME_OPS, *POINT_OPS)  # plan-wide; shapes are a column's own (ADR-0024)
SAME_DIRECTION = ("level", "diff", "pct", "yoy_diff", "yoy_pct", "ma", "log", "sqrt", "cap")
CAP_QUANTILES = (0.01, 0.99)

OP_LABELS = {
    "level": "as explored", "lag": "lagged", "diff": "change on the period before",
    "pct": "growth on the period before", "yoy_diff": "change on a year before",
    "yoy_pct": "growth on a year before", "ma": "moving average", "log": "logarithm",
    "sqrt": "square root", "square": "square", "cap": "capped at the 1st and 99th percentiles",
    "knee": "shaped with knees", "smooth": "monotone smooth", "custom": "custom expression",
}


@dataclass
class Factor:
    """One column of the library: ``transform`` of ``source``, then lagged ``lag`` periods."""

    name: str
    source: str
    transform: str = "level"
    param: int = 0  # periods: 1 for diff / pct, the year for yoy, the window for ma; knees
    lag: int = 0
    points: list[list[float]] = field(default_factory=list)  # shaped: (x, value), fitted once
    expr: str = ""  # custom: an expression of x, the source column (ADR-0024)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def needs_order(self) -> bool:
        return self.lag > 0 or self.transform in TIME_OPS

    def describe(self) -> str:
        text = OP_LABELS.get(self.transform, self.transform)
        if self.transform == "ma":
            text = f"{self.param}-period moving average"
        if self.transform == "cap" and self.points:
            text = f"capped to [{self.points[0][0]:.4g}, {self.points[-1][0]:.4g}]"
        if self.transform == "knee" and self.points:
            knots = ", ".join(f"{x:.4g}" for x, _ in self.points[1:-1])
            text = f"{len(self.points) - 2} knee(s) at {knots}" if knots else "capped, straight"
        if self.transform == "custom":
            text = f"custom: {self.expr}"
        if self.transform == "smooth" and self.points:
            text = f"monotone smooth through {len(self.points)} points"
        return text + (f", lagged {self.lag}" if self.lag else "")


def factor_name(source: str, transform: str, param: int, lag: int) -> str:
    name = source
    if transform != "level":
        name += f"__{transform}" + (str(param) if transform in ("ma", "knee") else "")
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


def own_factor(col: str, spec: dict[str, Any], period: int | None) -> Factor | None:
    """The factor a column's own ``log``, ``change`` or ``lag`` makes (ADR-0024): a function of
    the column alone. None for a kind fitted after the split, or one that cannot run."""
    kind = spec.get("kind")
    lag = int(spec.get("lag") or 0)
    if kind == "log":
        return Factor(name=factor_name(col, "log", 0, 0), source=col, transform="log")
    if kind == "custom":
        expr = str(spec.get("expr") or "").strip()
        return (Factor(name=factor_name(col, "custom", 0, 0), source=col, transform="custom",
                       expr=expr) if expr else None)
    if kind == "lag" and lag > 0:
        return Factor(name=factor_name(col, "level", 0, lag), source=col, lag=lag)
    if kind == "change":
        op = spec.get("op") or "diff"
        param = 1 if op in ("diff", "pct") else int(period or 0)
        if param < 1:
            return None  # a year-on-year change needs to know how many periods make a year
        return Factor(name=factor_name(col, op, param, lag), source=col, transform=op,
                      param=param, lag=lag)
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


def custom_values(x: pd.Series, expr: str, source: str) -> pd.Series:
    """A custom expression of one column (ADR-0024): ``x`` (or the column's own name) is the
    column; nothing else is in scope, the outcome never. Raises ``UnsafeExpressionError`` (a
    ``ValueError``) for anything the safe expression rules refuse."""
    from .transforms import evaluate_expr

    names = {"x": x.to_numpy(dtype=float)}
    if source.isidentifier():
        names[source] = names["x"]
    values = evaluate_expr(expr, pd.DataFrame(names))
    out = pd.Series(np.broadcast_to(values, (len(x),)).astype(float), index=x.index)
    return out.where(x.notna())


def _compute(frame: pd.DataFrame, f: Factor) -> pd.Series | None:
    x = pd.to_numeric(frame[f.source], errors="coerce").astype(float)
    if f.transform == "custom":
        try:
            return custom_values(x, f.expr, f.source).replace([np.inf, -np.inf], np.nan)
        except (ValueError, TypeError, ArithmeticError):
            return None
    if f.transform in SHAPE_OPS:
        return _through(x, f.points) if f.points else None
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
    own = {c: (s if isinstance(s, dict) else s.model_dump())
           for c, s in (fc.get("columns") or {}).items() if c in columns}
    own = {c: s for c, s in own.items() if s.get("kind") != "library"}  # widened by the plan
    replaced: set[str] = set()  # columns that enter only through their own factor
    new: dict[str, pd.Series] = {}
    skipped = 0
    for col in columns:
        if col in protected or not pd.api.types.is_numeric_dtype(frame[col]) \
                or pd.api.types.is_bool_dtype(frame[col]) or frame[col].nunique(dropna=True) <= 2:
            continue  # a category or a flag enters as it is
        if col in own:  # the column's own transformation, decided at the gates (ADR-0024)
            f = own_factor(col, own[col], period)
            if f is None:
                continue  # as it is, or fitted after the split (transforms.library)
            values = _compute(frame, f)
            if values is None or values.nunique(dropna=True) < 2:
                notes.append(f"{col}: its {own[col]['kind']} could not be computed; it enters "
                             "as it is")
                continue
            new[f.name] = values
            factors.append(f)
            replaced.add(col)
            continue
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
    factors = [f for f in factors if not (f.source in replaced and f.transform == "level"
                                          and not f.lag)]
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
            and (f.transform not in SHAPE_OPS or f.points)
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


# --- shaped factors (ADR-0024) --------------------------------------------------------------
def _through(x: pd.Series, points: list[list[float]]) -> pd.Series:
    """A shaped factor: the line through ``points``, flat beyond the first and the last."""
    xs = np.array([pt[0] for pt in points], dtype=float)
    ys = np.array([pt[1] for pt in points], dtype=float)
    values = np.interp(x.to_numpy(dtype=float), xs, ys)
    return pd.Series(np.where(x.isna().to_numpy(), np.nan, values), index=x.index)


def shapeable(s: pd.Series) -> bool:
    """A column a cap or a knee means something for: numeric, not a flag, many values."""
    return (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            and s.nunique(dropna=True) > 10)


def _fit_line(u: np.ndarray, y: np.ndarray, knots: list[float], classification: bool
              ) -> tuple[np.ndarray, float, float]:
    """A straight line in ``u`` that may bend at ``knots``: coefficients, intercept and the
    log-likelihood (logistic for a 0/1 outcome, Gaussian otherwise)."""
    X = np.column_stack([u] + [np.maximum(u - k, 0.0) for k in knots])
    if classification:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=1e4, max_iter=1000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return m.coef_[0], float(m.intercept_[0]), float(np.sum(y * np.log(p)
                                                               + (1 - y) * np.log(1 - p)))
    A = np.column_stack([np.ones(len(u)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta[1:], float(beta[0]), float(-0.5 * len(y) * np.log(max(float(resid @ resid)
                                                                     / len(y), 1e-300)))


def fit_shape(x: pd.Series, y: Any, *, cap: bool = True, knees: int = 0,
              knots: list[float] | None = None, direction: str | None = None,
              classification: bool = True, grid: int = 9,
              quantiles: tuple[float, float] = CAP_QUANTILES) -> dict | None:
    """Fit one column's shape on the rows given (the development sample, ADR-0024).

    The column is capped at its ``quantiles`` (the 1st and 99th percentiles), then a line
    that may bend at up to ``knees`` knees is fitted, each segment held to ``direction`` (the expected sign, else the
    data's) so the factor is monotonic. Knees are tried at the deciles; the number used is
    the one with the best BIC (a knee costs its slope; its place is one of
    nine), so asking for two is asking for at most two. Knots a person gave
    are kept as given. Returns the points the factor runs through, the observed and fitted
    curves, and what was decided, or None when the column cannot be shaped."""
    s = pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce").astype(float).to_numpy()
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(s) & np.isfinite(yv)
    xv, yv = s[ok], yv[ok]
    if len(xv) < 30 or (classification and len(np.unique(yv)) < 2):
        return None
    lo, hi = (np.quantile(xv, quantiles) if cap else (xv.min(), xv.max()))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return None
    u = (np.clip(xv, lo, hi) - lo) / (hi - lo)
    direction_from = "expected sign" if direction in ("+", "-") else "the data"
    if direction not in ("+", "-"):
        coef, _, _ = _fit_line(u, yv, [], classification)
        direction = "+" if coef[0] >= 0 else "-"
    sign = 1.0 if direction == "+" else -1.0

    def monotonic(coef: np.ndarray) -> bool:
        return bool(np.all(sign * np.cumsum(coef) >= -1e-9))

    def held(coef: np.ndarray) -> np.ndarray | None:
        """The fit with a segment that turns slightly the wrong way laid flat (at most a
        tenth of the steepest slope), or None when a segment turns clearly the wrong way."""
        slopes = np.cumsum(coef)
        wrong = sign * slopes < 0
        if not wrong.any():
            return coef
        if np.max(np.abs(slopes[wrong])) > 0.1 * np.max(np.abs(slopes)):
            return None
        slopes = np.where(wrong, 0.0, slopes)
        return np.diff(np.concatenate([[0.0], slopes]))

    given = sorted((k - lo) / (hi - lo) for k in (knots or []) if lo < k < hi)
    if given:  # kept where a person put them; a slight wrong-way slope is laid flat too
        coef, b0, ll = _fit_line(u, yv, given, classification)
        kept = held(coef)
        chosen, origin = (given, coef if kept is None else kept, b0), "given"
    else:
        cuts = np.unique(np.round(np.quantile(u, np.linspace(0.1, 0.9, grid)), 6))
        cuts = [float(c) for c in cuts if 0 < c < 1]
        best, best_bic = None, np.inf
        for k in range(0, max(0, int(knees)) + 1):
            for combo in itertools.combinations(cuts, k):
                coef, b0, ll = _fit_line(u, yv, list(combo), classification)
                bic = -2 * ll + (k + 2) * np.log(len(u))  # a knee costs its slope
                kept = held(coef)
                if kept is not None and bic < best_bic:
                    best, best_bic = (list(combo), kept, b0), bic
        if best is None:  # not even a straight line goes the expected way
            coef, b0, _ = _fit_line(u, yv, [], classification)
            best = ([], coef, b0)
        chosen, origin = best, "fitted"
    knots_u, coef, b0 = chosen

    def g(uu: np.ndarray) -> np.ndarray:
        return np.column_stack([uu] + [np.maximum(uu - k, 0.0) for k in knots_u]) @ coef

    at = np.array([0.0, *knots_u, 1.0])
    points = [[float(lo + a * (hi - lo)), float(v)] for a, v in zip(at, g(at), strict=True)]
    line = np.linspace(0.0, 1.0, 61)
    fitted = g(line) + b0
    if classification:
        fitted = 1.0 / (1.0 + np.exp(-fitted))
    bands = pd.qcut(pd.Series(xv), q=min(12, len(np.unique(xv))), duplicates="drop")
    seen = pd.DataFrame({"x": xv, "y": yv, "band": bands}).groupby("band", observed=True).agg(
        x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
    return {
        "cap": [lo, hi], "knots": [float(lo + k * (hi - lo)) for k in knots_u],
        "knees_asked": int(knees), "knees_used": len(knots_u), "origin": origin,
        "direction": direction, "direction_from": direction_from,
        "monotonic": monotonic(coef), "points": points, "rows": int(len(xv)),
        "observed": [[float(r.x), float(r.y), int(r.n)] for r in seen.itertuples()],
        "curve": [[float(lo + a * (hi - lo)), float(v)] for a, v in zip(line, fitted,
                                                                       strict=True)],
    }


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
    if row.get("transform") in ("knee", "smooth"):
        return "+"  # a knee or a smooth factor is already turned to rise with the outcome
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
