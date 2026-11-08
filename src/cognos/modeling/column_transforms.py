"""How each column enters the model, decided in two steps (ADR-0024).

A scorecard needs every factor to rank risk in order, a loss regression a straight-line
relationship that a few extreme values do not drive, a loss forecast a driver that does not
drift on its own. The *intent* of the model (``intent_of``) says which property a column must
have and which transformations may give it.

1. **Whether** (explore, the data gate). ``review`` reads every row explore has, checks each
   column for the property its intent needs and recommends a kind, never a parameter: the
   sample is not split yet, so nothing measured here may reach a model.
2. **How** (ideate, the design gate). ``fit`` places the parameters on the development
   sample alone, cut exactly as the model stage cuts it; a person may change them there.

The model stage fits again, on its own development sample, with what was decided
(``library``), and freezes the points: the test sample and the holdout read them, never an
outcome. ``log``, ``change`` and ``lag`` are functions of the column alone
(``factors.own_factor``) and are computed with the rest of the library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import ColumnTransform
from . import factors as factors_mod

KINDS: dict[str, dict[str, str]] = {
    "as_is": {"label": "as it is", "family": "none",
              "hint": "enters the model as explored"},
    "cap": {"label": "capped", "family": "fitted",
            "hint": "held inside its lower and upper percentiles, so a few extreme values "
                    "cannot drive the slope"},
    "knee": {"label": "bends at knees", "family": "fitted",
             "hint": "a line that may bend once or twice and keeps one direction"},
    "smooth": {"label": "monotone smooth", "family": "fitted",
               "hint": "a local (LOWESS) curve held to one direction"},
    "log": {"label": "logarithm", "family": "point",
            "hint": "straightens a long right tail"},
    "change": {"label": "change", "family": "time",
               "hint": "its change over a period or a year instead of its level"},
    "lag": {"label": "lagged", "family": "time",
            "hint": "the value some periods earlier"},
    "library": {"label": "the plan's lags and changes", "family": "plan",
                "hint": "every lag and change the plan allows competes in the screen"},
    "custom": {"label": "custom expression", "family": "point",
               "hint": "your own expression of x, the column: arithmetic and np functions "
                       "such as np.clip, np.minimum, np.where, np.log1p"},
}

INTENTS: dict[str, dict[str, Any]] = {
    "rank": {"label": "rank ordering",
             "need": "Each variable should move the outcome one way, so the score ranks risk "
                     "in order. A variable that turns, bends or has a long tail is reshaped.",
             "kinds": ("as_is", "cap", "knee", "smooth", "log", "custom")},
    "linear": {"label": "a straight-line relationship",
               "need": "Each variable should relate to the outcome along a straight line that "
                       "a few extreme values do not drive.",
               "kinds": ("as_is", "log", "cap", "knee", "custom")},
    "stationary": {"label": "drivers that do not drift",
                   "need": "Each driver should move with the outcome, not drift on its own; a "
                           "driver whose effect arrives later enters lagged.",
                   "kinds": ("library", "as_is", "change", "lag", "log", "custom")},
}

FITTED = tuple(k for k, v in KINDS.items() if v["family"] == "fitted")
BANDS = 12


def intent_of(task: str | None, shape: str | None = None) -> str | None:
    """The property the model's intent asks of each column, or None when there is none."""
    if task == "timeseries" or shape == "series":
        return "stationary"
    return {"classification": "rank", "regression": "linear"}.get(task or "")


def normalize(spec: dict[str, Any] | str) -> dict[str, Any]:
    """A column's transformation with every parameter filled in (``ColumnTransform``)."""
    raw = {"kind": spec} if isinstance(spec, str) else {k: v for k, v in spec.items()
                                                          if k in ColumnTransform.model_fields}
    return ColumnTransform(**raw).model_dump()


def check_expr(expr: str, col: str) -> str:
    """A custom expression, cleaned, or ``ValueError`` saying why it cannot be used: it must
    pass the safe expression rules and name no column but ``x`` (or ``col`` itself)."""
    from .transforms import UnsafeExpressionError, validate_expr

    expr = " ".join(str(expr or "").split())
    if not expr:
        raise ValueError("write an expression of x, for example np.clip(x, 0, 5)")
    try:
        validate_expr(expr, {"x", col})
    except UnsafeExpressionError as exc:
        raise ValueError(f"{exc}; use x for {col}, numbers and np functions") from exc
    return expr


def lay_over(plan: dict, own: dict[str, Any] | None, source: str = "human") -> dict:
    """The plan with the transformations decided at the gates laid over its own."""
    own = {c: normalize(s) for c, s in (own or {}).items()}
    if own:
        plan["factors"]["columns"] = {**(plan["factors"].get("columns") or {}), **own}
        plan.setdefault("sources", {})["factors.columns"] = source
    return plan


def decided(own: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The columns given a transformation of their own (not kept as they are, not left to the
    plan's library), their specs filled in: the ones the design gate sets."""
    out = {}
    for col, spec in (own or {}).items():
        spec = normalize(spec if isinstance(spec, dict | str) else spec.model_dump())
        if spec["kind"] not in ("as_is", "library"):
            out[col] = spec
    return out


def _bands(x: np.ndarray, y: np.ndarray, q: int = BANDS) -> list[list[float]]:
    bands = pd.qcut(pd.Series(x), q=min(q, len(np.unique(x))), duplicates="drop")
    seen = pd.DataFrame({"x": x, "y": y, "band": bands}).groupby("band", observed=True).agg(
        x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
    return [[float(r.x), float(r.y), int(r.n)] for r in seen.itertuples()]


def _lowess(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    """A local regression of ``y`` on ``x``: (x, fitted) pairs, sorted by x."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    span = float(np.ptp(x)) or 1.0
    return lowess(y, x, frac=frac, it=0, delta=0.01 * span, return_sorted=True)


def _thin(pairs: np.ndarray, n: int = 60) -> list[list[float]]:
    if len(pairs) > n:
        pairs = pairs[np.linspace(0, len(pairs) - 1, n).round().astype(int)]
    return [[float(a), float(b)] for a, b in pairs]


def _one_way(x: np.ndarray, y: np.ndarray, sign: float) -> tuple[bool | None, str]:
    """Whether the outcome moves the column's way through five equal bands. A step the other
    way counts only beyond two standard errors of the difference, so noise is not a turn.
    None when the outcome barely moves across the bands at all."""
    bands = pd.qcut(pd.Series(x), q=5, duplicates="drop").to_numpy()
    g = pd.Series(y).groupby(bands, observed=True)
    means, var, n = g.mean().to_numpy(), g.var().fillna(0.0).to_numpy(), g.size().to_numpy()
    if len(means) < 2:
        return True, ""
    se = np.sqrt(var[:-1] / n[:-1] + var[1:] / n[1:])
    steps = sign * np.diff(means)
    spread = sign * (means[-1] - means[0])
    if spread <= 2 * float(np.sqrt(var[0] / n[0] + var[-1] / n[-1])) and np.all(
            np.abs(steps) <= 2 * se):
        return None, "the outcome barely moves across five bands"
    wrong = steps < -2 * se
    if not wrong.any():
        return True, ""
    first = int(np.argmax(wrong)) + 1
    return False, f"band {first} to {first + 1} of {len(means)} goes the other way"


# --- step one: whether, on every row explore has ----------------------------------------
def review(df: pd.DataFrame, y: Any, columns: list[str], *, intent: str,
           expected: dict[str, str], rows: int = 4000, seed: int = 0,
           suspects: tuple[str, ...] = ()) -> dict[str, Any]:
    """What each column looks like against the outcome, whether it has the property
    ``intent`` needs, and the kind of transformation recommended. Advice read on every row
    explore has (the sample is not split yet): a kind, never a parameter."""
    spec = INTENTS[intent]
    yv = np.asarray(y, dtype=float)
    take = np.arange(len(df))
    if len(take) > rows:
        take = np.sort(np.random.default_rng(seed).choice(take, rows, replace=False))
    binary = len(np.unique(yv[np.isfinite(yv)])) <= 2
    out, plain = [], []
    for col in columns:
        if col not in df.columns:
            continue
        if col in suspects or not factors_mod.shapeable(df[col]):
            plain.append(col)  # a leakage suspect is left for validation to judge as it is
            continue
        x = pd.to_numeric(df[col].iloc[take], errors="coerce").to_numpy(dtype=float)
        yy = yv[take]
        ok = np.isfinite(x) & np.isfinite(yy)
        entry = (_review_one(col, x[ok], yy[ok], intent=intent, expected=expected.get(col),
                             binary=binary) if ok.sum() >= 30 else None)
        if entry is None:
            plain.append(col)
            continue
        if intent == "stationary":  # time order matters: read the whole column, in order
            entry["checks"] += _time_checks(df[col], yv)
            entry["recommended"] = _recommend(intent, entry["checks"])
        out.append(entry)
    return {"intent": intent, "label": spec["label"], "need": spec["need"],
            "kinds": {k: KINDS[k]["label"] for k in spec["kinds"]},
            "hints": {k: KINDS[k]["hint"] for k in spec["kinds"]},
            "columns": out, "as_they_are": plain, "rows": int(len(take))}


def _review_one(col: str, x: np.ndarray, y: np.ndarray, *, intent: str, expected: str | None,
                binary: bool) -> dict[str, Any] | None:
    fit_ = factors_mod.fit_shape(x, y, cap=True, knees=2, direction=expected,
                                 classification=binary, grid=5)
    if fit_ is None:
        return None
    sign = 1.0 if fit_["direction"] == "+" else -1.0
    skew = float(pd.Series(x).skew() or 0.0)
    checks: list[dict[str, Any]] = []
    if intent == "rank":
        way, where = _one_way(x, y, sign)
        verb = "rises" if sign > 0 else "falls"
        checks.append({"id": "one_way", "label": "Ranks in order", "ok": way is not False,
                       "text": (f"the outcome {verb} through all five bands" if way else
                                f"it turns: {where}" if way is False else where)})
        if way is None:
            checks.append({"id": "flat", "label": "Moves the outcome", "ok": False,
                           "text": f"{where}: the screen will judge it, a transformation "
                                   "would only fit noise"})
    if intent in ("rank", "linear"):
        knees = fit_["knees_used"]
        checks.append({"id": "straight", "label": "Straight", "ok": not knees,
                       "text": ("a straight line does as well as a bent one" if not knees else
                                f"a line with {knees} bend{'s' if knees > 1 else ''} fits "
                                "better, after paying for the extra slopes")})
    if intent == "linear":
        long_right = skew >= 1 and bool(np.nanmin(x) >= 0)
        checks.append({"id": "skew", "label": "Even spread", "ok": not long_right,
                       "text": (f"a long right tail (skew {skew:.1f}); a logarithm straightens it"
                                if long_right else f"skew {skew:.1f}")})
    tail = abs(skew) >= 2
    checks.append({"id": "tails", "label": "No extreme values", "ok": not tail,
                   "text": (f"a long tail (skew {skew:.1f}): a few values could drive the slope"
                            if tail else f"skew {skew:.1f}")})
    curve = _lowess(np.clip(x, *fit_["cap"]), y, 0.4)
    if binary:
        curve[:, 1] = np.clip(curve[:, 1], 0.0, 1.0)
    entry: dict[str, Any] = {
        "column": col, "direction": fit_["direction"], "direction_from": fit_["direction_from"],
        "range": [float(np.min(x)), float(np.max(x))], "cap": fit_["cap"],
        "observed": _bands(x, y), "curve": _thin(curve), "rate": binary,
        "checks": checks, "rows": int(len(x)),
    }
    if not binary:  # a cloud of points says something only when the outcome is not 0/1
        pick = np.random.default_rng(1).choice(len(x), min(400, len(x)), replace=False)
        entry["points"] = [[float(x[i]), float(y[i])] for i in pick]
    entry["recommended"] = _recommend(intent, checks)
    return entry


def _time_checks(s: pd.Series, y: np.ndarray) -> list[dict[str, Any]]:
    x = pd.to_numeric(s, errors="coerce").astype(float).reset_index(drop=True)
    acf = float(x.autocorr(1)) if x.notna().sum() > 3 else 0.0
    acf = 0.0 if not np.isfinite(acf) else acf
    drift = acf >= 0.9
    checks = [{"id": "drift", "label": "Does not drift", "ok": not drift,
               "text": (f"the level drifts (autocorrelation {acf:.2f}); its change is steadier"
                        if drift else f"autocorrelation {acf:.2f}")}]
    yy = pd.Series(np.asarray(y, dtype=float))

    def r(k: int) -> float:
        v = x.shift(k).corr(yy)
        return abs(float(v)) if v is not None and np.isfinite(v) else 0.0

    base = r(0)
    best, best_k = base, 0
    for k in range(1, 5):
        if r(k) > best + 0.05:
            best, best_k = r(k), k
    checks.append({"id": "lead", "label": "Acts at once", "ok": best_k == 0,
                   "text": (f"the outcome follows it {best_k} period(s) later (|r| {best:.2f} "
                            f"against {base:.2f} at once)" if best_k else
                            f"strongest at once (|r| {base:.2f})")})
    return checks


def _recommend(intent: str, checks: list[dict[str, Any]]) -> dict[str, str]:
    """The kind the failed checks call for, in the order a modeler would reach for it."""
    bad = {c["id"]: c["text"] for c in checks if not c["ok"]}
    if "flat" in bad:
        return {"kind": "as_is", "why": bad["flat"]}
    if intent == "rank":
        if "one_way" in bad:
            kind = "knee" if "straight" in bad else "smooth"
            why = bad["one_way"] + ("; a line with knees that keeps one direction fits it"
                                    if kind == "knee" else
                                    "; no clean bend, so a smooth curve held to one direction")
        elif "straight" in bad:
            kind, why = "knee", f"it ranks in order but bends: {bad['straight']}"
        elif "tails" in bad:
            kind, why = "cap", bad["tails"]
        else:
            kind, why = "as_is", "it ranks in order along a straight line"
    elif intent == "linear":
        if "skew" in bad:
            kind, why = "log", bad["skew"]
        elif "straight" in bad:
            kind, why = "knee", bad["straight"]
        elif "tails" in bad:
            kind, why = "cap", bad["tails"]
        else:
            kind, why = "as_is", "it relates to the outcome along a straight line"
    else:  # a driver: its lags and changes compete in the screen, which picks on the outcome
        found = "; ".join(bad.values())
        kind, why = "library", ((found + "; the plan's changes and lags compete with its level "
                                 "in the screen") if found else
                                "it does not drift and acts at once; its lags and changes still "
                                "compete with its level in the screen")
    return {"kind": kind, "why": why}


# --- step two: how, on the development sample -------------------------------------------
def fit(dev: pd.DataFrame, y: Any, col: str, spec: dict[str, Any], *, direction: str | None,
        classification: bool, period: int | None = None) -> dict[str, Any] | None:
    """Fit one column's transformation on the development sample ``dev`` (ADR-0024).

    Returns the parameters as fitted (knots placed, percentiles read), the factor that carries
    them, the observed outcome per band and the curve the model would see, or None when the
    column cannot be transformed this way. A ``log``, ``change`` or ``lag`` is read from the
    library column ``dev`` already carries."""
    spec = normalize(spec)
    kind = spec["kind"]
    yv = np.asarray(y, dtype=float)
    entry: dict[str, Any] = {"kind": kind, "label": KINDS[kind]["label"], "axis": col}
    if kind in FITTED:
        quantiles = ((spec["lower"] / 100, spec["upper"] / 100) if kind == "cap"
                     else factors_mod.CAP_QUANTILES)
        shaped = factors_mod.fit_shape(
            dev[col], yv, cap=True, knees=spec["knees"] if kind == "knee" else 0,
            knots=spec["knots"] if kind == "knee" else None, direction=direction,
            classification=classification, quantiles=quantiles)
        if shaped is None:
            return None
        lo, hi = shaped["cap"]
        entry.update({k: shaped[k] for k in ("cap", "observed", "curve", "rows", "direction",
                                             "direction_from", "monotonic", "origin")})
        entry["knots"] = shaped["knots"] if kind == "knee" else []
        param = 0
        if kind == "cap":
            points = [[lo, lo], [hi, hi]]
        elif kind == "knee":
            points = shaped["points"]
            param = shaped["knees_used"]  # spec keeps the knots asked for; entry the fitted
            entry["knees_used"] = param
        else:
            points, curve = _smooth(dev[col], yv, lo, hi, spec["span"], shaped["direction"],
                                    classification)
            if not points:
                return None
            entry.update({"curve": curve, "monotonic": True, "origin": "fitted"})
        factor = factors_mod.Factor(name=factors_mod.factor_name(col, kind, param, 0),
                                    source=col, transform=kind, param=param, points=points)
    else:
        factor = factors_mod.own_factor(col, spec, period)
        if factor is None or factor.name not in dev.columns:
            return None
        t = pd.to_numeric(dev[factor.name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(yv)
        if ok.sum() < 30 or np.ptp(t[ok]) == 0:
            return None
        t, yy = t[ok], yv[ok]
        coef, b0, _ = factors_mod._fit_line(t, yy, [], classification)
        line = np.linspace(t.min(), t.max(), 61)
        fitted = coef[0] * line + b0
        if classification:
            fitted = 1.0 / (1.0 + np.exp(-fitted))
        entry.update({"observed": _bands(t, yy), "rows": int(len(t)), "knots": [],
                      "curve": [[float(a), float(b)] for a, b in zip(line, fitted, strict=True)],
                      "direction": "+" if coef[0] >= 0 else "-", "direction_from": "the data",
                      "monotonic": True, "origin": "computed",
                      "axis": f"{col}: {factor.describe()}"})
    entry["factor"] = factor.to_dict()
    entry["spec"] = spec
    entry["caption"] = caption(col, entry)
    return entry


def _smooth(x: pd.Series, y: np.ndarray, lo: float, hi: float, span: float, direction: str,
            classification: bool) -> tuple[list[list[float]], list[list[float]]]:
    """A LOWESS curve of the outcome on the capped column, held to ``direction`` (isotonic),
    as the points a factor runs through (log-odds for a 0/1 outcome) and the curve to draw."""
    from sklearn.isotonic import IsotonicRegression

    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(y)
    xc, yy = np.clip(xv[ok], lo, hi), y[ok]
    if len(np.unique(xc)) < 5:
        return [], []
    sm = _lowess(xc, yy, span)
    iso = IsotonicRegression(increasing=direction == "+", out_of_bounds="clip")
    iso.fit(sm[:, 0], sm[:, 1])
    grid = np.unique(np.quantile(xc, np.linspace(0.0, 1.0, 13)))
    vals = iso.predict(grid)
    line = np.linspace(lo, hi, 61)
    curve = iso.predict(line)
    if classification:
        p = np.clip(vals, 1e-4, 1 - 1e-4)
        vals = np.log(p / (1 - p))
        curve = np.clip(curve, 0.0, 1.0)
    elif direction != "+":
        vals = -vals  # turned to rise with the outcome, like a knee
    points = [[float(a), float(b)] for a, b in zip(grid, vals, strict=True)]
    return points, [[float(a), float(b)] for a, b in zip(line, curve, strict=True)]


def caption(col: str, entry: dict[str, Any]) -> str:
    """One line for a transformed column: what was fitted and which way it runs."""
    kind = entry["kind"]
    parts = []
    if kind in FITTED and entry.get("cap"):
        parts.append(f"capped to [{entry['cap'][0]:.4g}, {entry['cap'][1]:.4g}]")
    if kind == "knee":
        knots = entry.get("knots") or []
        if knots:
            parts.append(f"{len(knots)} knee{'s' if len(knots) > 1 else ''} at "
                         + ", ".join(f"{k:.4g}" for k in knots)
                         + (" (given)" if entry.get("origin") == "given"
                            else " (placed on the development sample)"))
        else:
            parts.append("straight: no knee earned its place")
    if kind == "smooth":
        parts.append(f"each local fit reads {entry['spec']['span']:.0%} of the rows")
    if kind in ("log", "change", "lag", "custom"):
        parts.append(entry["axis"].split(": ", 1)[-1])
    way = "rises" if entry.get("direction") == "+" else "falls"
    parts.append(f"the outcome {way} with it ({entry.get('direction_from', 'the data')})")
    if not entry.get("monotonic", True):
        parts.append("not monotonic: no fit keeps one direction")
    return " · ".join(parts)


def sketch(df: pd.DataFrame, y: Any, col: str, kind: str, *, direction: str | None,
           classification: bool, period: int | None = None, rows: int = 4000,
           seed: int = 0, expr: str = "") -> dict[str, Any] | None:
    """What a kind would do to a column, for step one's preview on the Explore page: the
    curve the model would see, with default parameters, on the rows explore has. Advice like
    the review: nothing it places is kept (the parameters are fitted at the design gate, on the
    development sample). None for ``library`` or a kind the column cannot take; a custom
    ``expr`` the safe expression rules refuse raises ``ValueError``."""
    if kind == "custom":
        expr = check_expr(expr, col)
    if kind == "library" or col not in df.columns:
        return None
    yv = np.asarray(y, dtype=float)
    frame = df
    if kind in ("change", "lag"):  # read in time order on the whole column, then sampled
        f = factors_mod.own_factor(col, normalize(kind), period)
        values = factors_mod._compute(df, f) if f is not None else None
        if values is None:
            return None
        frame = df.assign(**{f.name: values.to_numpy()})
    take = np.arange(len(frame))
    if len(take) > rows:
        take = np.sort(np.random.default_rng(seed).choice(take, rows, replace=False))
    frame, yv = frame.iloc[take], yv[take]
    if kind not in ("as_is", "log", "custom"):
        out = fit(frame, yv, col, {"kind": kind}, direction=direction,
                  classification=classification, period=period)
        if out is not None:
            out["caption"] = out["caption"].replace(
                "(placed on the development sample)", "(a preview: placed again at the design "
                "gate, on the development sample)")
        return out
    def through(v: pd.Series) -> pd.Series | None:
        if kind == "as_is":
            return v
        if kind == "custom":
            return factors_mod.custom_values(v, expr, col)
        return factors_mod._base(v, "log", 0)

    x = pd.to_numeric(frame[col], errors="coerce").astype(float)
    t = through(x)
    if t is None:
        return None
    t = t.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() < 30 or np.ptp(t[ok]) == 0:
        return None
    coef, b0, _ = factors_mod._fit_line(t[ok], yv[ok], [], classification)
    lo, hi = np.quantile(xv[ok], [0.005, 0.995])
    grid = np.linspace(lo, hi, 61)
    tg = through(pd.Series(grid)).to_numpy(dtype=float)
    fitted = coef[0] * tg + b0
    if classification:
        fitted = 1.0 / (1.0 + np.exp(-fitted))
    way = "rises" if coef[0] >= 0 else "falls"
    return {"kind": kind, "label": KINDS[kind]["label"], "axis": col,
            "observed": _bands(xv[ok], yv[ok]), "knots": [], "rows": int(ok.sum()),
            "curve": [[float(a), float(b)] for a, b in zip(grid, fitted, strict=True)],
            "caption": ({"as_is": "a straight line in the column as it is",
                         "log": "a straight line in its logarithm"}.get(kind)
                        or f"a straight line in {expr}") + f" · the outcome {way} with it",
            **({"expr": expr} if kind == "custom" else {})}


def library(dev: pd.DataFrame, y: Any, columns: list[str], specs: dict[str, Any], *,
            expected: dict[str, str], classification: bool, period: int | None = None
            ) -> tuple[list[factors_mod.Factor], dict[str, dict], list[str]]:
    """The model stage's fit (ADR-0024): every transformed column on the development sample
    ``dev`` only. Returns the fitted factors (their points frozen, so ``apply_library`` computes
    them on any other sample without an outcome), a report per column and notes."""
    made: list[factors_mod.Factor] = []
    report: dict[str, dict] = {}
    notes: list[str] = []
    for col, spec in decided(specs).items():
        if col not in columns or col not in dev.columns:
            continue
        entry = fit(dev, y, col, spec, direction=expected.get(col),
                    classification=classification, period=period)
        if entry is None:
            notes.append(f"{col}: its {KINDS[spec['kind']]['label']} could not be fitted on the "
                         "development sample; it enters as it is")
            continue
        report[col] = entry
        if spec["kind"] in FITTED:
            made.append(factors_mod.Factor(**entry["factor"]))
            if not entry.get("monotonic", True):
                notes.append(f"{col}: no {KINDS[spec['kind']]['label']} fit keeps one direction")
    fitted = [c for c, e in report.items() if e["kind"] in FITTED]
    if fitted:
        notes.append(f"{len(fitted)} column(s) fitted on the development sample ({len(dev)} "
                     "rows); their points are frozen for every other sample")
    return made, report, notes
