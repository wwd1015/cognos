"""The test plan's mechanics: cutting the samples and re-sampling the development sample.

A plan (``modeling/plan.py``) names up to three samples. *Development* is what models are fitted
on, and what ``validation`` re-samples (k-fold, grouped, or a rolling forecast origin). *Test* is
out of sample for every fit, and the candidate pool is compared on it, so it is used to choose.
The sealed *holdout* is cut first and scored once, for the champion, after the choice.

Everything here is mechanical: the same plan and the same rows give the same cut.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

TESTS = ("out_of_time", "random", "none")
VALIDATIONS = ("none", "kfold", "stratified_kfold", "group_kfold", "rolling_origin",
               "sliding_window")
ORDERED = ("rolling_origin", "sliding_window")  # need the rows in time order

LABELS = {
    "out_of_time": "Out of time: the latest periods are held back",
    "random": "Random: rows are held back at random",
    "none": "None",
    "kfold": "K-fold cross-validation",
    "stratified_kfold": "Stratified k-fold (each fold keeps the event rate)",
    "group_kfold": "Grouped k-fold (one group never sits on both sides)",
    "rolling_origin": "Rolling origin (fit to date t, forecast the next block, move on)",
    "sliding_window": "Sliding window (as rolling origin, with a fixed-length history)",
}


def split_samples(df: pd.DataFrame, ev: dict[str, Any], *, datetime_col: str | None,
                  random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(development, test, holdout). Cut from the end of the ordering: the holdout last, the
    test sample before it. The ordering is time when the plan says so, else a seeded shuffle."""
    has_date = bool(datetime_col) and datetime_col in df.columns
    by_time = (ev.get("test") == "out_of_time" or ev.get("validation") in ORDERED
               or (ev.get("test") == "none" and has_date))
    if by_time:  # without a date column the row order is taken as time
        ordered = (df.sort_values(datetime_col, kind="stable") if has_date else df
                   ).reset_index(drop=True)
    else:
        ordered = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(ordered)
    n_hold = int(round(n * float(ev.get("holdout_fraction") or 0.0)))
    n_test = 0 if ev.get("test") == "none" else int(round(n * float(ev.get("test_fraction") or 0.0)))
    n_dev = max(0, n - n_hold - n_test)
    dev = ordered.iloc[:n_dev].reset_index(drop=True)
    test = ordered.iloc[n_dev:n_dev + n_test].reset_index(drop=True)
    holdout = ordered.iloc[n_dev + n_test:].reset_index(drop=True)
    return dev, test, holdout


def make_splitter(ev: dict[str, Any], *, n: int, random_state: int):
    """The re-sampling scheme for the development sample, fitted to its size. Returns
    ``(splitter, note)``; the splitter is None when the plan has no validation."""
    kind = ev.get("validation") or "none"
    folds = max(2, int(ev.get("folds") or 5))
    if kind == "none" or n < 4:
        return None, ""
    folds = max(2, min(folds, n // 2))
    if kind == "kfold":
        return KFold(n_splits=folds, shuffle=True, random_state=random_state), ""
    if kind == "stratified_kfold":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), ""
    if kind == "group_kfold":
        return GroupKFold(n_splits=folds), ""
    # A rolling forecast origin: every block is forecast from the history before it.
    horizon = max(1, int(ev.get("horizon") or max(1, n // (folds + 2))))
    note = ""
    min_history = max(8, 2 * horizon)
    while folds > 2 and folds * horizon + min_history > n:
        folds -= 1
    if folds * horizon + 2 > n:
        horizon = max(1, (n - 2) // folds)
        note = f"horizon shortened to {horizon} period(s): the development sample has {n} rows"
    elif folds != max(2, int(ev.get("folds") or 5)):
        note = f"{folds} forecast origin(s): the development sample has {n} rows"
    window = None
    if kind == "sliding_window":
        window = int(ev.get("window") or max(min_history, n - folds * horizon))
    return TimeSeriesSplit(n_splits=folds, test_size=horizon, max_train_size=window), note


def blocks(splitter, n: int, groups=None, y=None) -> list[dict[str, int]]:
    """The size of every fit / validate block, for the record."""
    if splitter is None:
        return []
    idx = pd.RangeIndex(n)
    out = []
    for tr, va in splitter.split(idx, y, groups):
        out.append({"fit_rows": int(len(tr)), "validate_rows": int(len(va)),
                    "validate_from": int(va.min()), "validate_to": int(va.max())})
    return out
