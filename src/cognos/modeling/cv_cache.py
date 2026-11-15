"""Validation results kept between stages: the same fit on the same rows is computed once.

The single-factor analysis is drawn at the design gate (every look at it, ADR-0025) and run
again by the model stage, on the same development sample; a changed rule re-runs a search most
of whose specifications were already validated. Re-sampling a specification is the expensive
part of all of them, and it is a pure function of what goes into it. This is a memo of that
function, nothing more: a result is reused only when everything it was computed from is
identical, so no recorded number can differ with or without it.

A key covers the specification (family, factors in order, every setting), the values of its
factor columns row by row, the outcome, the exact rows of every validation block, the metric
and the versions of the libraries that fit. A family that reads columns beyond its factors
(the hazard families read event timing) is never kept.

Heavy and binary, so it lives beside the search cache (``stages/model/cv_cache.joblib``) and
never travels in an export. Losing the file costs time, never a result.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = 1
MAX_ENTRIES = 4000  # oldest out first: a run's searches are a few hundred specifications each


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return h.hexdigest()


def _libraries() -> bytes:
    import sklearn
    import statsmodels

    return f"{VERSION}|{np.__version__}|{pd.__version__}|{sklearn.__version__}|{statsmodels.__version__}".encode()


class CVCache:
    """A memo of ``cv_score`` results on disk. Not a record: nothing reads it for a number
    that was not just as computable without it."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: dict[str, Any] = {}
        self.hits = 0
        self._dirty = False
        self._guard = threading.Lock()
        if self.path is not None and self.path.exists():
            try:
                import joblib

                stored = joblib.load(self.path)
                if stored.get("version") == VERSION:
                    self.entries = dict(stored["entries"])
            except Exception:  # an unreadable cache is an empty one
                self.entries = {}

    # --- keys ----------------------------------------------------------------------
    @staticmethod
    def sample_key(dev: pd.DataFrame, y: np.ndarray, folds: list[tuple], metric: str,
                   is_classification: bool) -> str:
        """What every specification validated on one sample shares: the outcome, the rows of
        each validation block, the metric and the libraries."""
        rows = [np.ascontiguousarray(np.asarray(part, dtype=np.int64)).tobytes()
                for fold in folds for part in fold]
        y = np.ascontiguousarray(np.asarray(y, dtype=float))
        return _digest(_libraries(), metric.encode(), b"1" if is_classification else b"0",
                       str(len(dev)).encode(), y.tobytes(), str(len(folds)).encode(), *rows)

    @staticmethod
    def column_key(column: pd.Series) -> bytes:
        """A column's values row by row, with its type (order matters: folds are positions)."""
        hashed = pd.util.hash_pandas_object(column, index=False).to_numpy()
        return str(column.dtype).encode() + np.ascontiguousarray(hashed).tobytes()

    @staticmethod
    def key(sample: str, family: str, features: list[str], hyperparams: dict,
            columns: list[bytes]) -> str:
        spec = json.dumps({"family": family, "features": list(features), "hp": hyperparams},
                          sort_keys=True, default=str).encode()
        return _digest(sample.encode(), spec, *columns)

    # --- the memo ------------------------------------------------------------------
    def get(self, key: str) -> Any | None:
        with self._guard:
            found = self.entries.get(key)
            if found is not None:
                self.hits += 1
            return found

    def put(self, key: str, value: Any) -> None:
        with self._guard:
            self.entries[key] = value
            self._dirty = True

    def save(self) -> None:
        """Write what was added (atomically: another process may be reading it)."""
        if self.path is None or not self._dirty:
            return
        import joblib

        with self._guard:
            keep = dict(list(self.entries.items())[-MAX_ENTRIES:])
            self._dirty = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            joblib.dump({"version": VERSION, "entries": keep}, tmp)
            os.replace(tmp, self.path)
        except OSError:  # a cache that cannot be written is only slower
            tmp.unlink(missing_ok=True)
