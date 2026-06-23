"""Reimplementation of the autoforge / autoresearch experiment protocol.

autoforge is not an importable library — it is a *protocol*: a training script prints metrics as
``name:    <number>`` lines, an agent greps them, keeps the change if the metric improved or reverts
it, and appends a row to ``results.tsv``. COGNOS ports that protocol so any external script or tool
can plug into the same ratchet the modeling stage uses internally.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..modeling.metrics import is_better

_METRIC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", re.M)


def parse_metrics(stdout: str) -> dict[str, float]:
    """Grep ``name:    <number>`` lines out of a script's stdout (the autoforge contract)."""
    return {m.group(1): float(m.group(2)) for m in _METRIC_RE.finditer(stdout)}


def format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"{k}:    {v:.6f}" for k, v in metrics.items())


@dataclass
class LedgerRow:
    commit: str
    metric_value: float
    status: str  # keep | discard | crash
    description: str
    extras: dict[str, float] = field(default_factory=dict)


class ExperimentLedger:
    """The ``results.tsv`` ledger: one row per attempt, status in {keep, discard, crash}."""

    def __init__(self, metric: str, extra_columns: list[str] | None = None) -> None:
        self.metric = metric
        self.extra_columns = extra_columns or []
        self.rows: list[LedgerRow] = []

    def append(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def header(self) -> str:
        cols = ["commit", "metric_value", "status", "description", *self.extra_columns]
        return "\t".join(cols)

    def to_tsv(self) -> str:
        lines = [self.header()]
        for r in self.rows:
            extras = "\t".join(f"{r.extras.get(c, 0.0):.6f}" for c in self.extra_columns)
            base = f"{r.commit}\t{r.metric_value:.6f}\t{r.status}\t{r.description}"
            lines.append(f"{base}\t{extras}" if extras else base)
        return "\n".join(lines) + "\n"

    @property
    def kept(self) -> list[LedgerRow]:
        return [r for r in self.rows if r.status == "keep"]


def run_ratchet(
    experiments: list[Callable[[], dict[str, float]]],
    *,
    metric: str,
    labels: list[str] | None = None,
) -> tuple[float | None, ExperimentLedger]:
    """Generic accept-if-better-else-discard loop over experiment callables.

    Each callable returns a metrics dict containing ``metric``. Returns (champion_value, ledger).
    Crashes are caught, logged as status='crash' with value 0, and the loop continues — exactly the
    autoforge behaviour (the agent never stops to ask).
    """
    ledger = ExperimentLedger(metric=metric)
    champion: float | None = None
    labels = labels or [f"exp{i}" for i in range(len(experiments))]
    for i, exp in enumerate(experiments):
        try:
            metrics = exp()
            value = metrics[metric]
            keep = is_better(metric, value, champion)
            if keep:
                champion = value
            extras = {k: v for k, v in metrics.items() if k != metric}
            ledger.append(LedgerRow(commit=labels[i], metric_value=value,
                                    status="keep" if keep else "discard",
                                    description=labels[i], extras=extras))
        except Exception as exc:
            ledger.append(LedgerRow(commit=labels[i], metric_value=0.0, status="crash",
                                    description=f"{labels[i]}: {type(exc).__name__}: {exc}"))
    return champion, ledger
