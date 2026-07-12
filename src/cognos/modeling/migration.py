"""Rating-transition (migration) credit model — the CreditMetrics-style loss-forecast framework.

The industry-standard framework when the risk driver is a **rating history** rather than a default
flag alone: estimate a one-period rating-transition matrix by the cohort method, power it up to the
forecast horizon, and read cumulative default probabilities (and expected loss) off the default
column. Banks reach for it when the internal default history is too short to calibrate long-run
PDs — the matrix is estimated on a long external agency history (e.g. S&P CreditPro-style
obligor-year data) and applied to the current book through the obligor's rating.

Estimation follows agency practice:

- **Cohort method** with Laplace smoothing: row-stochastic counts of ``rating → next_rating`` over
  the outcome window, one row per live rating, with the default state absorbing.
- **Withdrawn-rating (NR) adjustment**: transitions into a withdrawn state are removed from the
  row denominator (the standard S&P/CreditPro treatment), and the withdrawal share is reported —
  never silently dropped.
- **Rank-order diagnostics**: the default column must be monotone in rating order and rows should
  be diagonally dominant (ratings are sticky); violations surface as findings, not crashes.

Two uses in COGNOS, mirroring the Merton structural layer (``modeling/structural.py``):

- **Hybrid (default)**: the horizon cumulative PD implied by the obligor's *current* rating becomes
  the engineered feature ``migration_pd`` feeding the reduced-form champion. The raw rating column
  is swapped out of the champion's features (the PD is an exact function of the rating, and keeping
  both would make the K-1 inference design perfectly collinear).
- **Pure-migration benchmark**: ``migration_pd`` is scored on the sealed holdout as a labelled
  challenger benchmark, never the deployed model.

The matrix is *fitted* (unlike the deterministic Merton solver), so the model stage estimates it on
the training partition only — the sealed holdout stays frozen. The fitted artifact is a plain dict
(JSON-serializable) so the scorer bundle can recompute ``migration_pd`` from the rating column at
serve time, exactly as it was computed in training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

#: Standard agency (S&P-style) letter scale, best to worst, excluding the default state.
AGENCY_SCALE = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


@dataclass
class MigrationSpec:
    """Column mapping + estimation parameters for the migration layer (fully serializable)."""

    rating_col: str
    next_rating_col: str
    scale: list[str] = field(default_factory=lambda: list(AGENCY_SCALE))  # best -> worst
    default_state: str = "D"
    withdrawn_states: list[str] = field(default_factory=lambda: ["NR"])
    horizon_periods: int = 1  # loss-forecast horizon in rating periods (matrix powers)
    smoothing: float = 0.5  # Laplace prior count per cell (keeps thin rows finite, never zero)
    monotone_pd: bool = True  # PAVA-monotonize the default column (rank-order by construction)
    pd_feature: str = "migration_pd"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MigrationSpec:
        return cls(**d)


def infer_scale(ratings: pd.Series, next_ratings: pd.Series, *, default_state: str = "D",
                withdrawn_states: tuple[str, ...] = ("NR",)) -> list[str]:
    """Order the observed rating states best→worst by their observed default rate.

    Fallback when no scale is configured. Ties (e.g. two grades with zero observed defaults) are
    broken by the mean rank of the *destination* states, so upgrades/downgrades still order the
    grades; remaining ties break alphabetically for determinism.
    """
    r = ratings.astype(str)
    nxt = next_ratings.astype(str)
    skip = {str(default_state), *map(str, withdrawn_states)}
    states = sorted(s for s in r.unique() if s not in skip and s != "nan")
    if not states:
        return []
    default_rate = {s: float((nxt[r == s] == default_state).mean()) for s in states}
    # Destination severity: provisional alphabetical rank of each live state; D = worst.
    prov = {s: i for i, s in enumerate(sorted(states))}
    prov[str(default_state)] = len(states) + 1.0
    dest_rank: dict[str, float] = {}
    for s in states:
        dests = nxt[(r == s) & (~nxt.isin(list(map(str, withdrawn_states))))]
        dest_rank[s] = float(np.mean([prov.get(d, len(states) / 2) for d in dests])) if len(dests) else 0.0
    return sorted(states, key=lambda s: (default_rate[s], dest_rank[s], s))


def _pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence (L2, weighted)."""
    v = np.asarray(values, dtype=float).copy()
    w = np.asarray(weights, dtype=float).copy()
    blocks = [[i] for i in range(len(v))]
    vals, wts = list(v), list(w)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            tot = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            wts[i] = tot
            blocks[i].extend(blocks[i + 1])
            del vals[i + 1], wts[i + 1], blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    out = np.empty_like(v)
    for val, idxs in zip(vals, blocks, strict=True):
        out[idxs] = val
    return out


def estimate_transition_matrix(ratings: pd.Series, next_ratings: pd.Series,
                               spec: MigrationSpec) -> dict:
    """Cohort-method one-period transition matrix with the NR adjustment and Laplace smoothing.

    Returns a dict with the row-stochastic ``matrix`` (rows = live ratings in ``spec.scale``,
    columns = live ratings + the absorbing default state), raw ``counts``, per-row support,
    withdrawal counts, and rank-order diagnostics.
    """
    scale = list(spec.scale)
    states = [*scale, spec.default_state]
    withdrawn = set(map(str, spec.withdrawn_states))
    r = ratings.astype(str).to_numpy()
    nxt = next_ratings.astype(str).to_numpy()

    idx_from = {s: i for i, s in enumerate(scale)}
    idx_to = {s: j for j, s in enumerate(states)}
    counts = np.zeros((len(scale), len(states)), dtype=float)
    n_withdrawn = np.zeros(len(scale), dtype=float)
    n_unknown = 0
    for a, b in zip(r, nxt, strict=True):
        i = idx_from.get(a)
        if i is None:
            n_unknown += 1
            continue
        if b in withdrawn:  # NR adjustment: out of the denominator, reported separately
            n_withdrawn[i] += 1
            continue
        j = idx_to.get(b)
        if j is None:
            n_unknown += 1
            continue
        counts[i, j] += 1

    smoothed = counts + float(spec.smoothing)
    matrix = smoothed / smoothed.sum(axis=1, keepdims=True)

    pd_one = matrix[:, -1]  # one-period PD per rating (default column)
    raw_monotone = bool(np.all(np.diff(pd_one) >= -1e-12))
    monotonized_rows: list[str] = []
    if spec.monotone_pd and not raw_monotone:
        # Rank-order by construction: weighted PAVA on the default column (standard agency-matrix
        # smoothing — a AAA PD above the AA PD is a small-sample artifact, not a credit view).
        # Live-state cells are rescaled proportionally so rows stay stochastic.
        pd_iso = _pava_increasing(pd_one, weights=counts.sum(axis=1) + 1.0)
        monotonized_rows = [scale[i] for i in range(len(scale))
                            if abs(pd_iso[i] - pd_one[i]) > 1e-12]
        live_scale = (1.0 - pd_iso) / np.clip(1.0 - pd_one, 1e-12, None)
        matrix[:, :-1] *= live_scale[:, None]
        matrix[:, -1] = pd_iso
        matrix /= matrix.sum(axis=1, keepdims=True)
        pd_one = matrix[:, -1]

    diag = np.diag(matrix[:, : len(scale)])
    diagnostics = {
        "default_col_monotone": bool(np.all(np.diff(pd_one) >= -1e-12)),
        "raw_default_col_monotone": raw_monotone,
        "monotonized_rows": monotonized_rows,
        "diagonally_dominant": bool(np.all(diag >= matrix[:, : len(scale)].max(axis=1) - 1e-12)),
        "min_row_support": int(counts.sum(axis=1).min()),
        "thin_rows": [scale[i] for i in range(len(scale)) if counts[i].sum() < 30],
    }
    return {
        "scale": scale,
        "states": states,
        "default_state": spec.default_state,
        "matrix": matrix.tolist(),
        "counts": counts.tolist(),
        "row_support": {scale[i]: int(counts[i].sum()) for i in range(len(scale))},
        "one_period_pd": {scale[i]: float(pd_one[i]) for i in range(len(scale))},
        "n_obs": int(counts.sum()),
        "n_withdrawn": int(n_withdrawn.sum()),
        "withdrawn_share": float(n_withdrawn.sum() / max(1.0, n_withdrawn.sum() + counts.sum())),
        "n_unknown_state": int(n_unknown),
        "diagnostics": diagnostics,
    }


def cumulative_default_curve(matrix: list[list[float]] | np.ndarray, scale: list[str],
                             periods: int) -> dict[str, list[float]]:
    """Cumulative PD term structure per rating: power up the matrix with default absorbing.

    ``matrix`` is (n_live × n_live+1) with the default column last; the absorbing default row
    [0, …, 0, 1] is appended before powering. Returns {rating: [cumPD_1, …, cumPD_periods]}.
    """
    m = np.asarray(matrix, dtype=float)
    n = len(scale)
    full = np.zeros((n + 1, n + 1))
    full[:n, :] = m
    full[n, n] = 1.0  # default is absorbing
    curve: dict[str, list[float]] = {s: [] for s in scale}
    p = np.eye(n + 1)
    for _ in range(int(periods)):
        p = p @ full
        for i, s in enumerate(scale):
            curve[s].append(float(p[i, n]))
    return curve


def fit_migration(df: pd.DataFrame, spec: MigrationSpec, *,
                  condition_col: str | None = None) -> dict:
    """Fit the migration model on (training) data: pooled matrix, term structure, serve-time map.

    When ``condition_col`` is given (e.g. a macro-regime column), per-regime conditional matrices
    are estimated alongside the pooled (through-the-cycle) one — the CreditMetrics-style stress
    axis for the loss forecast. The returned dict is JSON-serializable and self-contained: the
    scorer recomputes ``migration_pd`` from it at serve time.
    """
    est = estimate_transition_matrix(df[spec.rating_col], df[spec.next_rating_col], spec)
    curve = cumulative_default_curve(est["matrix"], est["scale"], spec.horizon_periods)
    pd_map = {s: curve[s][-1] for s in est["scale"]}
    support = np.array([est["row_support"][s] for s in est["scale"]], dtype=float)
    pds = np.array([pd_map[s] for s in est["scale"]], dtype=float)
    fill = float(np.average(pds, weights=np.maximum(support, 1.0)))  # unseen rating -> book mean

    conditional: dict[str, dict] = {}
    if condition_col and condition_col in df.columns:
        for value, part in df.groupby(df[condition_col].astype(str)):
            c_est = estimate_transition_matrix(part[spec.rating_col], part[spec.next_rating_col], spec)
            c_curve = cumulative_default_curve(c_est["matrix"], c_est["scale"], spec.horizon_periods)
            conditional[str(value)] = {
                "matrix": c_est["matrix"],
                "one_period_pd": c_est["one_period_pd"],
                "cumulative_pd": {s: c_curve[s][-1] for s in c_est["scale"]},
                "n_obs": c_est["n_obs"],
            }

    return {
        "spec": spec.to_dict(),
        "estimate": est,
        "term_structure": curve,
        "pd_map": pd_map,
        "fill": fill,
        "condition_col": condition_col if conditional else None,
        "conditional": conditional,
    }


def augment_frame(df: pd.DataFrame, model: dict) -> tuple[pd.DataFrame, dict]:
    """Add the ``migration_pd`` column (horizon cumulative PD implied by the current rating).

    Target-hidden by construction: the mapping reads the rating column only. Ratings outside the
    fitted scale get the support-weighted book-mean PD (``fill``); the count is reported.
    """
    spec = MigrationSpec.from_dict(model["spec"])
    ratings = df[spec.rating_col].astype(str)
    mapped = ratings.map(model["pd_map"])
    n_unmapped = int(mapped.isna().sum())
    out = df.copy()
    out[spec.pd_feature] = mapped.fillna(float(model["fill"])).astype(float)
    info = {"n_rows": int(len(df)), "n_unmapped": n_unmapped,
            "mean_migration_pd": float(out[spec.pd_feature].mean()) if len(df) else None}
    return out, info


def expected_loss_forecast(ratings: pd.Series, model: dict, *, lgd: float = 0.45,
                           ead: np.ndarray | None = None) -> dict:
    """Portfolio expected-loss forecast: EL_r = EAD_r × LGD × cumPD_r(horizon), by rating band.

    Reported per rating band and in total, under the pooled (through-the-cycle) matrix and under
    each conditional (e.g. regime) matrix when present — the migration analogue of a baseline vs.
    downturn loss forecast. EL is a *report* (it never feeds champion selection); PDs come from the
    fitted matrix, so read it next to the outcomes-analysis calibration section.
    """
    spec = MigrationSpec.from_dict(model["spec"])
    r = ratings.astype(str).reset_index(drop=True)
    w = pd.Series(np.ones(len(r)) if ead is None else np.asarray(ead, dtype=float))
    total_ead = float(w.sum()) or 1.0

    def _forecast(pd_by_rating: dict[str, float]) -> dict:
        obligor_pd = r.map(pd_by_rating).fillna(float(model["fill"])).to_numpy(dtype=float)
        el = float(np.sum(w.to_numpy() * obligor_pd) * lgd)
        bands = []
        for s in model["estimate"]["scale"]:
            mask = (r == s).to_numpy()
            if not mask.any():
                continue
            band_ead = float(w[mask].sum())
            bands.append({
                "rating": s,
                "n_obligors": int(mask.sum()),
                "ead": band_ead,
                "ead_share": band_ead / total_ead,
                "cumulative_pd": float(pd_by_rating.get(s, model["fill"])),
                "expected_loss": float(band_ead * pd_by_rating.get(s, model["fill"]) * lgd),
            })
        return {"expected_loss": el, "expected_loss_rate": el / total_ead, "bands": bands}

    out = {
        "horizon_periods": spec.horizon_periods,
        "lgd": float(lgd),
        "total_ead": total_ead,
        "n_obligors": int(len(r)),
        "pooled": _forecast(model["pd_map"]),
        "conditional": {},
        "note": "EL = EAD × LGD × cumulative migration PD at the horizon; pooled matrix is "
                "through-the-cycle, conditional matrices are the regime-stress axis.",
    }
    for value, cond in (model.get("conditional") or {}).items():
        out["conditional"][value] = _forecast(cond["cumulative_pd"])
    return out
