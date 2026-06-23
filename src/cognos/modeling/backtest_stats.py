"""Backtest-overfitting statistics: Probability of Backtest Overfitting (PBO via CSCV) and the
Deflated Sharpe Ratio (DSR).

These defend against the #1 failure of autonomous search loops: multiple-testing / data-snooping
makes the best-of-N candidate look great in-sample and fail out-of-sample. Implemented as pure
functions over performance matrices / return series so they are trivially unit-testable.

References: Bailey, Borwein, López de Prado, Zhu — "The Probability of Backtest Overfitting" (2017);
López de Prado — "The Deflated Sharpe Ratio" (2014).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.stats import norm, rankdata

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd)


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > benchmark) given the observed track record (skew/kurtosis adjusted)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return float("nan")
    sr = sharpe_ratio(r)
    skew = float(((r - r.mean()) ** 3).mean() / (r.std(ddof=0) ** 3 + 1e-12))
    kurt = float(((r - r.mean()) ** 4).mean() / (r.std(ddof=0) ** 4 + 1e-12))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null of zero true skill across ``n_trials`` independent trials."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    sigma = math.sqrt(var_sr)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, var_sr: float | None = None) -> dict:
    """DSR = PSR benchmarked against the expected max Sharpe from ``n_trials`` (selection-bias adjusted)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sr = sharpe_ratio(r)
    if var_sr is None:
        # Variance of the SR estimator under iid normal returns (López de Prado).
        var_sr = (1 + 0.5 * sr**2) / max(1, n - 1)
    sr0 = expected_max_sharpe(var_sr, max(2, n_trials))
    dsr = probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
    return {
        "sharpe": sr,
        "sr0_expected_max": sr0,
        "n_trials": int(n_trials),
        "deflated_sharpe": dsr,
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0),
    }


def pbo_cscv(perf_matrix: np.ndarray, n_splits: int = 8) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.

    ``perf_matrix`` is (T observations x N strategies); higher entries are better (e.g. per-sample
    -loss or per-period returns). Returns the fraction of IS/OOS partitions where the in-sample-best
    strategy lands below the OOS median — i.e. the probability that selecting the best backtest is
    overfit.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "n_strategies": int(M.shape[1] if M.ndim == 2 else 0),
                "note": "Need >=2 strategies for PBO."}
    T, N = M.shape
    S = max(4, min(n_splits, T))
    S = S - (S % 2)  # even
    if S < 4:
        return {"pbo": float("nan"), "n_combinations": 0, "n_strategies": N, "note": "Too few observations."}
    rows_per = T // S
    blocks = [np.arange(i * rows_per, (i + 1) * rows_per) for i in range(S)]

    logits: list[float] = []
    overfits = 0
    for combo in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in combo])
        is_perf = M[is_rows].sum(axis=0)
        oos_perf = M[oos_rows].sum(axis=0)
        n_star = int(np.argmax(is_perf))
        ranks = rankdata(oos_perf)  # ascending; higher = better OOS (average method handles ties)
        rel_rank = ranks[n_star] / (N + 1)
        w = min(max(rel_rank, 1e-6), 1 - 1e-6)
        lg = math.log(w / (1 - w))
        logits.append(lg)
        # Strict: only count as overfit when the IS-best is *below* the OOS median. Tied/equally-good
        # libraries land at logit==0 and contribute no overfitting evidence (avoids false alarms).
        if lg < 0:
            overfits += 1
    n_comb = len(logits)
    return {
        "pbo": float(overfits / n_comb) if n_comb else float("nan"),
        "n_combinations": n_comb,
        "n_strategies": N,
        "mean_logit": float(np.mean(logits)) if logits else float("nan"),
    }
