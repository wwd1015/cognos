"""Portfolio-level simulation: Vasicek one-factor loss distribution + macro-scenario stress.

Connects the obligor-level PD model to the questions a bank actually asks of it:

- **Vasicek / ASRF**: given each obligor's PD, an LGD assumption, and an asset correlation ρ, the
  one-factor model draws a systematic factor Z per simulation, conditions each PD on it,
  Φ((Φ⁻¹(pd) + √ρ·Z)/√(1−ρ)), draws idiosyncratic defaults, and accumulates the portfolio loss
  distribution → EL, UL, VaR, ES. The closed-form **Basel IRB capital** K per obligor is computed
  alongside as the analytic cross-check (same math at the 99.9th percentile of Z).
- **Macro-scenario stress**: shock the champion's macro covariates per scenario (CCAR-flavoured
  baseline / adverse / severely adverse), re-score deterministically through the deployed scorer,
  and report the scenario PD and expected-loss deltas.

Everything is seeded or closed-form, so results are bit-reproducible (determinism in the plumbing).
Simulation outputs are *reported* — they never feed champion selection, so the frozen metric /
sealed holdout guarantees are untouched. PDs in are the model's scores; if the model is
miscalibrated the simulation inherits it, which is why the outcomes-analysis calibration section
sits next to these numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_SIM_BLOCK = 1000  # simulations per vectorized block (bounds memory at ~block × n_obligors)


def basel_asset_correlation(pd_: np.ndarray) -> np.ndarray:
    """Basel IRB corporate asset-correlation formula: ρ(PD) ∈ [0.12, 0.24], decreasing in PD."""
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    w = (1 - np.exp(-50 * p)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def basel_irb_capital(pd_: np.ndarray, *, lgd: float = 0.45, confidence: float = 0.999,
                      rho: float | np.ndarray | None = None) -> np.ndarray:
    """Closed-form per-obligor capital requirement K (unexpected-loss share of EAD).

    K = LGD·[Φ((Φ⁻¹(PD) + √ρ·Φ⁻¹(conf)) / √(1−ρ)) − PD]. No maturity adjustment — this is the
    ASRF core, documented as such.
    """
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    r = basel_asset_correlation(p) if rho is None else np.broadcast_to(np.asarray(rho, float), p.shape)
    cond = norm.cdf((norm.ppf(p) + np.sqrt(r) * norm.ppf(confidence)) / np.sqrt(1 - r))
    return lgd * np.maximum(cond - p, 0.0)


def vasicek_loss_simulation(pd_: np.ndarray, *, lgd: float = 0.45, ead: np.ndarray | None = None,
                            rho: float | None = None, confidence: float = 0.999,
                            n_sims: int = 20000, seed: int = 42) -> dict:
    """Seeded one-factor Monte Carlo of the portfolio loss distribution (exact Bernoulli defaults).

    Returns EL/UL/VaR/ES as fractions of total EAD, plus loss quantiles and the Monte-Carlo
    standard error of EL so a reader can judge convergence.
    """
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    n = len(p)
    w = (np.ones(n) if ead is None else np.asarray(ead, dtype=float))
    total_ead = float(w.sum()) or 1.0
    r = basel_asset_correlation(p) if rho is None else np.full(n, float(rho))
    inv_pd = norm.ppf(p)
    sr, sq = np.sqrt(r), np.sqrt(1 - r)

    rng = np.random.default_rng(seed)
    losses = np.empty(n_sims, dtype=float)
    done = 0
    while done < n_sims:
        block = min(_SIM_BLOCK, n_sims - done)
        z = rng.standard_normal(block)[:, None]  # systematic factor per simulation
        cond_pd = norm.cdf((inv_pd[None, :] + sr[None, :] * z) / sq[None, :])
        defaults = rng.uniform(size=(block, n)) < cond_pd
        losses[done:done + block] = (defaults * w[None, :]).sum(axis=1) * lgd / total_ead
        done += block

    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    el = float(losses.mean())
    return {
        "n_obligors": int(n),
        "n_sims": int(n_sims),
        "lgd": float(lgd),
        "asset_correlation": ("basel_formula" if rho is None else float(rho)),
        "confidence": float(confidence),
        "expected_loss": el,
        "unexpected_loss_std": float(losses.std()),
        "var": var,
        "expected_shortfall": float(tail.mean()) if len(tail) else var,
        "loss_quantiles": {q: float(np.quantile(losses, float(q)))
                           for q in ("0.5", "0.9", "0.99", "0.999")},
        "mc_stderr_el": float(losses.std() / np.sqrt(n_sims)),
        "irb_capital_mean": float(np.average(basel_irb_capital(
            p, lgd=lgd, confidence=confidence, rho=rho), weights=w)),
        "note": "losses as a fraction of total EAD; PDs are model scores (see calibration section)",
    }


def stress_scenarios(eval_df: pd.DataFrame, scorer_path: str, scenarios: list[dict], *,
                     lgd: float = 0.45, ead: np.ndarray | None = None) -> dict:
    """Deterministic macro-scenario stress: shock covariates, re-score, report PD/EL deltas.

    A scenario is ``{"name": str, "shocks": {column: {"add": x} | {"mul": x} | {"set": x}}}``.
    Unknown columns are reported, never silently ignored into a wrong answer.
    """
    from ..runtime.score import score_frame

    w = np.ones(len(eval_df)) if ead is None else np.asarray(ead, dtype=float)
    total_ead = float(w.sum()) or 1.0
    base_scores = np.asarray(score_frame(eval_df, scorer_path), dtype=float)
    base_pd = float(np.average(base_scores, weights=w))
    out = {"baseline": {"mean_pd": base_pd, "expected_loss": base_pd * lgd}, "scenarios": []}

    for sc in scenarios:
        name = str(sc.get("name", "scenario"))
        shocks: dict = sc.get("shocks") or {}
        shocked = eval_df.copy()
        missing = []
        for col, spec in shocks.items():
            if col not in shocked.columns:
                missing.append(col)
                continue
            base_col = pd.to_numeric(shocked[col], errors="coerce")
            if "set" in spec:
                shocked[col] = float(spec["set"])
            elif "add" in spec:
                shocked[col] = base_col + float(spec["add"])
            elif "mul" in spec:
                shocked[col] = base_col * float(spec["mul"])
        scores = np.asarray(score_frame(shocked, scorer_path), dtype=float)
        mean_pd = float(np.average(scores, weights=w))
        entry = {
            "name": name,
            "shocks": shocks,
            "mean_pd": mean_pd,
            "delta_pd": mean_pd - base_pd,
            "expected_loss": mean_pd * lgd,
            "delta_expected_loss": (mean_pd - base_pd) * lgd,
            "worst_decile_mean_pd": float(np.mean(np.sort(scores)[-max(1, len(scores) // 10):])),
        }
        if missing:
            entry["missing_columns"] = missing
        out["scenarios"].append(entry)
    out["note"] = f"EL as PD×LGD (lgd={lgd}) per unit EAD ({total_ead:.0f} total); deterministic re-scoring"
    return out
