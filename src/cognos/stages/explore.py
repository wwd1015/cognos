"""Stage 1 — Data exploration.

Profiles the dataset (shape, dtypes, missingness, distributions, correlations) and flags data-quality
risks the downstream modeling/validation stages must respect — especially **target-leakage suspects**
(features near-perfectly correlated with the target), which the research identified as the #1 way
automated systems silently overfit.

v1: the engine computes the profile; the **Data Analyst** agent then recommends keep/exclude for
every leakage suspect (and any other column that should not be an input) and raises questions for
the sponsor. Exclusions take effect only when the human accepts them at ``gate_data``; downstream
stages read the filtered view via ``RunContext.profile()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target, select_features
from .base import Stage, attach_recommendation, questions_from, register_stage

LEAKAGE_CORR = 0.98
HIGH_MISSING = 0.30


@register_stage
class ExploreStage(Stage):
    name = "explore"
    description = "Profile the dataset and flag data-quality / leakage risks."

    def run(self, ctx: RunContext) -> StageResult:
        df = ctx.load_dataset()
        cfg = ctx.config
        target = cfg.data.target
        if target not in df.columns:
            return StageResult(
                stage=self.name, verdict=Verdict.FAIL,
                summary=f"Target column '{target}' not found in dataset.",
            )

        features = select_features(df, cfg)
        y = coerce_target(df, cfg)
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        # --- basic profile -------------------------------------------------------
        dtypes = {c: str(df[c].dtype) for c in df.columns}
        missing = {c: float(df[c].isna().mean()) for c in df.columns}
        numeric_cols = df[features].select_dtypes(include=["number", "bool"]).columns.tolist()
        numeric_summary = {
            c: {
                "mean": float(df[c].mean()),
                "std": float(df[c].std()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }
            for c in numeric_cols
        }

        # --- target relationship + leakage detection -----------------------------
        corrs: list[dict] = []
        leakage: list[str] = []
        yv = np.asarray(y, dtype=float)
        for c in numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.nanstd(col) == 0:
                res.add_finding(Finding(id=f"const-{c}", severity=Severity.LOW, category="data-quality",
                                        message=f"Feature '{c}' is constant.", location=c))
                continue
            r = float(np.corrcoef(np.nan_to_num(col), yv)[0, 1])
            corrs.append({"feature": c, "corr": r})
            if abs(r) >= LEAKAGE_CORR:
                leakage.append(c)
                res.add_finding(Finding(
                    id=f"leak-{c}", severity=Severity.HIGH, category="leakage",
                    message=f"Feature '{c}' has |corr|={abs(r):.3f} with target — possible target leakage.",
                    location=c, suggestion="Confirm this feature is available at prediction time; drop if it leaks.",
                ))
        corrs.sort(key=lambda d: abs(d["corr"]), reverse=True)

        for c, frac in missing.items():
            if frac >= HIGH_MISSING:
                res.add_finding(Finding(id=f"missing-{c}", severity=Severity.MEDIUM, category="data-quality",
                                        message=f"Column '{c}' is {frac:.0%} missing.", location=c))

        if cfg.task.is_classification:
            counts = pd.Series(y).value_counts().to_dict()
            target_summary = {"classes": {str(k): int(v) for k, v in counts.items()},
                              "positive_rate": float(np.mean(y))}
            minority = min(counts.values()) / len(y)
            if minority < 0.05:
                res.add_finding(Finding(id="imbalance", severity=Severity.MEDIUM, category="data-quality",
                                        message=f"Severe class imbalance (minority share {minority:.1%})."))
        else:
            target_summary = {"mean": float(np.mean(y)), "std": float(np.std(y)),
                              "min": float(np.min(y)), "max": float(np.max(y))}

        if not features:
            res.verdict = Verdict.FAIL
            res.summary = "No usable feature columns after exclusions."
            return res

        profile = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "target": target,
            "task": cfg.task.value,
            "features": features,
            "numeric_features": numeric_cols,
            "categorical_features": [c for c in features if c not in numeric_cols],
            "dtypes": dtypes,
            "missing": missing,
            "numeric_summary": numeric_summary,
            "target_summary": target_summary,
            "top_correlations": corrs[:15],
            "leakage_suspects": leakage,
        }
        # --- the Data Analyst's recommendation (engine-checked) --------------------
        res.payload = profile
        constant = [c for c in numeric_cols if f"const-{c}" in {f.id for f in res.findings}]
        out = ctx.recommend("data_analyst", {
            "columns": list(df.columns),
            "features": features,
            "leakage_suspects": leakage,
            "top_correlations": corrs[:15],
            "missing_high": {c: round(f, 4) for c, f in missing.items() if f >= HIGH_MISSING},
            "constant_features": constant,
            "target_summary": target_summary,
            "dtypes": {c: dtypes[c] for c in features},
            "current_exclusions": list(ctx.overrides.exclude_columns),
        }, fresh={"explore": res})
        profile["recommended_exclusions"] = [d.column for d in out.column_decisions
                                             if d.decision == "exclude"]
        profile["column_decisions"] = [d.model_dump() for d in out.column_decisions]
        profile["questions"] = questions_from(out.questions_for_sponsor, category="data",
                                              reentry="explore", prefix="data")
        attach_recommendation(profile, ctx, "data_analyst", out)
        ref = ctx.save_json("stages/explore/profile.json", profile)
        res.add_artifact(ref)
        res.payload = profile
        res.metrics = {"n_rows": profile["n_rows"], "n_features": len(features),
                       "n_leakage_suspects": len(leakage)}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS
        res.summary = (
            f"Profiled {profile['n_rows']} rows x {profile['n_cols']} cols; {len(features)} features; "
            f"{len(leakage)} leakage suspect(s); {len(res.findings)} data-quality finding(s). "
            f"Data Analyst recommends excluding {len(profile['recommended_exclusions'])} column(s)."
        )
        return res
