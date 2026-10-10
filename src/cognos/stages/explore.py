"""Stage 1 — Data exploration.

Fetches the dataset from its source (a file, SQLite, Snowflake or a plugin connector; the
snapshot and its provenance are kept in the run), profiles it (shape, dtypes, missingness,
distributions, correlations) and flags data-quality risks the downstream stages must respect —
especially **target-leakage suspects** (features near-perfectly correlated with the target), the
#1 way automated systems silently overfit.

The **Data Analyst** agent reads the data against the confirmed business intent in two steps.
First it *requests analyses* — a registered tool (built in or from a plugin) or Python it writes
— and the engine runs them, keeping every result and every script as an artifact
(``stages/explore/analyses/``). When the profile leaves the dependent variable open, the analyst
names it in the first round. When the run has several sources, the analyst first proposes how
they join (``data_linker``): the engine measures the links, runs the chosen plan and keeps one
row per row of the base table. Then it *recommends*: the target, the features worth considering,
keep/exclude for every leakage suspect, and questions for the sponsor. The target and the
exclusions (and the join) take effect only when the human accepts them at ``gate_data``; downstream stages read
the filtered view via ``RunContext.profile()``.
"""

from __future__ import annotations

import re
import shutil

import numpy as np
import pandas as pd

from .. import linking
from ..analysis import run as analysis
from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..datautil import coerce_target, select_features
from .base import Stage, attach_recommendation, questions_from, register_stage

LEAKAGE_CORR = 0.98
HIGH_MISSING = 0.30

# Name tokens that mark a column as an outcome. Used only to rank target candidates when the
# profile leaves the dependent variable open; the analyst chooses and the human confirms.
_OUTCOME_HINTS = {"default", "defaulted", "target", "label", "outcome", "bad", "event", "flag",
                  "loss", "lgd", "churn", "churned", "fraud", "chargeoff", "delinquent",
                  "delinquency", "response", "converted", "y"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t}


def target_candidates(df: pd.DataFrame, cfg, intent: str) -> list[dict]:
    """Columns that could be the dependent variable, most outcome-like first. Mechanical: a
    name that reads like an outcome, a binary coding, and words shared with the stated intent."""
    dc = cfg.data
    skip = {dc.datetime_col, dc.event_time_col, cfg.migration.next_rating_col,
            *dc.drop_columns, *dc.protected_attributes}
    words = {w for w in _tokens(intent) if len(w) >= 4}
    out = []
    for col in df.columns:
        s = df[col]
        if col in skip or not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            continue
        n_unique = int(s.nunique())
        if n_unique < 2 or (n_unique == len(s) and pd.api.types.is_integer_dtype(s)):
            continue  # constant, or a row identifier
        toks = _tokens(col)
        binary = n_unique == 2
        why, score = [], 0
        if toks & _OUTCOME_HINTS:
            score += 3
            why.append("its name reads like an outcome")
        if binary:
            score += 2
            why.append("it is binary")
        if toks & words:
            score += 2
            why.append("the business intent uses the same words")
        out.append({"column": col, "binary": binary, "n_unique": n_unique,
                    "mean": round(float(pd.to_numeric(s, errors="coerce").mean()), 4),
                    "score": score, "why": "; ".join(why) or "a numeric column"})
    out.sort(key=lambda c: -c["score"])
    scored = [c for c in out if c["score"] > 0]
    return (scored or out)[:8]


def infer_task(series: pd.Series) -> str:
    """The task a target column implies: two values is classification, a number is regression."""
    return "classification" if series.nunique() == 2 else "regression"


def _intent_text(ctx: RunContext) -> str:
    intake = ctx.get("intake")
    brief = ((intake.payload or {}).get("brief") if intake is not None else None) or []
    parts = [b.get("value") or "" for b in brief if b.get("field") in (
        "objective", "default_definition", "use_case")]
    d = ctx.config.design
    return " ".join([*parts, d.default_definition, d.use_case, ctx.config.description])


def linked_dataset(ctx: RunContext) -> tuple[pd.DataFrame, dict]:
    """Several sources as one modelling table, and how it was made. The plan in force is the
    data-gate decision, else the profile's stated join, else the Data Analyst's proposal. The
    engine runs it and writes the snapshot every later stage reads."""
    import hashlib
    import json
    from datetime import UTC, datetime

    from ..fsutil import atomic_write

    base = ctx.base_config
    tables, provenance = linking.snapshot(ctx)
    target = base.data.target or ctx.overrides.target or ""
    prof = linking.profile(tables, base.data.join, target)
    rationale: dict = {}
    if ctx.overrides.join:
        source = "decision"
        base_table, steps = ctx.overrides.join["base"], list(ctx.overrides.join["steps"])
    elif base.data.join:
        source = "profile"
        base_table = base.data.base or base.data.join[0].left
        stated = {(lk["a"], tuple(lk["a_on"]), lk["b"], tuple(lk["b_on"])): lk["id"]
                  for lk in prof["links"]}
        steps, problems = linking.resolve(base_table, [
            {"link": stated[(j.left, tuple(j.left_on), j.right, tuple(j.right_on))],
             "many": j.many, "unmatched": j.unmatched} for j in base.data.join], prof)
        if problems:
            raise linking.LinkError("data.join: " + "; ".join(problems))
    else:
        source = "agent"
        out = ctx.recommend("data_linker", {
            "tables": prof["tables"], "links": prof["links"], "target": target,
            "base_table": base.data.base or "",
        })
        base_table = out.base_table
        steps, _ = linking.resolve(base_table, [
            {"link": j.link, "many": j.many, "unmatched": j.unmatched} for j in out.joins], prof)
        rationale = {"base": out.base_rationale, "joins": {j.link: j.reason for j in out.joins},
                     "unmatched": {j.link: j.unmatched_reason for j in out.joins},
                     "left_out": [t.model_dump() for t in out.left_out],
                     "concerns": list(out.concerns)}
    df, report = linking.apply(tables, base_table, steps)
    joined = {base_table, *(s["right"] for s in steps)}
    path = ctx.data_dir / "dataset.parquet"
    df.to_parquet(path, index=False)
    plan = {"base": base_table, "steps": steps, "source": source,
            "digest": linking.plan_digest(base_table, steps)}
    atomic_write(ctx.data_dir / "source.json", json.dumps({
        "kind": "linked", "connector": "Linked sources", "origin": "cognos",
        "location": " + ".join(sorted(joined, key=[t["name"] for t in prof["tables"]].index)),
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": provenance, "join": plan,
    }, indent=1, default=str))
    by_name = {p["name"]: p for p in provenance}
    info = {
        "tables": [{**t, **{k: by_name[t["name"]].get(k) for k in (
            "connector", "kind", "location", "query", "snapshot_sha256")}}
            for t in prof["tables"]],
        "links": prof["links"], "plan": plan, "rationale": rationale, "report": report,
        "left_out": [t["name"] for t in prof["tables"] if t["name"] not in joined],
    }
    ctx.save_json("stages/explore/join.json", info)
    return df, info


@register_stage
class ExploreStage(Stage):
    name = "explore"
    description = "Fetch and profile the dataset; settle the target; flag data-quality / leakage risks."

    def run(self, ctx: RunContext) -> StageResult:
        base = ctx.base_config
        linked = None
        if linking.is_linked(base):
            from ..agents.runner import AgentRunError

            try:
                df, linked = linked_dataset(ctx)
            except (linking.LinkError, AgentRunError) as exc:
                return StageResult(stage=self.name, verdict=Verdict.FAIL,
                                   summary=f"The sources could not be joined: {exc}")
        else:
            df = ctx.load_dataset()
        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        shutil.rmtree(ctx.stages_dir / "explore" / "analyses", ignore_errors=True)

        # --- the dependent variable: the profile, a data-gate decision, or the analyst ----------
        target = base.data.target or ctx.overrides.target or ""
        target_source = "profile" if base.data.target else ("decision" if target else "agent")
        target_rationale = ""
        candidates = [] if base.data.target else target_candidates(df, base, _intent_text(ctx))
        if target and target not in df.columns:
            return StageResult(
                stage=self.name, verdict=Verdict.FAIL,
                summary=f"Target column '{target}' not found in dataset.",
            )
        if not target and not candidates:
            return StageResult(stage=self.name, verdict=Verdict.FAIL,
                               summary="No column can serve as the dependent variable (no numeric "
                                       "column varies). Set data.target in the profile.")

        def settled(col: str):
            """The config once ``col`` is the target (task inferred where the profile is silent)."""
            task = (base.task.value if base.task else None) or (
                ctx.overrides.task if col == ctx.overrides.target else None) or infer_task(df[col])
            return ctx._effective_config(base, ctx.overrides).with_target(col, task)

        def correlations(cfg) -> list[dict]:
            y = pd.to_numeric(df[cfg.data.target], errors="coerce").to_numpy(dtype=float)
            out = []
            for c in df[select_features(df, cfg)].select_dtypes(include=["number", "bool"]).columns:
                col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                if np.nanstd(col) == 0:
                    continue
                out.append({"feature": c, "corr": float(np.corrcoef(np.nan_to_num(col),
                                                                    np.nan_to_num(y))[0, 1])})
            return sorted(out, key=lambda d: abs(d["corr"]), reverse=True)

        # --- step 1: the analyst requests analyses; the engine runs them --------------------------
        from .. import plugins
        from ..agents.runner import AgentRunError
        from ..engine import events

        reg = plugins.registry(base.plugins)
        tools = [{"name": t.name, "description": t.description, "params": t.params,
                  "needs_target": t.needs_target, "origin": t.origin}
                 for t in reg.tools_for("explore")]
        author = f"the Data Analyst agent ({ctx.runner.provider['id']})"
        with_missing = [c for c in df.columns if df[c].isna().any()]
        records: list[dict] = []
        res.payload = {"analyses": records}
        for round_no in range(1, base.analysis.rounds + (0 if target else 1) + 1):
            cfg = settled(target) if target else None
            try:
                out = ctx.recommend("data_scout", {
                    "round": round_no, "columns": list(df.columns),
                    "dtypes": {c: str(df[c].dtype) for c in df.columns}, "n_rows": int(len(df)),
                    "target": target, "task": cfg.task.value if cfg else None,
                    "target_basis": target_rationale, "target_candidates": candidates,
                    "datetime_col": base.data.datetime_col,
                    "columns_with_missing": with_missing[:30],
                    "top_correlations": correlations(cfg)[:10] if cfg else [],
                    "tools": tools, "allow_code": base.analysis.allow_code,
                    "max_requests": base.analysis.max_requests,
                    "analyses": [{k: r[k] for k in ("id", "kind", "tool", "purpose", "status",
                                                    "error", "summary")} for r in records],
                }, fresh={"explore": res})
            except AgentRunError as exc:
                if not target:
                    return StageResult(stage=self.name, verdict=Verdict.FAIL,
                                       summary=f"The dependent variable could not be settled: {exc}")
                res.add_finding(Finding(
                    id="analysis-requests-failed", severity=Severity.LOW, category="analysis",
                    message=f"The Data Analyst's analysis requests could not be obtained: {exc}"))
                break
            if not target:
                target, target_rationale = out.target_column, out.target_rationale
                cfg = settled(target)
            env = {"target": target, "task": cfg.task.value, "features": select_features(df, cfg),
                   "datetime_col": base.data.datetime_col}
            reqs = [{"purpose": r.purpose, "tool": r.tool, "code": r.code,
                     "params": {p.name: p.value for p in r.params}} for r in out.requests]
            if reqs:
                records += analysis.execute(ctx, reqs, df, env, start=len(records) + 1,
                                            author=author)
                events.publish(ctx.run_dir, "progress",
                               f"Ran {len(reqs)} analysis request(s) for the Data Analyst.",
                               step="explore")
            if out.done:
                break

        cfg = settled(target)
        task = cfg.task.value
        if cfg.task.is_classification and not set(pd.unique(df[target].dropna())) <= {0, 1}:
            return StageResult(
                stage=self.name, verdict=Verdict.FAIL,
                summary=f"Target '{target}' has two values but is not coded 0/1; recode it (the "
                        "event as 1) or choose another target.")
        features = select_features(df, cfg)
        y = coerce_target(df, cfg)

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
            "task": task,
            "target_source": target_source,
            "target_candidates": candidates,
            "source": ctx.data_source(),
            "linking": linked,
            "analyses": records,
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
            "target": target, "task": task, "target_source": target_source,
            "target_basis": target_rationale, "target_candidates": candidates,
            "analyses": [{k: r[k] for k in ("id", "kind", "tool", "purpose", "status", "error",
                                            "summary")} for r in records],
        }, fresh={"explore": res})
        profile["target_rationale"] = out.target_rationale or target_rationale
        profile["feature_candidates"] = [c.model_dump() for c in out.feature_candidates]
        profile["recommended_exclusions"] = [d.column for d in out.column_decisions
                                             if d.decision == "exclude"]
        profile["column_decisions"] = [d.model_dump() for d in out.column_decisions]
        profile["questions"] = questions_from(out.questions_for_sponsor, category="data",
                                              reentry="explore", prefix="data")
        attach_recommendation(profile, ctx, "data_analyst", out)
        ref = ctx.save_json("stages/explore/profile.json", profile)
        res.add_artifact(ref)
        res.payload = profile
        for step in (linked or {}).get("report", {}).get("steps", []):
            if step["dropped_rows"]:
                res.add_finding(Finding(
                    id=f"join-dropped-{step['right']}", category="data-quality",
                    severity=Severity.MEDIUM if step["match_rate"] < 0.9 else Severity.LOW,
                    message=f"{step['dropped_rows']} base row(s) ({1 - step['match_rate']:.1%}) "
                            f"had no match in {step['right']} ({'; '.join(step['on'])}) and were "
                            "dropped: the model is built without them.", location=step["right"],
                    suggestion="Confirm the dropped rows are not a distinct group the model "
                               "must cover."))
            elif step["match_rate"] < 1:
                res.add_finding(Finding(
                    id=f"join-match-{step['right']}", category="data-quality",
                    severity=Severity.MEDIUM if step["match_rate"] < 0.8 else Severity.LOW,
                    message=f"{1 - step['match_rate']:.1%} of the rows found no match in "
                            f"{step['right']} ({'; '.join(step['on'])}) and have missing values "
                            "in its columns.", location=step["right"]))
        failed = [r for r in records if r["status"] != "ok"]
        if failed:
            res.add_finding(Finding(
                id="analysis-failed", severity=Severity.LOW, category="analysis",
                message=f"{len(failed)} requested analysis(es) did not run "
                        f"({', '.join(r['id'] for r in failed)}); see each one for the reason."))
        n_code = sum(1 for r in records if r["kind"] == "code")
        res.metrics = {"n_rows": profile["n_rows"], "n_features": len(features),
                       "n_leakage_suspects": len(leakage), "n_analyses": len(records),
                       "n_code_analyses": n_code}
        res.verdict = Verdict.WARN if res.findings else Verdict.PASS
        res.summary = (
            f"Profiled {profile['n_rows']} rows x {profile['n_cols']} cols; {len(features)} features; "
            f"{len(leakage)} leakage suspect(s); {len(res.findings)} data-quality finding(s). "
            f"Data Analyst recommends excluding {len(profile['recommended_exclusions'])} column(s)."
            + (f" Target '{target}' ({task}) proposed by the Data Analyst."
               if target_source == "agent" else "")
            + (f" {len(records)} analysis(es) run, {n_code} from agent-written code."
               if records else "")
            + (f" Built from {len(linked['plan']['steps']) + 1} joined source(s)"
               + (" as the Data Analyst proposed." if linked["plan"]["source"] == "agent"
                  else ".") if linked else "")
        )
        return res
