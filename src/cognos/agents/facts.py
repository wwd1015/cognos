"""Facts: the only numbers an agent may state.

Every agent context carries a flat ``facts`` map (``"model.cv_mean": 0.7812``) built from engine
artifacts. Claims and findings cite fact ids; the engine rejects unknown ids. The writer's prose
carries ``{{fact:<id>}}`` placeholders that the engine renders — an agent never types a metric, so a
hallucinated number cannot reach the white paper (no LLM math).
"""

from __future__ import annotations

import math
import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"\{\{fact:([A-Za-z0-9_.\-]+)\}\}")
# A bare number with two or more decimals (e.g. 0.78, 12.345) or a percentage with decimals —
# metric-like values that must come through a placeholder instead.
BARE_METRIC_RE = re.compile(r"(?<![\w.{])\d+\.\d{2,}(?![\w}])|\b\d+\.\d+\s?%")


def _clean(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    if isinstance(value, list | tuple):
        return ", ".join(str(v) for v in value) if value else "none"
    try:
        f = float(value)
        return round(f, 4) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return str(value)


def _put(facts: dict[str, Any], key: str, value: Any) -> None:
    v = _clean(value)
    if v is not None:
        facts[key] = v


def collect(ctx, *, fresh: dict | None = None,
            prefixes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Facts from stage results — ``fresh`` (the calling stage's provisional result) wins over
    what is on disk — filtered to the ``prefixes`` an agent may see (its independence scope)."""
    fresh = fresh or {}

    def get(stage: str):
        return fresh[stage] if stage in fresh else ctx.get(stage)

    f: dict[str, Any] = {}
    it = get("intake")
    if it is not None and it.payload:
        p = it.payload
        _put(f, "intake.development_mode", p.get("kind"))
        _put(f, "intake.clarity", p.get("clarity"))
        _put(f, "intake.n_documents", len(p.get("documents", [])))
        brief = p.get("brief", [])
        _put(f, "intake.n_fields_decided", sum(1 for e in brief if e.get("value")))
        _put(f, "intake.n_fields_missing", sum(1 for e in brief if not e.get("value")))
        _put(f, "intake.n_open_questions", len(p.get("questions", [])))
        _put(f, "intake.n_blocking_questions", p.get("n_blocking"))
        up = p.get("update") or {}
        _put(f, "intake.update_scope", up.get("scope"))
        if up:
            _put(f, "intake.n_change_items", len(up.get("change_items", [])))
        prior = (p.get("prior") or {}).get("run") or {}
        for k in ("champion_family", "metric", "cv_mean", "cv_std", "holdout_metric", "n_train",
                  "validation_verdict"):
            _put(f, f"prior.{k}", prior.get(k))
        if prior:
            _put(f, "prior.n_features", len(prior.get("features", [])))
    ex = get("explore")
    if ex is not None and ex.payload:
        p = ex.payload
        _put(f, "explore.n_rows", p.get("n_rows"))
        _put(f, "explore.n_features", len(p.get("features", [])))
        _put(f, "explore.leakage_suspects", p.get("leakage_suspects", []))
        ts = p.get("target_summary") or {}
        _put(f, "explore.positive_rate", ts.get("positive_rate"))
        for c in p.get("top_correlations", [])[:12]:
            _put(f, f"explore.corr.{c['feature']}", c["corr"])
        for col, frac in (p.get("missing") or {}).items():
            if frac and frac >= 0.05:
                _put(f, f"explore.missing.{col}", frac)
        _put(f, "explore.target", p.get("target"))
        _put(f, "explore.task", p.get("task"))
        analyses = [a for a in p.get("analyses", []) if a.get("status") == "ok"]
        if p.get("analyses") is not None:
            _put(f, "explore.n_analyses", len(analyses))
            _put(f, "explore.n_code_analyses", sum(1 for a in analyses if a["kind"] == "code"))
        # What each analysis found. A tool's numbers are the engine's; a script's are the
        # engine's execution of code the analyst wrote, kept and re-run at validation.
        for a in analyses[:16]:
            for key, value in list((a.get("summary") or {}).items())[:8]:
                _put(f, f"explore.analysis.{a['id']}.{key}", value)
    ide = get("ideate")
    if ide is not None and ide.payload:
        ds = ide.payload.get("data_structure") or {}
        for k in ("shape", "n_events", "event_rate", "events_per_variable"):
            _put(f, f"ideate.{k}", ds.get(k))
        _put(f, "ideate.families", ide.payload.get("families", []))
        _put(f, "ideate.n_open_questions", len(ide.payload.get("open_questions", [])))
    mo = get("model")
    if mo is not None and mo.payload:
        p = mo.payload
        champ = p.get("champion") or {}
        _put(f, "model.metric", p.get("metric"))
        _put(f, "model.champion_family", champ.get("family"))
        _put(f, "model.champion_label", p.get("champion_label"))
        _put(f, "model.n_champion_features", len(champ.get("features", [])))
        _put(f, "model.cv_mean", p.get("cv_mean"))
        _put(f, "model.cv_std", p.get("cv_std"))
        _put(f, "model.n_candidates_tried", p.get("n_candidates_tried"))
        _put(f, "model.n_train", p.get("n_train"))
        _put(f, "model.holdout_metric", p.get("holdout_metric"))
        _put(f, "model.n_holdout", p.get("n_holdout"))
        _put(f, "model.holdout_evaluations", p.get("holdout_evaluations"))
        for name, v in (p.get("coefficients") or {}).items():
            _put(f, f"model.coef.{name}", v)
        for name, v in (p.get("pvalues") or {}).items():
            _put(f, f"model.p.{name}", v)
        diag = p.get("diagnostics") or {}
        _put(f, "model.diagnostics_passed", diag.get("n_passed"))
        _put(f, "model.diagnostics_run", diag.get("n_run"))
        for t in diag.get("tests", []) or []:
            if not t.get("skipped"):
                _put(f, f"model.test.{t['name']}", "passed" if t.get("passed") else "failed")
        bench = p.get("challenger_benchmark") or {}
        _put(f, "model.challenger_benchmark_score", bench.get("benchmark_score"))
    bt = get("backtest")
    if bt is not None and bt.payload:
        p = bt.payload
        _put(f, "backtest.evaluation_sample", p.get("evaluation_sample"))
        _put(f, "backtest.oos_metric", p.get("oos_metric"))
        oa = p.get("outcomes_analysis") or {}
        for k in ("gini", "ks", "auc", "expected_calibration_error", "psi", "psi_label",
                  "observed_default_rate", "mean_predicted_pd"):
            _put(f, f"backtest.{k}", oa.get(k))
        wf = p.get("walk_forward") or {}
        _put(f, "backtest.walk_forward_mean", wf.get("mean"))
        _put(f, "backtest.walk_forward_std", wf.get("std"))
        port = p.get("portfolio_analysis") or {}
        _put(f, "backtest.portfolio_expected_loss", port.get("expected_loss"))
        _put(f, "backtest.portfolio_var", port.get("var"))
    va = get("validate")
    if va is not None and va.payload:
        p = va.payload
        _put(f, "validate.verdict", va.verdict.value)
        _put(f, "validate.overall_score", p.get("overall_score"))
        for axis, v in (p.get("rubric") or {}).items():
            _put(f, f"validate.rubric.{axis}", v)
        _put(f, "validate.n_findings", len(va.findings))
    co = get("comply")
    if co is not None and co.payload:
        p = co.payload
        _put(f, "comply.n_outstanding", len(p.get("outstanding_human_steps", [])))
        for k, e in (p.get("sr11_7") or {}).items():
            _put(f, f"comply.sr11_7.{k}", e.get("status"))
        fl = p.get("fair_lending") or {}
        _put(f, "comply.disparate_impact", fl.get("disparate_impact"))
    if prefixes is not None:
        f = {k: v for k, v in f.items() if k.startswith(prefixes)}
    return f


def unknown_refs(ids: list[str], facts: dict[str, Any]) -> list[str]:
    return [i for i in ids if i not in facts]


def render(text: str, facts: dict[str, Any]) -> str:
    """Replace ``{{fact:id}}`` placeholders with formatted fact values."""

    def sub(m: re.Match) -> str:
        v = facts.get(m.group(1))
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:,.0f}"
        return str(v) if v is not None else "n/a"

    return PLACEHOLDER_RE.sub(sub, text)


def placeholder_ids(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def bare_metrics(text: str) -> list[str]:
    """Metric-like numbers typed directly into prose (outside placeholders)."""
    return BARE_METRIC_RE.findall(PLACEHOLDER_RE.sub("", text))
