"""Engine re-checks on agent output — the "engine disposes" half of every recommendation.

A non-empty error list rejects the answer; the runner feeds the errors back and the agent retries.
These rules are deterministic and cheap, and they are the same for every backend (the heuristic
agents must pass them too — the tests hold them to it).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import facts as facts_mod
from .contracts import (
    SECTIONS,
    AgentFinding,
    Claim,
    ColumnDecision,
    Contract,
    DataAnalystOutput,
    DesignLeadOutput,
    ExperimentProposal,
    ModelerOutput,
    ValidatorOutput,
    WriterOutput,
)

Check = Callable[[Contract, dict[str, Any]], list[str]]


def _cited(out: Contract) -> list[str]:
    ids: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, Claim | AgentFinding | ColumnDecision):
            ids.extend(o.evidence)
        if isinstance(o, Contract):
            for name in type(o).model_fields:
                walk(getattr(o, name))
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(out)
    return ids


def default(out: Contract, sl: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    unknown = facts_mod.unknown_refs(_cited(out), sl.get("facts", {}))
    if unknown:
        errs.append(f"evidence cites unknown fact id(s) {sorted(set(unknown))[:8]}; cite only keys "
                    "of context.facts")
    expected = {c["id"] for c in sl.get("challenges", [])}
    responses = getattr(out, "responses_to_challenges", None)
    if responses is not None:
        got = [r.challenge_id for r in responses]
        missing = expected - set(got)
        extra = set(got) - expected
        if missing:
            errs.append(f"responses_to_challenges is missing challenge id(s) {sorted(missing)}; "
                        "answer every challenge")
        if extra:
            errs.append(f"responses_to_challenges names unknown challenge id(s) {sorted(extra)}")
    return errs


def data_analyst(out: DataAnalystOutput, sl: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    columns = set(sl.get("columns", []))
    decided = [d.column for d in out.column_decisions]
    for c in decided:
        if c not in columns:
            errs.append(f"column_decisions names unknown column {c!r}")
    for s in sl.get("leakage_suspects", []):
        if s not in decided:
            errs.append(f"leakage suspect {s!r} has no keep/exclude decision")
    target = sl.get("project", {}).get("target")
    if any(d.column == target and d.decision == "exclude" for d in out.column_decisions):
        errs.append("the target column cannot be excluded")
    features = set(sl.get("features", []))
    excluded = {d.column for d in out.column_decisions if d.decision == "exclude"}
    if features and features <= excluded:
        errs.append("excluding every feature leaves nothing to model")
    if len(decided) != len(set(decided)):
        errs.append("each column may appear in column_decisions only once")
    return errs


def design_lead(out: DesignLeadOutput, sl: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    assessed = {f["framework"]: f for f in sl.get("framework_assessment", [])}
    chosen = [f.framework for f in out.frameworks]
    for fid in assessed:
        if fid not in chosen:
            errs.append(f"framework {fid!r} from framework_assessment has no decision")
    for f in out.frameworks:
        a = assessed.get(f.framework)
        if a is None:
            errs.append(f"unknown framework {f.framework!r}")
        elif f.decision == "primary" and a.get("applicable") is False:
            errs.append(f"framework {f.framework!r} is not applicable to this data and cannot be primary")
    fittable = set(sl.get("fittable_families", []))
    seen: set[tuple[str, str]] = set()
    if not out.slate:
        errs.append("slate must contain at least one specification")
    for item in out.slate:
        if item.family not in fittable:
            errs.append(f"slate family {item.family!r} is not engine-fittable here; choose from "
                        f"{sorted(fittable)}")
        key = (item.family, item.feature_strategy)
        if key in seen:
            errs.append(f"duplicate slate item {key}")
        seen.add(key)
    if sl.get("interpretability") == "required":
        for item in out.slate:
            if item.family in ("random_forest", "gradient_boosting") and item.role != "challenger":
                errs.append(f"interpretability is required: {item.family} may only be a challenger")
    return errs


def modeler(out: ModelerOutput, sl: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    admissible = {c["id"]: c for c in sl.get("admissible_set", [])}
    pick = admissible.get(out.champion)
    if pick is None:
        return [f"champion {out.champion!r} is not in admissible_set; choose one of "
                f"{sorted(admissible)}"]
    if sl.get("interpretability") == "required" and pick.get("role") == "challenger":
        errs.append(f"interpretability is required: {out.champion} ({pick['family']}) is a "
                    "challenger benchmark and cannot be the champion")
    feats = set(pick.get("features", []))
    for s in out.sign_checks:
        if s.feature not in feats:
            errs.append(f"sign_checks names {s.feature!r}, which is not a feature of {out.champion}")
    return errs


def experiment(out: ExperimentProposal, sl: dict[str, Any]) -> list[str]:
    if out.stop:
        return []
    allowed = set(sl.get("allowed_families", []))
    if out.family not in allowed:
        return [f"family {out.family!r} is not allowed; choose from {sorted(allowed)} or stop"]
    return []


def validator(out: ValidatorOutput, sl: dict[str, Any]) -> list[str]:
    ids = [f.id for f in out.findings]
    errs = [f"duplicate finding id {i!r}" for i in {i for i in ids if ids.count(i) > 1}]
    for f in out.findings:
        if not f.evidence:
            errs.append(f"finding {f.id!r} cites no evidence; an effective challenge rests on facts")
    return errs


def writer(out: WriterOutput, sl: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    got = [s.section for s in out.sections]
    for name in SECTIONS:
        if got.count(name) != 1:
            errs.append(f"section {name!r} must appear exactly once")
    known = sl.get("facts", {})
    for s in out.sections:
        unknown = facts_mod.unknown_refs(facts_mod.placeholder_ids(s.markdown), known)
        if unknown:
            errs.append(f"section {s.section!r} uses unknown fact placeholder(s) {unknown[:6]}")
        bare = facts_mod.bare_metrics(s.markdown)
        if bare:
            errs.append(f"section {s.section!r} types metric value(s) {bare[:4]} directly; write "
                        "{{fact:<id>}} placeholders instead")
    return errs


CHECKS: dict[str, Check] = {
    "data_analyst": data_analyst,
    "design_lead": design_lead,
    "modeler": modeler,
    "experiment": experiment,
    "validator": validator,
    "writer": writer,
}


def run_checks(agent: str, out: Contract, sl: dict[str, Any], extra: Check | None = None) -> list[str]:
    errs = default(out, sl)
    if agent in CHECKS:
        errs += CHECKS[agent](out, sl)
    if extra is not None:
        errs += extra(out, sl)
    return errs
