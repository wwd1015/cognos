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
    DataScoutOutput,
    DesignLeadOutput,
    ExperimentProposal,
    FeatureCandidate,
    IntakeAnalystOutput,
    ModelerOutput,
    ToolRequestOutput,
    ValidatorOutput,
    WriterOutput,
)

Check = Callable[[Contract, dict[str, Any]], list[str]]


def _cited(out: Contract) -> list[str]:
    ids: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, Claim | AgentFinding | ColumnDecision | FeatureCandidate):
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


def intake_analyst(out: IntakeAnalystOutput, sl: dict[str, Any]) -> list[str]:
    """The brief is grounded and complete: a 'stated' position quotes the documents the agent was
    shown, every undecided required field is asked about, and nothing decided is asked again."""
    from ..engagement import squash

    errs: list[str] = []
    fields = {f["field"]: f for f in sl.get("fields", [])}
    got = [e.field for e in out.brief]
    for name in fields:
        if got.count(name) != 1:
            errs.append(f"brief must contain field {name!r} exactly once")
    for name in set(got) - set(fields):
        errs.append(f"brief names {name!r}, which is not a field of this engagement")
    docs = [sl.get("intent_document") or {}, *sl.get("supporting_documents", []),
            *(sl.get("prior_model") or {}).get("documents", [])]
    corpus = squash("\n".join(str(d.get("text") or "") for d in docs))
    asked = {q.field for q in out.interview if q.field != "none"}
    blocking = {q.field for q in out.interview if q.blocking}
    for e in out.brief:
        if e.basis == "missing":
            if e.value.strip():
                errs.append(f"brief field {e.field!r} is 'missing' but carries a value; leave it "
                            "empty or mark it inferred")
            continue
        if not e.value.strip():
            errs.append(f"brief field {e.field!r} is {e.basis!r} but has no value")
        if e.basis == "stated":
            quote = squash(e.quote)
            if len(quote) < 3 or quote not in corpus:
                errs.append(f"brief field {e.field!r} is 'stated' but its quote is not in the "
                            "documents; copy a passage verbatim, or mark it inferred or missing")
    stated = {e.field for e in out.brief if e.basis == "stated"}
    for name, f in fields.items():
        if f.get("decided"):
            if name in asked:
                errs.append(f"field {name!r} is already decided; do not ask about it again")
        elif f.get("required") and name not in stated and name not in blocking:
            errs.append(f"required field {name!r} is not stated in the documents; add a blocking "
                        "interview question for it")
    for name in asked - set(fields):
        errs.append(f"interview question targets {name!r}, which is not a field of this engagement")
    targeted = [q.field for q in out.interview if q.field != "none"]
    for name in {n for n in targeted if targeted.count(n) > 1}:
        errs.append(f"ask one question per field; {name!r} has several")
    texts = [" ".join(q.question.lower().split()) for q in out.interview]
    if len(texts) != len(set(texts)):
        errs.append("interview questions must be distinct")
    answered = {" ".join(a["question"].lower().split()) for a in sl.get("sponsor_answers", [])}
    for t in texts:
        if t in answered:
            errs.append("an interview question repeats one the sponsor already answered")
    if out.clarity == "clear" and any(q.blocking for q in out.interview):
        errs.append("clarity cannot be 'clear' while a blocking interview question is open")
    if sl.get("kind") == "update":
        if out.update_scope == "not_applicable":
            errs.append("a model update needs update_scope: recalibrate, re_estimate or redevelop")
        if not out.scope_rationale.strip():
            errs.append("a model update needs scope_rationale")
        if "requested_changes" in stated and not out.change_items:
            errs.append("the update request lists changes; restate each one in change_items")
        families = set(sl.get("engine_families", []))
        if out.incumbent_family and out.incumbent_family not in families:
            errs.append(f"incumbent_family {out.incumbent_family!r} is not an engine family; "
                        f"choose from {sorted(families)} or leave it empty")
    else:
        if out.update_scope != "not_applicable" or out.change_items or out.incumbent_family:
            errs.append("this is a new model development: update_scope must be not_applicable, "
                        "with no change_items and no incumbent_family")
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
    fixed = sl.get("target")
    if fixed and out.target and out.target != fixed:
        errs.append(f"the target is {fixed!r}; to argue for another column, raise it in "
                    "uncertainties or questions_for_sponsor")
    named = [c.column for c in out.feature_candidates]
    for c in named:
        if c not in features:
            errs.append(f"feature_candidates names {c!r}, which is not a candidate feature")
        elif c in excluded:
            errs.append(f"{c!r} is both a feature candidate and excluded; choose one")
    if len(named) != len(set(named)):
        errs.append("each column may appear in feature_candidates only once")
    return errs


def data_scout(out: DataScoutOutput, sl: dict[str, Any]) -> list[str]:
    """Analysis requests the engine can run: a known tool with known parameters and columns, or
    a script that passes the restricted-code check."""
    from ..analysis import sandbox

    errs: list[str] = []
    columns = set(sl.get("columns", []))
    if not sl.get("target"):
        candidates = {c["column"] for c in sl.get("target_candidates", [])}
        if not out.target_column:
            errs.append("no target is fixed: name the dependent variable in target_column")
        elif out.target_column not in candidates:
            errs.append(f"target_column {out.target_column!r} cannot be the dependent variable "
                        f"here; choose from {sorted(candidates)}")
    elif out.target_column and out.target_column != sl["target"]:
        errs.append(f"the target is already {sl['target']!r}; leave target_column empty")
    tools = {t["name"]: t for t in sl.get("tools", [])}
    limit = sl.get("max_requests", 8)
    if len(out.requests) > limit:
        errs.append(f"at most {limit} analyses per round; you asked for {len(out.requests)}")
    target_known = bool(sl.get("target") or out.target_column)
    for i, r in enumerate(out.requests, 1):
        where = f"request {i}"
        if bool(r.tool) == bool(r.code.strip()):
            errs.append(f"{where}: give a tool or code, not both and not neither")
            continue
        if not r.purpose.strip():
            errs.append(f"{where}: say what question it answers in purpose")
        if r.code.strip():
            if not sl.get("allow_code"):
                errs.append(f"{where}: writing code is switched off for this run; use a tool")
            else:
                errs += [f"{where}: {e}" for e in sandbox.validate(r.code)]
            continue
        tool = tools.get(r.tool)
        if tool is None:
            errs.append(f"{where}: unknown tool {r.tool!r}; choose from {sorted(tools)}")
            continue
        if tool.get("needs_target") and not target_known:
            errs.append(f"{where}: {r.tool} needs the target")
        for p in r.params:
            if p.name not in tool.get("params", {}):
                errs.append(f"{where}: {r.tool} has no parameter {p.name!r}")
            elif p.name == "column" and p.value not in columns:
                errs.append(f"{where}: unknown column {p.value!r}")
    return errs[:12]


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


def tool_request(out: ToolRequestOutput, sl: dict[str, Any]) -> list[str]:
    """Tool requests the engine can run: offered to this stage, known parameters, within the
    round limit, and not a repeat of a run already made."""
    errs: list[str] = []
    tools = {t["name"]: t for t in sl.get("tools", [])}
    limit = sl.get("max_requests", 8)
    if len(out.requests) > limit:
        errs.append(f"at most {limit} tool runs per round; you asked for {len(out.requests)}")
    ran = {(r["tool"], tuple(sorted((r.get("params") or {}).items())))
           for r in sl.get("tool_runs", []) if isinstance(r.get("params"), dict)}
    seen = set()
    for i, r in enumerate(out.requests, 1):
        where = f"request {i}"
        if r.tool not in tools:
            errs.append(f"{where}: {r.tool!r} is not a tool offered to this stage; choose from "
                        f"{sorted(tools)}")
            continue
        if not r.purpose.strip():
            errs.append(f"{where}: say what question the run answers (purpose)")
        unknown = [p.name for p in r.params if p.name not in tools[r.tool].get("params", {})]
        if unknown:
            errs.append(f"{where}: {r.tool} has no parameter(s) {unknown}; it takes "
                        f"{sorted(tools[r.tool].get('params', {}))}")
        key = (r.tool, tuple(sorted((p.name, p.value) for p in r.params)))
        if key in seen or key in ran:
            errs.append(f"{where}: {r.tool} with these parameters is already requested or run")
        seen.add(key)
    return errs


CHECKS: dict[str, Check] = {
    "intake_analyst": intake_analyst,
    "data_analyst": data_analyst,
    "data_scout": data_scout,
    "design_lead": design_lead,
    "modeler": modeler,
    "experiment": experiment,
    "validator": validator,
    "writer": writer,
}


def run_checks(agent: str, out: Contract, sl: dict[str, Any], extra: Check | None = None) -> list[str]:
    errs = default(out, sl)
    if isinstance(out, ToolRequestOutput):
        errs += tool_request(out, sl)
    if agent in CHECKS:
        errs += CHECKS[agent](out, sl)
    if extra is not None:
        errs += extra(out, sl)
    return errs
