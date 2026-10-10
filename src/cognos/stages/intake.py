"""Stage 0 — Intake: understand the business intent before any data is touched.

The engine reads the engagement's documents (the business intent document, supporting material
and, for a model update, the existing model's artifacts), inventories them, and parses the intent
template. The **Intake Analyst** agent fills the engagement brief, judges whether the goal is clear
and interviews the sponsor where it is not: every question becomes an open gap, each answer
re-runs this stage, and the loop ends when no blocking question is left. The human confirms the
brief at ``gate_intent``; what the sponsor's document states then becomes part of the effective
config (the design brief and the intended use), exactly as a gate answer does.

A run without an intent document still works: the brief is assembled from the profile, and the
interview covers what the profile leaves open.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .. import engagement as eg
from ..artifacts import Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..modeling.fit import DEFAULT_FAMILIES
from .base import Stage, attach_recommendation, register_stage

INTENT_CHARS = 24000  # what the agent is shown of the intent document
EXCERPT_CHARS = 5000  # ... and of every other document


def _clip(text: str | None, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n[… truncated …]"


def _engine_families() -> list[str]:
    fams: list[str] = []
    for group in DEFAULT_FAMILIES.values():
        fams += [f for f in group if f not in fams]
    for f in ("probit", "cloglog", "ridge_logit", "lasso_logit", "hazard_logit", "hazard_cloglog",
              "random_forest", "gradient_boosting"):
        if f not in fams:
            fams.append(f)
    return fams


def question_id(field: str, text: str) -> str:
    """Stable ids: a core design question shares its id with the one ideate would raise, so the
    sponsor is asked once; a free question is keyed by its text."""
    if field in eg.CORE:
        return f"design-{field}"
    if field != "none":
        return f"intent-{field}"
    return f"intent-{hashlib.sha1(text.strip().encode()).hexdigest()[:6]}"


@register_stage
class IntakeStage(Stage):
    name = "intake"
    description = ("Read the business intent and the prior model's artifacts; interview the "
                   "sponsor on what is unclear.")

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.config
        kind = cfg.engagement.kind
        res = StageResult(stage=self.name, verdict=Verdict.PASS)
        docs = eg.load_documents(cfg, ctx.run_dir)
        intent = next((d for d in docs if d["role"] == "intent"), None)
        intent_text = _clip(intent["text"], INTENT_CHARS) if intent and intent["readable"] else ""
        parsed = eg.parse_intent(intent_text)
        fields = eg.fields_for(kind)

        # --- what is already decided: the profile, then answers recorded on the run ----------
        base = ctx.base_config
        profile = {f: str(getattr(base.design, f, "") or "").strip() for f in eg.CORE}
        profile.update({f: str(getattr(base.compliance, f, "") or "").strip()
                        for f in eg.COMPLIANCE_FIELDS})
        profile["objective"] = base.description.strip()  # what the profile says the model is for
        answers = {a["id"]: (a["answer"] or "accepted as an assumption")
                   for a in ctx.sponsor_answers()}
        answered: dict[str, str] = {}
        for f in fields:
            if f.id in ctx.overrides.design or f.id in ctx.overrides.compliance:
                answered[f.id] = ctx.overrides.design.get(f.id) or ctx.overrides.compliance[f.id]
            elif question_id(f.id, "") in answers:
                answered[f.id] = answers[question_id(f.id, "")]

        if intent is None:
            res.add_finding(Finding(
                id="intake-no-intent", severity=Severity.LOW, category="intent",
                message="No business intent document was supplied; the brief is assembled from "
                        "the profile and the sponsor's answers.",
                suggestion="Write one on the template (cognos intent-template) and start the run "
                           "with it."))
        for d in docs:
            if not d["readable"]:
                res.add_finding(Finding(
                    id=f"intake-unreadable-{hashlib.sha1(d['path'].encode()).hexdigest()[:6]}",
                    severity=Severity.MEDIUM if d["role"] == "intent" else Severity.LOW,
                    category="intent", location=d["name"],
                    message=f"'{d['name']}' ({d['role']}) could not be read as text: {d['note']}.",
                    suggestion="Supply it as .md, .txt, .docx or source code."))

        prior_docs = [d for d in docs if d["prior"]]
        prior_texts = [d["text"] for d in prior_docs if d["readable"]]
        prior_run = eg.prior_run_summary(ctx.run_dir) if kind == "update" else None
        mentions = eg.family_mentions(prior_texts)
        likely = (prior_run or {}).get("champion_family") or (mentions[0]["family"] if mentions
                                                              else None)
        inventory = [{k: d[k] for k in ("name", "role", "path", "readable", "note", "chars",
                                        "sha256")} for d in docs]
        prior: dict[str, Any] | None = None
        if kind == "update":
            prior = {"family_mentions": mentions, "likely_family": likely, "run": prior_run,
                     "n_artifacts": len(prior_docs),
                     "n_code_files": sum(1 for d in prior_docs if d["role"] == "code")}
            if not prior_texts and prior_run is None:
                res.add_finding(Finding(
                    id="intake-no-prior-text", severity=Severity.MEDIUM, category="intent",
                    message="None of the existing model's artifacts could be read; the update is "
                            "scoped from the request alone.",
                    suggestion="Supply the white paper or the development code as text."))

        res.payload = {"kind": kind, "documents": inventory, "brief": [], "questions": [],
                       "prior": prior}
        tool_runs = ctx.consult_tools(
            "intake_analyst", {"kind": kind, "documents": inventory}, res=res,
            inputs={"documents": [{k: d[k] for k in ("name", "role", "text")}
                                  for d in docs if d["readable"]]})
        out = ctx.recommend("intake_analyst", {
            "tool_runs": tool_runs,
            "kind": kind,
            "fields": [{"field": f.id, "label": f.label, "asks": f.hint, "required": f.required,
                        "decided": answered.get(f.id) or profile.get(f.id) or None,
                        "template_value": parsed.get(f.id)} for f in fields],
            "intent_document": ({"name": intent["name"], "text": intent_text}
                                if intent_text else None),
            "supporting_documents": [{"name": d["name"], "text": _clip(d["text"], EXCERPT_CHARS)}
                                     for d in docs if d["role"] == "supporting" and d["readable"]],
            "prior_model": ({**{k: prior[k] for k in ("family_mentions", "likely_family")},
                             "earlier_cognos_run": ({k: prior_run.get(k) for k in (
                                 "run_id", "project", "champion_family", "champion_label",
                                 "features", "validation_verdict")} if prior_run else None),
                             "documents": [{"name": d["name"], "role": d["role"],
                                            "text": _clip(d["text"], EXCERPT_CHARS)}
                                           for d in prior_docs if d["readable"]]}
                            if prior is not None else None),
            "engine_families": _engine_families(),
        }, fresh={"intake": res})

        # --- engine disposes: the brief ---------------------------------------------------------
        entries = {e.field: e for e in out.brief}
        brief: list[dict[str, Any]] = []
        design_from_intent: dict[str, str] = {}
        compliance_from_intent: dict[str, str] = {}
        for f in fields:
            e = entries.get(f.id)
            stated = " ".join(e.value.split()) if e is not None and e.basis == "stated" else ""
            inferred = " ".join(e.value.split()) if e is not None and e.basis == "inferred" else ""
            if f.id in answered and answered[f.id] != stated:
                value, basis = answered[f.id], "answered"
            elif stated:
                value, basis = stated, "stated"
            elif profile.get(f.id):
                value, basis = profile[f.id], "profile"
            elif inferred:
                value, basis = inferred, "inferred"
            else:
                value, basis = "", "missing"
            brief.append({"field": f.id, "label": f.label, "value": value, "basis": basis,
                          "required": f.required, "quote": e.quote if e is not None else ""})
            if not stated:
                continue
            if f.id in eg.CORE and stated != profile.get(f.id):
                design_from_intent[f.id] = stated
                if profile.get(f.id):
                    res.add_finding(Finding(
                        id=f"intake-conflict-{f.id}", severity=Severity.MEDIUM, category="intent",
                        location=f.id,
                        message=f"The intent document and the profile disagree on "
                                f"{f.label.lower()}: “{stated[:120]}” against "
                                f"“{profile[f.id][:120]}”.",
                        suggestion="Accepting the brief takes the document; answer the field "
                                   "to set it yourself."))
            elif f.id == "interpretability":
                level = eg.interpretability_of(stated)
                if level and level != base.design.interpretability:
                    design_from_intent[f.id] = level
            elif f.id in eg.COMPLIANCE_FIELDS and stated != profile.get(f.id):
                compliance_from_intent[f.id] = stated

        open_fields = {b["field"] for b in brief if b["basis"] in ("missing", "inferred")}
        questions = []
        for q in out.interview:
            if q.field != "none" and q.field not in open_fields:
                continue  # stated or answered: nothing left to ask
            questions.append({
                "id": question_id(q.field, q.question), "question": q.question,
                "category": "intent", "source": "agent",
                "design_field": q.field if q.field in eg.CORE else None, "reentry": "intake",
                "field": q.field, "blocking": q.blocking, "why_it_matters": q.why_it_matters})
        n_blocking = sum(1 for q in questions if q["blocking"])
        update = None
        if kind == "update":
            update = {"scope": out.update_scope, "rationale": out.scope_rationale,
                      "change_items": [c.model_dump() for c in out.change_items],
                      "incumbent_family": out.incumbent_family or None}

        payload: dict[str, Any] = {
            "kind": kind, "kind_label": eg.KIND_LABEL[kind],
            "objective": out.restated_objective, "clarity": out.clarity,
            "brief": brief, "questions": questions, "n_blocking": n_blocking,
            "design_from_intent": design_from_intent,
            "compliance_from_intent": compliance_from_intent,
            "update": update, "prior": prior, "documents": inventory,
        }
        attach_recommendation(payload, ctx, "intake_analyst", out)
        res.add_artifact(ctx.save_json("stages/intake/brief.json", payload))
        res.add_artifact(ctx.save_text("stages/intake/brief.md", self._brief_md(cfg, payload),
                                       kind="markdown"))
        # The documents as read, by reference: later stages and the export work from this text.
        res.add_artifact(ctx.save_json("stages/intake/corpus.json", {"documents": [
            {k: d[k] for k in ("name", "role", "sha256", "text")} for d in docs if d["readable"]]}))

        if n_blocking:
            res.add_finding(Finding(
                id="intake-open-questions", severity=Severity.MEDIUM, category="intent",
                message=f"{n_blocking} blocking question(s) about the business intent are open.",
                suggestion="Answer them at the intent gate; intake re-reads the brief with each "
                           "answer."))
        res.payload = payload
        n_decided = sum(1 for b in brief if b["value"])
        res.metrics = {"n_documents": len(docs), "n_fields_decided": n_decided,
                       "n_open_questions": len(questions), "n_blocking_questions": n_blocking}
        noisy = n_blocking or any(f.severity != Severity.LOW for f in res.findings)
        res.verdict = Verdict.WARN if noisy else Verdict.PASS
        res.summary = (
            f"{eg.KIND_LABEL[kind]}: read {len(docs)} document(s); {n_decided}/{len(brief)} brief "
            f"field(s) decided; intent {out.clarity.replace('_', ' ')}; {len(questions)} interview "
            f"question(s), {n_blocking} blocking."
            + (f" Update scope: {out.update_scope.replace('_', ' ')}." if update else ""))
        return res

    @staticmethod
    def _brief_md(cfg, p: dict[str, Any]) -> str:
        basis = {"stated": "stated in the intent document", "answered": "answered by the sponsor",
                 "profile": "from the project profile", "inferred": "inferred, to be confirmed",
                 "missing": "open"}
        lines = [f"# Engagement brief — {cfg.name}", "",
                 f"- **Development mode:** {p['kind_label']}",
                 f"- **Intent:** {p['clarity'].replace('_', ' ')}",
                 f"- **Objective (restated):** {p['objective']}", "",
                 "## What the sponsor decided", ""]
        for b in p["brief"]:
            if not b["value"] and not b["required"]:
                lines.append(f"- **{b['label']}:** _not stated_")
                continue
            lines.append(f"- **{b['label']}** ({basis[b['basis']]}): "
                         + (b["value"] or "_open — see the interview_"))
        up = p.get("update")
        if up:
            lines += ["", "## Update request", "",
                      f"- **Scope:** {up['scope'].replace('_', ' ')} — {up['rationale']}",
                      f"- **Existing model family:** {up['incumbent_family'] or 'not identified'}",
                      ""]
            lines += [f"- {c['change']} _({c['type'].replace('_', ' ')}; from "
                      f"{c['affects']})_" for c in up["change_items"]] or ["_No change listed._"]
        lines += ["", "## Interview", ""]
        lines += [f"- {'**[blocking]** ' if q['blocking'] else ''}{q['question']} "
                  f"_({q['why_it_matters']})_" for q in p["questions"]] or ["_Nothing open._"]
        lines += ["", "## Documents received", ""]
        lines += [f"- `{d['name']}` — {d['role']}"
                  + ("" if d["readable"] else f" (not readable: {d['note']})")
                  for d in p["documents"]] or ["_None._"]
        return "\n".join(lines) + "\n"
