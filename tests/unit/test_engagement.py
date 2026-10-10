"""Engagements: the intent template, reading documents, the config rules, and the Intake
Analyst's contract (the deterministic agent must pass the engine's checks)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from cognos import engagement as eg
from cognos.agents import checks, heuristic
from cognos.agents.contracts import CONTRACTS
from cognos.config import CognosConfig
from cognos.engine.state import RunState

FILLED = {
    "objective": "Rank middle-market borrowers by default risk at origination.",
    "use_case": "origination underwriting",
    "segment": "C&I middle-market",
    "default_definition": "90+ DPD or nonaccrual",
    "horizon": "12 months",
}
UPDATE = {
    **FILLED,
    "prior_model": "PD v1, in production since 2024",
    "update_reason": "The annual review found calibration drift.",
    "requested_changes": "- Refresh the development data through the latest vintage\n"
                         "- Recalibrate to the long-run default rate",
}


# --- the template ---------------------------------------------------------------------------
def test_template_round_trips_and_an_untouched_template_is_empty():
    assert eg.parse_intent(eg.render_template("new")) == {}  # hints are not answers
    assert eg.parse_intent(eg.render_template("new", FILLED, "pd")) == FILLED
    assert eg.parse_intent(eg.render_template("update", UPDATE)) == UPDATE


def test_update_template_is_the_same_document_plus_the_change_request():
    new, update = eg.render_template("new"), eg.render_template("update")
    assert "## Requested changes" not in new and "## Requested changes" in update
    assert all(f"## {f.label}" in update for f in eg.fields_for("new"))
    with pytest.raises(ValueError):
        eg.render_template("rebuild")


def test_shipped_templates_match_the_code():
    root = Path(__file__).resolve().parents[2] / "docs" / "templates"
    assert (root / "business_intent.md").read_text(encoding="utf-8") == eg.render_template("new")
    assert (root / "model_update_request.md").read_text(encoding="utf-8") == eg.render_template("update")


def test_parser_reads_aliases_numbering_and_bold_headings_and_skips_placeholders():
    text = ("# Intent\n\n## 1. Use case\nsurveillance\n\n**Segment**\nCRE income-producing\n\n"
            "### Horizon\nTBD\n\n## Something else\nignored\n\n## Purpose\nWatch the book.\n")
    assert eg.parse_intent(text) == {"use_case": "surveillance", "segment": "CRE income-producing",
                                     "objective": "Watch the book."}


def test_small_readers():
    assert eg.bullets("- a\n- b\n  more\n1. c") == ["a", "b more", "c"]
    assert eg.interpretability_of("Flexible, but required for adverse action") == "flexible"
    assert eg.interpretability_of("whatever works") is None
    assert eg.infer_role("score.py") == "code" and eg.infer_role("Validation report.pdf") == "validation"
    assert eg.infer_role("model_whitepaper.docx") == "whitepaper" and eg.infer_role("x.bin") == "other"
    mentions = eg.family_mentions(["A logistic regression scorecard.", "LogisticRegression()",
                                   "benchmarked against XGBoost"])
    assert mentions[0] == {"family": "logit", "mentions": 3}
    assert eg.columns_mentioned(["leverage", "dscr", "size"], ["X = df[['leverage', 'DSCR']]"]) == [
        "leverage", "dscr"]


# --- reading documents ----------------------------------------------------------------------
def test_read_text_handles_word_notebooks_and_binaries(tmp_path):
    docx = tmp_path / "intent.docx"
    xml = ('<w:document><w:body>'
           '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Use case</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>origination </w:t></w:r><w:r><w:t>&amp; renewal</w:t></w:r></w:p>'
           '</w:body></w:document>')
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", xml)
    text, note = eg.read_text(docx)
    assert note == "" and eg.parse_intent(text) == {"use_case": "origination & renewal"}

    nb = tmp_path / "dev.ipynb"
    nb.write_text(json.dumps({"cells": [{"source": ["import x\n", "fit()"]}]}), encoding="utf-8")
    assert eg.read_text(nb)[0] == "import x\nfit()"

    blob = tmp_path / "model.pkl"
    blob.write_bytes(b"\x00\x01\x02binary")
    text, note = eg.read_text(blob)
    assert text is None and "binary" in note
    assert eg.read_text(tmp_path / "nope.md")[0] is None  # never raises


# --- config ---------------------------------------------------------------------------------
def _raw(**engagement):
    return {"name": "t", "task": "classification", "data": {"target": "y"},
            "engagement": engagement}


def test_an_update_needs_the_existing_model_and_a_request():
    with pytest.raises(ValueError, match="existing model"):
        CognosConfig.from_dict(_raw(kind="update", intent="request.md"))
    with pytest.raises(ValueError, match="update request"):
        CognosConfig.from_dict(_raw(kind="update", prior_artifacts=["wp.md"]))
    with pytest.raises(ValueError, match="only in a model update"):
        CognosConfig.from_dict(_raw(kind="new", prior_run="r1"))
    cfg = CognosConfig.from_dict(_raw(kind="update", intent="request.md",
                                      prior_artifacts=["wp.md", {"path": "x.txt", "role": "code"}]))
    assert [(a.path, a.role) for a in cfg.engagement.prior_artifacts] == [("wp.md", "auto"),
                                                                           ("x.txt", "code")]
    assert CognosConfig.from_dict({k: v for k, v in _raw().items() if k != "engagement"}
                                  ).engagement.kind == "new"


def test_a_run_recorded_before_intake_reads_it_as_skipped():
    old = {"run_id": "r", "steps": {"explore": {"status": "done"}, "gate_data": {"status": "done"}}}
    state = RunState.model_validate(old)
    assert state.status_of("intake") == "skipped" and state.status_of("gate_intent") == "skipped"
    assert state.status_of("explore") == "done"


# --- the Intake Analyst's contract ------------------------------------------------------------
def _slice(kind: str, values: dict, decided: dict | None = None, **extra) -> dict:
    text = eg.render_template(kind, values)
    parsed = eg.parse_intent(text)
    return {
        "kind": kind,
        "fields": [{"field": f.id, "label": f.label, "required": f.required,
                    "decided": (decided or {}).get(f.id), "template_value": parsed.get(f.id)}
                   for f in eg.fields_for(kind)],
        "intent_document": {"name": "intent.md", "text": text},
        "supporting_documents": [], "prior_model": None,
        "engine_families": ["logit", "probit", "gradient_boosting"],
        "facts": {}, "sponsor_answers": [], "challenges": [], **extra,
    }


def _checked(sl: dict, **changes) -> list[str]:
    raw = {**heuristic.intake_analyst(sl), **changes}
    return checks.run_checks("intake_analyst", CONTRACTS["intake_analyst"].model_validate(raw), sl)


def test_heuristic_intake_passes_the_checks_and_asks_only_what_is_open():
    sl = _slice("new", {"objective": FILLED["objective"], "use_case": FILLED["use_case"]})
    out = heuristic.intake_analyst(sl)
    assert _checked(sl) == []
    assert {q["field"] for q in out["interview"] if q["blocking"]} == {
        "segment", "default_definition", "horizon"}
    assert out["clarity"] != "clear" and out["update_scope"] == "not_applicable"
    # the profile (or an earlier answer) already decides a field: it is not asked again
    sl = _slice("new", {"objective": FILLED["objective"]},
                decided={"use_case": "x", "segment": "x", "default_definition": "x", "horizon": "x"})
    out = heuristic.intake_analyst(sl)
    assert _checked(sl) == [] and out["clarity"] == "clear"
    assert [q["field"] for q in out["interview"]] == ["success_criteria"]


def test_heuristic_update_scopes_the_request_and_names_the_incumbent():
    sl = _slice("update", UPDATE, prior_model={"likely_family": "logit", "family_mentions": [],
                                               "documents": []})
    out = heuristic.intake_analyst(sl)
    assert _checked(sl) == []
    assert [c["type"] for c in out["change_items"]] == ["data_refresh", "recalibration"]
    assert out["update_scope"] == "re_estimate" and out["incumbent_family"] == "logit"
    deep = _slice("update", {**UPDATE, "requested_changes": "- Redevelop on a hazard framework"},
                  prior_model={"likely_family": "tobit", "documents": []})
    out = heuristic.intake_analyst(deep)
    assert out["update_scope"] == "redevelop" and out["incumbent_family"] == ""


def test_engine_rejects_an_ungrounded_or_incomplete_brief():
    sl = _slice("new", FILLED)
    good = heuristic.intake_analyst(sl)

    def with_entry(field, **kw):
        return [{**e, **kw} if e["field"] == field else e for e in good["brief"]]

    # a "stated" position must quote the documents
    errs = _checked(sl, brief=with_entry("horizon", value="36 months", quote="thirty-six months"))
    assert any("quote is not in the documents" in e for e in errs)
    # a required field that is not stated needs a blocking question
    errs = _checked(sl, brief=with_entry("horizon", value="", basis="missing", quote=""))
    assert any("'horizon'" in e and "blocking" in e for e in errs)
    # ... and an inferred one is still to be confirmed
    errs = _checked(sl, brief=with_entry("horizon", basis="inferred", quote=""))
    assert any("'horizon'" in e for e in errs)
    # every field appears once
    assert any("exactly once" in e for e in _checked(sl, brief=good["brief"][1:]))
    # "clear" and a blocking question cannot both be true
    q = {"question": "Which book?", "field": "none", "why_it_matters": "scope", "blocking": True}
    assert any("clear" in e for e in _checked(sl, interview=[q], clarity="clear"))
    # a new development carries no change request
    assert any("new model development" in e for e in _checked(sl, update_scope="redevelop"))
    # nothing decided is asked again
    decided = _slice("new", {}, decided={f.id: "x" for f in eg.fields_for("new")})
    ask = {"question": "What is the use case?", "field": "use_case", "why_it_matters": "x",
           "blocking": False}
    assert any("already decided" in e for e in _checked(decided, interview=[ask]))


def test_engine_rejects_an_update_without_a_scope():
    sl = _slice("update", UPDATE, prior_model={"likely_family": None, "documents": []})
    errs = _checked(sl, update_scope="not_applicable", scope_rationale="", change_items=[],
                    incumbent_family="tobit")
    assert any("update_scope" in e for e in errs) and any("scope_rationale" in e for e in errs)
    assert any("change_items" in e for e in errs) and any("incumbent_family" in e for e in errs)
