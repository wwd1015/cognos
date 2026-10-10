"""Development modes end to end: a new model development from a business intent document, the
Intake Analyst's interview, and a model update from the existing model's artifacts."""

from __future__ import annotations

import json
import zipfile

import pytest

from cognos import engagement as eg
from cognos import service
from cognos.agents import slices
from cognos.cli import main
from cognos.config import CognosConfig
from cognos.engine import Engine, GateError

BRIEF = {
    "objective": "Rank middle-market borrowers by default risk at origination.",
    "use_case": "origination underwriting",
    "segment": "C&I middle-market",
    "default_definition": "90+ DPD or nonaccrual",
    "horizon": "12 months",
    "interpretability": "Preferred: credit officers must be able to explain a decline.",
    "intended_use": "Underwriters, as one input to the credit memo.",
}


def _intent(tmp_path, values, kind="new", name="intent.md") -> str:
    path = tmp_path / name
    path.write_text(eg.render_template(kind, values, "pd"), encoding="utf-8")
    return str(path)


def _with(cfg: CognosConfig, **engagement) -> CognosConfig:
    return service.with_engagement(cfg, engagement)


# --- new model development ------------------------------------------------------------------
def test_new_development_reads_the_intent_and_the_accepted_brief_shapes_the_config(
        make_config, runs_dir, tmp_path):
    cfg = _with(make_config("commercial"), intent=_intent(tmp_path, BRIEF))
    eng = Engine(cfg, runs_root=runs_dir, mode="interactive")
    state = eng.run_until_idle()
    assert state.status_of("gate_intent") == "awaiting" and state.status_of("explore") == "pending"
    res = eng.results()["intake"]
    p = res.payload
    assert p["kind"] == "new" and p["clarity"] == "clear" and p["n_blocking"] == 0
    assert {b["field"]: b["basis"] for b in p["brief"]}["horizon"] == "stated"
    assert p["recommendation"]["agent"] == "intake_analyst"
    # the documents live in the run, and the run's config points at its own copy
    assert eng.config.engagement.intent == "inputs/intent/intent.md"
    assert (eng.run_dir / "inputs" / "intent" / "intent.md").is_file()
    assert Engine.load(eng.run_dir).config.engagement.intent == "inputs/intent/intent.md"
    assert "Rank middle-market" in (eng.run_dir / "stages/intake/brief.md").read_text(encoding="utf-8")
    corpus = json.loads((eng.run_dir / "stages/intake/corpus.json").read_text(encoding="utf-8"))
    assert corpus["documents"][0]["role"] == "intent"

    eng.submit_gate("gate_intent", "accept")  # nothing blocking: no reason needed
    state = eng.run_until_idle()
    assert state.overrides.design == {
        "use_case": BRIEF["use_case"], "segment": BRIEF["segment"],
        "default_definition": BRIEF["default_definition"], "horizon": BRIEF["horizon"],
        "interpretability": "preferred"}
    assert state.overrides.compliance == {"intended_use": BRIEF["intended_use"]}
    assert state.status_of("gate_data") == "awaiting"
    ctx = eng.context()
    assert ctx.config.design.horizon == "12 months" and ctx.config.design.interpretability == "preferred"
    assert ctx.config.compliance.intended_use == BRIEF["intended_use"]
    assert ctx.base_config.design.horizon == ""  # the profile itself is never edited
    # every later agent works from the confirmed brief
    project = slices.build(ctx, "data_analyst", {})["project"]
    assert project["engagement"]["objective"] == BRIEF["objective"]
    assert project["engagement"]["brief"]["segment"] == BRIEF["segment"]
    assert project["development_mode"] == "new"


def test_interview_loop_answers_rerun_intake_until_nothing_blocks(make_config, runs_dir, tmp_path):
    partial = {k: BRIEF[k] for k in ("objective", "use_case", "segment")}
    eng = Engine(_with(make_config("commercial"), intent=_intent(tmp_path, partial)),
                 runs_root=runs_dir, mode="interactive")
    state = eng.run_until_idle()
    open_ids = {g.id for g in state.open_gaps()}
    assert {"design-horizon", "design-default_definition"} <= open_ids
    assert all(g.stage == "intake" and g.category == "intent" for g in state.open_gaps())
    assert eng.results()["intake"].payload["clarity"] == "needs_clarification"

    with pytest.raises(GateError, match="blocking interview question"):
        eng.submit_gate("gate_intent", "accept")
    with pytest.raises(GateError, match="nothing to apply"):
        eng.submit_gate("gate_intent", "edit")

    # one answer: the stage re-reads the brief and the gate re-opens with one question fewer
    eng.submit_gate("gate_intent", "edit", {"answers": {"design-horizon": "12-month"}})
    state = eng.run_until_idle()
    assert state.steps["intake"].runs == 2 and state.status_of("gate_intent") == "awaiting"
    assert "interview question(s) answered" in state.steps["intake"].rerun_reason
    p = eng.results()["intake"].payload
    assert {b["field"]: b["basis"] for b in p["brief"]}["horizon"] == "answered"
    assert [q["id"] for q in p["questions"] if q["blocking"]] == ["design-default_definition"]

    # going ahead with a blocking question open takes a reason, and the question stays open
    eng.submit_gate("gate_intent", "accept", reason="definition is with the credit policy team")
    state = eng.run_until_idle()
    assert state.status_of("gate_data") == "awaiting"
    assert state.gap("design-default_definition").status == "open"
    assert state.overrides.design["use_case"] == BRIEF["use_case"]

    # once the intent is confirmed, a design answer re-enters at design, not at the interview
    eng.submit_gate("gate_data", "accept")
    eng.run_until_idle()
    state = eng.answer_gap("design-default_definition", "90+ DPD")
    assert state.status_of("ideate") == "stale" and state.status_of("intake") == "done"
    assert state.status_of("explore") == "done"
    state = eng.run_until_idle()
    ideate = eng.results()["ideate"].payload
    assert not [q for q in ideate["open_questions"] if q["id"].startswith("design-")]


def test_send_back_at_the_intent_gate_challenges_the_intake_analyst(make_config, runs_dir, tmp_path):
    eng = Engine(_with(make_config("commercial"), intent=_intent(tmp_path, BRIEF)),
                 runs_root=runs_dir, mode="interactive")
    eng.run_until_idle()
    eng.submit_gate("gate_intent", "send_back", {"message": "The segment excludes agricultural loans."})
    state = eng.run_until_idle()
    challenge = state.challenges[0]
    assert challenge.target_stage == "intake" and challenge.status == "answered"
    assert state.status_of("gate_intent") == "awaiting"


def test_autonomous_run_without_an_intent_document_still_completes(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "completed"
    intake = eng.results()["intake"]
    assert [f.id for f in intake.findings][0] == "intake-no-intent"
    first = state.decisions[0]
    assert (first.gate, first.seat) == ("gate_intent", "express")
    # the four sponsor decisions are asked once, by intake; ideate does not ask them again
    core = [g for g in state.gaps if g.design_field]
    assert sorted(g.design_field for g in core) == ["default_definition", "horizon", "segment",
                                                     "use_case"]
    assert {g.stage for g in core} == {"intake"}
    assert "engagement" in eng.results()["document"].payload["concepts"]


def test_a_missing_document_refuses_the_run_before_it_exists(make_config, runs_dir, tmp_path):
    cfg = _with(make_config("classification"), intent=str(tmp_path / "nowhere.md"))
    with pytest.raises(FileNotFoundError, match="business intent document not found"):
        Engine(cfg, runs_root=runs_dir, run_id="r1")
    assert not (tmp_path / "runs" / "r1" / "state.json").exists()


def test_unreadable_and_free_form_documents_are_findings_not_crashes(make_config, runs_dir, tmp_path):
    blob = tmp_path / "deck.bin"
    blob.write_bytes(b"\x00\x01\x02\x03")
    free = tmp_path / "memo.txt"
    free.write_text("We would like a better PD model for the commercial book.", encoding="utf-8")
    eng = Engine(_with(make_config("classification"), intent=str(free), supporting=[str(blob)]),
                 runs_root=runs_dir, mode="interactive")
    eng.run_until_idle()
    res = eng.results()["intake"]
    assert res.verdict.value == "WARN"
    assert any(f.id.startswith("intake-unreadable") for f in res.findings)
    # a memo that does not follow the template states nothing the deterministic agent can quote
    # (the profile's description stands in for the objective; the four design decisions are open)
    assert res.payload["clarity"] == "unclear" and res.payload["n_blocking"] == 4


# --- model update ---------------------------------------------------------------------------
@pytest.fixture
def baseline(make_config, runs_dir, apply_brief):
    """An earlier COGNOS run of the model: the existing model an update starts from."""
    cfg = apply_brief(make_config("commercial"))
    eng = Engine(cfg, runs_root=runs_dir)
    assert eng.run_until_idle().status == "completed"
    return cfg, eng


def test_model_update_reads_the_prior_model_and_records_the_change(baseline, runs_dir, tmp_path):
    cfg, old = baseline
    prior_family = old.results()["model"].payload["champion"]["family"]
    request = _intent(tmp_path, {
        **BRIEF, "prior_model": "PD v1, in production since 2024",
        "update_reason": "The annual review found calibration drift.",
        "requested_changes": "- Refresh the development data through the latest vintage\n"
                             "- Re-estimate the coefficients",
        "must_not_change": "The segment and the default definition.",
    }, kind="update", name="request.md")
    code = tmp_path / "score.py"
    code.write_text("from sklearn.linear_model import LogisticRegression\n"
                    "FEATURES = ['leverage', 'interest_coverage']\n", encoding="utf-8")
    paper = tmp_path / "pd_v1_whitepaper.docx"
    with zipfile.ZipFile(paper, "w") as z:
        z.writestr("word/document.xml", "<w:p><w:r><w:t>A logistic regression scorecard."
                                        "</w:t></w:r></w:p>")
    upd = _with(cfg, kind="update", intent=request, prior_artifacts=[str(code), str(paper)],
                prior_run=old.run_id)
    eng = Engine(upd, runs_root=runs_dir)
    state = eng.run_until_idle()
    assert state.status == "completed"
    results = eng.results()

    p = results["intake"].payload
    assert p["kind"] == "update" and p["clarity"] == "clear"
    assert {d["name"]: d["role"] for d in p["documents"]} == {
        "request.md": "intent", "score.py": "code", "pd_v1_whitepaper.docx": "whitepaper"}
    assert p["update"]["scope"] == "re_estimate"
    assert [c["affects"] for c in p["update"]["change_items"]] == ["explore", "ideate"]
    # an earlier COGNOS run is read mechanically and outranks phrases counted in the artifacts
    assert p["prior"]["run"]["run_id"] == old.run_id
    assert p["update"]["incumbent_family"] == prior_family
    assert {m["family"] for m in p["prior"]["family_mentions"]} == {"logit"}
    assert json.loads((eng.run_dir / "inputs" / "prior_run.json").read_text(encoding="utf-8"))[
        "champion_family"] == prior_family

    # the design keeps the existing specification in front
    ideate = results["ideate"].payload
    assert ideate["incumbent"]["family"] == prior_family and ideate["incumbent"]["features_in_data"]
    assert ideate["hypotheses"][0]["family"] == prior_family
    assert "Existing model's family" in ideate["hypotheses"][0]["rationale"]

    # the prior model's scores reach the agents that read results, never the modeler
    ctx = eng.context()
    validator = slices.build(ctx, "validator", {})
    modeler = slices.build(ctx, "modeler", {})
    assert validator["facts"]["prior.holdout_metric"] == round(
        old.results()["model"].payload["holdout_metric"], 4)
    assert not [k for k in modeler["facts"] if k.startswith("prior.")]
    assert modeler["project"]["engagement"]["update"]["scope"] == "re_estimate"
    assert "holdout" not in json.dumps(modeler["project"]["engagement"])

    # the white paper carries the change record and the engine's side-by-side numbers
    doc = (eng.run_dir / "docs" / "engagement.md").read_text(encoding="utf-8")
    assert "## Model change record" in doc and "Refresh the development data" in doc
    assert "## Existing model against this update" in doc and old.run_id in doc
    assert "Model change record" in (eng.run_dir / "stages/document/whitepaper.md").read_text(
        encoding="utf-8")
    assert results["review"].verdict.value != "BLOCK"
    assert next(r for r in service.list_runs(runs_dir) if r["run_id"] == eng.run_id)["kind"] == "update"


def test_update_scoped_from_documents_alone_and_a_missing_prior_run_is_refused(
        make_config, runs_dir, tmp_path, apply_brief):
    cfg = apply_brief(make_config("commercial"))
    request = _intent(tmp_path, {**BRIEF, "prior_model": "PD v1",
                                 "update_reason": "New capital rule.",
                                 "requested_changes": "- Redevelop on a new framework"},
                      kind="update")
    paper = tmp_path / "methodology.md"
    paper.write_text("# PD v1\nThe model is a probit regression on leverage and liquidity.\n",
                     encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="prior run not found"):
        Engine(_with(cfg, kind="update", intent=request, prior_run="no-such-run"),
               runs_root=runs_dir)
    eng = Engine(_with(cfg, kind="update", intent=request, prior_artifacts=[str(paper)]),
                 runs_root=runs_dir, mode="interactive")
    eng.run_until_idle()
    p = eng.results()["intake"].payload
    assert p["update"]["scope"] == "redevelop" and p["update"]["incumbent_family"] == "probit"
    assert p["prior"]["run"] is None and p["prior"]["n_artifacts"] == 1


# --- the front ends -------------------------------------------------------------------------
def test_service_uploads_template_and_demo_intent(tmp_path):
    root = tmp_path / "runs"
    assert "## Business objective" in service.intent_template("new")
    assert "## Requested changes" in service.intent_template("update")
    data_url = "data:text/markdown;base64," + __import__("base64").b64encode(
        eg.render_template("new", BRIEF).encode()).decode()
    saved = service.save_upload("../my intent.md", data_url, root)
    assert saved.endswith("my_intent.md") and "_uploads" in saved  # no path escapes the uploads

    cfg = service.demo_config("cni", root, n=800, search_budget=6)
    assert eg.parse_intent(open(cfg.engagement.intent, encoding="utf-8").read())["horizon"] == "12-month PD"
    run_id = service.create_run(cfg, mode="interactive", root=root, engagement={"intent": saved})
    state = service.run_until_idle(run_id, root)
    assert state.status_of("gate_intent") == "awaiting"
    brief = service.results(run_id, root)["intake"].payload["brief"]
    assert next(b for b in brief if b["field"] == "objective")["value"] == BRIEF["objective"]
    assert not [r for r in service.list_runs(root) if r["run_id"] == "_uploads"]
    # a new development never carries a prior model, whatever the form sent along
    assert service.with_engagement(cfg, {"kind": "new", "prior_run": None}).engagement.prior_run is None


def test_export_carries_the_brief_and_the_documents_as_read(make_config, runs_dir, tmp_path):
    eng = Engine(_with(make_config("classification"), intent=_intent(tmp_path, BRIEF)),
                 runs_root=runs_dir)
    eng.run_until_idle()
    path = service.export_run(eng.run_id, tmp_path / "out", runs_dir)
    with zipfile.ZipFile(path) as z:
        names = {n.split("/", 1)[1] for n in z.namelist()}
    assert {"stages/intake/brief.json", "stages/intake/brief.md", "stages/intake/corpus.json",
            "docs/engagement.md"} <= names
    assert not [n for n in names if n.startswith(("data/", "inputs/"))]


def test_cli_intent_template_and_run_with_documents(tmp_path, capsys):
    from cognos import synth

    out = tmp_path / "intent.md"
    assert main(["intent-template", "--kind", "update", "--name", "pd", "-o", str(out)]) == 0
    assert "## Reason for the update" in out.read_text(encoding="utf-8")
    assert main(["intent-template", "-o", str(out)]) == 1  # does not overwrite
    assert main(["intent-template"]) == 0 and "## Business objective" in capsys.readouterr().out

    data = tmp_path / "data.csv"
    synth.GENERATORS["classification"](n=240).to_csv(data, index=False)
    cfg = CognosConfig.from_dict({
        "name": "cli", "task": "classification", "data": {"path": str(data), "target": "target"},
        "search": {"max_candidates": 6, "cv_folds": 3}})
    profile = tmp_path / "cognos.yaml"
    cfg.to_yaml(profile)
    intent = _intent(tmp_path, BRIEF, name="brief.md")
    runs = tmp_path / "runs"
    assert main(["run", "--config", str(profile), "--intent", intent, "--runs-dir", str(runs),
                 "--run-id", "r1"]) == 0
    assert service.results("r1", runs)["intake"].payload["clarity"] == "clear"
    assert main(["explain", "--config", str(profile)]) == 0
    assert "development mode: new" in capsys.readouterr().out
    # an update without the existing model, or with a missing document, is refused up front
    assert main(["run", "--config", str(profile), "--kind", "update", "--intent", intent,
                 "--runs-dir", str(runs)]) == 1
    assert main(["run", "--config", str(profile), "--intent", str(tmp_path / "gone.md"),
                 "--runs-dir", str(runs)]) == 1
    assert "refused" in capsys.readouterr().out
