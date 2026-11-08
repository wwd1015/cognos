"""The design stage in seven steps (ADR-0025): the single-factor analysis is shown at the design
gate, on the development sample, once the variable rules are set. Every look is kept, rules
changed after a look are counted, exceptions carry a reason, and the model stage runs the same
screen again as the record. Models go through the plan's rules as a funnel and may be ranked by
a blend of metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cognos import service
from cognos.engine import GateError
from cognos.modeling import plan as plan_mod
from cognos.modeling import selection
from cognos.modeling.fit import Candidate
from cognos.stages.validate import ValidateStage


def _at_design_gate(runs_dir) -> str:
    cfg = service.template_config("built-in:pd_logistic_scorecard", "pd", root=runs_dir)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=runs_dir)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_intent", "accept", reason="t", root=runs_dir,
                        background=False)
    service.run_until_idle(run_id, runs_dir)
    info = service.data_info(run_id, runs_dir, with_catalog=True)
    service.submit_gate(run_id, "gate_source", "accept",
                        service.data_payload(info["preselected"], root=runs_dir),
                        root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    service.submit_gate(run_id, "gate_data", "accept", root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    return run_id


def test_the_design_gate_shows_the_screen_the_model_stage_then_runs(runs_dir):
    run_id = _at_design_gate(runs_dir)
    screen = service.results(run_id, runs_dir)["ideate"].payload["screen"]
    assert not screen.get("error") and screen["shortlist"]
    assert [line["rule"] for line in screen["screen"]][:2] == ["not_fitted", "max_missing"]
    dropped = [r for r in screen["rows"] if r["status"] == "dropped"]
    assert dropped and all(r["dropped_by"] and r["reason"] for r in dropped)

    # a look recomputes in place; changing a rule after a look is a revision, with who and what
    first = service.preview_screen(run_id, root=runs_dir, by="dana")
    assert first["shortlist"] == screen["shortlist"]  # the plan in force: the stored screen
    strict = service.preview_screen(run_id, {"factors": {"max_p_value": 0.01}}, root=runs_dir,
                                    by="dana")
    assert len(strict["shortlist"]) <= len(first["shortlist"])
    revised = service.screen_revisions(run_id, runs_dir)
    assert len(revised) == 1 and revised[0]["by"] == "dana"
    assert revised[0]["changed"] == {"max_p_value": [0.1, 0.01]}
    with pytest.raises(ValueError, match="cannot be run"):
        service.preview_screen(run_id, {"evaluation": {"test_fraction": 0.9}}, root=runs_dir)

    # an exception needs a factor the screen saw and a reason, and records who set it
    gone = next(r["name"] for r in screen["rows"] if r["status"] == "dropped"
                and r["dropped_by"] != "not_fitted")
    kept = screen["shortlist"][0]
    with pytest.raises(GateError, match="needs a reason"):
        service.submit_gate(run_id, "gate_design", "edit",
                            {"screen_exceptions": {gone: {"keep": True, "reason": " "}}},
                            root=runs_dir, background=False)
    with pytest.raises(GateError, match="not a factor"):
        service.submit_gate(run_id, "gate_design", "edit",
                            {"screen_exceptions": {"nope": {"keep": True, "reason": "x"}}},
                            root=runs_dir, background=False)
    service.submit_gate(run_id, "gate_design", "edit", {"screen_exceptions": {
        gone: {"keep": True, "reason": "underwriting policy requires it"},
        kept: {"keep": False, "reason": "not available at origination"}}},
        root=runs_dir, by="dana", background=False)
    st = service.state(run_id, runs_dir)
    assert st.overrides.screen_exceptions[gone] == {
        "keep": True, "reason": "underwriting policy requires it", "by": "dana"}
    service.run_until_idle(run_id, runs_dir)

    model = service.results(run_id, runs_dir)["model"].payload
    assert gone in model["shortlist"] and kept not in model["shortlist"]
    by = {e["factor"]: e for e in model["exceptions"]}
    assert by[gone]["exception"] == "keep" and by[gone]["overruled"]  # the rule it overruled
    assert "underwriting policy" in by[gone]["reason"]
    assert by[kept]["exception"] == "drop"

    # the model funnel: every rule in order, each model leaving at the first one it breaks
    funnel = model["model_funnel"]
    assert funnel["lines"][0]["rule"] == "all"
    assert "max_vif" in [line["rule"] for line in funnel["lines"]]
    for a, b in zip(funnel["lines"], funnel["lines"][1:], strict=False):
        assert b["before"] == a["after"]
    assert set(funnel["kept"]) <= set(model["evaluated_ids"])

    # validation reads the revisions and the exceptions as findings a reviewer sees
    ctx = service.engine(run_id, runs_dir).context()
    record = ValidateStage._screen_record(ctx, model)
    assert len(record["revisions"]) == 1 and len(record["exceptions"]) == 2
    assert {f.id for f in record["findings"]} == {"screen-revised", "screen-exceptions"}


def _evaluation(i: str, auc: float, n: int, spread: float) -> selection.Evaluation:
    from cognos.modeling.metrics import CVResult

    cv = CVResult(mean=auc, std=spread, oof_pred=np.zeros(1), fold_scores=[auc], n_folds=1)
    return selection.Evaluation(id=i, candidate=Candidate(family="logit",
                                                          features=[f"f{k}" for k in range(n)]),
                                cv=cv, scores={"validation": {"roc_auc": auc}})


def test_a_blend_ranks_by_the_weighted_share_of_models_each_one_beats():
    evs = [_evaluation("a", 0.80, 6, 0.04), _evaluation("b", 0.79, 2, 0.01),
           _evaluation("c", 0.70, 3, 0.02)]
    rules = [{"metric": "roc_auc", "sample": "validation"}]
    assert [e.id for e in selection.rank(evs, rules)] == ["a", "b", "c"]
    blend = [{"metric": "roc_auc", "sample": "validation", "weight": 0.5},
             {"metric": "n_factors", "sample": "validation", "weight": 0.3},
             {"metric": "fold_spread", "sample": "validation", "weight": 0.2}]
    ranked = selection.rank(evs, blend, mode="weights")
    assert [e.id for e in ranked] == ["b", "a", "c"]  # nearly as good, a third of the factors
    assert ranked[0].info["blend"] == {"score": 0.75, "parts": [0.5, 1.0, 1.0]}


def test_a_plan_says_what_a_blend_and_an_inflation_limit_need():
    base = {"evaluation": {"test": "random", "validation": "kfold", "test_fraction": 0.2,
                           "holdout_fraction": 0.0, "pool_size": 3, "rank_mode": "weights",
                           "rank_by": [{"metric": "roc_auc", "sample": "validation"}]},
            "selection": {"methods": ["stepwise"], "criterion": "validation", "max_factors": 4,
                          "max_vif": 0.5},
            "factors": {"criterion": "validation"}, "options": {}, "shape": {}}
    bad = plan_mod.problems(base)
    assert any("weight" in b for b in bad) and any("inflation" in b for b in bad)
    base["evaluation"]["rank_by"] = [{"metric": "n_factors", "sample": "validation",
                                      "weight": 1}]
    assert any("must be a score" in b for b in plan_mod.problems(base))
    assert plan_mod.rank_words({"rank_mode": "weights", "rank_by": [
        {"metric": "roc_auc", "sample": "validation", "weight": 3},
        {"metric": "n_factors", "sample": "validation", "weight": 1}]}) == \
        "a blend of roc_auc (validation) 75%, fewer factors 25%"


def test_variance_inflation_sets_a_model_aside_and_the_funnel_says_where():
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    dev = pd.DataFrame({"a": a, "b": a + 0.1 * rng.normal(size=500), "c": rng.normal(size=500)})
    inflation = selection.vif(dev, ["a", "b", "c"])
    assert inflation["a"] > 50 and inflation["c"] < 1.1
    models = [{"id": "m1", "name": "Atlas", "violations": ["a has variance inflation 98.0"]},
              {"id": "m2", "name": "Birch", "violations": []},
              {"id": "m3", "name": "Cedar", "violations": ["c has p-value 0.400"]}]
    funnel = selection.model_funnel(models, {"max_vif": 5.0, "max_p_value": 0.1})
    lines = {line["rule"]: line for line in funnel["lines"]}
    assert lines["max_vif"]["dropped"] == [{"id": "m1", "name": "Atlas",
                                            "reason": "a has variance inflation 98.0"}]
    assert lines["max_p_value"]["after"] == 1 and funnel["kept"] == ["m2"]


def test_the_developer_sets_the_model_form_and_proposes_one_the_engine_cannot_fit(runs_dir):
    run_id = _at_design_gate(runs_dir)
    ideate = service.results(run_id, runs_dir)["ideate"].payload
    fws = {f["framework"]: f for f in ideate["framework_assessment"]}
    assert fws["reduced_form_pd"]["editable"] and "logit" in fws["reduced_form_pd"]["families"]
    assert not fws["structural_merton"]["editable"]

    def edit(payload, by="dana"):
        service.submit_gate(run_id, "gate_design", "edit", payload, root=runs_dir, by=by,
                            background=False)

    with pytest.raises(GateError, match="needs a reason"):
        edit({"framework_roles": {"ml_challenger": "rejected"}})
    with pytest.raises(GateError, match="exactly one"):
        edit({"framework_roles": {"reduced_form_pd": "candidate"}, "model_form_reason": "x"})
    with pytest.raises(GateError, match="cannot be set here"):
        edit({"framework_roles": {"structural_merton": "candidate"}, "model_form_reason": "x"})
    with pytest.raises(GateError, match="only be a benchmark"):
        edit({"framework_roles": {"ml_challenger": "candidate"}, "model_form_reason": "x"})
    with pytest.raises(GateError, match="needs a reason"):
        edit({"proposed_forms": [{"name": "Bayesian hierarchical logit", "reason": " "}]})

    # machine learning is a benchmark here: ticking its families fits them, and none can win
    slate = [*({"family": f, "feature_strategy": "all", "role": "candidate", "priority": 0.5,
                "rationale": "t"} for f in ideate["families"]),
             *({"family": f, "feature_strategy": "all", "role": "challenger", "priority": 0.5,
                "rationale": "t"} for f in fws["ml_challenger"]["families"])]
    edit({"slate": slate, "proposed_forms": [{"name": "Bayesian hierarchical logit",
                                              "reason": "pools thin segments"}]})
    service.run_until_idle(run_id, runs_dir)
    ideate = service.results(run_id, runs_dir)["ideate"].payload
    assert set(ideate["benchmark_families"]) == {"random_forest", "gradient_boosting"}
    assert ideate["proposed_forms"][0]["by"] == "dana"
    brief = service.read_artifact(run_id, "stages/ideate/design_brief.md", runs_dir)
    assert "Bayesian hierarchical logit" in brief and "not fitted" in brief

    service.submit_gate(run_id, "gate_design", "accept", root=runs_dir, background=False)
    service.run_until_idle(run_id, runs_dir)
    model = service.results(run_id, runs_dir)["model"].payload
    ranking = json.loads(service.read_artifact(run_id, "stages/model/ranking.json", runs_dir))
    assert {"random_forest", "gradient_boosting"} <= {r["family"] for r in ranking}
    assert all(a["role"] == "challenger" for a in model["leaders"]
               if a["family"] in ("random_forest", "gradient_boosting"))
    assert not any(a["family"] in ("random_forest", "gradient_boosting")
                   for a in model["pool"])

    # set to not used, with a reason: its families leave the slate, and validation reads it
    service.reopen(run_id, "gate_design", runs_dir)
    edit({"framework_roles": {"ml_challenger": "rejected"},
          "model_form_reason": "no benchmark wanted for a recalibration"}, by="lee")
    st = service.state(run_id, runs_dir)
    assert st.overrides.framework_roles["ml_challenger"] == {
        "decision": "rejected", "reason": "no benchmark wanted for a recalibration", "by": "lee"}
    assert not {"random_forest", "gradient_boosting"} & {h["family"] for h in st.overrides.slate}
    service.run_until_idle(run_id, runs_dir)
    ml = next(c for c in service.results(run_id, runs_dir)["ideate"].payload["framework_choices"]
              if c["framework"] == "ml_challenger")
    assert ml["decision"] == "rejected" and ml["source"] == "developer"
    ctx = service.engine(run_id, runs_dir).context()
    assert {f.id for f in ValidateStage._model_form_findings(ctx)} == {
        "model-form-set", "model-form-proposed"}


def test_the_page_says_why_a_model_form_cannot_be_applied():
    from cognos.ui import design_steps as dz

    rec = {"reduced_form_pd": "primary", "ml_challenger": "challenger"}
    assert dz.model_form_problem(rec, rec, "") is None  # the recommendation needs no reason
    assert "No framework" in dz.model_form_problem(
        {**rec, "reduced_form_pd": "candidate"}, rec, "why")
    assert "2 frameworks" in dz.model_form_problem({**rec, "ml_challenger": "primary"}, rec, "x")
    assert "write your reason" in dz.model_form_problem(
        {**rec, "ml_challenger": "rejected"}, rec, " ")


def test_the_table_s_must_carry_cells_become_the_plan_s_must_include():
    from cognos.ui import app as ui

    plan = {"selection": {"must_include": []}, "factors": {}, "evaluation": {}}
    assert ui.plan_patch(plan, {"must::drawn": "drawn__log", "must::limit": None}) == {
        "selection": {"must_include": ["drawn__log"]}}
    assert ui.plan_patch(plan, {"must::drawn": None}) == {}  # nothing flagged: no change


def test_a_new_strength_metric_after_a_look_is_a_revision():
    plan = {"factors": {"max_p_value": 0.1}, "evaluation": {"rank_by": [{"metric": "roc_auc"}]}}
    rules = plan_mod.screen_rules(plan)
    assert rules["metric"] == "roc_auc"
    looks = [{"at": "1", "by": "a", "rules": {"max_p_value": 0.1}},  # a look from before
             {"at": "2", "by": "a", "rules": rules},
             {"at": "3", "by": "b", "rules": {**rules, "metric": "brier"}}]
    revised = plan_mod.screen_revisions(looks)
    assert [r["changed"] for r in revised] == [{"metric": ["roc_auc", "brier"]}]
