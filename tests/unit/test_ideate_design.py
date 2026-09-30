"""Ideate's design layer: data-structure assessment, framework selection, MD triangulation.

These are the commercial-risk "senior modeler" behaviors: pick the econometric framework from the
data structure, refuse to silently assume unanswered design points, and never seed the parsimonious
strategy with a leakage suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognos import synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator


def _cni_config(tmp_path, *, design: dict | None = None, **overrides) -> CognosConfig:
    df = synth.make_cni_portfolio_dataset(n=1200)
    csv = tmp_path / "cni.csv"
    df.to_csv(csv, index=False)
    raw = {
        "name": "test_cni", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome"]},
        "design": design or {},
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 6, "cv_folds": 3},
    }
    raw.update(overrides)
    return CognosConfig.from_dict(raw)


def _run_ideate(cfg, runs_dir):
    orch = Orchestrator(cfg, runs_root=runs_dir)
    orch.run_stage("explore")
    return orch, orch.run_stage("ideate")


ANSWERED = {
    "use_case": "origination underwriting", "horizon": "12-month PD",
    "default_definition": "90+ DPD or nonaccrual within 12 months",
    "segment": "C&I middle-market",
}


def test_framework_assessment_on_cni_panel(tmp_path, runs_dir):
    cfg = _cni_config(tmp_path, design=ANSWERED)
    _, res = _run_ideate(cfg, runs_dir)
    fw = {f["framework"]: f for f in res.payload["framework_assessment"]}

    assert fw["reduced_form_pd"]["applicable"] is True
    assert fw["reduced_form_pd"]["role"] == "primary"
    # vintage-indexed data => a discrete-time hazard reading is on the table
    assert fw["discrete_time_hazard"]["applicable"] == "partial"
    # private obligors, no market observables => structural Merton considered but rejected
    assert fw["structural_merton"]["applicable"] is False
    assert fw["structural_merton"]["role"] == "rejected"
    # regression guard: 'operating_margin' must NOT read as a rating column
    assert fw["transition_matrix"]["applicable"] is False
    # interpretability defaults to required => trees are challengers, not candidates
    assert fw["ml_challenger"]["role"] == "challenger"

    assert res.payload["data_structure"]["shape"] == "panel"
    assert res.payload["data_structure"]["n_events"] is not None
    assert any(a.path.endswith("design_brief.md") for a in res.artifacts)


def test_structural_framework_applicable_with_market_columns(tmp_path, runs_dir):
    rng = np.random.default_rng(0)
    n = 300
    df = synth.make_classification_dataset(n=n)
    df["equity_vol"] = rng.uniform(0.1, 0.9, n)
    df["market_cap"] = rng.gamma(2.0, 1e8, n)
    csv = tmp_path / "mkt.csv"
    df.to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "mkt", "task": "classification",
        "data": {"path": str(csv), "target": "target"},
        "metric": {"name": "roc_auc"}, "search": {"max_candidates": 4, "cv_folds": 3},
    })
    _, res = _run_ideate(cfg, runs_dir)
    fw = {f["framework"]: f for f in res.payload["framework_assessment"]}
    assert fw["structural_merton"]["applicable"] is True
    # no datetime column => hazard framework rejected with a pointer to datetime_col
    assert fw["discrete_time_hazard"]["applicable"] is False


def test_open_questions_from_unanswered_design(tmp_path, runs_dir):
    cfg = _cni_config(tmp_path)  # empty design brief
    _, res = _run_ideate(cfg, runs_dir)
    ids = {q["id"] for q in res.payload["open_questions"]}
    assert {"design-use_case", "design-horizon", "design-default_definition",
            "design-segment"} <= ids
    assert any(f.id == "design-open-questions" for f in res.findings)


def test_answered_design_removes_questions(tmp_path, runs_dir):
    cfg = _cni_config(tmp_path, design=ANSWERED)
    _, res = _run_ideate(cfg, runs_dir)
    ids = {q["id"] for q in res.payload["open_questions"]}
    assert not any(i.startswith("design-") for i in ids)
    # data-driven questions can remain (e.g. low events-per-variable on this sample)
    assert res.payload["design"]["use_case"] == "origination underwriting"


def test_leakage_suspect_never_seeds_top_features(tmp_path, runs_dir):
    df = synth.make_cni_portfolio_dataset(n=1200)  # keep dpd_at_outcome this time
    csv = tmp_path / "leaky.csv"
    df.to_csv(csv, index=False)
    cfg = CognosConfig.from_dict({
        "name": "leaky_cni", "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id"]},
        "metric": {"name": "roc_auc"}, "search": {"max_candidates": 4, "cv_folds": 3},
    })
    orch, res = _run_ideate(cfg, runs_dir)
    assert "dpd_at_outcome" in orch.ctx.require("explore").payload["leakage_suspects"]
    assert "dpd_at_outcome" not in res.payload["clean_top_features"]
    assert any(q["id"] == "data-leakage" for q in res.payload["open_questions"])


def test_epv_check_fires_on_thin_events(tmp_path, runs_dir):
    cfg = _cni_config(tmp_path, design=ANSWERED)  # ~4.5% default rate, 12 features
    _, res = _run_ideate(cfg, runs_dir)
    epv = res.payload["data_structure"]["events_per_variable"]
    assert epv is not None and epv < 10
    assert any(f.id == "design-epv" for f in res.findings)
    # parsimonious strategies must outrank the same family's all-features spec
    by_family = {}
    for h in res.payload["hypotheses"]:
        by_family.setdefault(h["family"], {})[h["feature_strategy"]] = h["priority"]
    assert all(p["top"] > p["all"] for p in by_family.values() if {"top", "all"} <= p.keys())


@pytest.mark.parametrize("interp,expected_role", [("required", "challenger"),
                                                  ("flexible", "candidate")])
def test_interpretability_policy_governs_tree_role(tmp_path, runs_dir, interp, expected_role):
    cfg = _cni_config(
        tmp_path, design={**ANSWERED, "interpretability": interp},
        search={"max_candidates": 6, "cv_folds": 3,
                "model_families": ["logit", "gradient_boosting"]},
    )
    _, res = _run_ideate(cfg, runs_dir)
    trees = [h for h in res.payload["hypotheses"] if h["family"] == "gradient_boosting"]
    assert trees and all(h["role"] == expected_role for h in trees)
    if interp == "required":
        # interpretable families are searched first when the budget binds
        assert res.payload["families"][0] == "logit"


def _design_lead_output(slate_extra: dict) -> dict:
    frameworks = [
        {"framework": f, "decision": d, "reason": "test"}
        for f, d in (("reduced_form_pd", "primary"), ("discrete_time_hazard", "candidate"),
                     ("structural_merton", "rejected"), ("transition_matrix", "rejected"),
                     ("ml_challenger", "challenger"))
    ]
    return {
        "summary": "Consider macro sensitivity for stress use.",
        "frameworks": frameworks,
        "slate": [{"family": "logit", "feature_strategy": "top", "role": "candidate",
                   "priority": 0.9, "rationale": "baseline"}, slate_extra],
        "sponsor_questions": [{"question": "Will the model feed stress-testing overlays?",
                               "category": "design", "design_field": "use_case",
                               "why_it_matters": "stress use needs macro sensitivity"}],
    }


def test_design_lead_widens_slate_and_questions(tmp_path, runs_dir, replay_dir):
    bogus = _design_lead_output({"family": "not_a_family", "feature_strategy": "top",
                                 "role": "candidate", "priority": 0.5, "rationale": "bogus"})
    good = _design_lead_output({"family": "gradient_boosting", "feature_strategy": "top",
                                "role": "challenger", "priority": 0.5,
                                "rationale": "ceiling benchmark"})
    cfg = _cni_config(tmp_path, design=ANSWERED,
                      agents={"replay_dir": replay_dir({"design_lead": [bogus, good]})})
    orch, res = _run_ideate(cfg, runs_dir)

    # the engine rejected the unfittable family and the agent's corrected answer was kept
    audit = [line for line in (orch.ctx.agents_dir / "audit.jsonl").read_text(
        encoding="utf-8").splitlines() if '"design_lead"' in line]
    assert '"invalid"' in audit[0] and '"ok"' in audit[1]
    assert any(q["source"] == "agent" for q in res.payload["open_questions"])
    assert [h["family"] for h in res.payload["hypotheses"]] == ["logit", "gradient_boosting"]
    # the widened family joins the search list so the ratchet can judge it on evidence
    assert "gradient_boosting" in res.payload["families"]
    assert "Design Lead" in res.payload["notes"]


def test_design_config_round_trips_yaml(tmp_path):
    cfg = _cni_config(tmp_path, design=ANSWERED)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    back = CognosConfig.from_yaml(path)
    assert back.design.use_case == "origination underwriting"
    assert back.design.unanswered() == []
    assert CognosConfig.from_dict({
        "name": "d", "task": "regression", "data": {"path": "x.csv", "target": "y"},
    }).design.unanswered() == ["use_case", "horizon", "default_definition", "segment"]
