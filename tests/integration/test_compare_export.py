"""Why a step went stale, what its re-run changed, comparing two runs, and exporting one.

All three are read from what the engine already records (state, stage results, config): nothing
is judged and nothing new is decided."""

from __future__ import annotations

import json
import zipfile

import pytest

from cognos import compare as cmp
from cognos import service
from cognos.artifacts import Finding, Severity, StageResult, Verdict
from cognos.cli import main
from cognos.engine import Engine, gates
from cognos.engine.graph import GATES


def _accept_all(eng: Engine, until: str | None = None):
    for _ in range(20):
        state = eng.run_until_idle()
        waiting = [g for g in GATES if state.status_of(g) == "awaiting"]
        if not waiting or waiting[0] == until:
            return state
        eng.submit_gate(waiting[0], "approve" if waiting[0] == "gate_signoff" else "accept", reason="ok")
    return eng.state


# --- why a step is stale, and what the re-run changed -------------------------------------------
def test_a_revised_decision_is_named_on_every_step_it_makes_stale(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    assert all(s.rerun_reason is None for s in eng.state.steps.values())  # nothing has re-run yet
    eng.reopen("gate_data")
    feature = eng.results()["explore"].payload["features"][0]
    eng.submit_gate("gate_data", "edit", {"exclude_columns": [feature]}, reason="not at origination")
    state = eng.state
    for step in ("ideate", "gate_design", "model", "gate_champion"):
        why = state.steps[step].rerun_reason
        assert state.status_of(step) == "stale" and why == state.steps[step].message
        assert "Review data decisions" in why and "1 column(s)" in why and feature in why, why
    assert state.steps["explore"].rerun_reason is None  # upstream of the decision: untouched

    # the reason survives the re-run, and the replaced result is kept beside the new one
    eng.run_until_idle()
    eng.submit_gate("gate_design", "accept")
    eng.run_until_idle()
    state = eng.state
    assert state.status_of("model") == "done" and feature in state.steps["model"].rerun_reason
    stage_dir = eng.run_dir / "stages" / "model"
    prev = json.loads((stage_dir / "result.prev.json").read_text(encoding="utf-8"))
    assert feature in prev["payload"]["raw_features"]
    assert feature not in eng.results()["model"].payload["raw_features"]

    changes = service.step_changes(eng.run_id, "model", runs_dir)
    assert changes is not None and changes["stage"] == "model"
    assert {r["key"] for r in changes["metrics"]} >= {"cv_mean", "holdout_metric"}
    assert all(r["direction"] == "higher" for r in changes["metrics"] if r["key"] == "cv_mean")  # roc_auc
    # explore never re-ran, so it has nothing to compare
    assert service.step_changes(eng.run_id, "explore", runs_dir) is None


def test_every_way_of_going_stale_says_why(make_config, runs_dir):
    eng = Engine(make_config("classification"), runs_root=runs_dir, mode="interactive")
    _accept_all(eng, until="gate_champion")
    eng.submit_gate("gate_champion", "send_back", {"message": "Try a simpler model first."})
    assert "sent back" in eng.state.steps["model"].rerun_reason
    assert "Try a simpler model first." in eng.state.steps["model"].rerun_reason


@pytest.mark.parametrize(("gate", "action", "payload", "reason", "expect"), [
    ("gate_data", "accept", {}, "", "recommended exclusions accepted"),
    ("gate_data", "edit", {"exclude_columns": []}, "", "exclusions changed to 0 column(s)"),
    ("gate_design", "edit", {"slate": [{"family": "logit"}], "answers": {"q1": "x"}}, "",
     "slate edited to 1 hypothesis(es) and 1 design question(s) answered"),
    ("gate_champion", "override", {"champion": "c3"}, "simpler", "champion overridden to c3 — simpler"),
    ("gate_validation", "send_back", {"message": "calibration drifts"}, "", "sent back to “Search & select champion” — calibration"),
])
def test_why_restates_the_decision(gate, action, payload, reason, expect):
    assert expect in gates.why(gate, action, payload, reason)


def test_a_long_reason_is_clipped_not_dropped():
    out = gates.why("gate_champion", "override", {"champion": "c1"}, "word " * 100)
    assert len(out) < 220 and out.endswith("…")


# --- comparing ------------------------------------------------------------------------------------
def _res(stage, verdict=Verdict.PASS, metrics=None, payload=None, findings=()):
    return StageResult(stage=stage, verdict=verdict, metrics=metrics or {}, payload=payload or {},
                       findings=list(findings))


def test_better_is_claimed_only_where_the_metric_has_a_direction():
    a = {"model": _res("model", metrics={"cv_mean": 0.80, "cv_std": 0.05, "n_candidates_tried": 8, "champion": "x"},
                       payload={"direction": "maximize"})}
    b = {"model": _res("model", metrics={"cv_mean": 0.84, "cv_std": 0.07, "n_candidates_tried": 12, "champion": "y"},
                       payload={"direction": "maximize"})}
    rows = {r["key"]: r for r in cmp.metric_rows(a, b)}
    assert rows["cv_mean"]["better"] == "b" and rows["cv_mean"]["delta"] == pytest.approx(0.04)
    assert rows["cv_std"]["better"] == "a"                     # lower is better by definition
    assert rows["n_candidates_tried"]["better"] is None        # a count: a difference, not a verdict
    assert rows["champion"]["delta"] is None and rows["champion"]["better"] is None

    # an error metric: the same numbers read the other way
    for side in (a, b):
        side["model"].payload["direction"] = "minimize"
    assert {r["key"]: r for r in cmp.metric_rows(a, b)}["cv_mean"]["better"] == "a"

    # runs on different metrics are not like for like: no verdict on the headline number
    a["model"].payload["direction"] = "maximize"
    assert {r["key"]: r for r in cmp.metric_rows(a, b)}["cv_mean"]["better"] is None


def test_missing_and_non_finite_numbers_are_never_subtracted():
    a = {"backtest": _res("backtest", metrics={"psi": float("nan"), "gini": 0.5})}
    b = {"backtest": _res("backtest", metrics={"psi": 0.1})}
    rows = {r["key"]: r for r in cmp.metric_rows(a, b)}
    assert rows["psi"]["delta"] is None and rows["psi"]["better"] is None
    assert rows["gini"]["b"] is None and rows["gini"]["delta"] is None


def test_findings_are_matched_by_stage_and_id():
    f = lambda i, sev=Severity.HIGH: Finding(id=i, severity=sev, message=f"finding {i}")  # noqa: E731
    a = {"validate": _res("validate", findings=[f("v1"), f("v2")])}
    b = {"validate": _res("validate", findings=[f("v2"), f("v3", Severity.CRITICAL)])}
    out = cmp.finding_changes(a, b)
    assert [r["id"] for r in out["new"]] == ["v3"] and [r["id"] for r in out["gone"]] == ["v1"]


def test_config_changes_ignore_where_the_run_lives():
    a = {"name": "p", "runs_dir": "/a", "data": {"path": "/a/x.csv", "target": "y"}, "search": {"max_candidates": 8}}
    b = {"name": "p", "runs_dir": "/b", "data": {"path": "/b/x.csv", "target": "y"}, "search": {"max_candidates": 20}}
    assert cmp.config_changes(a, b) == [{"key": "search.max_candidates", "a": 8, "b": 20}]


def test_two_runs_that_decided_differently(make_config, runs_dir, capsys, apply_brief):
    cfg = apply_brief(make_config("classification"))
    first = Engine(cfg, runs_root=runs_dir, mode="interactive")
    _accept_all(first)
    second = Engine(cfg, runs_root=runs_dir, mode="interactive")
    second.run_until_idle()
    feature = second.results()["explore"].payload["features"][0]
    second.submit_gate("gate_data", "edit", {"exclude_columns": [feature]}, reason="not at origination")
    _accept_all(second)

    c = service.compare(first.run_id, second.run_id, runs_dir)
    assert (c["a"]["run_id"], c["b"]["run_id"]) == (first.run_id, second.run_id) and not c["caveats"]
    data_gate = next(d for d in c["decisions"] if d["gate"] == "gate_data")
    assert data_gate["different"] and (data_gate["a"]["action"], data_gate["b"]["action"]) == ("accept", "edit")
    assert not next(d for d in c["decisions"] if d["gate"] == "gate_design")["different"]  # same action
    excl = next(o for o in c["overrides"] if o["what"] == "Excluded columns")
    assert excl["b"] == [feature] and excl["a"] == []
    assert c["config"] == []  # the same profile: only the decision differs
    assert any(r["key"] == "cv_mean" for r in c["metrics"])
    assert service.previous_run(second.run_id, runs_dir) == first.run_id
    assert service.previous_run(first.run_id, runs_dir) is None

    # swapping the runs flips every claim
    back = service.compare(second.run_id, first.run_id, runs_dir)
    flip = {"a": "b", "b": "a", "same": "same", None: None}
    assert [flip[r["better"]] for r in c["metrics"]] == [r["better"] for r in back["metrics"]]

    assert main(["compare", first.run_id, second.run_id, "--runs-dir", runs_dir]) == 0
    out = capsys.readouterr().out
    assert "Decided differently" in out and "Excluded columns" in out and feature in out
    assert main(["compare", first.run_id, "nope", "--runs-dir", runs_dir]) == 1


def test_an_unfinished_or_re_running_run_is_flagged(make_config, runs_dir):
    cfg = make_config("classification")
    done = Engine(cfg, runs_root=runs_dir)
    done.run_until_idle()
    partial = Engine(cfg, runs_root=runs_dir, mode="interactive")
    partial.run_until_idle()  # stops at the first gate
    c = service.compare(done.run_id, partial.run_id, runs_dir)
    assert any("has not finished" in x and partial.run_id in x for x in c["caveats"])


# --- exporting ------------------------------------------------------------------------------------
def test_export_is_readable_complete_and_never_contains_the_data(make_config, runs_dir, tmp_path, capsys):
    eng = Engine(make_config("classification"), runs_root=runs_dir)
    eng.run_until_idle()
    assert (eng.run_dir / "data" / "holdout.parquet").exists()  # the thing that must not leave

    path = service.export_run(eng.run_id, root=runs_dir)
    assert path.name == f"{eng.run_id}.zip" and path.parent.name == "_exports"
    with zipfile.ZipFile(path) as z:
        names = [n.removeprefix(f"{eng.run_id}/") for n in z.namelist()]
        manifest = json.loads(z.read(f"{eng.run_id}/EXPORT.json"))
        state = json.loads(z.read(f"{eng.run_id}/state.json"))
    assert not [n for n in names if n.startswith(("data/", "models/")) or n.endswith((".parquet", ".joblib", ".npz"))]
    assert {"state.json", "config.yaml", "docs/model_card.md", "docs/decisions.md", "stages/model/result.json",
            "stages/validate/result.json", "agents/audit.jsonl", "EXPORT.json"} <= set(names)
    assert not [n for n in names if n.startswith("agents/") and n != "agents/audit.jsonl"]  # opt-in only
    assert state["run_id"] == eng.run_id and len(state["decisions"]) == 5

    # the manifest lists every other file with a hash that matches what is in the zip
    import hashlib
    with zipfile.ZipFile(path) as z:
        for f in manifest["files"]:
            assert hashlib.sha256(z.read(f"{eng.run_id}/{f['path']}")).hexdigest() == f["sha256"], f["path"]
    assert {f["path"] for f in manifest["files"]} == set(names) - {"EXPORT.json"}
    assert any("holdout" in x for x in manifest["not_included"])

    # the export lives under _exports/, which is not a run
    assert [r["run_id"] for r in service.list_runs(runs_dir)] == [eng.run_id]

    out = tmp_path / "handover.zip"
    assert main(["export", eng.run_id, "--runs-dir", runs_dir, "-o", str(out), "--with-agent-io"]) == 0
    assert "never included" in capsys.readouterr().out
    with zipfile.ZipFile(out) as z:
        assert any(n.endswith(".input.json") for n in z.namelist())
        assert not any(n.endswith(".parquet") for n in z.namelist())
    assert not list(tmp_path.glob(".*.tmp"))  # written atomically: no temp file left behind
    assert main(["export", "nope", "--runs-dir", runs_dir]) == 1
