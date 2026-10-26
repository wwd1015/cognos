"""Trial runs (ADR-0017): a decision taken differently in a copy, with nothing lost and
nothing hidden. A trial run stops before validation, every look at the test sample is counted
for the model, and a deleted trial run keeps its line."""

from __future__ import annotations

import json
import zipfile

import pytest

from cognos import service, synth, trials
from cognos.agents import facts as facts_mod
from cognos.cli import main
from cognos.config import CognosConfig
from cognos.engine import GateError
from cognos.engine.graph import GATES


def _to(run_id: str, root, until: str | None = None):
    """Accept every gate until the run waits at ``until`` (or stops)."""
    for _ in range(24):
        state = service.run_until_idle(run_id, root)
        waiting = [g for g in GATES if state.status_of(g) == "awaiting"]
        if not waiting or waiting[0] == until:
            return state
        service.submit_gate(run_id, waiting[0], "accept", reason="as proposed", root=root,
                            background=False)
    return service.state(run_id, root)


@pytest.fixture
def main_run(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("regression", root, n=260, search_budget=6)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    _to(run_id, root, until="gate_champion")
    return root, run_id


def test_a_trial_run_is_a_copy_that_waits_at_the_decision_and_leaves_its_parent_alone(main_run):
    root, run_id = main_run
    before = (root / run_id / "stages" / "backtest" / "result.json").read_text(encoding="utf-8")
    steps = {s: v.status for s, v in service.state(run_id, root).steps.items()}

    trial = service.trial_run(run_id, "gate_candidates", "a smaller pool", root=root, by="ana")
    st = service.state(trial, root)
    assert st.is_trial and st.status == "awaiting" and st.status_of("gate_candidates") == "awaiting"
    assert (st.lineage.parent, st.lineage.gate, st.lineage.note, st.lineage.by) == (
        run_id, "gate_candidates", "a smaller pool", "ana")
    assert st.status_of("model") == "done" and st.status_of("validate") == "pending"
    assert st.package is None and not (root / trial / "packages").exists()
    # the work before the gate came across: results, data and the search cache
    for rel in ("stages/model/result.json", "stages/model/search_cache.joblib",
                "data/train.parquet", "config.yaml"):
        assert (root / trial / rel).exists(), rel

    parent = service.state(run_id, root)
    assert not parent.is_trial and parent.lineage.model == run_id  # it is the main run now
    assert {s: v.status for s, v in parent.steps.items()} == steps
    assert (root / run_id / "stages" / "backtest" / "result.json").read_text(
        encoding="utf-8") == before
    assert [r["role"] for r in service.list_runs(root) if r["run_id"] in (run_id, trial)] \
        .count("trial") == 1


def test_a_trial_run_needs_a_reason_a_developer_and_a_decision_it_has_reached(main_run):
    root, run_id = main_run
    with pytest.raises(GateError, match="what this trial run is for"):
        service.trial_run(run_id, "gate_design", "  ", root=root)
    with pytest.raises(GateError, match="model developer"):
        service.trial_run(run_id, "gate_design", "try", root=root, seat="reviewer")
    with pytest.raises(GateError, match="not a developer decision"):
        service.trial_run(run_id, "gate_validation", "try", root=root)
    assert service.trial_gates(run_id, root) == ["gate_intent", "gate_source", "gate_data",
                                                 "gate_design", "gate_candidates",
                                                 "gate_champion"]
    fresh = service.create_run(service.demo_config("regression", root, n=150, search_budget=4),
                               mode="interactive", provider="heuristic", root=root)
    with pytest.raises(GateError, match="has not reached"):
        service.trial_run(fresh, "gate_design", "too early", root=root)
    assert [r["run_id"] for r in service.list_runs(root)].count(run_id) == 1
    assert service.trial_record(run_id, root)["n_runs"] == 1  # nothing was made


def test_a_trial_run_stops_before_validation_and_every_look_is_counted(main_run):
    root, run_id = main_run
    pool = service.results(run_id, root)["model"].payload["pool_ids"]
    main_looks = service.results(run_id, root)["backtest"].payload["test_evaluations"]

    trial = service.trial_run(run_id, "gate_candidates", "drop the last member", root=root)
    service.submit_gate(trial, "gate_candidates", "edit", {"pool": pool[:-1]}, root=root,
                        reason="smaller", background=False)
    state = _to(trial, root)
    assert state.status == "completed" and state.status_of("gate_champion") == "done"
    assert state.status_of("validate") == "pending"  # ... and it stays that way
    assert service.engine(trial, root).ready_steps() == []
    assert not (root / trial / "stages" / "validate").exists()
    with pytest.raises(GateError, match="does not run on a trial run"):
        service.engine(trial, root).run_stage("validate")

    rec = service.trial_record(run_id, root)
    by_id = {r["run_id"]: r for r in rec["lines"]}
    assert rec["main"] == run_id and rec["n_runs"] == 2 and rec["n_trials"] == 1
    assert by_id[trial]["role"] == "trial" and by_id[trial]["changed"] == ["pool"]
    assert by_id[trial]["stands"] == "trial finished at the champion"
    # the trial run's own looks: its pool, not the comparison it copied from its parent
    assert by_id[trial]["test_evaluations"] == len(pool) - 1
    assert by_id[run_id]["test_evaluations"] == main_looks
    assert rec["test_evaluations"] == main_looks + len(pool) - 1
    assert rec["holdout_scored"] == []
    assert service.trial_record(trial, root)["lines"] == rec["lines"]  # one record, either way in


def test_validation_and_the_white_paper_report_the_trial_runs(main_run):
    root, run_id = main_run
    pool = service.results(run_id, root)["model"].payload["pool_ids"]
    trial = service.trial_run(run_id, "gate_candidates", "drop the last member", root=root)
    service.submit_gate(trial, "gate_candidates", "edit", {"pool": pool[:-1]}, root=root,
                        reason="smaller", background=False)
    _to(trial, root)
    total = service.trial_record(run_id, root)["test_evaluations"]

    _to(run_id, root, until="gate_validation")
    validate = service.results(run_id, root)["validate"]
    tried = validate.payload["trials"]
    assert tried["n_runs"] == 2 and tried["n_other_runs"] == 1
    assert tried["test_evaluations"] == total > tried["test_evaluations_this_run"]
    finding = next(f for f in validate.findings if f.id == "trial-runs")
    # no sealed holdout here: the test result was reached with every look in view
    assert finding.severity.value == "LOW" and f"used {total} time(s)" in finding.message
    assert "best of several tries" in finding.suggestion
    saved = json.loads((root / run_id / "stages/validate/trials.json").read_text("utf-8"))
    assert {r["run_id"] for r in saved["lines"]} == {run_id, trial}
    known = facts_mod.collect(service.engine(run_id, root).context())
    assert known["validate.test_evaluations_model"] == total
    assert known["validate.runs_of_this_model"] == 2

    # deleting the trial run removes its files and keeps its line and its count
    blocked = service.deletion_preview([run_id], root)[0]["blocked"]
    assert "main run of a model with trial runs" in blocked
    assert not any(r["blocked"] for r in service.deletion_preview([run_id, trial], root))
    service.delete_runs([trial], root, by="ana")
    assert not (root / trial).exists()
    rec = service.trial_record(run_id, root)
    gone = next(r for r in rec["lines"] if r["run_id"] == trial)
    assert gone["deleted_by"] == "ana" and gone["deleted_at"] and gone["note"] == "drop the last member"
    assert rec["n_runs"] == 2 and rec["n_deleted"] == 1 and rec["test_evaluations"] == total
    assert [e["what"] for e in rec["events"]] == ["trial", "deleted"]
    assert "files deleted" in trials.describe(gone)

    service.submit_gate(run_id, "gate_validation", "accept", reason="read", root=root,
                        background=False)
    service.run_until_idle(run_id, root)
    paper = (root / run_id / "docs" / "methodology.md").read_text(encoding="utf-8")
    assert "## Alternatives tried" in paper and trial in paper and "drop the last member" in paper
    assert service.results(run_id, root)["review"].verdict.value != "BLOCK"
    with zipfile.ZipFile(service.export_run(run_id, root=root)) as z:
        shipped = json.loads(z.read(f"{run_id}/MODEL_RECORD.json"))
    assert [r["run_id"] for r in shipped["lines"]] == [run_id, trial]


def test_the_cli_makes_lists_and_promotes_trial_runs(main_run, capsys):
    root, run_id = main_run
    assert main(["trial", "--run", run_id, "--gate", "gate_champion", "--note", "another look",
                 "--runs-dir", str(root)]) == 0
    trial = next(r["run_id"] for r in service.list_runs(root) if r["role"] == "trial")
    assert main(["trials", "--run", run_id, "--runs-dir", str(root)]) == 0
    out = capsys.readouterr().out
    assert trial in out and "another look" in out and "2 run(s) of this model" in out
    assert main(["promote", "--run", run_id, "--runs-dir", str(root)]) == 1  # already the main run
    assert "already the main run" in capsys.readouterr().out


# --- with a sealed holdout: only the main run opens it ------------------------------------------
@pytest.fixture
def sealed_run(tmp_path):
    df = synth.make_cni_portfolio_dataset(n=2400)
    csv = tmp_path / "cni.csv"
    df.to_csv(csv, index=False)
    root = tmp_path / "runs"
    cfg = CognosConfig.from_dict({
        "name": "sealed", "task": "classification", "mode": "interactive",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome", "default_quarter"]},
        "search": {"max_candidates": 6, "cv_folds": 3, "model_families": ["logit"]},
    })
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    _to(run_id, root, until="gate_champion")
    return root, run_id


def test_a_trial_run_cannot_open_the_holdout_and_a_promoted_one_inherits_what_was_opened(
        sealed_run):
    root, run_id = sealed_run
    backtest = service.results(run_id, root)["backtest"].payload
    first = backtest["champion_id"]
    other = next(c["id"] for c in backtest["candidates"] if c["id"] != first)

    trial = service.trial_run(run_id, "gate_champion", "the simpler model", root=root)
    assert (root / trial / "data" / "holdout.parquet").exists()  # copied, and left closed
    service.submit_gate(trial, "gate_champion", "override", {"champion": other}, root=root,
                        reason="simpler", background=False)
    assert _to(trial, root).status == "completed"
    assert service.state(trial, root).overrides.champion == other
    assert service.trial_record(run_id, root)["holdout_scored"] == []

    # the main run goes to validation: the holdout is scored once, and the trial run is a
    # note for the record, not a finding that lowers the verdict
    _to(run_id, root, until="gate_validation")
    validate = service.results(run_id, root)["validate"]
    assert validate.payload["holdout_scored"] == [first]
    note = next(f for f in validate.findings if f.id == "trial-runs")
    assert note.severity.value == "INFO" and "cannot open the sealed holdout" not in note.message
    assert "No trial run can open the sealed holdout" in note.message
    assert not [f for f in validate.findings if f.id == "holdout-reuse"]

    # the developer prefers the trial run: it becomes the main run and is validated in turn
    with pytest.raises(GateError, match="model developer"):
        service.promote_trial(trial, root=root, seat="approver")
    service.promote_trial(trial, "one factor fewer, same result", root=root, by="ana",
                          background=False)
    old = service.state(run_id, root)
    assert old.is_trial and old.lineage.replaced_by == trial
    new = _to(trial, root, until="gate_validation")
    assert not new.is_trial and new.lineage.promoted_by == "ana"
    second = service.results(trial, root)["validate"]
    # the holdout was already opened for this model: the second champion is counted
    assert second.payload["holdout_scored"] == [first, other]
    assert second.payload["holdout_evaluations"] == 2
    assert any(f.id == "holdout-reuse" for f in second.findings)

    rec = service.trial_record(trial, root)
    assert rec["main"] == trial and rec["holdout_scored"] == [first, other]
    assert [e["what"] for e in rec["events"]] == ["trial", "promoted"]
    # the replaced run keeps its results and can go no further
    assert (root / run_id / "stages" / "validate" / "result.json").exists()
    service.engine(run_id, root).reopen("gate_champion")
    service.submit_gate(run_id, "gate_champion", "accept", root=root, reason="again",
                        background=False)
    assert service.engine(run_id, root).ready_steps() == []
    with pytest.raises(GateError, match="already the main run"):
        service.promote_trial(trial, root=root)
