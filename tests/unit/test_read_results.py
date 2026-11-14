"""One reader for the recorded results of a run (``artifacts.read_result`` and friends)."""

from __future__ import annotations

import pytest

from cognos import service
from cognos.artifacts import (
    StageResult,
    Verdict,
    read_payload,
    read_raw,
    read_result,
    read_results,
    result_path,
)
from cognos.context import RunContext


def _write(run_dir, stage, text=None, name="result.json"):
    path = result_path(run_dir, stage, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        text = StageResult(stage=stage, verdict=Verdict.PASS, payload={"k": stage}).model_dump_json()
    path.write_text(text, encoding="utf-8")


def test_a_stage_without_a_result_reads_as_none(tmp_path):
    assert read_result(tmp_path, "model") is None
    assert read_results(tmp_path, ["explore", "model"]) == {}
    assert read_raw(tmp_path, "model") == {}
    assert read_payload(tmp_path, "model") == {}


def test_results_come_back_in_the_order_asked(tmp_path):
    for stage in ("model", "explore"):
        _write(tmp_path, stage)
    got = read_results(tmp_path, ["explore", "ideate", "model"])
    assert list(got) == ["explore", "model"]
    assert got["model"].payload == {"k": "model"}
    assert read_payload(tmp_path, "explore") == {"k": "explore"}


def test_a_damaged_result_raises_unless_the_reader_is_tolerant(tmp_path):
    _write(tmp_path, "explore")
    _write(tmp_path, "model", text="{ not json")
    with pytest.raises(ValueError):
        read_result(tmp_path, "model")
    with pytest.raises(ValueError):
        read_results(tmp_path, ["explore", "model"])
    assert read_result(tmp_path, "model", tolerant=True) is None
    assert list(read_results(tmp_path, ["explore", "model"], tolerant=True)) == ["explore"]
    assert read_raw(tmp_path, "model") == {}  # plain readers never raise


def test_the_previous_result_is_read_by_name(tmp_path):
    _write(tmp_path, "model", name="result.prev.json")
    assert read_result(tmp_path, "model") is None
    assert read_result(tmp_path, "model", "result.prev.json").stage == "model"


def test_service_and_context_read_through_the_one_reader(tmp_path, monkeypatch):
    run_dir = tmp_path / "r1"
    _write(run_dir, "explore")
    _write(run_dir, "model", text="{ not json")
    with pytest.raises(ValueError):
        service.results("r1", tmp_path)  # a page must not show a damaged stage as not run
    assert list(service._results_in(run_dir)) == ["explore"]  # a comparison leaves it out

    ctx = RunContext.__new__(RunContext)
    ctx._results, ctx.run_dir = {}, run_dir
    assert ctx.get("explore").payload == {"k": "explore"}
    assert ctx.get("ideate") is None
    with pytest.raises(ValueError):
        ctx.get("model")
