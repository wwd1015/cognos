"""Data sources, the plugin registry, analysis tools, chart specs and the restricted code runner."""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from cognos import datasources, plugins, synth
from cognos.agents import checks, heuristic
from cognos.agents.contracts import CONTRACTS
from cognos.analysis import charts, sandbox
from cognos.config import CognosConfig, SourceConfig
from cognos.stages.explore import infer_task, target_candidates

EXAMPLES = str(Path(__file__).resolve().parents[2] / "examples" / "plugins")


@pytest.fixture
def frame() -> pd.DataFrame:
    return synth.GENERATORS["commercial"](n=400)


def _cfg(source: dict | None = None, **data) -> CognosConfig:
    return CognosConfig.from_dict({"name": "t", "data": {"source": source, **data}})


# --- data sources ---------------------------------------------------------------------------
def test_file_source_reads_common_formats_and_records_provenance(tmp_path, frame):
    for name, write in (("d.csv", lambda p: frame.to_csv(p, index=False)),
                        ("d.parquet", lambda p: frame.to_parquet(p, index=False)),
                        ("d.tsv", lambda p: frame.to_csv(p, index=False, sep="\t")),
                        ("d.jsonl", lambda p: frame.to_json(p, orient="records", lines=True))):
        path = tmp_path / name
        write(path)
        df, prov = datasources.load(_cfg({"kind": "file", "path": str(path), "limit": 50}))
        assert list(df.columns) == list(frame.columns) and len(df) == 50, name
        assert prov["kind"] == "file" and prov["location"] == name and prov["n_rows"] == 50
    # the older data.path + data.format still means a file source
    df, _ = datasources.load(_cfg(None, path=str(tmp_path / "d.parquet"), format="parquet"))
    assert len(df) == len(frame)
    with pytest.raises(datasources.SourceError, match="not found"):
        datasources.load(_cfg({"kind": "file", "path": str(tmp_path / "nope.csv")}))
    with pytest.raises(datasources.SourceError, match="unknown data source"):
        datasources.load(_cfg({"kind": "oracle", "table": "t"}))


def test_sqlite_source_reads_a_table_or_one_read_only_query(tmp_path, frame):
    db = tmp_path / "book.db"
    con = sqlite3.connect(db)
    frame.to_sql("loans", con, index=False)
    con.close()
    df, prov = datasources.load(_cfg({"kind": "sqlite", "path": str(db), "table": "loans"}))
    assert len(df) == len(frame) and prov["table"] == "loans"
    df, prov = datasources.load(_cfg({
        "kind": "sqlite", "path": str(db), "limit": 25,
        "query": "SELECT leverage, \"default\" FROM loans WHERE leverage > 0;"}))
    assert list(df.columns) == ["leverage", "default"] and len(df) == 25
    assert "WHERE leverage > 0" in prov["query"]
    for bad in ("DROP TABLE loans", "SELECT 1; DELETE FROM loans", "select * from loans; --",
                "WITH x AS (SELECT 1) UPDATE loans SET leverage = 0"):
        with pytest.raises(datasources.SourceError):
            datasources.statement(SourceConfig(kind="sqlite", query=bad))
    with pytest.raises(datasources.SourceError, match="not a table name"):
        datasources.statement(SourceConfig(kind="sqlite", table="loans; drop table x"))
    with pytest.raises(datasources.SourceError, match="table or a query"):
        datasources.statement(SourceConfig(kind="sqlite"))


def test_snowflake_is_unavailable_without_its_driver(monkeypatch):
    monkeypatch.setitem(sys.modules, "snowflake", None)
    ok, why = datasources._snowflake_available()
    assert not ok and "snowflake-connector-python" in why
    with pytest.raises(datasources.SourceUnavailable):
        datasources.load(_cfg({"kind": "snowflake", "table": "DB.S.T"}))


def test_snowflake_connects_from_the_environment_and_keeps_secrets_out(monkeypatch, frame):
    seen: dict = {}

    class Cursor:
        description = [(c,) for c in frame.columns]

        def execute(self, sql):
            seen["sql"] = sql

        def fetchall(self):
            return [tuple(r) for r in frame.head(20).itertuples(index=False)]

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            seen["closed"] = True

    connector = types.ModuleType("snowflake.connector")
    connector.connect = lambda **kw: seen.update(args=kw) or Connection()
    pkg = types.ModuleType("snowflake")
    pkg.connector = connector
    monkeypatch.setitem(sys.modules, "snowflake", pkg)
    monkeypatch.setitem(sys.modules, "snowflake.connector", connector)
    for k, v in {"SNOWFLAKE_ACCOUNT": "acme", "SNOWFLAKE_USER": "svc", "MY_ROLE": "ANALYST",
                 "SNOWFLAKE_PASSWORD": "s3cret"}.items():
        monkeypatch.setenv(k, v)
    assert datasources._snowflake_available() == (True, "")
    cfg = _cfg({"kind": "snowflake", "table": "RISK.LOANS.ORIG", "limit": 1000,
                "options": {"warehouse": "WH", "role": "env:MY_ROLE"}})
    df, prov = datasources.load(cfg)
    assert len(df) == 20 and seen["closed"]
    assert seen["sql"] == "SELECT * FROM (SELECT * FROM RISK.LOANS.ORIG) AS cognos_src LIMIT 1000"
    assert seen["args"] == {"account": "acme", "user": "svc", "warehouse": "WH", "role": "ANALYST",
                            "password": "s3cret", "application": "COGNOS"}
    assert "s3cret" not in str(prov) and "s3cret" not in cfg.model_dump_json()
    monkeypatch.delenv("SNOWFLAKE_PASSWORD")
    with pytest.raises(datasources.SourceUnavailable, match="no Snowflake credential"):
        datasources.load(cfg)


# --- plugins --------------------------------------------------------------------------------
def test_plugins_add_tools_and_sources_and_a_broken_one_is_reported(monkeypatch, tmp_path, frame):
    monkeypatch.syspath_prepend(EXAMPLES)
    (tmp_path / "broken_plugin.py").write_text("raise RuntimeError('boom')", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = plugins.registry(["credit_tools", "broken_plugin", "no_such_plugin"], refresh=True)
    assert reg.loaded == ["credit_tools"]
    assert {p["plugin"] for p in reg.problems} == {"broken_plugin", "no_such_plugin"}
    tool = reg.tools["information_value"]
    assert tool.origin == "credit_tools" and reg.sources["fixed_width"].origin == "credit_tools"
    out = tool.fn(frame, {"column": "leverage"}, {"target": "default", "task": "classification"})
    assert out["summary"]["information_value"] > 0 and charts.normalize(out["chart"])
    listing = plugins.public_list(["credit_tools"])
    assert "information_value" in {t["name"] for t in listing["tools"]}
    assert {"file", "sqlite", "snowflake", "fixed_width"} <= {s["kind"] for s in listing["sources"]}
    # COGNOS_PLUGINS names plugins too; the built-ins are always there
    monkeypatch.setenv("COGNOS_PLUGINS", "credit_tools")
    assert "information_value" in plugins.registry(refresh=True).tools
    monkeypatch.delenv("COGNOS_PLUGINS")
    assert "information_value" not in plugins.registry(refresh=True).tools


def test_builtin_tools_return_numbers_tables_and_valid_charts(frame):
    reg = plugins.registry()
    env = {"target": "default", "task": "classification", "datetime_col": "vintage",
           "features": [c for c in frame.columns if c != "default"]}
    frame = frame.copy()
    frame.loc[frame.index[:40], "leverage"] = None
    for name, params in (("distribution", {"column": "leverage"}),
                         ("distribution", {"column": "sector"}), ("target_distribution", {}),
                         ("target_relationship", {"column": "leverage", "bins": "5"}),
                         ("target_relationship", {"column": "sector"}),
                         ("correlation_matrix", {}), ("missingness", {}), ("time_trend", {})):
        out = reg.tools[name].fn(frame, params, env)
        assert out["summary"] and out["table"]["rows"], name
        spec = charts.normalize(out["chart"])
        assert spec["kind"] in charts.KINDS and spec["title"]
    assert reg.tools["target_distribution"].fn(frame, {}, env)["summary"]["n_events"] > 0
    assert reg.tools["missingness"].fn(frame, {}, env)["summary"]["worst_share"] == 0.1
    with pytest.raises(ValueError, match="unknown column"):
        reg.tools["distribution"].fn(frame, {"column": "nope"}, env)


def test_chart_specs_are_validated_and_bounded():
    ok = charts.normalize({"kind": "bar", "x": ["a", "b"], "series": [{"name": "n", "y": [1, float("nan")]}]})
    assert ok["series"][0]["y"] == [1, None]
    big = charts.normalize({"kind": "line", "x": list(range(5000)),
                            "series": [{"name": "y", "y": list(range(5000))}]})
    assert len(big["x"]) == charts.MAX_POINTS
    for bad in ({"kind": "pie", "x": [1], "series": [{"y": [1]}]},
                {"kind": "bar", "x": [1, 2], "series": [{"y": [1]}]},
                {"kind": "bar", "x": [], "series": []},
                {"kind": "heatmap", "x": ["a"], "y": ["a", "b"], "z": [[1]]}):
        with pytest.raises(ValueError):
            charts.normalize(bad)


# --- agent-written code ---------------------------------------------------------------------
GOOD = """
# Does the default rate differ by sector?
by = df.groupby("sector")[TARGET].agg(["mean", "size"]).reset_index()
emit_table("default rate by sector", by)
emit_value("max_sector_rate", by["mean"].max())
emit_chart("bar", by["sector"], {"event rate": by["mean"]}, title="Default rate by sector",
           y_format="%")
"""


def test_restricted_runner_executes_an_analysis_and_returns_only_what_it_emits(tmp_path, frame):
    data = tmp_path / "d.parquet"
    frame.to_parquet(data)
    assert sandbox.validate(GOOD) == []
    out = sandbox.run(GOOD, data, target="default")
    assert out["ok"] and out["values"]["max_sector_rate"] > 0
    assert out["tables"][0]["columns"] == ["sector", "mean", "size"]
    assert charts.normalize(out["charts"][0])["y_format"] == "%"
    assert sandbox.run(GOOD, data, target="default") == out  # deterministic
    # a script that fails, or never finishes, is a result, not an exception
    assert "KeyError" in sandbox.run("emit_value('x', df['nope'].mean())", data)["error"]
    assert "timed out" in sandbox.run("while True:\n    pass\nemit_value('x', 1)", data,
                                      timeout_s=2)["error"]
    # the process has no file, network or interpreter access even where the check is bypassed
    sneaky = "import importlib\nemit_value('x', 1)"
    assert "not allowed" in sandbox.run(sneaky, data)["error"]


@pytest.mark.parametrize("code, why", [
    ("import os\nemit_value('x', 1)", "import of 'os'"),
    ("import subprocess as sp\nemit_value('x', 1)", "import of 'subprocess'"),
    ("from pathlib import Path\nemit_value('x', 1)", "import of 'pathlib'"),
    ("emit_value('x', open('/etc/passwd').read())", "'open'"),
    ("df.to_csv('/tmp/out.csv')\nemit_value('x', 1)", ".to_csv"),
    ("pd.read_csv('other.csv')\nemit_value('x', 1)", ".read_csv"),
    ("emit_value('x', df.query('leverage > 1').shape[0])", ".query"),
    ("emit_value('x', eval('1+1'))", "'eval'"),
    ("emit_value('x', df.__class__.__name__)", "private attribute"),
    ("b = __builtins__\nemit_value('x', 1)", "__builtins__"),
    ("x = df['leverage'].mean()", "returns nothing"),
    ("emit_value('x', 1", "syntax error"),
    ("", "empty"),
])
def test_scripts_that_reach_outside_the_data_are_rejected_before_running(code, why):
    assert any(why in e for e in sandbox.validate(code)), sandbox.validate(code)


# --- the target, and the analyst's requests ---------------------------------------------------
def test_target_candidates_rank_outcome_like_columns(frame):
    cfg = CognosConfig.from_dict({"name": "t", "data": {"datetime_col": "vintage"}})
    frame = frame.assign(row_id=range(len(frame)), utilization_flag=(frame["leverage"] > 3).astype(int))
    ranked = target_candidates(frame, cfg, "predict payment default within 12 months")
    assert ranked[0]["column"] == "default" and ranked[0]["binary"]
    assert "row_id" not in [c["column"] for c in ranked]  # an identifier is never an outcome
    assert "sector" not in [c["column"] for c in ranked]  # nor is a text column
    assert infer_task(frame["default"]) == "classification"
    assert infer_task(frame["leverage"]) == "regression"
    # a profile may leave both the target and the task open, and settle them later
    assert cfg.task is None and cfg.metric.name == "auto"
    done = cfg.with_target("default", "classification")
    assert (done.data.target, done.task.value, done.metric.name) == ("default", "classification", "roc_auc")
    fixed = CognosConfig.from_dict({"name": "t", "task": "regression", "data": {"target": "y"}})
    assert fixed.with_target("default", "classification") is fixed  # the profile wins


def _scout_slice(**over) -> dict:
    reg = plugins.registry()
    return {"columns": ["leverage", "sector", "default", "vintage"], "target": "default",
            "target_candidates": [], "allow_code": True, "max_requests": 8,
            "datetime_col": "vintage", "columns_with_missing": [], "analyses": [],
            "top_correlations": [{"feature": "leverage", "corr": 0.3}],
            "tools": [{"name": t.name, "params": t.params, "needs_target": t.needs_target}
                      for t in reg.tools.values()],
            "facts": {}, "sponsor_answers": [], "challenges": [], **over}


def _scout(sl: dict, **raw) -> list[str]:
    base = {"target_column": "", "requests": [], "done": True}
    out = CONTRACTS["data_scout"].model_validate({**base, **raw})
    return checks.run_checks("data_scout", out, sl)


def test_heuristic_scout_asks_for_the_standard_visuals_and_settles_an_open_target():
    sl = _scout_slice()
    out = heuristic.data_scout(sl)
    assert _scout(sl, **out) == [] and out["done"]
    assert [r["tool"] for r in out["requests"]] == [
        "target_distribution", "time_trend", "target_relationship", "correlation_matrix"]
    assert not any(r["code"] for r in out["requests"])  # the offline analyst never writes code
    open_ = _scout_slice(target="", target_candidates=[{"column": "default", "why": "binary"}])
    first = heuristic.data_scout(open_)
    assert _scout(open_, **first) == [] and first["target_column"] == "default" and not first["done"]
    assert heuristic.data_scout(_scout_slice(analyses=[{"id": "a1"}]))["requests"] == []


def test_engine_rejects_requests_it_cannot_run():
    sl = _scout_slice()

    def req(**kw):
        return {"purpose": "why", "tool": "", "code": "", "params": [], **kw}

    assert _scout(sl, requests=[req(tool="distribution", params=[{"name": "column", "value": "leverage"}])]) == []
    assert _scout(sl, requests=[req(code=GOOD)]) == []
    cases = {
        "unknown tool": req(tool="magic"),
        "not both and not neither": req(tool="distribution", code=GOOD),
        "unknown column": req(tool="distribution", params=[{"name": "column", "value": "nope"}]),
        "no parameter": req(tool="missingness", params=[{"name": "depth", "value": "3"}]),
        "import of 'os'": req(code="import os\nemit_value('x', 1)"),
        "what question": req(tool="missingness", purpose=" "),
    }
    for why, request in cases.items():
        assert any(why in e for e in _scout(sl, requests=[request])), why
    assert any("switched off" in e for e in _scout(_scout_slice(allow_code=False), requests=[req(code=GOOD)]))
    assert any("at most" in e for e in _scout(_scout_slice(max_requests=1),
                                              requests=[req(tool="missingness")] * 2))
    open_ = _scout_slice(target="", target_candidates=[{"column": "default"}])
    assert any("name the dependent variable" in e for e in _scout(open_))
    assert any("cannot be the dependent variable" in e for e in _scout(open_, target_column="sector"))
    assert any("needs the target" in e for e in _scout(
        _scout_slice(target="", target_candidates=[]), requests=[req(tool="time_trend")]))
    assert any("already" in e for e in _scout(sl, target_column="leverage"))
