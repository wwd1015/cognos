"""Catalog sources (ADR-0022): the catalog answers search, describe, links and defaults behind
one contract, a plugin can supply it, and its fit to the intent is advice with its name on it.
Links drawn on the join canvas are measured by the engine before they can be joined."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognos import catalog, plugins, service
from cognos.engine.gates import GateError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "plugins"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("COGNOS_RUNS_DIR", str(root))
    monkeypatch.delenv("COGNOS_CATALOG", raising=False)
    return root


def test_the_yaml_catalog_is_the_built_in_source(runs_dir):
    src = catalog.source(runs_dir)
    assert isinstance(src, catalog.YamlCatalog) and src.info()["kind"] == "default"
    assert src.search("equity volatility")[0].id == "market_equity"
    assert [t.id for t in src.search("")][:2] == ["obligor_quarters", "facilities"]
    assert src.search("no such words here") == []
    assert src.describe("obligors").keys == ["obligor_id"]
    with pytest.raises(KeyError):
        src.describe("nothing")
    assert {(lk.left, lk.right) for lk in src.links(["obligor_quarters", "facilities"])}
    assert src.defaults() == ["obligor_quarters", "macro_quarterly"]


def test_the_engine_says_whether_tables_fit_and_why(runs_dir):
    src = catalog.source(runs_dir)
    intent = {"objective": "Probability of default for commercial obligors", "fields": {}}
    fits = {f.table: f for f in catalog.fit(src, ["obligor_quarters", "facilities"], intent,
                                            runs_dir)}
    assert fits["obligor_quarters"].level == "fits" and fits["obligor_quarters"].origin == "engine"
    assert "outcome" in fits["obligor_quarters"].reasons[0]
    assert fits["facilities"].level == "fits" and "documents" in fits["facilities"].reasons[0]
    # a base table without an outcome: another chosen table carries it
    upside = catalog.fit(src, ["obligors", "obligor_quarters"], intent, runs_dir)
    assert upside[0].level == "partly" and "obligor_quarters" in upside[0].reasons[0]


def test_a_plugin_source_supplies_the_catalog(runs_dir, tmp_path, monkeypatch):
    from cognos import synth

    data = tmp_path / "loans.parquet"
    synth.GENERATORS["commercial"](n=300).to_parquet(data, index=False)
    export = tmp_path / "export.json"
    export.write_text(json.dumps({
        "system": "Acme metadata",
        "datasets": [{"name": "loans", "label": "Commercial loans", "summary": "One loan each.",
                      "grain": "one row per loan", "primary_key": ["obligor_id"],
                      "targets": ["default"], "steward": "Lending",
                      "location": {"kind": "file", "path": str(data)},
                      "fields": [{"name": "default", "definition": "Defaulted within a year"}]}],
        "recommended": ["loans"]}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setenv("COGNOS_PLUGINS", "metadata_catalog")
    monkeypatch.setenv("METADATA_EXPORT", str(export))
    plugins.registry(refresh=True)
    try:
        assert {c["name"]: c["available"] for c in plugins.public_list()["catalogs"]} == {
            "metadata": True}
        monkeypatch.setenv("COGNOS_CATALOG", "metadata")
        src = catalog.source(runs_dir)
        assert src.name == "metadata" and src.origin == "metadata_catalog"
        assert catalog.listing(runs_dir)["tables"][0]["title"] == "Commercial loans"
        assert catalog.data_spec(["loans"], root=runs_dir)["source"]["path"] == str(data)
        payload = service.data_payload(["loans"], root=runs_dir)
        assert payload["catalog_source"] == "metadata" and payload["fit"][0]["level"] == "fits"
        monkeypatch.setenv("COGNOS_CATALOG", "nowhere")
        with pytest.raises(ValueError, match="no plugin registers"):
            catalog.source(runs_dir)
        monkeypatch.delenv("METADATA_EXPORT")
        monkeypatch.setenv("COGNOS_CATALOG", "metadata")
        with pytest.raises(ValueError, match="not available"):
            catalog.source(runs_dir)
    finally:
        monkeypatch.delenv("COGNOS_PLUGINS")
        plugins.registry(refresh=True)
        sys.modules.pop("metadata_catalog", None)


@pytest.fixture
def two_files(tmp_path):
    """A base table and a second one keyed by a differently named column that matches 85% of
    the base rows: the engine does not pair them by itself."""
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"id": np.arange(n), "x": rng.normal(size=n),
                         "default": (rng.random(n) < 0.2).astype(int)})
    extra = pd.DataFrame({"cust_ref": np.arange(340), "y": rng.normal(size=340)})
    paths = []
    for name, df in (("base", base), ("extra", extra)):
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def test_a_link_drawn_at_the_data_gate_is_measured_and_joins(runs_dir, two_files):
    cfg = service.template_config("built-in:pd_logistic_scorecard", "drawn link")
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic")
    service.run_until_idle(run_id)
    service.submit_gate(run_id, "gate_intent", "accept", reason="test", background=False)
    service.run_until_idle(run_id)
    service.provide_data(run_id, added=[{"name": p.stem, "kind": "file", "path": str(p)}
                                        for p in two_files], target="default",
                         background=False)
    st = service.run_until_idle(run_id)
    assert st.status_of("gate_data") == "awaiting"
    linked = service.results(run_id)["explore"].payload["linking"]
    assert not any({lk["a"], lk["b"]} == {"base", "extra"} for lk in linked["links"])
    # measured as it is drawn, nothing recorded
    lk = service.measure_link(run_id, "base", ["id"], "extra", ["cust_ref"])
    assert lk["a_in_b"] == pytest.approx(0.85) and lk["relation"] == "one-to-one"
    with pytest.raises(ValueError, match="no column"):
        service.measure_link(run_id, "base", ["nope"], "extra", ["cust_ref"])
    with pytest.raises(GateError, match="cannot be used"):
        service.submit_gate(run_id, "gate_data", "edit", {"join": {
            "base": "base", "links": ["D1"],
            "drawn": [{"id": "D1", "a": "base", "a_on": ["nope"], "b": "extra",
                       "b_on": ["cust_ref"]}]}}, background=False)
    st = service.submit_gate(run_id, "gate_data", "edit", {"join": {
        "base": "base", "links": ["D9"], "rules": {"D9": {"unmatched": "drop"}},
        "drawn": [{"id": "D9", "a": "base", "a_on": ["id"], "b": "extra",
                   "b_on": ["cust_ref"]}]}}, by="maria", background=False)
    assert st.overrides.drawn_links == [{"a": "base", "a_on": ["id"], "b": "extra",
                                         "b_on": ["cust_ref"], "by": "maria"}]
    step = st.overrides.join["steps"][0]
    assert step["link"] == "D1" and step["unmatched"] == "drop"  # renumbered, its own rule
    assert st.decisions[-1].payload["join"]["drawn"][0]["id"] == "D1"
    st = service.run_until_idle(run_id)
    p = service.results(run_id)["explore"].payload
    drawn = next(lk for lk in p["linking"]["links"] if lk["id"] == "D1")
    assert drawn["drawn"] and drawn["by"] == "maria"
    assert p["n_rows"] == 340 and "y" in p["features"]  # joined, unmatched rows dropped
    # new data starts the data work over, drawn links included
    service.reopen(run_id, "gate_source")
    service.provide_data(run_id, added=[{"name": "base", "kind": "file",
                                         "path": str(two_files[0])}],
                         target="default", background=False)
    assert service.state(run_id).overrides.drawn_links == []
