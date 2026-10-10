"""A run over several sources of different kinds: the join is proposed, shown, confirmed or
changed at the data gate, recorded in the white paper and kept as the dataset snapshot."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cognos import service
from cognos.engine import GateError


@pytest.fixture
def at_data_gate(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("linked", root, n=600, search_budget=4)
    assert {s.kind for s in cfg.data.sources} == {"file", "sqlite"}
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    service.run_until_idle(run_id, root)
    service.submit_gate(run_id, "gate_intent", "accept", root=root, background=False)
    service.run_until_idle(run_id, root)
    assert service.state(run_id, root).status_of("gate_data") == "awaiting"
    return root, run_id


def test_the_analyst_proposes_the_join_and_the_engine_runs_it(at_data_gate):
    root, run_id = at_data_gate
    ep = service.results(run_id, root)["explore"].payload
    linked = ep["linking"]
    assert linked["plan"]["source"] == "agent" and linked["plan"]["base"] == "loans"
    assert {s["right"] for s in linked["plan"]["steps"]} == {"financials", "macro"}
    fin = next(s for s in linked["report"]["steps"] if s["right"] == "financials")
    assert fin["match_rate"] == pytest.approx(0.97, abs=0.01) and fin["dropped_rows"] == 18
    # rows without financials are dropped, counted and raised as a finding: never silently
    assert ep["n_rows"] == 582 and linked["report"]["base_rows"] == 600
    assert "debt_to_ebitda" in ep["features"]
    assert any(f.id == "join-dropped-financials"
               for f in service.results(run_id, root)["explore"].findings)
    run_dir = root / run_id
    assert len(pd.read_parquet(run_dir / "data" / "dataset.parquet")) == 582
    src = json.loads((run_dir / "data" / "source.json").read_text(encoding="utf-8"))
    assert src["kind"] == "linked" and len(src["sources"]) == 3 and src["join"]["digest"]
    assert all((run_dir / "data" / "sources" / f"{n}.parquet").exists()
               for n in ("loans", "financials", "macro"))
    audit = [a["agent"] for a in service.audit(run_id, root)]
    assert "data_linker" in audit


def test_the_developer_changes_the_join_at_the_gate(at_data_gate):
    root, run_id = at_data_gate
    linked = service.results(run_id, root)["explore"].payload["linking"]
    fin = next(lk["id"] for lk in linked["links"] if "financials" in (lk["a"], lk["b"]))
    with pytest.raises(GateError, match="unknown link"):
        service.submit_gate(run_id, "gate_data", "edit", {"join": {"base": "loans", "links": ["L99"]}},
                            root=root, background=False)
    st = service.submit_gate(run_id, "gate_data", "edit",
                             {"join": {"base": "loans", "links": [fin], "unmatched": "keep"}},
                             root=root, background=False)
    assert st.status_of("explore") == "stale" and "join" in st.steps["explore"].rerun_reason
    service.run_until_idle(run_id, root)
    ep = service.results(run_id, root)["explore"].payload
    assert ep["linking"]["plan"]["source"] == "decision" and ep["linking"]["left_out"] == ["macro"]
    assert "gdp_growth" not in ep["dtypes"] and ep["n_rows"] == 600  # unmatched rows kept
    st = service.submit_gate(run_id, "gate_data", "accept", root=root, background=False)
    assert st.status_of("gate_data") == "done"
    assert st.decisions[-1].payload["reviewed_join"] == ep["linking"]["plan"]["digest"]


def test_accepting_fixes_the_join_and_the_white_paper_prints_it(at_data_gate):
    root, run_id = at_data_gate
    st = service.submit_gate(run_id, "gate_data", "accept", root=root, background=False)
    assert st.overrides.join["base"] == "loans" and len(st.overrides.join["steps"]) == 2
    for _ in range(8):
        st = service.run_until_idle(run_id, root)
        waiting = [s for s, v in st.steps.items() if v.status == "awaiting"]
        if not waiting:
            break
        seat = {"gate_validation": "reviewer", "gate_signoff": "approver"}.get(waiting[0])
        service.submit_gate(run_id, waiting[0], "accept", reason="test", root=root,
                            background=False, seat=seat)
    assert service.state(run_id, root).status_of("document") == "done"
    text = "\n".join(p.read_text(encoding="utf-8") for p in (root / run_id / "docs").rglob("*.md"))
    assert "Sources and join" in text and "loans.obligor_id = financials.obligor_id" in text


def test_a_stated_join_is_run_without_asking_the_agent(tmp_path):
    root = tmp_path / "runs"
    cfg = service.demo_config("linked", root, n=400, search_budget=4)
    raw = cfg.model_dump(mode="json")
    raw["data"]["base"] = "loans"
    raw["data"]["join"] = [{"left": "loans", "left_on": "obligor_id", "right": "financials",
                            "right_on": "obligor_id"}]
    run_id = service.create_run(raw, mode="autonomous", provider="heuristic", root=root)
    service.run_until_idle(run_id, root)
    ep = service.results(run_id, root)["explore"].payload
    assert ep["linking"]["plan"]["source"] == "profile" and ep["linking"]["left_out"] == ["macro"]
    assert "data_linker" not in [a["agent"] for a in service.audit(run_id, root)]
    assert service.state(run_id, root).overrides.join is None  # the profile stays the authority
