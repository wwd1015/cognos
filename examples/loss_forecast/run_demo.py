#!/usr/bin/env python3
"""COGNOS worked example — a quarterly loss-forecasting model, from factors to a champion.

A portfolio net charge-off rate on macro drivers, one row per quarter. The series is short and
the drivers act through lags and changes, so the build is the classic one (ADR-0016):

  1. THE PLAN. The engine reads the shape of the data (one row per period) and the intent and
     recommends how to split it: the latest quarters held back as the test sample, a rolling
     forecast origin for validation, no sealed third sample (too few quarters, and it says so).
     The developer edits the plan at the design gate: here, the ranking and the size limit.
  2. SINGLE-FACTOR ANALYSIS. Eight explored columns become about two hundred factors (lags,
     changes, year-on-year moves, moving averages). Each is fitted alone, and the developer's
     rules cut the library to a shortlist; every dropped factor keeps the rule and the reason.
  3. FROM ONE FACTOR TO SEVERAL. Forward selection starts from the best single factor and adds
     one at a time while the validation error falls and the model stays significant, correctly
     signed and free of near-duplicate factors. Best subset checks the path against every
     combination. Every model evaluated lands in one ledger.
  4. THE CANDIDATE POOL. The search does not end on a champion. The developer reviews the
     leaders and adds a simpler model from the ledger by its id.
  5. THE CHAMPION, ON OUTCOMES. Every pool member forecasts the held-back quarters and each
     scenario. Models that ranked together part ways here, and the choice is made on that page.

Run:  python examples/loss_forecast/run_demo.py
(Requires `pip install -e .`. Runs offline with the deterministic agents; LLM agents recommend
on top of the same engine checks, never instead.)
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

from cognos import service

warnings.filterwarnings("ignore")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _next_gate(run_id: str, root: Path) -> str | None:
    state = service.run_until_idle(run_id, root)
    return next((g for g, s in state.steps.items() if s.status == "awaiting"), None)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="cognos-loss-"))
    cfg = service.demo_config("loss_forecast", root)
    run_id = service.create_run(cfg, mode="interactive", provider="heuristic", root=root)
    for gate in ("gate_intent", "gate_data"):
        assert _next_gate(run_id, root) == gate
        service.submit_gate(run_id, gate, "accept", reason="as proposed", root=root,
                            background=False)

    # --- 1. the plan ---------------------------------------------------------------------
    banner("1. THE PLAN — recommended from the data and the intent, confirmed by the developer")
    assert _next_gate(run_id, root) == "gate_design"
    plan = service.results(run_id, root)["ideate"].payload["plan"]
    shape = plan["shape"]
    print(f"  Data: {shape['n_rows']} rows, one per period ({shape['period']} a year) → "
          f"'{shape['shape']}'")
    for field, label in (("test", "Test sample"), ("validation", "Validation"),
                         ("holdout_fraction", "Sealed holdout"), ("folds", "Forecast origins"),
                         ("methods", "Selection"), ("criterion", "Step criterion")):
        value = (plan["evaluation"] | plan["selection"])[field]
        print(f"  {label:17s} {str(value):28s} [{plan['sources'].get(field, '')}] "
              f"{plan['reasons'].get(field, '')}")
    print(f"  The data also allow: validation {plan['options']['validation']}, "
          f"test {plan['options']['test']}")
    edit = {"evaluation": {"pool_size": 3}, "factors": {"max_p_value": 0.05}}
    service.submit_gate(run_id, "gate_design", "edit", {"plan": edit}, root=root,
                        reason="a tighter screen and three candidates", background=False)
    assert _next_gate(run_id, root) == "gate_design"  # re-opens on the plan in force
    plan = service.results(run_id, root)["ideate"].payload["plan"]
    print(f"\n  The developer's edit: pool of {plan['evaluation']['pool_size']} "
          f"[{plan['sources']['pool_size']}], single-factor p ≤ {plan['factors']['max_p_value']} "
          f"[{plan['sources']['factors.max_p_value']}]")
    service.submit_gate(run_id, "gate_design", "accept", reason="plan confirmed", root=root,
                        background=False)

    # --- 2. single-factor analysis ---------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_candidates"
    model = service.results(run_id, root)["model"].payload
    banner("2. SINGLE-FACTOR ANALYSIS — the library, and the rules that shorten it")
    lib, samples = model["library"], model["samples"]
    print(f"  Samples: {samples['development']} development quarters, {samples['test']} test, "
          f"{samples['holdout']} sealed; {len(samples['validation_blocks'])} forecast origins")
    print(f"  Library: {lib['n_factors']} factors from {lib['n_sources']} columns "
          f"({', '.join(lib['transforms'])})")
    for note in lib["notes"]:
        print(f"    note: {note}")
    for rule in model["screen"]:
        if rule["before"] != rule["after"]:
            print(f"    {rule['rule'].replace('_', ' '):18s} {str(rule['setting'] or ''):6s} "
                  f"{rule['before']:>4d} → {rule['after']:<4d}")
    record = json.loads((root / run_id / "stages/model/single_factor.json").read_text("utf-8"))
    print("  Shortlist (fitted alone: sign, p-value, validation RMSE):")
    for row in (r for r in record["rows"] if r["status"] == "kept"):
        print(f"    {row['name']:34s} {row['sign']}  p={row['p_value']:.4f}  "
              f"rmse={row['validation']:.4f}")
    lost = next(r for r in record["rows"] if r["dropped_by"] == "enforce_sign")
    print(f"  A factor the rules dropped: {lost['name']} — {lost['reason']}")

    # --- 3. multi-factor selection ---------------------------------------------------------
    banner("3. FROM ONE FACTOR TO SEVERAL — each method's path")
    for path in model["paths"]:
        print(f"  {path['method'].replace('_', ' ')} ({path['family']}; criterion "
              f"{path['criterion']}): {path['note']}")
        for s in path["steps"]:
            change = f"{s['change']:+.4f}" if s["change"] is not None else "      "
            print(f"    {s['step']}. {s['action']:5s} {str(s['factor'] or ''):34s} "
                  f"→ {s['criterion']:.4f} {change}  ({s['tried']} tried, "
                  f"{s['rejected_by_constraints']} refused by the constraints)")
    print(f"  {model['n_candidates_tried']} models evaluated in one ledger; the test sample was "
          f"scored {model['test_evaluations']} times during the search")

    # --- 4. the candidate pool -------------------------------------------------------------
    banner("4. THE CANDIDATE POOL — no champion yet")
    for a in model["leaders"][:6]:
        mark = "pool" if a["id"] in model["pool_ids"] else "    "
        print(f"  {mark} {a['id']}  {a['family']:6s} rmse={a['score']:.4f}  "
              f"{', '.join(a['features'])}"
              + (f"   ✗ {a['violations'][0]}" if a["violations"] else ""))
    # Does the third factor earn its place out of sample? Add the forward path's two-factor
    # model: it is in the ledger, so it can join the pool by its id.
    two_factor = model["paths"][0]["steps"][1]["model"]
    pool = [*model["pool_ids"], two_factor]
    service.submit_gate(run_id, "gate_candidates", "edit", {"pool": pool}, root=root,
                        reason="add the two-factor model: does the third factor earn its place?",
                        background=False)
    assert _next_gate(run_id, root) == "gate_candidates"
    print(f"\n  The developer adds {two_factor} from the ledger (the two-factor step of the "
          "forward path).\n  The search is not re-run; the new member is fitted, tested and saved.")
    service.submit_gate(run_id, "gate_candidates", "accept", reason="pool confirmed", root=root,
                        background=False)

    # --- 5. the champion, on outcomes ------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_champion"
    backtest = service.results(run_id, root)["backtest"].payload
    banner(f"5. THE CHAMPION — chosen on the {backtest['evaluation_sample'].replace('_', ' ')} "
           "sample and the scenarios")
    print(f"  {'candidate':9s} {'validation':>10s} {'test rmse':>10s} {'R²':>6s} {'bias':>8s}   "
          "severely adverse Δ")
    for c in backtest["candidates"]:
        o = c["outcomes"]
        severe = next(s["delta"] for s in c["scenarios"] if s["name"] == "severely_adverse")
        mark = "  ← recommended" if c["id"] == backtest["champion_id"] else ""
        print(f"  {c['id']:9s} {c['development_score']:10.4f} {o['rmse']:10.4f} {o['r2']:6.3f} "
              f"{o['bias']:+8.4f}   {severe:+.3f}{mark}")
    print(f"  Outcomes Analyst: {backtest['champion_rationale']}")
    print("  The test sample was the only out-of-sample evidence and it was used to choose: "
          f"sealed holdout = {backtest['sealed_holdout']}")
    service.submit_gate(run_id, "gate_champion", "accept", reason="agreed", root=root,
                        background=False)

    assert _next_gate(run_id, root) == "gate_validation"
    validate = service.results(run_id, root)["validate"]
    banner(f"VALIDATION — {validate.verdict.value}")
    for f in validate.findings:
        print(f"  [{f.severity.value}] {f.message}")
    print(f"\n  Run directory: {root / run_id}")
    print("  Open it in the workbench:  cognos ui --runs-dir " + str(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
