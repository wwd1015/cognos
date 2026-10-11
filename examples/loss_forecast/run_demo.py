#!/usr/bin/env python3
"""COGNOS worked example — a quarterly loss-forecasting model, from factors to a champion.

A portfolio net charge-off rate on macro drivers, one row per quarter. The series is short and
the drivers act through lags and changes, so the build is the classic one (ADR-0016):

  1. THE PLAN. The engine reads the shape of the data (one row per period) and the intent and
     recommends how to split it: the latest quarters held back as the test sample, a rolling
     forecast origin for validation, no sealed third sample (too few quarters, and it says so).
     The developer sets their own criteria at the design gate: here the primary metric gets a
     second line, and single factors must be significant at 5%.
  2. SINGLE-FACTOR ANALYSIS. Eight explored columns become about two hundred factors (lags,
     changes, year-on-year moves, moving averages). Each is fitted alone, and the developer's
     rules cut the library to a shortlist; every dropped factor keeps the rule and the reason.
  3. FROM ONE FACTOR TO SEVERAL. One method, the one the plan names: forward selection starts
     from the best single factor and adds one at a time while the validation error falls and
     the model stays significant, correctly signed and free of near-duplicate factors. Every
     model evaluated lands in one ranked ledger, each with a name a person can say.
  4. THE CANDIDATE POOL. The search does not end on a champion. The developer reviews the
     leaders, adds more than the engine proposed, and gives one a name of their own.
  5. THE CHAMPION, ON OUTCOMES. Every pool member forecasts the held-back quarters and each
     scenario. Models that ranked together part ways here, and the choice is made on that page.

Run:  python examples/loss_forecast/run_demo.py [--save-sample]
(Requires `pip install -e .`. Runs offline with the deterministic agents; LLM agents recommend
on top of the same engine checks, never instead.)

`--save-sample` refreshes examples/loss_forecast/sample_output/ (this walkthrough as text, the
white paper's methodology page, and the single-factor table).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from cognos import service

warnings.filterwarnings("ignore")


HERE = Path(__file__).parent
LINES: list[str] = []


def say(text: str = "") -> None:
    LINES.append(text)
    print(text)


def banner(title: str) -> None:
    say("\n" + "=" * 78)
    say(f"  {title}")
    say("=" * 78)


def _next_gate(run_id: str, root: Path) -> str | None:
    state = service.run_until_idle(run_id, root)
    return next((g for g, s in state.steps.items() if s.status == "awaiting"), None)


def main(save: bool = False) -> int:
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
    say(f"  Data: {shape['n_rows']} rows, one per period ({shape['period']} a year) → "
          f"'{shape['shape']}'")
    for field, label in (("test", "Test sample"), ("validation", "Validation"),
                         ("holdout_fraction", "Sealed holdout"), ("folds", "Forecast origins"),
                         ("methods", "Selection"), ("criterion", "Step criterion")):
        value = (plan["evaluation"] | plan["selection"])[field]
        say(f"  {label:17s} {str(value):28s} [{plan['sources'].get(field, '')}] "
              f"{plan['reasons'].get(field, '')}")
    say(f"  The data also allow: validation {plan['options']['validation']}, "
          f"test {plan['options']['test']}")
    say("  Primary metric    " + ", then ".join(
        f"{r['metric']} ({r['sample']})" for r in plan["evaluation"]["rank_by"])
        + f"   [{plan['sources'].get('rank_by', '')}]")
    say(f"  Metrics to rank on: {plan['options']['metric']}; samples: {plan['options']['sample']}")
    edit = {"evaluation": {"rank_by": [{"metric": "rmse", "sample": "validation"},
                                       {"metric": "r2", "sample": "in_sample"}]},
            "factors": {"max_p_value": 0.05}}
    service.submit_gate(run_id, "gate_design", "edit", {"plan": edit}, root=root,
                        reason="rank on validation RMSE then in-sample fit; a tighter screen",
                        background=False)
    assert _next_gate(run_id, root) == "gate_design"  # re-opens on the plan in force
    plan = service.results(run_id, root)["ideate"].payload["plan"]
    say("\n  The developer's criteria: rank by " + ", then ".join(
        f"{r['metric']} ({r['sample']})" for r in plan["evaluation"]["rank_by"])
        + f" [{plan['sources']['rank_by']}]; single-factor p ≤ {plan['factors']['max_p_value']} "
        f"[{plan['sources']['factors.max_p_value']}]")
    service.submit_gate(run_id, "gate_design", "accept", reason="plan confirmed", root=root,
                        background=False)

    # --- 2. single-factor analysis ---------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_candidates"
    model = service.results(run_id, root)["model"].payload
    banner("2. SINGLE-FACTOR ANALYSIS — the library, and the rules that shorten it")
    lib, samples = model["library"], model["samples"]
    say(f"  Samples: {samples['development']} development quarters, {samples['test']} test, "
          f"{samples['holdout']} sealed; {len(samples['validation_blocks'])} forecast origins")
    say(f"  Library: {lib['n_factors']} factors from {lib['n_sources']} columns "
          f"({', '.join(lib['transforms'])})")
    for note in lib["notes"]:
        say(f"    note: {note}")
    for rule in model["screen"]:
        if rule["before"] != rule["after"]:
            say(f"    {rule['rule'].replace('_', ' '):18s} {str(rule['setting'] or ''):6s} "
                  f"{rule['before']:>4d} → {rule['after']:<4d}")
    record = json.loads((root / run_id / "stages/model/single_factor.json").read_text("utf-8"))
    say("  Shortlist (fitted alone: sign, p-value, validation RMSE):")
    for row in (r for r in record["rows"] if r["status"] == "kept"):
        say(f"    {row['name']:34s} {row['sign']}  p={row['p_value']:.4f}  "
              f"rmse={row['validation']:.4f}")
    lost = next(r for r in record["rows"] if r["dropped_by"] == "enforce_sign")
    say(f"  A factor the rules dropped: {lost['name']} — {lost['reason']}")

    # --- 3. multi-factor selection ---------------------------------------------------------
    banner("3. FROM ONE FACTOR TO SEVERAL — each method's path")
    for path in model["paths"]:
        say(f"  {path['method'].replace('_', ' ')} ({path['family']}; criterion "
              f"{path['criterion']}): {path['note']}")
        for s in path["steps"]:
            change = f"{s['change']:+.4f}" if s["change"] is not None else "      "
            say(f"    {s['step']}. {s['action']:5s} {str(s['factor'] or ''):34s} "
                  f"→ {s['criterion']:.4f} {change}  ({s['tried']} tried, "
                  f"{s['rejected_by_constraints']} refused by the constraints)")
    say(f"  {model['n_candidates_tried']} models evaluated in one ledger; the test sample was "
          f"scored {model['test_evaluations']} times during the search")

    # --- 4. the candidate pool -------------------------------------------------------------
    banner("4. THE CANDIDATE POOL — no champion yet")
    for a in model["leaders"][:6]:
        mark = "pool" if a["id"] in model["pool_ids"] else "    "
        say(f"  {mark} {a['name']:12s} {a['id']}  rmse={a['score']:.4f}  "
              f"{', '.join(a['features'])}"
              + (f"   ✗ {a['violations'][0]}" if a["violations"] else ""))
    # More than the engine proposed: the forward path's two-factor model (does the third
    # factor earn its place out of sample?) and the next leader. One gets the team's own name.
    two_factor = model["paths"][0]["steps"][1]["model"]
    another = next(a["id"] for a in model["leaders"] if a["id"] not in model["pool_ids"]
                   and a["id"] != two_factor and not a["violations"])
    pool = list(dict.fromkeys([*model["pool_ids"], two_factor, another]))
    lead = model["pool_ids"][0]
    service.submit_gate(run_id, "gate_candidates", "edit",
                        {"pool": pool, "names": {two_factor: "Two-driver base"}}, root=root,
                        reason="add the two-factor model and one more leader", background=False)
    assert _next_gate(run_id, root) == "gate_candidates"
    model = service.results(run_id, root)["model"].payload
    say(f"\n  The developer widens the pool to {len(pool)}: "
        + ", ".join(model["names"][i] for i in pool) + ".")
    say(f"  “Two-driver base” is the developer's name for {two_factor}; "
        f"{model['names'][lead]} and the rest keep theirs.")
    say("  The search is not re-run; the new members are fitted, tested and saved.")
    service.submit_gate(run_id, "gate_candidates", "accept", reason="pool confirmed", root=root,
                        background=False)

    # --- 5. the champion, on outcomes ------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_champion"
    backtest = service.results(run_id, root)["backtest"].payload
    banner(f"5. THE CHAMPION — chosen on the {backtest['evaluation_sample'].replace('_', ' ')} "
           "sample and the scenarios")
    say(f"  {'candidate':16s} {'validation':>10s} {'test rmse':>10s} {'R²':>6s} {'bias':>8s}   "
          "severely adverse Δ")
    for c in backtest["candidates"]:
        o = c["outcomes"]
        severe = next(s["delta"] for s in c["scenarios"] if s["name"] == "severely_adverse")
        mark = "  ← recommended" if c["id"] == backtest["champion_id"] else ""
        say(f"  {c['name']:16s} {c['development_score']:10.4f} {o['rmse']:10.4f} {o['r2']:6.3f} "
              f"{o['bias']:+8.4f}   {severe:+.3f}{mark}")
    say(f"  Outcomes Analyst: {backtest['champion_rationale']}")
    say("  The test sample was the only out-of-sample evidence and it was used to choose: "
          f"sealed holdout = {backtest['sealed_holdout']}")
    service.submit_gate(run_id, "gate_champion", "accept", reason="agreed", root=root,
                        background=False)

    assert _next_gate(run_id, root) == "gate_validation"
    validate = service.results(run_id, root)["validate"]
    banner(f"VALIDATION — {validate.verdict.value}")
    for f in validate.findings:
        say(f"  [{f.severity.value}] {f.message}")
    say(f"\n  Run directory: {root / run_id}")
    say("  Open it in the workbench:  cognos ui --runs-dir " + str(root))
    if save:
        out = HERE / "sample_output"
        out.mkdir(exist_ok=True)
        service.submit_gate(run_id, "gate_validation", "accept", root=root, background=False,
                            reason="findings read; accepted for the worked example")
        service.run_until_idle(run_id, root)  # the white paper is written after validation
        text = "\n".join(LINES).replace(str(root), "<workdir>").replace(run_id, "<run_id>")
        (out / "walkthrough.txt").write_text(text + "\n", encoding="utf-8")
        shutil.copy(root / run_id / "docs" / "methodology.md", out / "methodology.md")
        shutil.copy(root / run_id / "stages" / "model" / "single_factor.json",
                    out / "single_factor.json")
        print(f"\n  Sample output refreshed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(save="--save-sample" in sys.argv))
