#!/usr/bin/env python3
"""COGNOS worked example — a logistic-regression PD model, from factors to a champion.

A commercial 12-month probability-of-default model on 2,400 obligors observed by vintage. The
same development plan as the loss-forecasting example (ADR-0016), and a different answer at
every step, because the data are different:

  1. THE PLAN. Dated rows of many obligors, enough of them: the latest vintages are held back
     as the test sample, validation is stratified k-fold, and there are rows to spare for a
     SEALED HOLDOUT that nobody chooses on. The developer adds best subset beside stepwise and
     asks every coefficient to be significant.
  2. SINGLE-FACTOR ANALYSIS. Each financial ratio and macro column, and the logarithm of the
     skewed ones, is fitted alone in a logit. Rules cut the list: significance, the expected
     credit sign, the better of two transformations, no two factors too alike.
  3. FROM ONE FACTOR TO SEVERAL. Stepwise adds the factor that lifts validation AUC most and
     drops any that stops earning its place. Best subset checks the path. Probit is fitted on
     the same factor sets, so the link is compared on equal terms.
  4. THE CANDIDATE POOL. Several scorecards within noise of each other on the development
     sample. The developer adds a smaller one from the ledger.
  5. THE CHAMPION, ON OUTCOMES. Each candidate is scored on the out-of-time vintages: Gini, KS,
     calibration, population stability, and the PD under an adverse scenario.
  6. THE SEALED HOLDOUT. Opened once, at validation, for the champion only.

Run:  python examples/logistic_pd/run_demo.py [--save-sample]
(Requires `pip install -e .`. Runs offline with the deterministic agents; LLM agents recommend
on top of the same engine checks, never instead.)

`--save-sample` refreshes examples/logistic_pd/sample_output/ (this walkthrough as text, the
white paper's methodology page, and the single-factor table).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from cognos import service, synth
from cognos.config import CognosConfig

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
LINES: list[str] = []

EXPECTED_SIGNS = {  # the direction each driver should move the PD
    "debt_to_ebitda": "+", "utilization_rate": "+", "unemployment_rate": "+",
    "interest_coverage": "-", "current_ratio": "-", "operating_margin": "-",
    "log_total_assets": "-", "collateral_coverage": "-", "gdp_growth": "-",
    "revenue_growth": "-",
}


def say(text: str = "") -> None:
    LINES.append(text)
    print(text)


def banner(title: str) -> None:
    say("\n" + "=" * 78)
    say(f"  {title}")
    say("=" * 78)


def _config(root: Path) -> CognosConfig:
    csv = root / "cni.csv"
    synth.make_cni_portfolio_dataset(n=2400).to_csv(csv, index=False)
    intent = root / "intent.md"
    design = {"use_case": "origination underwriting", "horizon": "12-month PD",
              "default_definition": "90+ DPD or nonaccrual within 12 months of origination",
              "segment": "C&I middle-market", "interpretability": "required"}
    from cognos import engagement

    intent.write_text(engagement.render_template("new", {
        "objective": "A 12-month probability of default for C&I middle-market obligors at "
                     "origination, as a scorecard a credit officer can read.",
        **design, "data_sources": "Synthetic C&I portfolio (cognos.synth)."},
        name="logistic_pd"), encoding="utf-8")
    return CognosConfig.from_dict({
        "name": "logistic_pd",
        "description": "C&I middle-market 12-month PD — logistic regression (worked example)",
        "task": "classification",
        "data": {"path": str(csv), "target": "default", "datetime_col": "vintage",
                 "drop_columns": ["obligor_id", "dpd_at_outcome", "default_quarter"]},
        "engagement": {"kind": "new", "intent": str(intent)},
        "design": design,
        "metric": {"name": "roc_auc"},
        "search": {"max_candidates": 8, "cv_folds": 5, "model_families": ["logit", "probit"]},
        "factors": {"expected_signs": EXPECTED_SIGNS},
        "stress": {"enabled": True, "scenarios": [
            {"name": "adverse",
             "shocks": {"unemployment_rate": {"add": 3.0}, "gdp_growth": {"add": -2.0}}},
            {"name": "severely_adverse",
             "shocks": {"unemployment_rate": {"add": 6.0}, "gdp_growth": {"set": -4.0}}}]},
        "compliance": {"risk_tier": "high", "jurisdictions": ["US"]},
        "runs_dir": str(root / "runs"),
    })


def _next_gate(run_id: str, root: Path) -> str | None:
    state = service.run_until_idle(run_id, root)
    return next((g for g, s in state.steps.items() if s.status == "awaiting"), None)


def main(save: bool = False) -> int:
    work = Path(tempfile.mkdtemp(prefix="cognos-logit-"))
    cfg = _config(work)
    root = work / "runs"
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
    say(f"  Data: {shape['n_rows']:,} rows over {shape['n_periods']} vintages → '{shape['shape']}'")
    for field, label in (("test", "Test sample"), ("validation", "Validation"),
                         ("holdout_fraction", "Sealed holdout"), ("methods", "Selection"),
                         ("criterion", "Step criterion")):
        value = (plan["evaluation"] | plan["selection"])[field]
        say(f"  {label:15s} {str(value):22s} [{plan['sources'].get(field, '')}] "
            f"{plan['reasons'].get(field, '')}")
    say(f"  The data also allow: validation {plan['options']['validation']}, "
        f"test {plan['options']['test']}")
    edit = {"selection": {"methods": ["stepwise", "best_subset"], "max_p_value": 0.05,
                          "max_factors": 5}}
    service.submit_gate(run_id, "gate_design", "edit", {"plan": edit}, root=root,
                        reason="check stepwise against best subset; every coefficient significant",
                        background=False)
    assert _next_gate(run_id, root) == "gate_design"  # re-opens on the plan in force
    plan = service.results(run_id, root)["ideate"].payload["plan"]
    say(f"\n  The developer's edit: {' + '.join(plan['selection']['methods'])} "
        f"[{plan['sources']['methods']}], every coefficient p ≤ {plan['selection']['max_p_value']} "
        f"[{plan['sources']['max_p_value']}], at most {plan['selection']['max_factors']} factors")
    service.submit_gate(run_id, "gate_design", "accept", reason="plan confirmed", root=root,
                        background=False)

    # --- 2. single-factor analysis ---------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_candidates"
    model = service.results(run_id, root)["model"].payload
    banner("2. SINGLE-FACTOR ANALYSIS — each factor alone in a logit, and the rules that cut the list")
    lib, samples = model["library"], model["samples"]
    say(f"  Samples: {samples['development']:,} development obligors, {samples['test']:,} test "
        f"(later vintages), {samples['holdout']:,} sealed (the latest); "
        f"{len(samples['validation_blocks'])} stratified folds")
    say(f"  Library: {lib['n_factors']} factors from {lib['n_sources']} columns "
        f"({', '.join(lib['transforms']) or 'as explored'})")
    for rule in model["screen"]:
        if rule["before"] != rule["after"]:
            say(f"    {rule['rule'].replace('_', ' '):18s} {str(rule['setting'] or ''):6s} "
                f"{rule['before']:>4d} → {rule['after']:<4d}")
    record = json.loads((root / run_id / "stages/model/single_factor.json").read_text("utf-8"))
    say("  Shortlist (fitted alone: sign, p-value, validation AUC):")
    for row in (r for r in record["rows"] if r["status"] == "kept"):
        say(f"    {row['name']:26s} {row['sign']:3s} p={row['p_value']:.4f}  auc={row['validation']:.4f}")
    say("  Dropped, with the rule and the reason:")
    for row in [r for r in record["rows"] if r["status"] == "dropped"][:6]:
        say(f"    {row['name']:26s} {row['dropped_by'].replace('_', ' '):16s} {row['reason']}")

    # --- 3. multi-factor selection ---------------------------------------------------------
    banner("3. FROM ONE FACTOR TO SEVERAL — each method's path")
    for path in model["paths"]:
        say(f"  {path['method'].replace('_', ' ')} ({path['family']}; criterion "
            f"{path['criterion']}): {path['note']}")
        for s in path["steps"]:
            change = f"{s['change']:+.4f}" if s["change"] is not None else "       "
            say(f"    {s['step']}. {s['action']:5s} {str(s['factor'] or ''):24s} → "
                f"{s['criterion']:.4f} {change}  ({s['tried']} tried, "
                f"{s['rejected_by_constraints']} refused by the constraints)")
    say(f"  {model['n_candidates_tried']} models evaluated in one ledger; the test sample was "
        f"scored {model['test_evaluations']} times during the search")

    # --- 4. the candidate pool -------------------------------------------------------------
    banner("4. THE CANDIDATE POOL — no champion yet")
    for a in model["leaders"][:7]:
        mark = "pool" if a["id"] in model["pool_ids"] else "    "
        say(f"  {mark} {a['id']}  {a['family']:6s} auc={a['score']:.4f}  {', '.join(a['features'])}"
            + (f"   ✗ {a['violations'][0]}" if a["violations"] else ""))
    stepwise = next(p for p in model["paths"] if p["method"] == "stepwise")
    smaller = next((s["model"] for s in stepwise["steps"]
                    if s["n_factors"] == 2 and s["model"] not in model["pool_ids"]), None)
    if smaller:
        pool = [*model["pool_ids"], smaller]
        service.submit_gate(run_id, "gate_candidates", "edit", {"pool": pool}, root=root,
                            reason="add the two-factor scorecard: how much do the others buy?",
                            background=False)
        assert _next_gate(run_id, root) == "gate_candidates"
        say(f"\n  The developer adds {smaller} from the ledger (the two-factor step of the "
            "stepwise path).\n  The search is not re-run; the new member is fitted, tested and saved.")
    service.submit_gate(run_id, "gate_candidates", "accept", reason="pool confirmed", root=root,
                        background=False)

    # --- 5. the champion, on outcomes ------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_champion"
    backtest = service.results(run_id, root)["backtest"].payload
    banner(f"5. THE CHAMPION — chosen on the {backtest['evaluation_sample'].replace('_', ' ')} "
           "vintages and the scenarios")
    say(f"  {'candidate':9s} {'factors':>7s} {'validation':>10s} {'test AUC':>9s} {'Gini':>6s} "
        f"{'KS':>6s} {'calib.':>7s} {'PSI':>6s}   severely adverse ΔPD")
    for c in backtest["candidates"]:
        o = c["outcomes"]
        severe = next(s["delta"] for s in c["scenarios"] if s["name"] == "severely_adverse")
        mark = "  ← recommended" if c["id"] == backtest["champion_id"] else ""
        say(f"  {c['id']:9s} {c['n_features']:7d} {c['development_score']:10.4f} "
            f"{c['metrics']['roc_auc']:9.4f} {o['gini']:6.3f} {o['ks']:6.3f} "
            f"{o['expected_calibration_error']:7.4f} {o['psi']:6.3f}   {severe:+.4f}{mark}")
    say(f"  Outcomes Analyst: {backtest['champion_rationale']}")
    say(f"  The test sample has been looked at {backtest['test_evaluations']} times. "
        f"Sealed holdout kept: {backtest['sealed_holdout']} (not opened yet)")
    service.submit_gate(run_id, "gate_champion", "accept", reason="agreed", root=root,
                        background=False)

    # --- 6. the sealed holdout ---------------------------------------------------------------
    assert _next_gate(run_id, root) == "gate_validation"
    validate = service.results(run_id, root)["validate"]
    banner(f"6. THE SEALED HOLDOUT — opened once, at validation ({validate.verdict.value})")
    vp = validate.payload
    champion = next(c for c in backtest["candidates"] if c["id"] == backtest["champion_id"])
    say(f"  Champion {vp['champion_id']}: validation AUC {champion['development_score']:.4f} → "
        f"test AUC {champion['metrics']['roc_auc']:.4f} → sealed-holdout AUC "
        f"{vp['holdout_metric']:.4f}")
    say(f"  Holdout scored for: {vp['holdout_scored']} ({vp['holdout_evaluations']} evaluation)")
    for f in validate.findings:
        say(f"  [{f.severity.value}] {f.message}")
    say(f"\n  Run directory: {root / run_id}")
    say("  Open it in the workbench:  cognos ui --runs-dir " + str(root))

    if save:
        out = HERE / "sample_output"
        out.mkdir(exist_ok=True)
        for gate in ("gate_validation",):  # the white paper is written after validation
            service.submit_gate(run_id, gate, "accept", root=root, background=False,
                                reason="findings read; accepted for the worked example")
        service.run_until_idle(run_id, root)
        text = "\n".join(LINES).replace(str(work), "<workdir>").replace(run_id, "<run_id>")
        (out / "walkthrough.txt").write_text(text + "\n", encoding="utf-8")
        shutil.copy(root / run_id / "docs" / "methodology.md", out / "methodology.md")
        shutil.copy(root / run_id / "stages" / "model" / "single_factor.json",
                    out / "single_factor.json")
        print(f"\n  Sample output refreshed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(save="--save-sample" in sys.argv))
