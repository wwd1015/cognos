"""Development modes, end to end and offline: a new model development, then an update of it.

    python examples/model_update/run_demo.py

1. The sponsor fills in the business intent template, leaving two sections empty.
2. A new model development starts from that document. The Intake Analyst reads it and asks about
   what is missing; the answers go back in, and the intent is confirmed.
3. A year later the sponsor files a model update request on the same template. The second run
   starts from the first run (the existing model), its scoring code, and the request.

Everything runs with the deterministic agents: no key, no network.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("COGNOS_PROVIDER", "heuristic")

from cognos import engagement, service, synth  # noqa: E402

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_output"

INTENT = {
    "objective": "Rank middle-market commercial borrowers by default risk so that underwriters "
                 "price and size new facilities consistently.",
    "use_case": "Origination underwriting",
    "segment": "C&I middle-market",
    "interpretability": "Required: a credit officer must be able to explain a decline.",
    "success_criteria": "Rank-ordering no worse than the current expert scorecard on an "
                        "out-of-time sample.",
    # "default_definition" and "horizon" are left empty on purpose: the interview asks.
}
ANSWERS = {"design-default_definition": "90+ days past due or nonaccrual",
           "design-horizon": "12 months from origination"}
REQUEST = {
    **INTENT,
    "default_definition": ANSWERS["design-default_definition"],
    "horizon": ANSWERS["design-horizon"],
    "prior_model": "Commercial PD v1 (the first run of this demo), in use for one year.",
    "update_reason": "The annual review found the model under-predicting default in the two "
                     "latest vintages.",
    "requested_changes": "- Refresh the development data through the latest vintage\n"
                         "- Re-estimate the coefficients on the refreshed sample\n"
                         "- Close validation finding V-12 on calibration",
    "must_not_change": "The segment, the default definition and the 12-month horizon.",
}


def _accept_rest(run_id: str, root: Path) -> None:
    """Accept every later gate as the agents recommend (the demo is about the first one)."""
    for _ in range(20):
        state = service.run_until_idle(run_id, root)
        waiting = [s for s, v in state.steps.items() if v.status == "awaiting"]
        if not waiting:
            return
        action = "approve" if waiting[0] == "gate_signoff" else "accept"
        service.submit_gate(run_id, waiting[0], action, reason="as recommended", root=root,
                            background=False)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="cognos_update_"))
    root = work / "runs"
    data = work / "commercial.csv"
    synth.GENERATORS["commercial"](n=1500).to_csv(data, index=False)
    profile = {
        "name": "commercial_pd", "description": "", "task": "classification",
        "data": {"path": str(data), "target": "default", "datetime_col": "vintage"},
        "search": {"max_candidates": 10, "cv_folds": 3},
        "compliance": {"risk_tier": "high", "jurisdictions": ["US"]},
        "runs_dir": str(root),
    }

    # --- 1. the sponsor's intent document ---------------------------------------------------
    intent = work / "business_intent.md"
    intent.write_text(engagement.render_template("new", INTENT, "Commercial PD"), encoding="utf-8")

    # --- 2. new model development: intake interviews, then the run proceeds -------------------
    v1 = service.create_run(profile, mode="interactive", root=root, engagement={"intent": str(intent)})
    state = service.run_until_idle(v1, root)
    brief = service.results(v1, root)["intake"].payload
    print(f"[new] intent is '{brief['clarity'].replace('_', ' ')}'; the Intake Analyst asks:")
    for gap in state.open_gaps():
        print(f"   ? [{gap.id}] {gap.question}")
    service.submit_gate(v1, "gate_intent", "edit", {"answers": ANSWERS}, root=root, background=False)
    state = service.run_until_idle(v1, root)
    brief = service.results(v1, root)["intake"].payload
    print(f"[new] after the answers: intent is '{brief['clarity']}', "
          f"{brief['n_blocking']} blocking question(s)")
    service.submit_gate(v1, "gate_intent", "accept", root=root, background=False)
    _accept_rest(v1, root)
    model = service.results(v1, root)["model"].payload
    print(f"[new] run {v1}: {service.state(v1, root).status}, champion {model['champion_label']}")
    shutil.copy(root / v1 / "stages/intake/brief.md", SAMPLE / "new_brief.md")

    # --- 3. model update: the request, the existing model's code, and the earlier run -----------
    request = work / "update_request.md"
    request.write_text(engagement.render_template("update", REQUEST, "Commercial PD"), encoding="utf-8")
    code = work / "score_v1.py"
    code.write_text("# Deployment scorer, Commercial PD v1\n"
                    f"FEATURES = {model['champion']['features']!r}\n", encoding="utf-8")
    v2 = service.create_run(profile, mode="interactive", root=root, engagement={
        "kind": "update", "intent": str(request), "prior_artifacts": [str(code)], "prior_run": v1})
    service.run_until_idle(v2, root)
    update = service.results(v2, root)["intake"].payload["update"]
    print(f"[update] scope: {update['scope'].replace('_', ' ')}; existing family: "
          f"{update['incumbent_family']}")
    for item in update["change_items"]:
        print(f"   - {item['change']}  ({item['type'].replace('_', ' ')}, from {item['affects']})")
    service.submit_gate(v2, "gate_intent", "accept", root=root, background=False)
    _accept_rest(v2, root)
    first = service.results(v2, root)["ideate"].payload["hypotheses"][0]
    print(f"[update] run {v2}: {service.state(v2, root).status}; first on the slate: "
          f"{first['family']} / {first['feature_strategy']}")
    shutil.copy(root / v2 / "stages/intake/brief.md", SAMPLE / "update_brief.md")
    shutil.copy(root / v2 / "docs/engagement.md", SAMPLE / "engagement.md")
    print(f"\nRuns: {root}\nSample output: {SAMPLE}")


if __name__ == "__main__":
    SAMPLE.mkdir(exist_ok=True)
    main()
