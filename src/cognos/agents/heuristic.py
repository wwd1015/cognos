"""Heuristic agents — the deterministic offline/demo path.

Each function is a pure function of the *same* slice an LLM agent receives and returns a dict that
satisfies the agent's contract and the engine's checks. They encode the v0.5 deterministic rules
(so an offline run reproduces v0.5 results) plus a conservative response to challenges: a challenge
that names a column makes the data analyst exclude it; any other challenge is acknowledged.
"""

from __future__ import annotations

import re
from typing import Any

# Expected sign on PD for common commercial-credit drivers (cognos-model playbook), matched on
# underscore tokens of the feature name.
SIGN_PRIORS: list[tuple[tuple[str, ...], str]] = [
    (("leverage", "debt_to_ebitda", "debt_ebitda", "debt"), "+"),
    (("utilization", "utilisation", "revolver"), "+"),
    (("unemployment", "dpd", "delinquency"), "+"),
    (("coverage", "icr", "dscr"), "-"),
    (("liquidity", "current_ratio", "quick_ratio", "cash"), "-"),
    (("margin", "profitability", "roa", "roe", "ebitda_margin"), "-"),
    (("size", "assets", "log_assets", "revenue"), "-"),
    (("collateral", "ltv_coverage"), "-"),
    (("gdp", "gdp_growth"), "-"),
    (("merton_dd", "distance_to_default", "dd"), "-"),
    (("migration_pd", "pd"), "+"),
]


def _tokens(name: str) -> list[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return toks + ["_".join(toks[i:i + 2]) for i in range(len(toks) - 1)]


def sign_prior(feature: str) -> str:
    toks = set(_tokens(feature))
    for keys, sign in SIGN_PRIORS:
        if toks & set(keys):
            return sign
    return "none"


def _ack(sl: dict[str, Any], changed: set[str] | None = None, note: str = "") -> list[dict]:
    changed = changed or set()
    out = []
    for c in sl.get("challenges", []):
        did = c["id"] in changed
        out.append({
            "challenge_id": c["id"],
            "response": (note or ("Recommendation revised to address this challenge." if did else
                                  "Reviewed; the deterministic agent found no rule-based change to "
                                  "make — a human or an LLM agent should weigh this challenge.")),
            "changed_recommendation": did,
        })
    return out


def _fact(sl: dict[str, Any], key: str) -> list[str]:
    return [key] if key in sl.get("facts", {}) else []


# --- explore -----------------------------------------------------------------------------
def data_analyst(sl: dict[str, Any]) -> dict[str, Any]:
    features = list(sl.get("features", []))
    suspects = list(sl.get("leakage_suspects", []))
    current = set(sl.get("current_exclusions", []))
    named: dict[str, str] = {}  # column -> challenge id that names it
    for c in sl.get("challenges", []):
        toks = set(re.split(r"[^A-Za-z0-9_]+", c["message"] + " " + (c.get("remedy") or "")))
        for col in features:
            if col in toks:
                named[col] = c["id"]
    decisions: list[dict] = []
    for col in suspects:
        if col in named or col in current:
            decisions.append({
                "column": col, "decision": "exclude",
                "reason": ("Excluded as requested in challenge " + named[col] if col in named else
                           "Excluded by an earlier human decision at the data gate."),
                "evidence": _fact(sl, f"explore.corr.{col}"),
            })
        else:
            decisions.append({
                "column": col, "decision": "keep",
                "reason": ("Near-deterministic relationship with the target, but whether the field "
                           "exists at prediction time cannot be confirmed from the data alone — kept "
                           "pending sponsor attestation; validation BLOCKs if the champion uses a "
                           "confirmed leak."),
                "evidence": _fact(sl, f"explore.corr.{col}"),
            })
    for col in sorted((current | set(named)) - set(suspects)):
        if col in features or col in current:
            decisions.append({"column": col, "decision": "exclude",
                              "reason": "Excluded by a human decision or challenge.",
                              "evidence": []})
    quality = [
        {"statement": f"Column '{col}' is {frac:.0%} missing; treat imputation as a modeling "
                      "assumption to document.",
         "evidence": _fact(sl, f"explore.missing.{col}"), "confidence": "high"}
        for col, frac in sorted((sl.get("missing_high") or {}).items())
    ]
    kept = [d["column"] for d in decisions if d["decision"] == "keep"]
    excluded = [d["column"] for d in decisions if d["decision"] == "exclude"]
    summary = (f"Profiled {len(features)} candidate features. "
               + (f"Recommend excluding {', '.join(excluded)}. " if excluded else "")
               + (f"{len(kept)} leakage suspect(s) kept pending sponsor confirmation of the "
                  "information set." if kept else "No unresolved leakage suspects."))
    return {
        "summary": summary,
        "uncertainties": [f"Whether '{c}' is known at prediction time." for c in kept],
        "responses_to_challenges": _ack(sl, {named[c] for c in named}),
        "column_decisions": decisions,
        "data_quality": quality,
        "questions_for_sponsor": [f"Is '{c}' available at the time a prediction is made, or is it "
                                  "recorded after the outcome?" for c in kept],
    }


# --- ideate ------------------------------------------------------------------------------
_ROLE_TO_DECISION = {"primary": "primary", "candidate": "candidate", "challenger": "challenger",
                     "available": "rejected", "rejected": "rejected"}


def design_lead(sl: dict[str, Any]) -> dict[str, Any]:
    frameworks = [{
        "framework": f["framework"],
        "decision": _ROLE_TO_DECISION.get(f.get("role", "rejected"), "rejected"),
        "reason": f.get("reason", ""),
    } for f in sl.get("framework_assessment", [])]
    slate = [{k: h[k] for k in ("family", "feature_strategy", "role", "priority", "rationale")}
             for h in sl.get("default_slate", [])]
    primary = next((f["framework"] for f in frameworks if f["decision"] == "primary"), "n/a")
    return {
        "summary": (f"Primary framework: {primary}. Ranked {len(slate)} engine-fittable "
                    f"specifications, interpretable families first when interpretability is "
                    f"{sl.get('interpretability', 'required')}."),
        "uncertainties": [q["question"] for q in sl.get("open_questions", [])][:3],
        "responses_to_challenges": _ack(sl),
        "frameworks": frameworks,
        "slate": slate,
        "transforms": [],
        "sponsor_questions": [],
    }


# --- model -------------------------------------------------------------------------------
def modeler(sl: dict[str, Any]) -> dict[str, Any]:
    admissible = sl.get("admissible_set", [])
    interp_required = sl.get("interpretability") == "required"
    pick = next((c for c in admissible if not (interp_required and c.get("role") == "challenger")),
                admissible[0])
    checks, concerns = [], []
    for s in pick.get("coefficient_signs", []):
        exp = sign_prior(s["feature"])
        obs = s["sign"]
        if exp == "none" or obs in ("0", "n/a"):
            assessment = "no_prior" if exp == "none" else "not_applicable"
        else:
            assessment = "consistent" if exp == obs else "wrong_sign"
        checks.append({"feature": s["feature"], "expected": exp, "observed": obs,
                       "assessment": assessment})
        if assessment == "wrong_sign":
            concerns.append({"statement": f"'{s['feature']}' has sign {obs}, against the {exp} "
                                          "prior — check collinearity or segment mixing.",
                             "evidence": [], "confidence": "medium"})
    return {
        "summary": (f"Recommend {pick['id']} ({pick['label']}): the ratchet's best "
                    "parsimony-adjusted candidate that satisfies the interpretability constraint."),
        "uncertainties": ["Choice made on cross-validation only; the sealed holdout is scored after "
                          "the choice."],
        "responses_to_challenges": _ack(sl),
        "champion": pick["id"],
        "rationale": (f"Best cross-validated {sl.get('metric', 'metric')} in the admissible set "
                      f"({len(admissible)} candidate(s) within one standard error of the best)."),
        "sign_checks": checks,
        "concerns": concerns,
    }


def experiment(sl: dict[str, Any]) -> dict[str, Any]:
    return {"stop": True, "rationale": "The deterministic agent does not propose experiments."}


# --- backtest ----------------------------------------------------------------------------
def outcomes_analyst(sl: dict[str, Any]) -> dict[str, Any]:
    f = sl.get("facts", {})
    gini, ks = f.get("backtest.gini"), f.get("backtest.ks")
    ece, psi = f.get("backtest.expected_calibration_error"), f.get("backtest.psi")
    sample = f.get("backtest.evaluation_sample", "holdout")
    if gini is None:
        disc = "Outcomes analysis does not apply (continuous target); see the out-of-sample metric."
    else:
        band = "strong" if gini >= 0.5 else ("adequate" if gini >= 0.3 else "weak")
        disc = f"Discrimination on the {sample} sample is {band} (Gini {gini}, KS {ks})."
    cal = ("Calibration within tolerance." if ece is not None and ece <= 0.10 else
           "Calibration gap exceeds 0.10 — recalibrate before use." if ece is not None else
           "Calibration not assessed.")
    stab = ("Population stable dev→evaluation (PSI below 0.10)." if psi is not None and psi < 0.10 else
            "Moderate population shift (PSI 0.10–0.25)." if psi is not None and psi <= 0.25 else
            "Significant population shift (PSI above 0.25)." if psi is not None else
            "Stability not assessed.")
    return {
        "summary": f"{disc} {cal}",
        "uncertainties": [],
        "responses_to_challenges": _ack(sl),
        "discrimination": disc,
        "calibration": cal,
        "stability": stab,
        "findings": [],
    }


# --- validate ----------------------------------------------------------------------------
def validator(sl: dict[str, Any]) -> dict[str, Any]:
    verdict = sl.get("engine_verdict", "PASS")
    engine_findings = sl.get("engine_findings", [])
    rec = {"PASS": "approve", "WARN": "approve_with_conditions"}.get(verdict, "send_back")
    return {
        "summary": (f"Engine rubric verdict {verdict} with {len(engine_findings)} finding(s); "
                    "the deterministic validator adds no findings of its own."),
        "uncertainties": [],
        "responses_to_challenges": _ack(sl),
        "assessment": ("The rubric re-derives leakage, overfitting, stability, diagnostics and "
                       "significance from the artifacts; see the engine findings."),
        "findings": [],
        "recommendation": rec,
        "conditions": [f["message"] for f in engine_findings
                       if f.get("severity") in ("MEDIUM", "HIGH")][:5],
    }


# --- comply ------------------------------------------------------------------------------
def risk_analyst(sl: dict[str, Any]) -> dict[str, Any]:
    steps = sl.get("outstanding_human_steps", [])
    sr = {k: v for k, v in sl.get("facts", {}).items() if k.startswith("comply.sr11_7.")}
    actions = [{"statement": s, "evidence": list(sr)[:1], "confidence": "high"} for s in steps[:4]]
    return {
        "summary": f"Development evidence is organised; {len(steps)} human step(s) remain.",
        "uncertainties": [],
        "responses_to_challenges": _ack(sl),
        "readiness": "ready_with_actions" if steps else "ready",
        "narrative": ("COGNOS assembles SR 11-7 development evidence; it does not adjudicate "
                      "compliance. Independent validation, a monitoring plan and governance "
                      "approval remain human responsibilities."),
        "priority_actions": actions,
    }


# --- document ----------------------------------------------------------------------------
def writer(sl: dict[str, Any]) -> dict[str, Any]:
    f = sl.get("facts", {})

    def ph(key: str, fallback: str = "n/a") -> str:
        return "{{fact:" + key + "}}" if key in f else fallback

    metric = f.get("model.metric", "the primary metric")
    exec_summary = (
        f"The champion is a {ph('model.champion_family', 'model')} specification selected from "
        f"{ph('model.n_candidates_tried')} evaluated candidates. Cross-validated {metric} is "
        f"{ph('model.cv_mean')} and the sealed-holdout {metric} is {ph('model.holdout_metric')}. "
        f"Independent validation returned {ph('validate.verdict')}."
    )
    frameworks = sl.get("framework_choices", [])
    rejected = [fw for fw in frameworks if fw.get("decision") == "rejected"]
    alt = ("\n".join(f"- **{fw['framework']}** — {fw.get('reason', '')}" for fw in rejected)
           or "No alternative framework was rejected.")
    decisions = sl.get("decisions", [])
    return {
        "summary": "Drafted the narrative sections from recorded facts and decisions.",
        "uncertainties": [],
        "responses_to_challenges": _ack(sl),
        "sections": [
            {"section": "executive_summary", "markdown": exec_summary},
            {"section": "methodology_rationale", "markdown": (
                "Candidates were searched with leakage-safe cross-validation under a fixed budget; "
                "the champion was chosen from the admissible set (within one standard error of the "
                "best) before the sealed holdout was scored.")},
            {"section": "alternatives_considered", "markdown": alt},
            {"section": "limitations", "markdown": (
                "The holdout estimate is single-shot; performance may drift with the population. "
                f"Leakage suspects flagged at exploration: {ph('explore.leakage_suspects', 'none')}.")},
            {"section": "use_and_monitoring", "markdown": (
                f"{len(decisions)} human decision(s) were recorded at the review gates. A production "
                "monitoring plan (PSI and calibration triggers, cadence, owner) is required before "
                "use.")},
        ],
    }


AGENTS = {
    "data_analyst": data_analyst,
    "design_lead": design_lead,
    "modeler": modeler,
    "experiment": experiment,
    "outcomes_analyst": outcomes_analyst,
    "validator": validator,
    "risk_analyst": risk_analyst,
    "writer": writer,
}


def recommend(agent: str, sl: dict[str, Any]) -> dict[str, Any]:
    return AGENTS[agent](sl)
