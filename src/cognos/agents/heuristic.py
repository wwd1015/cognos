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


# --- intake ------------------------------------------------------------------------------
# Change-request wording -> change type and the earliest stage it alters. First match wins.
_CHANGE_TYPES: list[tuple[str, str, str]] = [
    (r"recalibrat", "recalibration", "model"),
    (r"redevelop|rebuild|new methodolog|new framework|replace the model|from scratch",
     "redevelopment", "ideate"),
    (r"extend|expand|new (portfolio|product|segment|geograph)|scope", "scope_change", "ideate"),
    (r"finding|remediat|audit|validation issue|mria|mra\b", "remediation", "model"),
    (r"variable|feature|driver|factor|coefficient|re-?estimat|specification|segment",
     "re_estimation", "ideate"),
    (r"data|refresh|vintage|sample|window|history|source", "data_refresh", "explore"),
    (r"document|white ?paper|model card", "documentation", "none"),
]
_SCOPE_DEPTH = {"documentation": 0, "recalibration": 1, "remediation": 2, "data_refresh": 2,
                "re_estimation": 2, "scope_change": 3, "redevelopment": 3}


def _change(text: str) -> dict[str, str]:
    for pattern, kind, stage in _CHANGE_TYPES:
        if re.search(pattern, text.lower()):
            return {"change": text, "type": kind, "affects": stage}
    return {"change": text, "type": "re_estimation", "affects": "ideate"}


def intake_analyst(sl: dict[str, Any]) -> dict[str, Any]:
    """Reads the intent template as written: a filled section is the sponsor's stated position
    (quoted from its first line), an empty required one becomes a blocking interview question.
    A free-form document that does not follow the template is all 'missing' to this agent."""
    from ..engagement import QUESTIONS, bullets

    update = sl.get("kind") == "update"
    answered = {" ".join(a["question"].lower().split()) for a in sl.get("sponsor_answers", [])}
    brief, interview, values = [], [], {}
    for f in sl.get("fields", []):
        name = f["field"]
        text = str(f.get("template_value") or "").strip()
        if text:
            first = next(line.strip() for line in text.splitlines() if line.strip())
            brief.append({"field": name, "value": " ".join(text.split())[:600], "basis": "stated",
                          "quote": first[:200]})
            values[name] = text
            continue
        brief.append({"field": name, "value": "", "basis": "missing", "quote": ""})
        ask = f.get("required") or name == "success_criteria"
        question = QUESTIONS.get(name)
        if (ask and question and not f.get("decided")
                and " ".join(question.lower().split()) not in answered):
            interview.append({
                "question": question, "field": name, "blocking": bool(f.get("required")),
                "why_it_matters": ("Development cannot be scoped without it." if f.get("required")
                                   else "It sets what 'good enough' means at sign-off.")})
    n_blocking = sum(q["blocking"] for q in interview)
    n_required = sum(1 for f in sl.get("fields", []) if f.get("required")) or 1
    clarity = ("clear" if not n_blocking else
               "needs_clarification" if n_blocking * 2 <= n_required else "unclear")
    decided = {f["field"]: f.get("decided") for f in sl.get("fields", [])}
    objective = values.get("objective") or decided.get("objective") or ""
    out: dict[str, Any] = {
        "summary": ((f"The intent document states {len(values)} of {len(brief)} brief fields. "
                     if sl.get("intent_document") else
                     "No business intent document was supplied; the brief rests on the profile. ")
                    + (f"{n_blocking} blocking question(s) for the sponsor before development "
                       "starts." if n_blocking else "The goal is clear enough to start.")),
        "uncertainties": ["The deterministic agent reads only the template's sections; a free-form "
                          "document needs an LLM agent or the sponsor's answers."]
        if sl.get("intent_document") and not values else [],
        "responses_to_challenges": _ack(sl),
        "restated_objective": (" ".join(objective.split())[:400]
                               or "Not stated; the sponsor has been asked."),
        "clarity": clarity,
        "brief": brief,
        "interview": interview,
        "update_scope": "not_applicable", "scope_rationale": "", "change_items": [],
        "incumbent_family": "",
    }
    if update:
        prior = sl.get("prior_model") or {}
        items = [_change(c) for c in bullets(values.get("requested_changes", ""))][:12]
        depth = max((_SCOPE_DEPTH[i["type"]] for i in items), default=2)
        if re.search(r"redevelop|rebuild", values.get("update_reason", "").lower()):
            depth = 3
        scope = {0: "recalibrate", 1: "recalibrate", 2: "re_estimate", 3: "redevelop"}[depth]
        family = prior.get("likely_family") or ""
        out.update({
            "update_scope": scope,
            "scope_rationale": (f"The deepest of the {len(items)} requested change(s) is "
                                f"{max(items, key=lambda i: _SCOPE_DEPTH[i['type']])['type'].replace('_', ' ')}."
                                if items else "No change list was given; a re-estimation of the "
                                              "existing specification is assumed until the "
                                              "sponsor says otherwise."),
            "change_items": items,
            "incumbent_family": family if family in sl.get("engine_families", []) else "",
        })
    return out


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
    confirmed = set()  # the sponsor answered that a suspect is recorded after the outcome
    for a in sl.get("sponsor_answers", []):
        text = (a["question"] + " " + a["answer"]).lower()
        if re.search(r"\b(after|post|not available|leak)", a["answer"].lower()):
            confirmed |= {col for col in suspects if col.lower() in text}
    decisions: list[dict] = []
    for col in suspects:
        if col in named or col in current or col in confirmed:
            decisions.append({
                "column": col, "decision": "exclude",
                "reason": ("Excluded as requested in challenge " + named[col] if col in named else
                           "The sponsor confirmed it is recorded after the outcome." if col in confirmed
                           else "Excluded by an earlier human decision at the data gate."),
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
    # Features worth considering: the strongest clean relationships the engine measured, with the
    # direction it saw. (An LLM analyst argues from the business intent; this one from the data.)
    candidates = []
    for c in sl.get("top_correlations", []):
        col = c["feature"]
        if col in excluded or col in suspects or col not in features or len(candidates) >= 8:
            continue
        prior = sign_prior(col)
        seen = "+" if c["corr"] > 0 else "-"
        candidates.append({
            "column": col, "relationship": seen,
            "rationale": (f"Moves with the target ({'rises' if seen == '+' else 'falls'} as it "
                          "increases)" + (", as the economic prior expects." if prior == seen else
                                          ", against the usual prior: check before relying on it."
                                          if prior != "none" else ".")),
            "evidence": _fact(sl, f"explore.corr.{col}")})
    target = sl.get("target") or ""
    return {
        "summary": summary,
        "uncertainties": [f"Whether '{c}' is known at prediction time." for c in kept],
        "responses_to_challenges": _ack(sl, {named[c] for c in named}),
        "target": target,
        "target_rationale": (sl.get("target_basis") or "") if target else "",
        "feature_candidates": candidates,
        "column_decisions": decisions,
        "data_quality": quality,
        "questions_for_sponsor": [f"Is '{c}' available at the time a prediction is made, or is it "
                                  "recorded after the outcome?" for c in kept],
    }


def data_linker(sl: dict[str, Any]) -> dict[str, Any]:
    """The join the measurements point to: the table holding the target (else the largest) as
    the base, then for each other table the link that matches most rows, same-named keys first."""
    from .. import linking

    prof = {"tables": sl.get("tables", []), "links": sl.get("links", [])}
    plan = linking.suggest(prof, sl.get("base_table") or None, sl.get("target") or "")
    links = {lk["id"]: lk for lk in prof["links"]}
    steps, _ = linking.resolve(plan["base"], plan["joins"], prof)
    joins, concerns = [], []
    for ch, step in zip(plan["joins"], steps, strict=False):
        lk = links[ch["link"]]
        on = ", ".join(f"{a} = {b}" for a, b in zip(step["left_on"], step["right_on"], strict=True))
        joins.append({"link": ch["link"], "many": ch["many"], "unmatched": ch["unmatched"],
                      "unmatched_reason": ch["unmatched_reason"],
                      "reason": f"{step['left']} and {step['right']} share the key {on} "
                                f"({lk['relation']})."})
        if step["expected_match"] < 0.9:
            concerns.append(f"Only {step['expected_match']:.0%} of the rows of {step['left']} "
                            f"find a match in {step['right']} (link {lk['id']}).")
        if ch["unmatched"] == "drop":
            concerns.append(f"Rows without a match in {step['right']} are dropped. Check that "
                            "they are not a group the model must cover (new obligors, a "
                            "segment missing from that source).")
        elif step["expected_match"] < 1:
            concerns.append(f"Rows without a match in {step['right']} are kept, with missing "
                            f"values in its columns. {ch['unmatched_reason']} The model "
                            "families do not take missing values: exclude those columns at "
                            "this gate, fill the gap in the source, or choose to drop the rows.")
        if ch["many"] == "aggregate":
            concerns.append(f"{step['right']} has several rows per key and is averaged; if the "
                            "rows are periods, check that none is dated after the observation.")
    holds = sl.get("target") and sl["target"] in next(
        (t["columns"] for t in prof["tables"] if t["name"] == plan["base"]), [])
    return {
        "base_table": plan["base"],
        "base_rationale": ("It holds the dependent variable." if holds else
                           "It has the most rows, so each of its rows is taken as one "
                           "observation."),
        "joins": joins,
        "left_out": [{"table": t, "reason": "No measured link connects it to the joined tables."}
                     for t in plan["left_out"]],
        "concerns": concerns,
    }


def data_scout(sl: dict[str, Any]) -> dict[str, Any]:
    """The standard look at a dataset, as tool calls. Round one settles the target when the
    profile leaves it open (the engine's first candidate); the visuals follow once it is known.
    The deterministic analyst never writes code."""
    target = sl.get("target") or ""
    out: dict[str, Any] = {"target_column": "", "target_rationale": "", "requests": [],
                           "done": True, "notes": ""}
    if not target:
        best = (sl.get("target_candidates") or [{}])[0]
        out.update(target_column=best.get("column", ""), done=False,
                   target_rationale=best.get("why", ""),
                   notes="Target taken from the engine's ranking of outcome-like columns.")
        return out
    if sl.get("analyses"):  # the standard set has run
        return out
    tools = {t["name"] for t in sl.get("tools", [])}
    reqs = [("target_distribution", {}, "How is the dependent variable distributed, and how many "
                                        "events are there?")]
    if sl.get("columns_with_missing"):
        reqs.append(("missingness", {}, "Which columns have gaps large enough to matter?"))
    if sl.get("datetime_col"):
        reqs.append(("time_trend", {}, "Is the target stable over time, and are the latest "
                                       "periods thin?"))
    for c in sl.get("top_correlations", [])[:3]:
        reqs.append(("target_relationship", {"column": c["feature"]},
                     f"How does the target move across {c['feature']}: direction, shape?"))
    reqs.append(("correlation_matrix", {}, "Which candidate features duplicate each other?"))
    out["requests"] = [{"purpose": why, "tool": name, "code": "",
                        "params": [{"name": k, "value": v} for k, v in params.items()]}
                       for name, params, why in reqs if name in tools][: sl.get("max_requests", 8)]
    return out


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


def tool_request(sl: dict[str, Any]) -> dict[str, Any]:
    """Run every tool offered to the stage once, with its default parameters. A tool was
    registered for this stage on purpose, so the deterministic agent uses it."""
    ran = {r["tool"] for r in sl.get("tool_runs", [])}
    reqs = [{"purpose": t["description"], "tool": t["name"], "params": []}
            for t in sl.get("tools", []) if t["name"] not in ran][: sl.get("max_requests", 8)]
    return {"requests": reqs, "done": True, "notes": ""}


AGENTS = {
    "intake_analyst": intake_analyst,
    "data_analyst": data_analyst,
    "data_scout": data_scout,
    "data_linker": data_linker,
    "design_lead": design_lead,
    "modeler": modeler,
    "experiment": experiment,
    "outcomes_analyst": outcomes_analyst,
    "validator": validator,
    "risk_analyst": risk_analyst,
    "writer": writer,
}


def recommend(agent: str, sl: dict[str, Any]) -> dict[str, Any]:
    if agent.endswith("_tools"):
        return tool_request(sl)
    return AGENTS[agent](sl)
