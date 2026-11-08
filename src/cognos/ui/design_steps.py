"""The design page in six steps (ADR-0025): the model form, how the data are cut, the
variables, the assessment plan, the single-factor analysis, then the summary and the decision.

Each step is confirmed before the next opens, and a confirmed step folds into one line. The
confirmations are the page's own (``dz-done``); nothing is recorded until the design gate is
decided, and the gate reads every field wherever it sits. The single-factor analysis opens only
once the assessment plan is confirmed, and every time it is drawn it is a look the engine
keeps (``service.preview_screen``). Pure functions of their inputs; the callbacks are in
``register_design_callbacks``.
"""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, ctx, html, no_update

from .. import service
from ..modeling import plan as plan_mod
from ..modeling import selection
from .components import empty, fmt, icon, pct, section, table

STEPS = {1: "Model form", 2: "Data treatment", 3: "Variables", 4: "Assessment plan",
         5: "Single-factor analysis", 6: "Summary and decision"}
GATE = "gate_design"
RULE_LABELS = {"force_out": "left out by you", "not_fitted": "could not be fitted",
               "max_missing": "too much missing", "max_p_value": "not significant",
               "enforce_sign": "sign against expectation", "min_strength": "too weak alone",
               "best_per_source": "another form of the column is stronger",
               "max_correlation": "correlated with a stronger one", "top_n": "beyond the shortlist"}
SHORT = {"force_out": "you", "not_fitted": "unfit", "max_missing": "missing",
         "max_p_value": "p-value", "enforce_sign": "sign", "min_strength": "strength",
         "best_per_source": "per column", "max_correlation": "corr", "top_n": "top n"}


def field(name: str) -> dict:
    return {"type": "gate-field", "gate": GATE, "field": name}


def words(v: Any) -> str:
    return str(v).replace("_", " ")


BY = {"engine": "recommended by the engine", "agent": "recommended by the Design agent",
      "profile": "set in the profile", "human": "set by the developer"}


def by(plan: dict, key: str) -> str | None:
    """Who set a choice of the plan, in words (None when nobody is on record)."""
    return BY.get((plan.get("sources") or {}).get(key, ""))


# --- the step frame -------------------------------------------------------------------------
def step(n: int, body: list, *, summary: str, editable: bool, description: str = "") -> Any:
    """One numbered step. On an editable page it carries a confirm button, a folded line for
    when it is confirmed and a note for while an earlier step is open; the page's callback
    shows the right one."""
    title = f"{n} · {STEPS[n]}"
    if not editable:
        return section(title, *body, description=description or None)
    last = n == max(STEPS)
    return section(
        title,
        html.Div([*body, dmc.Group([
            dmc.Button(f"Confirm {STEPS[n].lower()}", id={"type": "dz-confirm", "n": n},
                       size="xs", leftSection=icon("tabler:check", 14)),
            dmc.Text("Nothing is recorded until the design is decided.", size="xs",
                     c="dimmed")], gap="sm", mt="md") if not last else None],
                 id={"type": "dz-body", "n": n}),
        html.Div(dmc.Group([
            dmc.ThemeIcon(icon("tabler:check", 12), size="sm", color="green", variant="light"),
            html.Span(summary, id={"type": "dz-sumtext", "n": n}, className="dz-sum"),
            dmc.Button("Change", id={"type": "dz-reopen", "n": n}, size="compact-xs",
                       variant="subtle")], gap="xs", wrap="nowrap"),
            id={"type": "dz-folded", "n": n}, style={"display": "none"}),
        html.Div(dmc.Text(f"Opens when step {n - 1} is confirmed.", size="sm", c="dimmed",
                          fs="italic"), id={"type": "dz-wait", "n": n},
                 style={"display": "none"}),
        right=html.Span(id={"type": "dz-badge", "n": n}, style={"flexShrink": 0}),
        description=description or None)


def visibility(done: list[int], editable_steps: list[int]) -> dict[int, str]:
    """Each step's state: ``done`` (folded), ``open`` (the first not confirmed) or ``wait``."""
    out, opened = {}, False
    for n in sorted(editable_steps):
        if n in done and not opened:
            out[n] = "done"
        elif not opened:
            out[n], opened = "open", True
        else:
            out[n] = "wait"
    return out


# --- 1 · the model form ---------------------------------------------------------------------
ROLE_WORDS = {"primary": "Main form", "candidate": "Also competes",
              "challenger": "Benchmark only", "rejected": "Not used"}
ROLE_COLORS = {"primary": "oxblood", "candidate": "green", "challenger": "yellow",
               "rejected": "gray"}
# What a developer needs to tick a family: what it is, which of the plan's rules can judge it
# (p-values, signs and stepwise need coefficient tests), and how it reads.
FAMILY_NOTES = {
    "logit": ("The standard scorecard: coefficients read as odds ratios and convert to points.",
              "p-value, sign, VIF, stepwise"),
    "probit": ("Normal link. Ranks borrowers almost exactly as logit does; rarely the better "
               "choice, useful as a check.", "p-value, sign, VIF, stepwise"),
    "cloglog": ("Asymmetric link that suits rare defaults; also the grouped-time hazard link.",
                "p-value, sign, VIF, stepwise"),
    "ridge_logit": ("Logit with shrunken coefficients, steadier when variables are correlated. "
                    "No p-values.", "sign, VIF"),
    "lasso_logit": ("Logit that shrinks weak coefficients to zero, so it also selects. "
                    "No p-values.", "sign, VIF"),
    "hazard_logit": ("Discrete-time hazard on the obligor-period panel, logit link: a PD for "
                     "every period of the horizon.", "p-value, sign, VIF, stepwise"),
    "hazard_cloglog": ("Discrete-time hazard, cloglog link: the proportional-hazards form.",
                       "p-value, sign, VIF, stepwise"),
    "ols": ("Linear regression: coefficients read directly in the outcome's units.",
            "p-value, sign, VIF, stepwise"),
    "ridge": ("Linear with shrunken coefficients, steadier under correlation. No p-values.",
              "sign, VIF"),
    "lasso": ("Linear that shrinks weak coefficients to zero. No p-values.", "sign, VIF"),
    "elasticnet": ("Between ridge and lasso. No p-values.", "sign, VIF"),
    "poisson": ("GLM for counts.", "p-value, sign, VIF, stepwise"),
    "gamma": ("GLM for positive, right-skewed amounts.", "p-value, sign, VIF, stepwise"),
    "tweedie": ("GLM for amounts with many zeros.", "p-value, sign, VIF, stepwise"),
    "random_forest": ("Many trees averaged: catches non-linear effects and interactions, hard "
                      "to explain.", "correlation only"),
    "gradient_boosting": ("Trees fitted one after another on what is left: usually the most "
                          "accurate, the hardest to explain.", "correlation only"),
}


def frameworks_in_force(p: dict) -> dict[str, dict]:
    """Each framework with its decision in force (the developer's, else the Design agent's)."""
    choices = {c["framework"]: c for c in p.get("framework_choices", [])}
    out = {}
    for f in p.get("framework_assessment", []):
        ch = choices.get(f["framework"], {})
        out[f["framework"]] = {**f, "decision": ch.get("decision", f["role"]),
                               "why": ch.get("reason") or f["reason"],
                               "mine": ch.get("source") == "developer", "by": ch.get("by")}
    return out


def slate_families(p: dict) -> list[str]:
    return list(dict.fromkeys(h["family"] for h in p.get("hypotheses", [])))


def _role_cell(f: dict, editable: bool) -> Any:
    if not f["applicable"]:
        return dmc.Text("Not possible with this data", size="sm", c="dimmed")
    if not f.get("editable", False):
        return dmc.Badge(ROLE_WORDS.get(f["decision"], words(f["decision"])),
                         color=ROLE_COLORS.get(f["decision"], "gray"), variant="light",
                         size="sm")
    return dmc.Select(id=field(f"framework::{f['framework']}"), value=f["decision"],
                      data=[{"value": k, "label": v} for k, v in ROLE_WORDS.items()],
                      disabled=not editable, size="xs", allowDeselect=False, w=150)


def model_form(p: dict, editable: bool, *, reason: str = "",
               proposed: list[dict] | None = None) -> list:
    fws = frameworks_in_force(p)
    order = {"primary": 0, "candidate": 1, "challenger": 2, "rejected": 3}
    rows = sorted(fws.values(), key=lambda f: (not f["applicable"],
                                                order.get(f["decision"], 4)))
    primary = next((f for f in rows if f["decision"] == "primary"), None)
    mine = [f for f in rows if f["mine"]]
    proposed = list(proposed or [])
    on = set(slate_families(p))

    def why(f: dict) -> Any:
        text = f["why"] if not f["mine"] else f"set by {f['by'] or 'the developer'}: {f['why']}"
        idle = (f["decision"] != "rejected" and f.get("families")
                and not on & set(f["families"]))
        return dmc.Text([text, html.Span(" None of its families is ticked below, so nothing of "
                                         "it is fitted.", className="dz-warn") if idle else ""],
                        size="xs")

    table_rows = [[dmc.Text(f["label"], size="sm", fw=600 if f["decision"] == "primary" else 400),
                   _role_cell(f, editable), why(f)] for f in rows]
    bench = set(p.get("benchmark_families") or [])
    fams = list(p.get("fittable_families") or sorted(on))
    stepped = [m for m in (p.get("plan") or {}).get("selection", {}).get("methods") or []
               if m != "grid"]
    walker = next((f for f in fams if f in on and selection.steppable(f)), None) \
        if stepped else None
    form_rows = [html.Div([
        dmc.TextInput(id=field(f"proposed::{i}::name"), value=f.get("name", ""),
                      placeholder="e.g. Bayesian hierarchical logit", disabled=not editable,
                      size="xs", style={"flex": "1 1 200px"}),
        dmc.TextInput(id=field(f"proposed::{i}::reason"), value=f.get("reason", ""),
                      placeholder="why it should be considered", disabled=not editable,
                      size="xs", style={"flex": "3 1 320px"}),
    ], className="dz-proposed") for i, f in enumerate([*proposed, {}] if editable else proposed)]
    return [
        dmc.Text((f"Recommended: {primary['label']}." if not primary["mine"] else
                  f"Main form: {primary['label']} (set by {primary['by'] or 'the developer'}).")
                 if primary else "No framework stands out for this data.", fw=600),
        dmc.Text("Main form: what the model is built as. Also competes: fitted and may win. "
                 "Benchmark only: fitted to show what a simpler model gives up, never the "
                 "champion. Change a role if you disagree; the families below follow it when "
                 "you apply.", size="xs", c="dimmed"),
        table(["Framework", "Role", "Why"], table_rows),
        dmc.Textarea(id=field("model_form_reason"), value=reason, disabled=not editable,
                     label="Your reason (only if you change a role)", autosize=True, minRows=1,
                     description="Leave it blank to keep the recommendation."
                     + (f" In force for {', '.join(f['label'] for f in mine)}." if mine else ""),
                     mt="xs") if editable or mine else None,
        html.Details([html.Summary(f"Propose another form (optional"
                                   f"{f', {len(proposed)} on record' if proposed else ''})",
                                   className="gate-fold"),
                      dmc.Text("A form the engine does not fit is recorded with your reason, "
                               "shown to validation and listed in the documentation as not "
                               "fitted.", size="xs", c="dimmed"),
                      *form_rows], open=bool(proposed),
                     className="gate-more") if editable or proposed else None,
        dmc.CheckboxGroup(dmc.Stack([
            dmc.Checkbox(value=fam, disabled=not editable, label=html.Span([
                html.B(fam, className="mono"),
                html.Span(" · benchmark only" if fam in bench else "", className="dz-dim"),
                html.Span(f" · {FAMILY_NOTES.get(fam, ('', ''))[0]}", className="dz-dim")]))
            for fam in fams], gap=4),
            id=field("families"), label="Model families to search",
            description="Ticked: what the engine recommends for this form. Each family is "
                        "searched with the variables from steps 3 to 5; more families make "
                        "the search longer."
                        + (f" {walker} chooses the variables (it walks the stepwise path); the "
                           "other families with p-values are refitted on the variable sets it "
                           "tried, so they differ only in their link." if walker else ""),
            value=[f for f in fams if f in on], mt="md"),
        html.Details([html.Summary("What each family is, and which rules can judge it",
                                   className="gate-fold"),
                      table(["Family", "What it is", "Rules that apply", "Can win"],
                            [[dmc.Text(fam, ff="monospace", size="sm"),
                              dmc.Text(FAMILY_NOTES.get(fam, ("", ""))[0], size="xs"),
                              dmc.Text(FAMILY_NOTES.get(fam, ("", ""))[1], size="xs"),
                              "benchmark only" if fam in bench else "yes"] for fam in fams])],
                     className="gate-more"),
    ]


def model_form_summary(p: dict, families: list[str] | None = None) -> str:
    primary = next((f["label"] for f in frameworks_in_force(p).values()
                    if f["decision"] == "primary"), "no main form")
    fams = [f for f in (families if families is not None else slate_families(p))]
    extra = len(p.get("proposed_forms") or [])
    return (f"{primary} · {len(fams)} famil{'y' if len(fams) == 1 else 'ies'}: {', '.join(fams)}"
            + (f" · {extra} form(s) proposed" if extra else ""))


# --- 2 · data treatment ---------------------------------------------------------------------
def timeline(test: str, test_fraction: Any, holdout_fraction: Any, validation: str,
             folds: Any) -> Any:
    """The rows as one bar, cut as the plan cuts them: development, test, holdout."""
    t = 0.0 if test == "none" else float(test_fraction or 0)
    h = float(holdout_fraction or 0)
    d = max(0.0, 1.0 - t - h)
    order = "in time order, oldest first" if test == "out_of_time" else "in a shuffled order"
    parts = [("Development", d, "dz-seg dev", f"fitted here; {words(validation)}"
              + (f" × {folds}" if validation != "none" and folds else "")),
             ("Test", t, "dz-seg test", "the pool is compared here"),
             ("Holdout", h, "dz-seg hold", "sealed: scored once, after the choice")]
    return html.Div([
        html.Div([html.Div(html.Span(f"{label} {share:.0%}"), className=cls,
                           style={"flexGrow": max(share, 0.0001)}, title=note)
                  for label, share, cls, note in parts if share > 0], className="dz-bar"),
        html.Div([html.Div([html.B(label), html.Span(f" {share:.0%} · {note}")],
                           className="dz-legend-row")
                  for label, share, _, note in parts if share > 0], className="dz-legend"),
        dmc.Text(f"Rows are cut {order}: development first, the holdout last.", size="xs",
                 c="dimmed"),
    ])


def data_treatment(plan: dict, editable: bool) -> list:
    ev, opts = plan["evaluation"], plan.get("options") or {}
    reasons = plan.get("reasons") or {}
    return [
        html.Div(timeline(ev["test"], ev["test_fraction"], ev["holdout_fraction"],
                          ev["validation"], ev["folds"]), id="dz-timeline"),
        dmc.SimpleGrid([
            _pick("evaluation.test", "Test sample", ev["test"], opts.get("test") or [], editable,
                  description=by(plan, "test")),
            _num("evaluation.test_fraction", "Test share of rows", ev["test_fraction"], editable,
                 min=0, max=0.6, step=0.05, decimalScale=2),
            _num("evaluation.holdout_fraction", "Sealed holdout share (0 = none)",
                 ev["holdout_fraction"], editable, min=0, max=0.4, step=0.05, decimalScale=2,
                 description=by(plan, "holdout_fraction")),
            _pick("evaluation.validation", "Validation inside development", ev["validation"],
                  opts.get("validation") or [], editable, description=by(plan, "validation")),
            _num("evaluation.folds", "Validation blocks", ev["folds"], editable, min=2, max=20),
            _num("evaluation.horizon", "Periods per block (rolling)", ev.get("horizon"),
                 editable, min=1),
        ], cols={"base": 1, "sm": 2, "md": 3}, mt="sm"),
        html.Details([html.Summary("Why the engine cut it this way", className="gate-fold"),
                      dmc.List([dmc.ListItem(f"{words(k).capitalize()}: {reasons[k]}")
                                for k in ("test", "holdout_fraction", "validation", "folds",
                                          "horizon") if reasons.get(k)], size="sm")],
                     className="gate-more"),
    ]


def treatment_summary(ev: dict) -> str:
    test = "no test sample" if ev.get("test") == "none" else \
        f"{words(ev.get('test'))} test {pct(ev.get('test_fraction'), 0)}"
    hold = f"holdout {pct(ev.get('holdout_fraction'), 0)}" if ev.get("holdout_fraction") \
        else "no holdout"
    val = "no validation" if ev.get("validation") == "none" else \
        f"{words(ev.get('validation'))} × {ev.get('folds')}"
    return f"{test} · {hold} · {val}"


# --- 3 · variables --------------------------------------------------------------------------
def variables(transforms: Any, plan: dict, factor_names: list[str], editable: bool) -> list:
    sel = plan["selection"]
    return [
        transforms if transforms is not None else dmc.Text(
            "No variable was chosen for a transformation at the data gate: every variable "
            "enters as it is.", size="sm", c="dimmed"),
        dmc.MultiSelect(id=field("plan::selection.must_include"),
                        label="Variables every model must carry", size="xs", mt="md",
                        value=list(sel.get("must_include") or []), searchable=True,
                        clearable=True, disabled=not editable, maw=640,
                        data=[{"value": n, "label": n} for n in factor_names],
                        description="A policy variable or an incumbent's driver: it passes the "
                                    "screen and starts every model."),
    ]


def variables_summary(n_transformed: int, must: list[str] | None) -> str:
    must = must or []
    return (f"{n_transformed} transformed · "
            + (f"every model carries {', '.join(must)}" if must else "no variable required"))


# --- 4 · the assessment plan ----------------------------------------------------------------
RANK_LINES = 3


def assessment(plan: dict, editable: bool) -> list:
    ev, sel, fc = plan["evaluation"], plan["selection"], plan["factors"]
    opts = plan.get("options") or {}
    metrics = [*(opts.get("metric") or []), *plan_mod.RANK_EXTRAS]
    rank = list(ev.get("rank_by") or [])
    rank += [{}] * (RANK_LINES - len(rank))
    weights = ev.get("rank_mode") == "weights"
    lines = []
    for i, r in enumerate(rank[:RANK_LINES]):
        lines.append(dmc.Group([
            dmc.Text(str(i + 1), ff="monospace", size="sm", w=14),
            dmc.Select(id=field(f"plan::rank.{i}.metric"), value=r.get("metric"), size="xs",
                       w=230, clearable=i > 0, disabled=not editable,
                       placeholder="add a line" if i else None,
                       data=[{"value": m, "label": plan_mod.RANK_EXTRAS.get(m, m)}
                             for m in metrics]),
            dmc.Select(id=field(f"plan::rank.{i}.sample"), size="xs", w=130,
                       value=r.get("sample") or "validation", disabled=not editable,
                       allowDeselect=False,
                       data=[{"value": s, "label": words(s)}
                             for s in opts.get("sample") or plan_mod.SAMPLES]),
            dmc.NumberInput(id=field(f"plan::rank.{i}.weight"), size="xs", w=110,
                            value=r.get("weight") if r.get("weight") is not None else
                            (1 if r.get("metric") else ""), min=0, step=0.1, decimalScale=2,
                            disabled=not editable, placeholder="weight",
                            leftSection=dmc.Text("w", size="xs", c="dimmed")),
        ], gap="xs", wrap="nowrap"))
    return [
        _head("Variables are kept when", "Each variable is fitted alone on the development "
              "sample; these rules then run in this order. Empty = off."),
        dmc.SimpleGrid([
            _pick("factors.criterion", "Strength read on", fc.get("criterion") or "validation",
                  opts.get("factor_criterion") or ["validation", "in_sample"], editable,
                  description=by(plan, "factors.criterion")),
            _num("factors.max_missing", "Missing at most", fc.get("max_missing"), editable,
                 min=0, max=1, step=0.05, decimalScale=2),
            _num("factors.max_p_value", "p-value at most", fc.get("max_p_value"), editable,
                 min=0, max=1, step=0.01, decimalScale=3),
            _yes("factors.enforce_sign", "Drop a wrong sign", fc.get("enforce_sign"), editable),
            _num("factors.best_per_source", "Forms kept per column", fc.get("best_per_source"),
                 editable, min=1),
            _num("factors.max_correlation", "Correlation at most", fc.get("max_correlation"),
                 editable, min=0, max=1, step=0.05, decimalScale=2),
            _num("factors.top_n", "Shortlist size", fc.get("top_n"), editable, min=1),
        ], cols={"base": 2, "md": 4}),
        _head("A model is kept when", "Every model the search builds goes through these; the "
              "candidate pool page shows each rule's funnel."),
        dmc.SimpleGrid([
            _num("selection.max_factors", "Factors at most", sel["max_factors"], editable,
                 min=1, max=12),
            _num("selection.max_p_value", "Every p-value at most", sel.get("max_p_value"),
                 editable, min=0, max=1, step=0.01, decimalScale=3),
            _yes("selection.enforce_sign", "Every sign as expected", sel.get("enforce_sign"),
                 editable),
            _num("selection.max_correlation", "Two factors correlated at most",
                 sel.get("max_correlation"), editable, min=0, max=1, step=0.05, decimalScale=2),
            _num("selection.max_vif", "Variance inflation at most", sel.get("max_vif"),
                 editable, min=1, step=0.5, decimalScale=1, description=by(plan, "max_vif")),
        ], cols={"base": 2, "md": 5}),
        _head("Models are built by", "One method is enough; several share one ranking."),
        dmc.SimpleGrid([
            dmc.MultiSelect(id=field("plan::selection.methods"), label="Method", size="xs",
                            value=list(sel.get("methods") or []), disabled=not editable,
                            description=by(plan, "methods"),
                            data=[{"value": m, "label": words(m)}
                                  for m in opts.get("methods") or []]),
            _pick("selection.criterion", "A step is kept when this improves", sel["criterion"],
                  opts.get("criterion") or [], editable, description=by(plan, "criterion")),
        ], cols={"base": 1, "sm": 2}),
        _head("What is left is ranked by", "In order: the first line decides and the next ones "
              "break ties. Blended: each line turns into the share of models a model beats, "
              "averaged by the weights."),
        dmc.Group([
            dmc.SegmentedControl(id=field("plan::evaluation.rank_mode"), size="xs",
                                 value="weights" if weights else "order",
                                 disabled=not editable,
                                 data=[{"value": "order", "label": "In order"},
                                       {"value": "weights", "label": "Blended by weight"}]),
            _num("evaluation.pool_size", "Candidates proposed", ev["pool_size"], editable,
                 min=1, max=20, w=150),
        ], gap="lg", align="flex-end"),
        dmc.Stack(lines, gap=6, mt="xs"),
    ]


def assessment_summary(plan: dict) -> str:
    fc, sel = plan["factors"], plan["selection"]
    var_rules = [f"p ≤ {fmt(fc['max_p_value'], 2)}" if fc.get("max_p_value") is not None
                 else None, "sign" if fc.get("enforce_sign") else None,
                 f"|corr| ≤ {fmt(fc['max_correlation'], 2)}"
                 if fc.get("max_correlation") is not None else None,
                 f"top {fc['top_n']}" if fc.get("top_n") else None]
    model_rules = [f"≤ {sel['max_factors']} factors",
                   f"VIF ≤ {fmt(sel['max_vif'], 1)}" if sel.get("max_vif") is not None else None,
                   f"every p ≤ {fmt(sel['max_p_value'], 2)}"
                   if sel.get("max_p_value") is not None else None]
    return (f"variables: {', '.join(x for x in var_rules if x) or 'no rule'} · models: "
            f"{', '.join(x for x in model_rules if x)} · ranked by "
            f"{plan_mod.rank_words(plan['evaluation'])}")


# --- 5 · the single-factor analysis ---------------------------------------------------------
def funnel(lines: list[dict], *, total: int, selected: str | None, prefix: str,
           extra: list[tuple[str, str, int]] = ()) -> Any:
    """The funnel: one bar per rule in the order they run, its height the share still in.
    A bar is a button that filters the table under it (``selected``)."""
    bars = [("all", "all", total, total)]
    for line in lines:  # every rule that ran, a rule that removed nothing included
        out = line["before"] - line["after"]
        if line["rule"] != "not_fitted" or out:
            bars.append((line["rule"], SHORT.get(line["rule"], line.get("label") or line["rule"]),
                         line["after"], out))
    cells = []
    for rule, label, left, out in bars:
        cells.append(html.Button([
            html.Div(html.Div(className="dz-fill", style={"height": f"{100 * left / max(total, 1):.0f}%"}),
                     className="dz-col"),
            html.Div(label, className="dz-lab"),
            html.Div(f"{left}" if rule == "all" else f"−{out}", className="dz-num"),
        ], id={"type": f"{prefix}-bar", "rule": rule}, n_clicks=0, title=RULE_LABELS.get(rule, label),
            className="dz-barbtn" + (" sel" if rule == (selected or "all") else "")))
    for rule, label, n in extra:
        cells.append(html.Button([
            html.Div(html.Div(className="dz-fill " + ("exc" if rule == "exceptions" else "kept"),
                              style={"height": f"{100 * n / max(total, 1):.0f}%"}),
                     className="dz-col"),
            html.Div(label, className="dz-lab"),
            html.Div(f"{n}" if rule == "kept" else f"+{n}", className="dz-num")],
            id={"type": f"{prefix}-bar", "rule": rule}, n_clicks=0,
            className="dz-barbtn" + (" sel" if rule == selected else "")))
    return html.Div(cells, className="dz-funnel")


def screen_rows(rows: list[dict], selected: str | None) -> list[dict]:
    """The rows a funnel bar stands for: everyone, the ones a rule removed, the ones kept,
    or the exceptions."""
    if not selected or selected == "all":
        return rows
    if selected == "kept":
        return [r for r in rows if r["status"] == "kept"]
    if selected == "exceptions":
        return [r for r in rows if r.get("exception")]
    return [r for r in rows if r.get("dropped_by") == selected
            or any(str(o).startswith(f"{selected}:") for o in r.get("overruled") or [])]


def p_text(p: Any) -> str:
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def screen_table(rows: list[dict], metric: str, chosen: str | None) -> Any:
    def strength(r: dict) -> Any:
        return r.get("in_sample") if r.get("ranked_on") == "in_sample" else (
            r.get("validation") if r.get("validation") is not None else r.get("in_sample"))

    order = sorted(rows, key=lambda r: (r["status"] != "kept", -(strength(r) or -1e9)))
    out = []
    for r in order:
        exc = r.get("exception")
        out.append(html.Div([
            html.Span(r["name"], className="tx-col", title=r["name"]),
            html.Span(fmt(strength(r), 3), className="dz-n"),
            html.Span(r.get("sign") or "", className="dz-n"),
            html.Span(p_text(r.get("p_value")), className="dz-n"),
            html.Span([html.Span({"keep": "kept by you", "drop": "left out by you"}[exc]
                                 if exc else ("kept" if r["status"] == "kept" else "dropped"),
                                 className="dz-exc" if exc else f"dz-st {r['status']}"),
                       html.Span(f" {(r.get('reason') or '').split(': ', 1)[-1]}"
                                 if r["status"] != "kept" or exc else "", className="dz-why")],
                      className="dz-fate", title=r.get("reason") or ""),
        ], id={"type": "sf-row", "name": r["name"]}, n_clicks=0,
            className="dz-row" + (" gone" if r["status"] != "kept" else "")
            + (" sel" if r["name"] == chosen else "")))
    head = html.Div([html.Span("Variable"), html.Span(metric, className="dz-n"),
                     html.Span("sign", className="dz-n"), html.Span("p", className="dz-n"),
                     html.Span("Fate")], className="dz-row dz-head")
    return html.Div([head, *out] if out else [empty("No variable in this group.")],
                    className="dz-table")


def exception_editor(row: dict | None, exceptions: dict, editable: bool) -> Any:
    if row is None:
        return dmc.Text("Click a variable to keep it although a rule drops it, or to leave it "
                        "out although the rules keep it. Each exception needs a reason.",
                        size="xs", c="dimmed")
    mine = exceptions.get(row["name"]) or {}
    dropped = row["status"] != "kept" and not mine.get("keep")
    return dmc.Stack([
        dmc.Group([dmc.Text(row["name"], ff="monospace", fw=700),
                   dmc.Text(row.get("reason") or ("kept by the rules" if row["status"] == "kept"
                                                  else ""), size="xs", c="dimmed")], gap="xs"),
        dmc.Group([
            dmc.SegmentedControl(id={"type": "sf-exc-kind", "i": 0}, size="xs",
                                 disabled=not editable,
                                 value=("keep" if mine.get("keep") else "drop" if mine
                                        else "none"),
                                 data=[{"value": "none", "label": "As the rules say"},
                                       {"value": "keep", "label": "Keep anyway"},
                                       {"value": "drop", "label": "Leave out"}]),
            dmc.TextInput(id={"type": "sf-exc-why", "i": 0}, size="xs", w=340,
                          value=mine.get("reason", ""), disabled=not editable,
                          placeholder=("why it stays despite the rule" if dropped
                                       else "why it goes although the rules keep it")),
            dmc.Button("Record", id={"type": "sf-exc-go", "i": 0}, size="xs", variant="light",
                       disabled=not editable),
        ], gap="xs"),
        html.Div(id="sf-exc-msg"),
    ], gap=6, className="dz-exc-edit")


def single_factor(screen: dict | None, exceptions: dict, editable: bool,
                  selected: str | None = None, chosen: str | None = None) -> list:
    if not screen:
        return [empty("No single-factor analysis yet.")]
    if screen.get("error"):
        return [dmc.Alert(f"The single-factor analysis could not be drawn: {screen['error']}. "
                          "The model stage runs it in any case.", color="yellow",
                          variant="light")]
    rows = screen["rows"]
    kept = [r for r in rows if r["status"] == "kept"]
    n_exc = len([r for r in rows if r.get("exception")])
    extra = [("kept", "kept", len(kept))] + ([("exceptions", "yours", n_exc)] if n_exc else [])
    chosen_row = next((r for r in rows if r["name"] == chosen), None)
    return [
        dmc.Group([
            dmc.Text(f"{len(rows)} variables → {len(kept)} kept", fw=650),
            dmc.Text(f"each fitted alone on the development sample ({fmt(screen['n_rows'])} "
                     f"rows); strength is {screen['metric']}", size="xs", c="dimmed"),
        ], gap="sm"),
        html.Div(funnel(screen["screen"], total=len(rows), selected=selected, prefix="sf",
                        extra=extra), id="sf-funnel"),
        html.Div(screen_table(screen_rows(rows, selected), screen["metric"], chosen),
                 id="sf-table"),
        html.Div(exception_editor(chosen_row, exceptions, editable), id="sf-edit",
                 style={"marginTop": "10px"}),
        dmc.Text("The engineered factors of the structural and migration frameworks join at "
                 "the model stage.", size="xs", c="dimmed") if screen.get("engineered") else None,
        dmc.Text("A preview: the model stage runs this screen again under the rules you decide, "
                 "and its result is the record.", size="xs", c="dimmed"),
    ]


def single_factor_summary(screen: dict | None, n_exc: int) -> str:
    if not screen or screen.get("error"):
        return "not drawn"
    kept = len(screen["shortlist"])
    return (f"{len(screen['rows'])} variables → {kept} kept"
            + (f" · {n_exc} exception(s) with reasons" if n_exc else " · no exception"))


# --- small builders -------------------------------------------------------------------------
def _head(title: str, note: str) -> Any:
    return dmc.Stack([dmc.Text(title, size="sm", fw=650), dmc.Text(note, size="xs", c="dimmed")],
                     gap=0, mt="md", mb=4)


def _pick(name: str, label: str, value: Any, choices: list[str], editable: bool, **kw) -> Any:
    return dmc.Select(id=field(f"plan::{name}"), label=label, value=value, size="xs",
                      data=[{"value": c, "label": words(c)} for c in choices],
                      allowDeselect=False, disabled=not editable, **kw)


def _num(name: str, label: str, value: Any, editable: bool, **kw) -> Any:
    return dmc.NumberInput(id=field(f"plan::{name}"), label=label,
                           value=value if value is not None else "", size="xs",
                           disabled=not editable, **kw)


def _yes(name: str, label: str, value: Any, editable: bool) -> Any:
    return dmc.Select(id=field(f"plan::{name}"), label=label, size="xs", allowDeselect=False,
                      value="yes" if value else "no", disabled=not editable,
                      data=[{"value": "yes", "label": "yes"}, {"value": "no", "label": "no"}])


def exceptions_of(value: str | None) -> dict[str, dict]:
    try:
        out = json.loads(value or "{}")
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


# --- the callbacks --------------------------------------------------------------------------
def register_design_callbacks(app, plan_patch) -> None:
    """The steps' confirmations, the timeline, the single-factor analysis, its funnel and the
    exceptions. ``plan_patch`` turns the page's fields into the plan edits (``ui/app.py``)."""

    def page_fields(values, ids) -> dict:
        return {i["field"]: v for i, v in zip(ids, values, strict=False) if i["gate"] == GATE}

    @app.callback(Output("dz-done", "data"),
                  Input({"type": "dz-confirm", "n": ALL}, "n_clicks"),
                  Input({"type": "dz-reopen", "n": ALL}, "n_clicks"),
                  State("dz-done", "data"), State("dz-key", "children"),
                  prevent_initial_call=True)
    def confirm(_c, _r, done, key):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
            return no_update
        have = list((done or {}).get("done") or []) if (done or {}).get("key") == key else []
        n = int(trig["n"])
        if trig["type"] == "dz-confirm":
            have = sorted(set(have) | {n})
        else:  # a step opened again: the ones after it wait for it
            have = [k for k in have if k < n]
        return {"key": key, "done": have}

    @app.callback(Output({"type": "dz-body", "n": ALL}, "style"),
                  Output({"type": "dz-folded", "n": ALL}, "style"),
                  Output({"type": "dz-wait", "n": ALL}, "style"),
                  Output({"type": "dz-badge", "n": ALL}, "children"),
                  Output({"type": "gate-act", "gate": GATE, "action": ALL}, "disabled"),
                  Input("dz-done", "data"), Input("dz-key", "children"),
                  State({"type": "dz-body", "n": ALL}, "id"))
    def show(done, key, ids):
        steps = [i["n"] for i in ids]
        have = list((done or {}).get("done") or []) if (done or {}).get("key") == key else []
        vis = visibility(have, steps)
        hidden = {"display": "none"}
        bodies = [({} if vis[n] == "open" else hidden) for n in steps]
        folded = [({} if vis[n] == "done" else hidden) for n in steps]
        waits = [({} if vis[n] == "wait" else hidden) for n in steps]
        badges = [dmc.Badge({"done": "confirmed", "open": "to confirm", "wait": "waiting"}[vis[n]],
                            size="xs", variant="light",
                            color={"done": "green", "open": "oxblood", "wait": "gray"}[vis[n]])
                  for n in steps]
        ready = all(vis[n] == "done" for n in steps if n < max(STEPS))
        acts = ctx.outputs_list[4]
        return bodies, folded, waits, badges, [not ready for _ in acts]

    @app.callback(Output("dz-timeline", "children"),
                  Input(field("plan::evaluation.test"), "value"),
                  Input(field("plan::evaluation.test_fraction"), "value"),
                  Input(field("plan::evaluation.holdout_fraction"), "value"),
                  Input(field("plan::evaluation.validation"), "value"),
                  Input(field("plan::evaluation.folds"), "value"), prevent_initial_call=True)
    def redraw_timeline(test, tf, hf, validation, folds):
        return timeline(test or "none", tf, hf, validation or "none", folds)

    @app.callback(Output("sf-body", "children"), Output("sf-data", "data"),
                  Output({"type": "dz-sumtext", "n": ALL}, "children"),
                  Input({"type": "dz-confirm", "n": ALL}, "n_clicks"),
                  Input({"type": "sf-exc-go", "i": ALL}, "n_clicks"),
                  State({"type": "gate-field", "gate": ALL, "field": ALL}, "value"),
                  State({"type": "gate-field", "gate": ALL, "field": ALL}, "id"),
                  State({"type": "sf-exc-kind", "i": ALL}, "value"),
                  State({"type": "sf-exc-why", "i": ALL}, "value"),
                  State("sf-chosen", "data"), State("ws-run", "data"),
                  State({"type": "dz-sumtext", "n": ALL}, "id"),
                  State("dz-tx-count", "children"), State("sf-data", "data"),
                  prevent_initial_call=True)
    def draw_screen(_c, _x, values, ids, kind, why, chosen, run_id, sum_ids, n_tx, drawn):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
            return no_update, no_update, [no_update] * len(sum_ids)
        fields = page_fields(values, ids)
        ideate = (service.results(run_id).get("ideate").payload or {})
        plan = ideate.get("plan") or {}
        exceptions = exceptions_of(fields.get("screen_exceptions"))
        message = None
        if trig["type"] == "sf-exc-go" and chosen:
            k, w = (kind or ["none"])[0], " ".join(str((why or [""])[0] or "").split())
            if k == "none":
                exceptions.pop(chosen, None)
            elif not w:
                message = "An exception needs a reason."
            else:
                exceptions[chosen] = {"keep": k == "keep", "reason": w}
        elif trig["type"] == "dz-confirm" and int(trig["n"]) != 4:
            # only the assessment plan draws the analysis: any other step restates its line
            patch = plan_patch(plan, fields)
            return no_update, no_update, _summaries(
                sum_ids, ideate, fields, patch, n_tx, screen=(drawn or {}).get("screen"),
                n_exc=len(exceptions))
        patch = plan_patch(plan, fields)
        try:
            screen = service.preview_screen(run_id, patch, exceptions)
        except ValueError as exc:
            return ([dmc.Alert(str(exc), color="red", variant="light")], no_update,
                    [no_update] * len(sum_ids))
        body = single_factor(screen, exceptions, True, chosen=chosen if message else None)
        if message:
            body.insert(0, dmc.Alert(message, color="red", variant="light"))
        sums = _summaries(sum_ids, ideate, fields, patch, n_tx, screen=screen,
                          n_exc=len(exceptions))
        return body, {"screen": screen, "exceptions": exceptions}, sums

    @app.callback(Output(field("screen_exceptions"), "value"),
                  Input("sf-data", "data"), prevent_initial_call=True)
    def keep_exceptions(data):
        return json.dumps((data or {}).get("exceptions") or {})

    @app.callback(Output("sf-funnel", "children", allow_duplicate=True),
                  Output("sf-table", "children", allow_duplicate=True),
                  Output("sf-edit", "children", allow_duplicate=True),
                  Output("sf-chosen", "data"),
                  Input({"type": "sf-bar", "rule": ALL}, "n_clicks"),
                  Input({"type": "sf-row", "name": ALL}, "n_clicks"),
                  State("sf-data", "data"), State("sf-selected", "data"),
                  State("sf-chosen", "data"), State("sf-editable", "data"),
                  prevent_initial_call=True)
    def filter_screen(_b, _r, data, selected, chosen, editable):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
            return no_update, no_update, no_update, no_update
        screen = (data or {}).get("screen")
        if not screen or screen.get("error"):
            return no_update, no_update, no_update, no_update
        exceptions = (data or {}).get("exceptions") or {}
        if trig["type"] == "sf-bar":
            selected = trig["rule"]
        else:
            chosen = trig["name"]
        rows = screen["rows"]
        n_exc = len([r for r in rows if r.get("exception")])
        extra = [("kept", "kept", len([r for r in rows if r["status"] == "kept"]))] + (
            [("exceptions", "yours", n_exc)] if n_exc else [])
        row = next((r for r in rows if r["name"] == chosen), None)
        return (funnel(screen["screen"], total=len(rows), selected=selected, prefix="sf",
                       extra=extra),
                screen_table(screen_rows(rows, selected), screen["metric"], chosen),
                exception_editor(row, exceptions, bool(editable)), chosen)

    @app.callback(Output("sf-selected", "data"), Input({"type": "sf-bar", "rule": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def remember_bar(_b):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not (ctx.triggered and ctx.triggered[0].get("value")):
            return no_update
        return trig["rule"]


def _summaries(sum_ids: list[dict], ideate: dict, fields: dict, patch: dict, n_tx: Any, *,
               screen: dict | None = None, n_exc: int = 0) -> list:
    """Each step's folded line, from the page as it stands."""
    plan = plan_mod.merge(ideate.get("plan") or {}, patch, "human") if ideate.get("plan") else {}
    out = []
    for i in sum_ids:
        n = i["n"]
        if n == 1:
            out.append(model_form_summary(ideate, fields.get("families")))
        elif n == 2 and plan:
            out.append(treatment_summary(plan["evaluation"]))
        elif n == 3 and plan:
            out.append(variables_summary(int(n_tx or 0), plan["selection"].get("must_include")))
        elif n == 4 and plan:
            out.append(assessment_summary(plan))
        elif n == 5 and screen is not None:
            out.append(single_factor_summary(screen, n_exc))
        else:
            out.append(no_update)
    return out
