"""From single factors to multi-factor models: forward, backward, stepwise, best subset, grid.

Every method goes through one ``Evaluator``, so a specification is fitted once however many
methods reach it, and every model any method looked at lands in one ledger with the same
scores: the ranking metric in sample, on the plan's validation sample and, only when the plan
ranks on it, on the test sample (each such evaluation is counted).

A method is a path. It starts from one factor (or from all of them), and each step is kept
only if the criterion improves and the model still passes the constraints: every coefficient
significant, every sign as expected, no two factors too alike. The path is recorded step by
step, so "what did the fourth factor add?" has an answer on the page.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .fit import (
    BINARY_GLM_LINKS,
    GLM_FAMILIES,
    HAZARD_FAMILIES,
    Candidate,
    SMBinaryGLM,
    build_inference_design,
    make_fit_predict,
)
from .metrics import CVResult, cv_score, metric_direction, score
from .search import ExperimentRecord, build_search_space

METHODS = ("grid", "forward", "backward", "stepwise", "best_subset")
CRITERIA = ("validation", "in_sample", "aic", "bic", "adj_r2")
# Families whose coefficients carry statsmodels inference: the ones a stepwise path can test.
STEP_FAMILIES = ("ols", "logit", *sorted(BINARY_GLM_LINKS))
_MINIMIZE_CRITERIA = ("aic", "bic")

METHOD_LABELS = {
    "grid": "Grid (every family on the strongest factors and on all of them)",
    "forward": "Forward selection (start from the best single factor, add one at a time)",
    "backward": "Backward elimination (start from all, drop one at a time)",
    "stepwise": "Stepwise (add one, then drop any that no longer helps)",
    "best_subset": "Best subset (every combination up to the size limit)",
}


def candidate_id(cand: Candidate, transforms: list | None = None) -> str:
    """A name that follows the specification, not its place in a ledger: the same id always
    means the same family, factors and settings, across re-runs and searches."""
    key = {"family": cand.family, "features": sorted(cand.features),
           "hp": {k: v for k, v in sorted(cand.hyperparams.items()) if k != "random_state"},
           "transforms": [t.to_dict() if hasattr(t, "to_dict") else t for t in transforms or []]}
    return "c" + hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()
                                ).hexdigest()[:6]


@dataclass
class Evaluation:
    """One specification and everything measured on it."""

    id: str
    candidate: Candidate
    cv: CVResult | None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # sample -> metric -> value
    info: dict[str, Any] = field(default_factory=dict)  # aic, bic, adj_r2, max_p, signs
    methods: list[str] = field(default_factory=list)
    error: str = ""

    def value(self, metric: str, sample: str) -> float | None:
        return (self.scores.get(sample) or {}).get(metric)


class Evaluator:
    """Fits and scores specifications on the development sample, once each."""

    def __init__(self, dev: pd.DataFrame, y: np.ndarray, *, task: str, metric: str,
                 is_classification: bool, splitter=None, groups=None, metrics: tuple[str, ...] = (),
                 test: pd.DataFrame | None = None, y_test: np.ndarray | None = None,
                 score_test: bool = False, random_state: int = 42,
                 time_budget_s: float | None = None):
        self.dev, self.y = dev, np.asarray(y)
        self.task, self.metric, self.is_clf = task, metric, is_classification
        self.splitter, self.groups = splitter, groups
        self.metrics = tuple(dict.fromkeys([metric, *metrics]))
        self.test, self.y_test, self.score_test = test, y_test, score_test
        self.random_state = random_state
        self.deadline = (time.monotonic() + time_budget_s) if time_budget_s else None
        self.cache: dict[str, Evaluation] = {}
        self.order: list[str] = []
        self.test_evaluations = 0

    @property
    def out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def _score_all(self, y, point, proba) -> dict[str, float]:
        out = {}
        for m in self.metrics:
            try:
                out[m] = float(score(m, y, point, y_proba=proba) if self.is_clf
                               else score(m, y, point))
            except Exception:  # a metric that does not apply to this sample is left out
                continue
        return out

    def evaluate(self, cand: Candidate, method: str = "") -> Evaluation:
        cid = candidate_id(cand)
        if cid in self.cache:
            ev = self.cache[cid]
            if method and method not in ev.methods:
                ev.methods.append(method)
            return ev
        ev = Evaluation(id=cid, candidate=cand, cv=None, methods=[method] if method else [])
        try:
            fit_predict = make_fit_predict(cand, is_classification=self.is_clf)
            if self.splitter is not None:
                ev.cv = cv_score(fit_predict, self.dev, self.y, metric=self.metric,
                                 is_classification=self.is_clf, splitter=self.splitter,
                                 groups=self.groups, random_state=self.random_state)
                ev.scores["validation"] = {self.metric: ev.cv.mean}
                seen = ~np.isnan(ev.cv.oof_pred)
                if len(self.metrics) > 1 and seen.any():  # the other metrics, pooled out of fold
                    pooled = ev.cv.oof_pred[seen]
                    point = (pooled >= 0.5).astype(int) if self.is_clf else pooled
                    extra = self._score_all(self.y[seen], point, pooled if self.is_clf else None)
                    ev.scores["validation"] = {**extra, self.metric: ev.cv.mean}
            predict = fit_predict(self.dev, self.y)
            point, proba = predict(self.dev)
            ev.scores["in_sample"] = self._score_all(self.y, point, proba)
            if self.score_test and self.test is not None and len(self.test):
                point, proba = predict(self.test)
                ev.scores["test"] = self._score_all(self.y_test, point, proba)
                self.test_evaluations += 1
        except Exception as exc:
            ev.error = f"{type(exc).__name__}: {exc}"[:200]
        self.cache[cid] = ev
        self.order.append(cid)
        return ev

    def inference(self, ev: Evaluation) -> dict[str, Any]:
        """Coefficients, p-values and information criteria on the full-rank design (families
        with statsmodels inference). Computed on demand and kept on the evaluation."""
        if "fitted" in ev.info or ev.error:
            return ev.info
        ev.info["fitted"] = False
        cand = ev.candidate
        if cand.family not in STEP_FAMILIES:
            return ev.info
        try:
            import statsmodels.api as sm

            design = build_inference_design(self.dev[cand.features])
            yv = self.y.astype(float)
            if self.is_clf:
                res = (sm.GLM(yv, design, family=SMBinaryGLM._family(cand.family)).fit(maxiter=200)
                       if cand.family in BINARY_GLM_LINKS
                       else sm.Logit(yv, design).fit(disp=0, maxiter=200))
            else:
                res = sm.OLS(yv, design).fit()
            params = {k: float(v) for k, v in res.params.items() if k != "const"}
            pvals = {k: float(v) for k, v in res.pvalues.items() if k != "const"}
            ev.info.update(
                fitted=True, coefficients=params, pvalues=pvals,
                max_p=max(pvals.values()) if pvals else None,
                aic=float(res.aic), bic=float(res.bic),
                adj_r2=float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else None)
        except Exception as exc:
            ev.info["inference_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return ev.info


# --- criterion and constraints --------------------------------------------------------------
def criterion_value(ev: Evaluation, criterion: str, evaluator: Evaluator) -> float | None:
    if ev.error:
        return None
    if criterion in ("validation", "in_sample"):
        return ev.value(evaluator.metric, criterion)
    return evaluator.inference(ev).get(criterion)


def better(criterion: str, metric: str, a: float | None, b: float | None) -> bool:
    """Is ``a`` an improvement on ``b``?"""
    if a is None or not np.isfinite(a):
        return False
    if b is None or not np.isfinite(b):
        return True
    if criterion in _MINIMIZE_CRITERIA:
        return a < b
    if criterion == "adj_r2":
        return a > b
    return a > b if metric_direction(metric) == "maximize" else a < b


def violations(ev: Evaluation, evaluator: Evaluator, rules: dict[str, Any],
               expected: dict[str, str], corr: pd.DataFrame | None) -> list[str]:
    """Why a model is not acceptable under the plan's constraints (empty when it is)."""
    out: list[str] = []
    feats = ev.candidate.features
    lim = rules.get("max_correlation")
    if lim is not None and corr is not None:
        for a, b in itertools.combinations([f for f in feats if f in corr.index], 2):
            if abs(float(corr.at[a, b])) > float(lim):
                out.append(f"{a} and {b} are correlated {float(corr.at[a, b]):+.2f}")
    if ev.error:
        return [*out, ev.error]
    if rules.get("max_p_value") is None and not (rules.get("enforce_sign") and expected):
        return out
    info = evaluator.inference(ev)
    if not info.get("fitted"):
        return out
    if rules.get("max_p_value") is not None:
        for name, p in info["pvalues"].items():
            if p > float(rules["max_p_value"]):
                out.append(f"{name} has p-value {p:.3f}")
    if rules.get("enforce_sign"):
        for name, coef in info["coefficients"].items():
            exp = expected.get(name)
            if exp and abs(coef) > 1e-12 and (coef > 0) != (exp == "+"):
                out.append(f"{name} has sign {'+' if coef > 0 else '-'}, expected {exp}")
    return out


# --- the methods ----------------------------------------------------------------------------
@dataclass
class Path:
    method: str
    family: str
    criterion: str
    steps: list[dict] = field(default_factory=list)
    final: str | None = None  # the id of the model the path ends on
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "family": self.family, "criterion": self.criterion,
                "steps": self.steps, "final": self.final, "note": self.note}


class _Run:
    """What every method shares: the evaluator, the rules and a budget of models."""

    def __init__(self, evaluator: Evaluator, family: str, factors: list[str], rules: dict,
                 expected: dict[str, str], corr: pd.DataFrame | None, hyperparams: dict):
        self.e, self.family, self.factors = evaluator, family, list(factors)
        self.rules, self.expected, self.corr = rules, expected, corr
        self.hp = hyperparams
        self.criterion = rules["criterion"]
        self.max_factors = max(1, int(rules.get("max_factors") or 4))
        self.budget = max(1, int(rules.get("max_models") or 400))
        self.must = [f for f in rules.get("must_include") or [] if f in self.factors]
        self.used = 0

    @property
    def spent(self) -> bool:
        return self.used >= self.budget or self.e.out_of_time

    def try_(self, feats: list[str], method: str) -> tuple[Evaluation, float | None, list[str]]:
        cand = Candidate(family=self.family, features=list(feats), hyperparams=dict(self.hp),
                         description=f"{self.family}/{method}")
        ev = self.e.evaluate(cand, method)
        self.used += 1
        bad = violations(ev, self.e, self.rules, self.expected, self.corr)
        return ev, criterion_value(ev, self.criterion, self.e), bad

    def is_better(self, a: float | None, b: float | None) -> bool:
        return better(self.criterion, self.e.metric, a, b)

    def start(self, path: Path, method: str) -> tuple[list[str], float | None]:
        """Where an additive path begins: empty, or on the factors every model must carry."""
        if not self.must:
            return [], None
        ev, value, _ = self.try_(self.must, method)
        self.step(path, "start", None, ev, value, 1, 0)
        return list(self.must), value

    def step(self, path: Path, action: str, factor: str | None, ev: Evaluation,
             value: float | None, tried: int, rejected: int) -> None:
        prev = path.steps[-1]["criterion"] if path.steps else None
        path.steps.append({
            "step": len(path.steps) + 1, "action": action, "factor": factor,
            "factors": list(ev.candidate.features), "n_factors": len(ev.candidate.features),
            "model": ev.id, "criterion": value,
            "change": (value - prev) if value is not None and prev is not None else None,
            "tried": tried, "rejected_by_constraints": rejected})
        path.final = ev.id


def _kinds(bad: list[str]) -> set[str]:
    """Violations with their numbers taken out, so the same one compares equal across models."""
    return {re.sub(r"[-+]?\d+\.\d+", "", b) for b in bad}


def _add_step(run: _Run, current: list[str], best: float | None, method: str):
    """The best acceptable one-factor addition, or None. A constraint the starting model
    already breaks (factors the developer said every model must carry) does not rule out an
    addition; a new one does."""
    pick, tried, rejected = None, 0, 0
    inherited = _kinds(run.try_(current, method)[2]) if current and current == run.must else set()
    for f in run.factors:
        if f in current or run.spent:
            continue
        ev, value, bad = run.try_([*current, f], method)
        tried += 1
        if _kinds(bad) - inherited:
            rejected += 1
            continue
        if run.is_better(value, pick[2] if pick else best):
            pick = (f, ev, value)
    return pick, tried, rejected


def _drop_step(run: _Run, current: list[str], best: float | None, method: str, *,
               must: bool = False):
    """The removal that helps most (or, when one must go, hurts least)."""
    pick, tried = None, 0
    for f in current:
        if len(current) <= 1 or run.spent:
            break
        if f in run.must:
            continue
        ev, value, _ = run.try_([c for c in current if c != f], method)
        tried += 1
        if value is None:
            continue
        if pick is None:
            if must or run.is_better(value, best):
                pick = (f, ev, value)
        elif run.is_better(value, pick[2]):
            pick = (f, ev, value)
    return pick, tried


def forward(run: _Run) -> Path:
    path = Path("forward", run.family, run.criterion)
    current, best = run.start(path, "forward")
    while len(current) < run.max_factors and not run.spent:
        pick, tried, rejected = _add_step(run, current, best, "forward")
        if pick is None:
            path.note = ("no further factor improves the criterion within the constraints"
                         if current else "no single factor passes the constraints")
            break
        f, ev, best = pick
        current.append(f)
        run.step(path, "add", f, ev, best, tried, rejected)
    else:
        path.note = ("stopped at the size limit" if len(current) >= run.max_factors
                     else "stopped at the budget")
    return path


def stepwise(run: _Run) -> Path:
    path = Path("stepwise", run.family, run.criterion)
    current, best = run.start(path, "stepwise")
    visited: set[frozenset] = set()
    while not run.spent:
        moved = False
        if len(current) < run.max_factors:
            pick, tried, rejected = _add_step(run, current, best, "stepwise")
            if pick is not None and frozenset([*current, pick[0]]) not in visited:
                f, ev, best = pick
                current.append(f)
                visited.add(frozenset(current))
                run.step(path, "add", f, ev, best, tried, rejected)
                moved = True
        if len(current) > 2:  # does an earlier factor still earn its place?
            pick, tried = _drop_step(run, current, best, "stepwise")
            if pick is not None and frozenset(c for c in current if c != pick[0]) not in visited:
                f, ev, best = pick
                current.remove(f)
                visited.add(frozenset(current))
                run.step(path, "drop", f, ev, best, tried, 0)
                moved = True
        if not moved:
            path.note = ("no step improves the criterion within the constraints"
                         if current else "no single factor passes the constraints")
            break
    else:
        path.note = "stopped at the budget"
    return path


def backward(run: _Run) -> Path:
    path = Path("backward", run.family, run.criterion)
    # Start from the strongest factors that can sit in one model: the shortlist is already
    # ordered by single-factor strength.
    start_size = min(len(run.factors), max(run.max_factors * 2, 8))
    current = list(dict.fromkeys([*run.must, *run.factors]))[:max(start_size, len(run.must))]
    if start_size < len(run.factors):
        path.note = f"started from the {start_size} strongest of {len(run.factors)} factors. "
    ev, best, _ = run.try_(current, "backward")
    run.step(path, "start", None, ev, best, 1, 0)
    while len(current) > 1 and not run.spent:
        over = len(current) > run.max_factors
        pick, tried = _drop_step(run, current, best, "backward", must=over)
        if pick is None:
            break
        f, ev, best = pick
        current.remove(f)
        run.step(path, "drop", f, ev, best, tried, 0)
    final_ev, _, bad = run.try_(current, "backward")
    path.final = final_ev.id
    path.note += ("ends on a model that breaks a constraint: " + "; ".join(bad[:2]) if bad
                  else "no further removal improves the criterion")
    return path


def best_subset(run: _Run) -> Path:
    path = Path("best_subset", run.family, run.criterion)
    # Size the pool of factors to the budget: every combination up to the size limit, each
    # carrying the factors every model must.
    pool = [f for f in run.factors if f not in run.must]
    room = max(0, run.max_factors - len(run.must))

    def count(n: int) -> int:
        return sum(math.comb(n, k) for k in range(1, min(room, n) + 1))

    while len(pool) > 1 and count(len(pool)) > run.budget:
        pool.pop()
    if len(pool) < len(run.factors) - len(run.must):
        path.note = (f"searched the {len(pool)} strongest of {len(run.factors)} factors "
                     f"(budget {run.budget} models). ")
    overall = None
    for k in range(0 if run.must else 1, min(room, len(pool)) + 1):
        pick, tried, rejected = None, 0, 0
        for combo in itertools.combinations(pool, k):
            if run.spent:
                break
            ev, value, bad = run.try_([*run.must, *combo], "best_subset")
            tried += 1
            if bad:
                rejected += 1
                continue
            if run.is_better(value, pick[1] if pick else None):
                pick = (ev, value)
        if pick is None:
            continue
        run.step(path, "size", None, pick[0], pick[1], tried, rejected)
        if run.is_better(pick[1], overall[1] if overall else None):
            overall = pick
    path.final = overall[0].id if overall else None
    path.note += ("best model of each size; the path ends on the best of them" if overall
                  else "no combination passes the constraints")
    return path


_RUNNERS = {"forward": forward, "backward": backward, "stepwise": stepwise,
            "best_subset": best_subset}


def run_methods(evaluator: Evaluator, *, families: list[str], factors: list[str],
                rules: dict[str, Any], expected: dict[str, str], corr: pd.DataFrame | None,
                all_columns: list[str], max_candidates: int, max_features: int | None,
                hazard_meta: dict | None, random_state: int) -> list[Path]:
    """Run every method of the plan over every family. Stepwise methods need a family with
    coefficient inference; the others go through the grid on the shortlist, so no family on the
    slate is left without a model."""
    methods = [m for m in rules.get("methods") or [] if m in METHODS]
    paths: list[Path] = []
    stepped = [m for m in methods if m != "grid"]
    grid_families = list(families) if "grid" in methods or not stepped else [
        f for f in families if f not in STEP_FAMILIES]
    # The path is walked with the first family that has coefficient tests (the slate is
    # ordered by priority). The other such families are fitted on the factor sets the paths
    # passed through, which is the comparison that matters: same factors, another link.
    walkers = [f for f in families if f in STEP_FAMILIES]
    for method in stepped if walkers else []:
        run = _Run(evaluator, walkers[0], factors, rules, expected, corr,
                   {"random_state": random_state})
        paths.append(_RUNNERS[method](run))
    visited = list(dict.fromkeys(tuple(step["factors"]) for p in paths for step in p.steps))
    for family in walkers[1:]:
        path = Path("same_factors", family, rules["criterion"],
                    note=f"the factor sets of the {walkers[0]} path(s), fitted as {family}")
        best = None
        for feats in visited:
            if evaluator.out_of_time:
                break
            run = _Run(evaluator, family, factors, rules, expected, corr,
                       {"random_state": random_state})
            ev, value, bad = run.try_(list(feats), "same_factors")
            if not bad and run.is_better(value, best[1] if best else None):
                best = (ev, value)
        path.final = best[0].id if best else None
        paths.append(path)
    if grid_families:
        space = build_search_space(evaluator.task, all_columns, evaluator.dev, evaluator.y,
                                   families=grid_families, max_features=max_features,
                                   random_state=random_state,
                                   hazard_meta=hazard_meta)[:max_candidates]
        for cand in space:  # the factors every model must carry
            cand.features = list(dict.fromkeys([*cand.features, *(
                f for f in rules.get("must_include") or [] if f in all_columns)]))
        path = Path("grid", ", ".join(grid_families), "validation")
        best = None
        for cand in space:
            if evaluator.out_of_time:
                path.note = "stopped at the time budget"
                break
            ev = evaluator.evaluate(cand, "grid")
            value = criterion_value(ev, rules["criterion"] if rules["criterion"] in (
                "validation", "in_sample") else "in_sample", evaluator)
            if better("validation", evaluator.metric, value, best[1] if best else None):
                best = (ev, value)
        path.final = best[0].id if best else None
        path.note = path.note or f"{len(space)} specification(s) from the slate"
        paths.append(path)
    return paths


# --- ranking and the ledger -----------------------------------------------------------------
def rank(evaluations: list[Evaluation], rank_by: list[dict[str, str]], *,
         complexity_penalty: float = 0.0, n_total: int = 1) -> list[Evaluation]:
    """Order the evaluated models by the plan's ranking, most important line first. A model
    with no score on a line ranks after every model that has one."""
    def key(ev: Evaluation):
        out = []
        for i, rule in enumerate(rank_by):
            v = ev.value(rule["metric"], rule["sample"])
            if v is None or not np.isfinite(v):
                out.append((1, 0.0))
                continue
            maximize = metric_direction(rule["metric"]) == "maximize"
            if i == 0 and complexity_penalty:
                pen = complexity_penalty * (len(ev.candidate.features) / max(1, n_total))
                v = v - pen if maximize else v + pen
            out.append((0, -v if maximize else v))
        return (*out, len(ev.candidate.features))
    return sorted([e for e in evaluations if not e.error], key=key)


def ledger(evaluator: Evaluator, ranked: list[Evaluation], rank_by: list[dict[str, str]]
           ) -> list[ExperimentRecord]:
    """Every model evaluated, in the order it was tried (the results.tsv of the search)."""
    first = rank_by[0]
    place = {ev.id: i for i, ev in enumerate(ranked)}
    out = []
    for i, cid in enumerate(evaluator.order):
        ev = evaluator.cache[cid]
        v = ev.value(first["metric"], first["sample"])
        out.append(ExperimentRecord(
            idx=i, label=ev.candidate.label(), family=ev.candidate.family,
            n_features=len(ev.candidate.features), metric=first["metric"],
            metric_value=float(v) if v is not None else 0.0,
            cv_std=float(ev.cv.std) if ev.cv is not None else 0.0,
            adjusted=float(place.get(cid, -1)) + 1,  # the model's rank (0 = not ranked)
            status="crash" if ev.error else ("keep" if place.get(cid) == 0 else "discard"),
            description=(f"{ev.id} via {'+'.join(ev.methods) or 'search'}"
                         + (f": {ev.error}" if ev.error else ""))))
    return out


def steppable(family: str) -> bool:
    return family in STEP_FAMILIES and family not in GLM_FAMILIES and family not in HAZARD_FAMILIES
