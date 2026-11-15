"""The declared shape of the ``model`` and ``backtest`` payloads (``cognos.payloads``), held
from both sides.

* Writers: every key a real run records (a classification run with a returns column, the
  rating migration demo, a time series) is declared, and so is every key of each pool member
  and backtest candidate row.
* Readers: a static look (``ast``) over ``src/cognos`` collects the string keys read from a
  recognisable payload source, and from a row taken from one (a pool member out of ``pool`` /
  ``leaders``, a candidate out of ``candidates``: a loop, a comprehension, ``next(...)``,
  ``[0]``), and asks that each is declared. It misses indirect reads (a payload or a row handed
  to a helper under another name) on purpose; floors on how many reads it recognises keep it
  from going blind without anyone noticing.
* Every key read is written by one of the runs, or named in ``WRITTEN_ELSEWHERE``.
* The reader check reports an undeclared key in a synthetic source, through the same function.
"""

from __future__ import annotations

import ast
import functools
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cognos import payloads, service, synth
from cognos.config import CognosConfig
from cognos.orchestrator import Orchestrator

SRC = Path(__file__).resolve().parents[2] / "src" / "cognos"
STAGES = ["explore", "ideate", "model", "backtest"]

# A reader that asks for a key no writer writes: named here so the test passes while saying so.
# {(path relative to src/cognos, function, stage, key): why}
KNOWN_GAPS: dict[tuple[str, str, str, str], str] = {}

# The reader check must keep recognising at least this many reads of payload keys and of row
# keys (it found a few more when these were set): a lower count means a pattern stopped being
# recognised.
MIN_READS = 230
MIN_ROW_READS = 110

# A key a reader asks for that none of the module's runs writes (with a value): which path
# writes it and why these runs do not take it. {(stage, key): why}; "champion" is pool.view.
WRITTEN_ELSEWHERE: dict[tuple[str, str], str] = {
    ("champion", "feature_importances"):
        "pool_builder.finalize records it for a tree family only; the trading run ranks a "
        "random_forest among its leaders, but the engine's pool leaves it out.",
    ("candidate", "error"):
        "stages/backtest.py records it only for a pool member whose scorer fails to score the "
        "comparison sample (the except branch); every member of these runs scores.",
}


# --- writers ------------------------------------------------------------------------------------
def _run(cfg: CognosConfig, root: Path, stages: list[str] = STAGES) -> dict:
    cfg.stages.enabled = list(stages)
    orch = Orchestrator(cfg, runs_root=str(root))
    for s in stages:
        orch.run_stage(s)
    out = {s: orch.ctx.require(s).payload for s in ("model", "backtest")}
    out["validate"] = orch.ctx.require("validate").payload if "validate" in stages else None
    out["run_dir"] = orch.ctx.run_dir
    return out


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> dict[str, dict[str, dict]]:
    """Three small runs through backtest, each bringing keys of its own: a classification run
    with a returns column (trading) and the ensemble benchmark, the rating migration demo
    (migration, stress, portfolio; on through validate, which scores the sealed holdout the
    champion view reports) and a time series (forecast analysis)."""
    out = {}
    root = tmp_path_factory.mktemp("trading")
    df = synth.make_classification_dataset(n=400)
    df["ret"] = np.random.default_rng(0).normal(0, 0.02, len(df))
    df.to_csv(root / "d.csv", index=False, encoding="utf-8")
    out["trading"] = _run(CognosConfig.from_dict({
        "name": "trading", "task": "classification",
        "data": {"path": str(root / "d.csv"), "target": "target", "drop_columns": ["ret"]},
        "metric": {"name": "roc_auc"},
        # the ensemble benchmark (ADR-0007) writes challenger_benchmark
        "search": {"max_candidates": 6, "cv_folds": 3, "ensemble": True},
        "backtest": {"returns_column": "ret"}}), root / "runs")
    root = tmp_path_factory.mktemp("migration")
    out["migration"] = _run(service.demo_config("migration", root, n=120, search_budget=4), root,
                            [*STAGES, "validate"])
    root = tmp_path_factory.mktemp("timeseries")
    out["timeseries"] = _run(service.demo_config("timeseries", root, search_budget=4), root)
    return out


def _undeclared(shape, payload: dict) -> set[str]:
    return set(payload) - set(shape.model_fields)


@pytest.mark.parametrize("run", ["trading", "migration", "timeseries"])
def test_every_key_a_writer_records_is_declared(recorded, run):
    model, backtest = recorded[run]["model"], recorded[run]["backtest"]
    payloads.ModelPayload.model_validate(model)
    payloads.BacktestPayload.model_validate(backtest)
    assert not _undeclared(payloads.ModelPayload, model), "model"
    assert not _undeclared(payloads.BacktestPayload, backtest), "backtest"
    for m in [*model["pool"], *model["leaders"]]:
        assert not _undeclared(payloads.PoolMember, m), m["id"]
    for c in backtest["candidates"]:
        assert not _undeclared(payloads.BacktestCandidate, c), c["id"]


def test_task_specific_keys_were_seen(recorded):
    """The runs reach the parts that only some tasks write, so their keys are checked too."""
    assert recorded["trading"]["backtest"]["trading"]
    assert recorded["migration"]["model"]["migration"]
    assert recorded["migration"]["backtest"]["stress_testing"]
    assert recorded["migration"]["backtest"]["portfolio_analysis"]
    assert recorded["timeseries"]["backtest"]["forecast_analysis"]


def test_the_champion_view_is_declared(recorded):
    from cognos import pool

    model, backtest = recorded["trading"]["model"], recorded["trading"]["backtest"]
    view = pool.view(tempfile.gettempdir(), model, backtest)
    payloads.ChampionView.model_validate(view)
    assert not _undeclared(payloads.ChampionView, view)


# --- readers ------------------------------------------------------------------------------------
_PAYLOAD_STAGES = ("model", "backtest")
_SOURCE_FUNCS = {"read_payload", "_payload"}  # f(run_dir, "<stage>")
_LOCAL_SOURCES = {"payload", "payload_of"}  # a local helper: f("<stage>") -> payload
_VIEW_MODULES = {"pool", "pool_mod"}
_R, _P, _L, _W = "result", "payload", "list", "row"
_MEMBER, _CANDIDATE = "member", "candidate"  # a pool member, a backtest candidate row
ROWS = {_MEMBER: payloads.PoolMember, _CANDIDATE: payloads.BacktestCandidate}
# The payload keys that hold rows: {payload stage: {key: row}}.
_LISTS = {"model": {"pool": _MEMBER, "leaders": _MEMBER},
          "champion": {"pool": _MEMBER, "leaders": _MEMBER, "admissible_set": _MEMBER},
          "backtest": {"candidates": _CANDIDATE}}
_COMPS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
# Parameters known to hold a payload, a list of rows or one row, by file and function ("*":
# every function in the file): a helper handed one under a name of its own is otherwise out of
# sight.
PARAMS: dict[tuple[str, str], dict[str, tuple[str, str]]] = {
    ("ui/panels/backtest.py", "*"): {"res": (_R, "backtest"), "p": (_P, "backtest")},
    ("ui/panels/model.py", "*"): {"res": (_R, "model"), "p": (_P, "model")},
    ("stages/document.py", "_oos_lines"): {"mp": (_P, "champion"), "bp": (_P, "backtest")},
    ("stages/document.py", "_selection_body"): {"mp": (_P, "champion"), "bp": (_P, "backtest")},
    ("stages/document.py", "_engagement_body"): {"mp": (_P, "champion")},
    ("stages/validate.py", "_screen_record"): {"model": (_P, "champion")},
    ("stages/validate.py", "_score_holdout"): {"model": (_P, "champion")},
    ("stages/backtest.py", "_trading_metrics"): {"model": (_P, "champion")},
    ("stages/backtest.py", "_pbo"): {"model": (_P, "champion")},
    ("engine/gates.py", "_rename"): {"model": (_P, "model")},
    ("pool.py", "champion_id"): {"model": (_P, "model"), "backtest": (_P, "backtest")},
    ("pool.py", "member"): {"model": (_P, "model")},
    ("pool.py", "view"): {"model": (_P, "model"), "backtest": (_P, "backtest")},
    ("trials.py", "_test_metric"): {"backtest": (_P, "backtest")},
    # rows and lists of rows
    ("pool.py", "detail"): {"entry": (_W, _MEMBER)},
    ("pool.py", "scorer_path"): {"entry": (_W, _MEMBER)},
    ("ui/panels/model.py", "tags"): {"c": (_W, _MEMBER)},  # each of the leaders
    ("ui/panels/backtest.py", "_champion_proposal"): {"cands": (_L, _CANDIDATE)},
    ("ui/panels/backtest.py", "_champion_evidence"): {"cands": (_L, _CANDIDATE)},
}
# A function that handles several stages' results under one name, and the variable: inside
# ``if gate == "<gate>":`` that variable is the result of the stage the gate reviews
# (``graph.STAGE_OF_GATE``). {(path, function): variable}
GATE_BRANCHES: dict[tuple[str, str], str] = {("engine/gates.py", "handle"): "res"}


@dataclass(frozen=True)
class Read:
    path: str
    function: str
    line: int
    stage: str  # "model", "backtest", "champion"; or a row: "member", "candidate"
    key: str


def _str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _empty(node) -> bool:
    return (isinstance(node, ast.Dict) and not node.keys) or (
        isinstance(node, ast.List | ast.Tuple) and not node.elts) or (
        isinstance(node, ast.Constant) and node.value is None)


def _index(node) -> bool:
    """A constant position: ``[0]``, ``[-1]``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _gate_of(test) -> str | None:
    """``gate == "<g>"``, alone or as one part of an ``and``."""
    parts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    for t in parts:
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "gate"
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq) and _str(t.comparators[0])):
            return _str(t.comparators[0])
    return None


class _Function:
    """One function's bindings (name -> kind, by line) and the reads made through them."""

    def __init__(self, node, outer: dict, path: str):
        self.node, self.path = node, path
        self.bindings: dict[str, list] = {k: [(0, kind) for _, kind in v[-1:]]
                                          for k, v in outer.items()}
        given = {**PARAMS.get((path, "*"), {}), **PARAMS.get((path, node.name), {})}
        for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self.bindings[a.arg] = [(0, given.get(a.arg))]
        self.parents = {c: n for n in [node, *_own_nodes(node)] for c in ast.iter_child_nodes(n)}
        self.branch_var = GATE_BRANCHES.get((path, node.name))

    def _ancestors(self, node):
        """(ancestor, the child of it the path came through), innermost first."""
        while node in self.parents:
            parent = self.parents[node]
            yield parent, node
            node = parent

    def _targets(self, target, iterable, line: int) -> list[tuple[str, tuple | None]]:
        """The names a ``for`` binds and what each is: an element of a list of rows (also
        through ``enumerate``); anything else binds to nothing known."""
        if isinstance(target, ast.Name):
            k = self.kind(iterable, line)
            return [(target.id, (_W, k[1]) if k and k[0] == _L else None)]
        names = [e.id for e in getattr(target, "elts", []) if isinstance(e, ast.Name)]
        if (isinstance(target, ast.Tuple) and len(target.elts) == 2 and len(names) == 2
                and isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name)
                and iterable.func.id == "enumerate" and iterable.args):
            k = self.kind(iterable.args[0], line)
            return [(names[0], None), (names[1], (_W, k[1]) if k and k[0] == _L else None)]
        return [(n, None) for n in names]

    def _name(self, node, line: int):
        """A comprehension's own variable, or the gate branch's result; ``False`` when neither
        applies (the function's bindings decide)."""
        came = node
        for anc, child in self._ancestors(node):
            if isinstance(anc, _COMPS):
                # the generators this name can see: an iterable sees only those before it
                gens = anc.generators
                if child in gens:
                    j = gens.index(child)
                    gens = gens[:j] if came is child.iter else gens[:j + 1]
                for gen in reversed(gens):
                    if node.id in _names(gen.target):
                        return dict(self._targets(gen.target, gen.iter, line))[node.id]
            if (self.branch_var == node.id and isinstance(anc, ast.If)
                    and child in anc.body and _gate_of(anc.test)):
                from cognos.engine.graph import STAGE_OF_GATE

                stage = STAGE_OF_GATE.get(_gate_of(anc.test))
                return (_R, stage) if stage in _PAYLOAD_STAGES else None
            came = child
        return False

    # what an expression is: ("result", stage), ("payload", stage), ("payload", "champion"),
    # ("list", row), ("row", row) or ("map", row) (rows keyed by id)
    def kind(self, node, line: int) -> tuple[str, str] | None:
        if isinstance(node, ast.Name):
            own = self._name(node, line)
            if own is not False:
                return own
            seen = [k for ln, k in self.bindings.get(node.id, []) if ln <= line]
            return seen[-1] if seen else None
        if isinstance(node, ast.Call):
            f, args = node.func, node.args
            stage = _str(args[0]) if args else None
            if isinstance(f, ast.Attribute):
                if f.attr in ("require", "get") and stage in _PAYLOAD_STAGES and len(args) == 1:
                    return ("result", stage)
                if f.attr == "champion" and not args:
                    return ("payload", "champion")
                if isinstance(f.value, ast.Name) and f.value.id in _VIEW_MODULES:
                    if f.attr in ("view", "from_results"):
                        return ("payload", "champion")
                    if f.attr == "member":
                        return (_W, _MEMBER)
                if f.attr == "get" and stage == "payload" and isinstance(f.value, ast.Call):
                    # load("stages/<stage>/result.json").get("payload")
                    inner = f.value.args[0] if f.value.args else None
                    for s in _PAYLOAD_STAGES:
                        if _str(inner) == f"stages/{s}/result.json":
                            return ("payload", s)
                if f.attr == "get" and args:
                    k = self.kind(f.value, line)
                    if k and k[0] == _P and stage in _LISTS.get(k[1], {}):
                        return (_L, _LISTS[k[1]][stage])
                    if k and k[0] == "map" and stage is None:
                        return (_W, k[1])
            if isinstance(f, ast.Name):
                if f.id == "get" and stage in _PAYLOAD_STAGES and len(args) == 1:
                    return ("result", stage)
                if f.id in _LOCAL_SOURCES and stage in _PAYLOAD_STAGES and len(args) == 1:
                    return ("payload", stage)
                if f.id in _SOURCE_FUNCS and len(args) >= 2 and _str(args[1]) in _PAYLOAD_STAGES:
                    return ("payload", _str(args[1]))
                if f.id == "dict" and len(args) == 1:
                    return self.kind(args[0], line)
                if f.id in ("list", "sorted", "reversed", "tuple") and args:
                    k = self.kind(args[0], line)
                    return k if k and k[0] == _L else None
                if f.id == "member" and self.path == "pool.py":
                    return (_W, _MEMBER)
                if f.id == "next" and args and isinstance(args[0], ast.GeneratorExp):
                    k = self.kind(args[0], line)  # next((m for m in rows if ...), None)
                    return (_W, k[1]) if k and k[0] == _L else None
            return None
        if isinstance(node, ast.Subscript) and _str(node.slice) in _PAYLOAD_STAGES:
            return ("result", _str(node.slice))
        if isinstance(node, ast.Subscript):
            k = self.kind(node.value, line)
            key = _str(node.slice)
            if k and k[0] == _P and key in _LISTS.get(k[1], {}):
                return (_L, _LISTS[k[1]][key])
            if k and k[0] == _L and _index(node.slice):
                return (_W, k[1])
            if k and k[0] == _L and isinstance(node.slice, ast.Slice):
                return k
            if k and k[0] == "map" and key is None and not isinstance(node.slice, ast.Slice):
                return (_W, k[1])
            return None
        if isinstance(node, ast.Attribute) and node.attr == "payload":
            k = self.kind(node.value, line)
            return ("payload", k[1]) if k and k[0] == "result" else None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            k = self.kind(node.values[0], line)
            if k and all(_empty(v) for v in node.values[1:]):
                return k
            return None
        if isinstance(node, ast.IfExp):
            a, b = self.kind(node.body, line), self.kind(node.orelse, line)
            if a and (b == a or _empty(node.orelse)):
                return a
            if b and _empty(node.body):
                return b
            return None
        if isinstance(node, ast.ListComp | ast.GeneratorExp) and len(node.generators) == 1:
            # [m for m in rows if ...]: some of the same rows
            gen = node.generators[0]
            if isinstance(node.elt, ast.Name) and _same(node.elt, gen.target):
                k = self.kind(gen.iter, line)
                return k if k and k[0] == _L else None
        if isinstance(node, ast.DictComp) and len(node.generators) == 1:
            # {m["id"]: m for m in rows}
            gen = node.generators[0]
            if isinstance(node.value, ast.Name) and _same(node.value, gen.target):
                k = self.kind(gen.iter, line)
                return ("map", k[1]) if k and k[0] == _L else None
        return None

    def bind(self) -> None:
        nodes = [n for n in _own_nodes(self.node)
                 if isinstance(n, ast.Assign | ast.AnnAssign | ast.For | ast.AsyncFor)]
        for n in sorted(nodes, key=lambda n: (n.lineno, n.col_offset)):
            if isinstance(n, ast.For | ast.AsyncFor):
                for name, k in self._targets(n.target, n.iter, n.lineno):
                    self.bindings.setdefault(name, []).append((n.lineno, k))
                continue
            if n.value is None:
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                pairs = ([(t, n.value)] if not isinstance(t, ast.Tuple) else
                         list(zip(t.elts, n.value.elts, strict=False))
                         if isinstance(n.value, ast.Tuple) else [])
                for name, value in pairs:
                    if isinstance(name, ast.Name):
                        self.bindings.setdefault(name.id, []).append(
                            (n.lineno, self.kind(value, n.lineno)))

    def reads(self, path: str):
        def read_of(node, line):
            k = self.kind(node, line)
            return k[1] if k and k[0] in (_P, _W) else None

        for n in _own_nodes(self.node):
            got = None
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and n.args and _str(n.args[0])):
                got = (read_of(n.func.value, n.lineno), _str(n.args[0]))
            elif (isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load)
                  and _str(n.slice)):
                got = (read_of(n.value, n.lineno), _str(n.slice))
            elif (isinstance(n, ast.Compare) and len(n.ops) == 1
                  and isinstance(n.ops[0], ast.In | ast.NotIn) and _str(n.left)):
                got = (read_of(n.comparators[0], n.lineno), _str(n.left))
            if got and got[0]:
                yield Read(path, self.node.name, n.lineno, got[0], got[1])


def _names(target) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _same(name, target) -> bool:
    return isinstance(target, ast.Name) and name.id == target.id


def _own_nodes(func):
    """The nodes of a function's body, not descending into nested functions or classes."""
    stack = list(func.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                stack.append(c)


def collect_reads(source: str, path: str = "<string>") -> list[Read]:
    """Every recognised read of a model / backtest / champion payload key in ``source``."""
    out: list[Read] = []

    def visit(node, outer):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                fn = _Function(child, outer, path)
                fn.bind()
                out.extend(fn.reads(path))
                visit(child, fn.bindings)
            else:
                visit(child, outer)

    visit(ast.parse(source), {})
    return out


def undeclared_reads(reads: list[Read]) -> list[Read]:
    allowed = {"model": payloads.declared("model"), "backtest": payloads.declared("backtest"),
               "champion": payloads.declared("model") | payloads.declared("champion"),
               **{row: frozenset(shape.model_fields) for row, shape in ROWS.items()}}
    return [r for r in reads if r.key not in allowed[r.stage]]


@functools.cache
def _all_reads() -> tuple[Read, ...]:
    out: list[Read] = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        out.extend(collect_reads(f.read_text(encoding="utf-8"), rel))
    return tuple(out)


def test_every_key_a_reader_asks_for_is_declared():
    reads = _all_reads()
    rows = [r for r in reads if r.stage in ROWS]
    for what, got in (("payload", [r for r in reads if r.stage not in ROWS]), ("row", rows)):
        files = sorted({r.path for r in got})
        print(f"\n{len(got)} {what} reads recognised in {len(files)} files: {', '.join(files)}")
    assert len(reads) - len(rows) >= MIN_READS, "the reader check recognises fewer reads than it did"
    assert len(rows) >= MIN_ROW_READS, "the reader check recognises fewer row reads than it did"
    bad = [r for r in undeclared_reads(reads)
           if (r.path, r.function, r.stage, r.key) not in KNOWN_GAPS]
    assert not bad, "\n".join(f"{r.path}:{r.line} {r.function}() reads {r.stage}[{r.key!r}]"
                              for r in bad)


def test_the_known_payload_parameters_still_exist():
    """A renamed helper would quietly drop out of the reader check."""
    for path, function in PARAMS:
        tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert function == "*" or function in names, f"{path}: {function}"
        if function != "*":
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == function)
            assert set(PARAMS[(path, function)]) <= {a.arg for a in fn.args.args}, function


def test_the_gate_branches_are_still_read():
    """The gate handler still names its result as it did, and the reads inside its branches
    are still attributed to the stage each gate reviews."""
    from cognos.engine.graph import STAGE_OF_GATE

    reads = _all_reads()
    for (path, function), var in GATE_BRANCHES.items():
        tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
        assert var in {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}, function
        gates = {_gate_of(n.test) for n in ast.walk(fn) if isinstance(n, ast.If)} - {None}
        stages = {STAGE_OF_GATE.get(g) for g in gates} & set(_PAYLOAD_STAGES)
        assert stages, function
        assert stages <= {r.stage for r in reads if (r.path, r.function) == (path, function)}


def test_known_gaps_are_still_read():
    """A gap that is no longer read (fixed, or the reader moved) leaves the list."""
    seen = {(r.path, r.function, r.stage, r.key) for r in undeclared_reads(_all_reads())}
    assert not set(KNOWN_GAPS) - seen


# --- the check catches what it is for ---------------------------------------------------------
_SYNTHETIC = '''
def stage(ctx, run_dir):
    bt = ctx.require("backtest").payload
    a = bt.get("champion_id")
    b = bt["no_such_backtest_key"]
    res = ctx.get("model")
    p = (res.payload or {}) if res else {}
    c = "no_such_model_key" in p
    champ = ctx.champion()
    d = champ.get("cv_fold_scores"), champ.get("no_such_view_key")
    rec = read_payload(run_dir, "backtest")
    e = rec.get("oos_metric_typo")
    p = {}
    f = p.get("unrelated")
    m, b = payload_of("model"), payload_of("backtest")
    g = m["pool_ids"], b.get("no_such_key_either")
    for c in (bt.get("candidates") or []):
        h = c["metrics"], c.get("no_such_candidate_key")
    leader = next((x for x in m["leaders"] if x["id"] == a), None)
    i = leader.get("no_such_member_key"), champ["admissible_set"][0]["score"]
    j = [c["label"] for c in b.get("candidates") or []]
    k = {c["id"]: c for c in bt.get("candidates") or []}.get(a)["flags"]
    m = {"pool": [{"id": "x"}]}
    n = [z.get("unrelated_too") for z in m["pool"]]


def handle(gate, results):
    res = results.get(gate)
    if gate == "gate_champion":
        p = res.payload or {}
        o = p.get("recommended_id")
    if gate == "gate_data":
        q = res.payload.get("unrelated_explore_key")
'''


def test_the_reader_check_reports_an_undeclared_key(monkeypatch):
    monkeypatch.setitem(GATE_BRANCHES, ("<string>", "handle"), "res")
    reads = collect_reads(_SYNTHETIC)
    assert {(r.stage, r.key) for r in reads} == {
        ("backtest", "champion_id"), ("backtest", "no_such_backtest_key"),
        ("model", "no_such_model_key"), ("champion", "cv_fold_scores"),
        ("champion", "no_such_view_key"), ("backtest", "oos_metric_typo"),
        ("model", "pool_ids"), ("backtest", "no_such_key_either"),
        ("backtest", "candidates"), ("candidate", "metrics"), ("candidate", "no_such_candidate_key"),
        ("model", "leaders"), ("member", "id"), ("member", "no_such_member_key"),
        ("champion", "admissible_set"), ("member", "score"), ("candidate", "label"),
        ("candidate", "id"), ("candidate", "flags"), ("backtest", "recommended_id")}
    assert {r.key for r in undeclared_reads(reads)} == {
        "no_such_backtest_key", "no_such_model_key", "no_such_view_key", "oos_metric_typo",
        "no_such_key_either", "no_such_candidate_key", "no_such_member_key"}



# --- read, but never written ------------------------------------------------------------------
def _written(recorded) -> dict[str, set[str]]:
    """The keys the runs record with a value, per stage; the champion view over every member of
    each pool (any of them can be the champion in force), with validate's payload when the run
    went that far; and per row, over every pool member, leader and backtest candidate."""
    from cognos import pool

    out: dict[str, set[str]] = {"model": set(), "backtest": set(), "champion": set(),
                                **{row: set() for row in ROWS}}
    for r in recorded.values():
        views = [pool.view(r["run_dir"], r["model"], r["backtest"], r["validate"], chosen=c)
                 for c in r["model"]["pool_ids"]]
        for stage, d in [("model", r["model"]), ("backtest", r["backtest"]),
                         *(("champion", v) for v in views),
                         *((_MEMBER, m) for m in [*r["model"]["pool"], *r["model"]["leaders"]]),
                         *((_CANDIDATE, c) for c in r["backtest"]["candidates"])]:
            out[stage] |= {k for k, v in d.items() if v is not None}
    return out


def _unwritten_reads(recorded) -> dict[tuple[str, str], list[Read]]:
    written = _written(recorded)
    out: dict[tuple[str, str], list[Read]] = {}
    for r in _all_reads():
        if r.key not in written[r.stage]:
            out.setdefault((r.stage, r.key), []).append(r)
    return out


def test_every_key_a_reader_asks_for_is_written(recorded):
    """A key a writer stopped writing while the declaration stayed: the original bug. Every key
    read must be recorded (not as None) by one of the runs, or be named in WRITTEN_ELSEWHERE."""
    missing = {k: v for k, v in _unwritten_reads(recorded).items()
               if k not in WRITTEN_ELSEWHERE
               and not any((r.path, r.function, r.stage, r.key) in KNOWN_GAPS for r in v)}
    assert not missing, "\n".join(
        f"{stage}[{key!r}] is read ({', '.join(f'{r.path}:{r.line}' for r in reads)}) "
        f"but no run writes it" for (stage, key), reads in missing.items())


def test_written_elsewhere_is_still_read_and_still_unwritten(recorded):
    """An entry whose key is no longer read, or that a run now writes, leaves the list."""
    unwritten = _unwritten_reads(recorded)
    assert not set(WRITTEN_ELSEWHERE) - set(unwritten)
