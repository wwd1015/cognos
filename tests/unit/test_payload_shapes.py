"""The declared shape of the ``model`` and ``backtest`` payloads (``cognos.payloads``), held
from both sides.

* Writers: every key a real run records (a classification run with a returns column, the
  rating migration demo, a time series) is declared, and so is every key of each pool member
  and backtest candidate row.
* Readers: a static look (``ast``) over ``src/cognos`` collects the string keys read from a
  recognisable payload source and asks that each is declared. It misses indirect reads (a
  payload handed to a helper under another name) on purpose; a floor on how many reads it
  recognises keeps it from going blind without anyone noticing.
* The reader check reports an undeclared key in a synthetic source, through the same function.
"""

from __future__ import annotations

import ast
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

# The reader check must keep recognising at least this many reads (it found a few more when
# this was set): a lower count means a pattern stopped being recognised.
MIN_READS = 215

# A key a reader asks for that none of the module's runs writes (with a value): which path
# writes it and why these runs do not take it. {(stage, key): why}; "champion" is pool.view.
WRITTEN_ELSEWHERE: dict[tuple[str, str], str] = {
    ("champion", "feature_importances"):
        "pool_builder.finalize records it for a tree family only; the trading run ranks a "
        "random_forest among its leaders, but the engine's pool leaves it out.",
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
_R, _P = "result", "payload"
# Parameters known to hold a payload, by file and function ("*": every function in the file):
# a helper handed the payload under a name of its own is otherwise out of sight.
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
}


@dataclass(frozen=True)
class Read:
    path: str
    function: str
    line: int
    stage: str  # "model", "backtest" or "champion"
    key: str


def _str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _empty(node) -> bool:
    return (isinstance(node, ast.Dict) and not node.keys) or (
        isinstance(node, ast.Constant) and node.value is None)


class _Function:
    """One function's bindings (name -> kind, by line) and the reads made through them."""

    def __init__(self, node, outer: dict, path: str):
        self.node = node
        self.bindings: dict[str, list] = {k: [(0, kind) for _, kind in v[-1:]]
                                          for k, v in outer.items()}
        given = {**PARAMS.get((path, "*"), {}), **PARAMS.get((path, node.name), {})}
        for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self.bindings[a.arg] = [(0, given.get(a.arg))]

    # what an expression is: ("result", stage), ("payload", stage) or ("payload", "champion")
    def kind(self, node, line: int) -> tuple[str, str] | None:
        if isinstance(node, ast.Name):
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
                if (f.attr in ("view", "from_results") and isinstance(f.value, ast.Name)
                        and f.value.id in _VIEW_MODULES):
                    return ("payload", "champion")
                if f.attr == "get" and stage == "payload" and isinstance(f.value, ast.Call):
                    # load("stages/<stage>/result.json").get("payload")
                    inner = f.value.args[0] if f.value.args else None
                    for s in _PAYLOAD_STAGES:
                        if _str(inner) == f"stages/{s}/result.json":
                            return ("payload", s)
            if isinstance(f, ast.Name):
                if f.id == "get" and stage in _PAYLOAD_STAGES and len(args) == 1:
                    return ("result", stage)
                if f.id in _LOCAL_SOURCES and stage in _PAYLOAD_STAGES and len(args) == 1:
                    return ("payload", stage)
                if f.id in _SOURCE_FUNCS and len(args) >= 2 and _str(args[1]) in _PAYLOAD_STAGES:
                    return ("payload", _str(args[1]))
                if f.id == "dict" and len(args) == 1:
                    return self.kind(args[0], line)
            return None
        if isinstance(node, ast.Subscript) and _str(node.slice) in _PAYLOAD_STAGES:
            return ("result", _str(node.slice))
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

    def bind(self) -> None:
        assigns = [n for n in _own_nodes(self.node) if isinstance(n, ast.Assign | ast.AnnAssign)]
        for n in sorted(assigns, key=lambda n: (n.lineno, n.col_offset)):
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
        def payload_stage(node, line):
            k = self.kind(node, line)
            return k[1] if k and k[0] == "payload" else None

        for n in _own_nodes(self.node):
            got = None
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and n.args and _str(n.args[0])):
                got = (payload_stage(n.func.value, n.lineno), _str(n.args[0]))
            elif (isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load)
                  and _str(n.slice)):
                got = (payload_stage(n.value, n.lineno), _str(n.slice))
            elif (isinstance(n, ast.Compare) and len(n.ops) == 1
                  and isinstance(n.ops[0], ast.In | ast.NotIn) and _str(n.left)):
                got = (payload_stage(n.comparators[0], n.lineno), _str(n.left))
            if got and got[0]:
                yield Read(path, self.node.name, n.lineno, got[0], got[1])


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
               "champion": payloads.declared("model") | payloads.declared("champion")}
    return [r for r in reads if r.key not in allowed[r.stage]]


def _all_reads() -> list[Read]:
    out = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        out.extend(collect_reads(f.read_text(encoding="utf-8"), rel))
    return out


def test_every_key_a_reader_asks_for_is_declared():
    reads = _all_reads()
    files = sorted({r.path for r in reads})
    print(f"\n{len(reads)} payload reads recognised in {len(files)} files: {', '.join(files)}")
    assert len(reads) >= MIN_READS, "the reader check recognises fewer reads than it did"
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
'''


def test_the_reader_check_reports_an_undeclared_key():
    reads = collect_reads(_SYNTHETIC)
    assert {(r.stage, r.key) for r in reads} == {
        ("backtest", "champion_id"), ("backtest", "no_such_backtest_key"),
        ("model", "no_such_model_key"), ("champion", "cv_fold_scores"),
        ("champion", "no_such_view_key"), ("backtest", "oos_metric_typo"),
        ("model", "pool_ids"), ("backtest", "no_such_key_either")}
    assert {r.key for r in undeclared_reads(reads)} == {
        "no_such_backtest_key", "no_such_model_key", "no_such_view_key", "oos_metric_typo",
        "no_such_key_either"}



# --- read, but never written ------------------------------------------------------------------
def _written(recorded) -> dict[str, set[str]]:
    """The keys the runs record with a value, per stage; the champion view over every member of
    each pool (any of them can be the champion in force), with validate's payload when the run
    went that far."""
    from cognos import pool

    out: dict[str, set[str]] = {"model": set(), "backtest": set(), "champion": set()}
    for r in recorded.values():
        views = [pool.view(r["run_dir"], r["model"], r["backtest"], r["validate"], chosen=c)
                 for c in r["model"]["pool_ids"]]
        for stage, d in [("model", r["model"]), ("backtest", r["backtest"]),
                         *(("champion", v) for v in views)]:
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
