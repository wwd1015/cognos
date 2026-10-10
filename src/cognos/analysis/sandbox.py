"""Run Python the Data Analyst wrote, without trusting it.

Two layers. **Before** anything runs, the script's syntax tree is checked against a short list of
what an analysis needs: it may import only numerical libraries, and it may not open files, reach
the network or the operating system, touch private attributes, or evaluate strings. **Then** it
runs in a separate Python process (isolated mode, an empty working directory, restricted
builtins, a wall-clock limit) that is given a copy of the dataset and nothing else, and hands back
values, tables and chart specs through the three ``emit_*`` functions.

This keeps honest mistakes and casual misuse out. It is not a security boundary against a
determined adversary writing Python; ``analysis.allow_code: false`` turns the capability off, and
every script is kept as an artifact and re-run at validation (ADR-0012).

Inside a script:
    df        the dataset (pandas DataFrame), every column including the target
    TARGET    the target column's name ("" when it is not decided yet)
    pd, np    pandas and numpy (also importable: math, statistics, scipy, statsmodels, sklearn)
    emit_value(name, number_or_text)
    emit_table(name, dataframe)                      # first 50 rows are kept
    emit_chart(kind, x, series, title="", x_title="", y_title="", y_format="")
        kind: bar | line | scatter | histogram;  series: {"name": [y, ...]} or [y, ...]
    emit_heatmap(x, y, z, title="")
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "statistics", "scipy", "statsmodels", "sklearn",
                   "itertools", "collections", "functools", "datetime", "re"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
                   "memoryview", "help", "__builtins__"}
# Methods that read or write outside the process, or evaluate strings.
FORBIDDEN_ATTRS = {
    "to_csv", "to_parquet", "to_pickle", "to_sql", "to_excel", "to_hdf", "to_feather", "to_json",
    "to_clipboard", "to_stata", "to_orc", "to_gbq", "to_html", "to_latex", "to_markdown",
    "read_csv", "read_parquet", "read_pickle", "read_sql", "read_sql_query", "read_sql_table",
    "read_excel", "read_hdf", "read_feather", "read_json", "read_html", "read_xml", "read_table",
    "read_fwf", "read_clipboard", "read_stata", "read_sas", "read_spss", "read_orc", "read_gbq",
    "eval", "query", "load", "save", "savez", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "dump", "system", "popen", "ctypeslib", "f2py", "testing", "distutils",
}
MAX_CODE_CHARS = 8000

_RUNNER = r'''
import builtins, json, math, sys
import numpy as np
import pandas as pd

code_path, data_path, out_path, target = sys.argv[1:5]
ALLOWED = set(sys.argv[5].split(","))
out = {"values": {}, "tables": [], "charts": [], "stdout": ""}
_printed = []

def _cell(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 6)

def emit_value(name, value):
    out["values"][str(name)[:60]] = _cell(value)

def emit_table(name, frame):
    frame = pd.DataFrame(frame).reset_index(drop=False) if not isinstance(frame, pd.DataFrame) else frame
    head = frame.head(50)
    out["tables"].append({"name": str(name)[:80], "columns": [str(c) for c in head.columns],
                          "rows": [[_cell(v) for v in row] for row in head.itertuples(index=False)],
                          "n_rows": int(len(frame))})

def emit_chart(kind, x, series, title="", x_title="", y_title="", y_format=""):
    if not isinstance(series, dict):
        series = {y_title or "value": series}
    out["charts"].append({"kind": kind, "title": title, "x_title": x_title, "y_title": y_title,
                          "y_format": y_format, "x": [_cell(v) for v in list(x)],
                          "series": [{"name": str(k), "y": [_cell(v) for v in list(v_)]}
                                     for k, v_ in series.items()]})

def emit_heatmap(x, y, z, title=""):
    out["charts"].append({"kind": "heatmap", "title": title, "x": [_cell(v) for v in list(x)],
                          "y": [_cell(v) for v in list(y)],
                          "z": [[_cell(v) for v in list(row)] for row in list(z)]})

def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name!r} is not allowed in an analysis script")
    return __import__(name, globals, locals, fromlist, level)

def _print(*args, **kwargs):
    _printed.append(" ".join(str(a) for a in args))

SAFE = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "divmod", "hash",
        "True", "False", "None", "ValueError", "TypeError", "KeyError", "IndexError",
        "ZeroDivisionError", "Exception", "ArithmeticError", "StopIteration"]
safe = {k: getattr(builtins, k) for k in SAFE if hasattr(builtins, k)}
safe["__import__"] = _import
safe["print"] = _print
np.random.seed(0)
df = pd.read_parquet(data_path)
scope = {"__builtins__": safe, "__name__": "analysis", "df": df, "pd": pd, "np": np,
         "TARGET": target, "emit_value": emit_value, "emit_table": emit_table,
         "emit_chart": emit_chart, "emit_heatmap": emit_heatmap}
status = {"ok": True, "error": ""}
try:
    exec(compile(open(code_path, encoding="utf-8").read(), "analysis.py", "exec"), scope)
except BaseException as exc:
    status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600]}
out["stdout"] = "\n".join(_printed)[:4000]
out.update(status)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def validate(code: str) -> list[str]:
    """Why a script may not run (empty when it may). Purely syntactic."""
    if not code.strip():
        return ["the script is empty"]
    if len(code) > MAX_CODE_CHARS:
        return [f"the script is longer than {MAX_CODE_CHARS} characters; split the analysis"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]
    errs: list[str] = []
    emits = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            if isinstance(node, ast.ImportFrom) and node.level:
                errs.append("relative imports are not allowed")
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    errs.append(f"import of {name!r} is not allowed (allowed: "
                                f"{', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                errs.append(f"{node.id!r} is not available in an analysis script")
            emits = emits or node.id.startswith("emit_")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                errs.append(f"private attribute {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                errs.append(f".{node.attr} is not allowed: a script reads only `df` and returns "
                            "results through emit_value / emit_table / emit_chart")
        elif isinstance(node, ast.Global | ast.Nonlocal | ast.AsyncFunctionDef | ast.ClassDef):
            errs.append(f"{type(node).__name__} is not allowed in an analysis script")
    if not errs and not emits:
        errs.append("the script returns nothing: call emit_value, emit_table or emit_chart")
    return list(dict.fromkeys(errs))[:8]


def sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def run(code: str, data_path: str | Path, *, target: str = "", timeout_s: float = 30.0) -> dict[str, Any]:
    """Execute a validated script against the dataset snapshot. Never raises: a failure is
    ``{"ok": False, "error": ...}``."""
    problems = validate(code)
    if problems:
        return {"ok": False, "error": "rejected before running: " + "; ".join(problems),
                "values": {}, "tables": [], "charts": [], "stdout": ""}
    with tempfile.TemporaryDirectory(prefix="cognos_analysis_") as tmp:
        work = Path(tmp)
        (work / "analysis.py").write_text(code, encoding="utf-8")
        (work / "runner.py").write_text(_RUNNER, encoding="utf-8")
        out_path = work / "out.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(work / "runner.py"), str(work / "analysis.py"),
                 str(Path(data_path).resolve()), str(out_path), target,
                 ",".join(sorted(ALLOWED_IMPORTS))],
                cwd=work, capture_output=True, text=True, timeout=timeout_s,
                env={"PATH": "", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg",
                     **{k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP") if k in os.environ}})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout_s:.0f}s", "values": {},
                    "tables": [], "charts": [], "stdout": ""}
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return {"ok": False, "error": "the script produced no result: " + " | ".join(tail)[:500],
                    "values": {}, "tables": [], "charts": [], "stdout": ""}
