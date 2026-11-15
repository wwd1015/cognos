"""The top-level docs name code that exists.

Every `` `path/file.py` ``, `` `path/file.py::symbol` `` and `` `module.symbol` `` in the docs a
contributor works from must resolve. ("Extending COGNOS" once pointed at a function removed
two commits earlier, and nothing noticed.)
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "cognos"
DOCS = ["CLAUDE.md", "ARCHITECTURE.md", "README.md", "FEATURES.md", "CONTEXT.md"]

# A module's dotted name by its last part: `plan.resolve` means `cognos.modeling.plan.resolve`.
MODULES: dict[str, list[str]] = {}
for _p in sorted(PKG.rglob("*.py")):
    _rel = _p.relative_to(PKG).with_suffix("")
    _parts = _rel.parts[:-1] if _rel.name == "__init__" else _rel.parts
    if _parts:
        MODULES.setdefault(_parts[-1], []).append(".".join(("cognos", *_parts)))

# Words that are a module's name and also the everyday name of an object: `state.gaps` is a
# field of the run state, `config.data` of the profile. Resolved on the class as well.
ALSO = {"state": ("cognos.engine.state", "RunState"), "config": ("cognos.config", "CognosConfig"),
        "overrides": ("cognos.engine.state", "Overrides")}
NOT_CODE = {"json", "jsonl", "yaml", "yml", "md", "py", "parquet", "csv", "html", "css", "js",
            "joblib", "toml", "txt", "lock", "prev"}

TICK = re.compile(r"`([^`\n]+)`")
PATH = re.compile(r"^(?:src/cognos/)?((?:[\w]+/)+[\w]+\.py)(?:::([\w.]+))?")
DOTTED = re.compile(r"^([a-z_]\w*)\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")


def _has(obj, chain: str) -> bool:
    for name in chain.split("."):
        fields = getattr(obj, "model_fields", None) or getattr(obj, "__dataclass_fields__", None)
        if hasattr(obj, name):
            obj = getattr(obj, name)
        elif fields and name in fields:
            return True  # a field of a model: what it holds is data, not code
        else:
            return False
    return True


def _module(path: str):
    rel = Path(path).with_suffix("")
    return importlib.import_module(".".join(("cognos", *rel.parts)))


def _with_submodules(mods: list[str]) -> list[str]:
    """`analysis.reproduce` names the package for a function in one of its modules."""
    every = [m for found in MODULES.values() for m in found]
    return mods + [m for m in every if any(m.startswith(pkg + ".") for pkg in mods)]


def _profile_key(stem: str, chain: str) -> bool:
    from cognos.config import CognosConfig

    field = CognosConfig.model_fields.get(stem)
    return bool(field) and chain.split(".")[0] in getattr(field.annotation, "model_fields", {})


FACTS = (PKG / "agents" / "facts.py").read_text(encoding="utf-8")


def _problem(ref: str) -> str | None:
    if m := PATH.match(ref):
        path, symbol = m.groups()
        if not (PKG / path).exists():
            return None if (ROOT / path).exists() or not symbol and "/" not in path else "no such file"
        return None if not symbol or _has(_module(path), symbol) else "no such name in that file"
    if m := DOTTED.match(ref):
        stem, chain = m.groups()
        if stem not in MODULES or chain.split(".")[0] in NOT_CODE:
            return None
        if any(_has(importlib.import_module(mod), chain) for mod in _with_submodules(MODULES[stem])):
            return None
        if _profile_key(stem, chain) or f'"{stem}.{chain}"' in FACTS:
            return None  # a key of the profile (`engagement.kind`) or a fact id (`model.cv_mean`)
        if stem in ALSO:
            mod, cls = ALSO[stem]
            if _has(getattr(importlib.import_module(mod), cls), chain.split(".")[0]):
                return None
        return "no such name in " + " or ".join(MODULES[stem])
    return None


@pytest.mark.parametrize("doc", DOCS)
def test_the_doc_names_code_that_exists(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    bad = sorted({f"`{ref}`: {why}" for ref in TICK.findall(text) if (why := _problem(ref))})
    assert not bad, f"{doc} names code that is not there:\n  " + "\n  ".join(bad)


def test_the_check_itself_catches_a_missing_name():
    assert _problem("modeling/plan.py::resolve") is None
    assert _problem("plan.resolve") is None
    assert _problem("state.json") is None  # a file, not a name
    assert _problem("modeling/plan.py::no_such_function") == "no such name in that file"
    assert _problem("modeling/no_such_module.py") == "no such file"
    assert _problem("plan.no_such_function").startswith("no such name in")
