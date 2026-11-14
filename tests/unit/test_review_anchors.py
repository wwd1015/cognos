"""The review stage resolves code anchors against the installed package, not a checkout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognos
from cognos.artifacts import ArtifactRef, Verdict
from cognos.okf import OKFBundle, OKFConcept
from cognos.stages import review


class _Ctx(SimpleNamespace):
    """The three things the review stage reads from a run context."""

    def resolve(self, rel: str) -> Path:
        return self.run_dir / rel

    def save_json(self, rel: str, obj) -> ArtifactRef:
        p = self.run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")
        return ArtifactRef(name=p.name, kind="json", path=rel)


def _ctx(tmp_path: Path, body: str) -> _Ctx:
    run_dir = tmp_path / "run"
    b = OKFBundle(run_dir / "docs")
    b.add(OKFConcept(name="backtest", type="backtest", title="Backtest", body=body))
    b.finalize("White paper")
    return _Ctx(run_dir=run_dir, docs_dir=run_dir / "docs")


GOOD = ("{@code:src/cognos/stages/backtest.py} and "
        "{@code:src/cognos/stages/stat_tests.py#run_battery}")


@pytest.fixture()
def wheel_install(tmp_path, monkeypatch):
    """A copy of the package with no repository around it, as a wheel installs it."""
    site = tmp_path / "site-packages"
    shutil.copytree(Path(cognos.__file__).parent, site / "cognos",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert not (site.parent / "pyproject.toml").exists()
    monkeypatch.setattr(review, "package_dir", lambda: site / "cognos")
    return site / "cognos"


def test_anchors_resolve_without_a_checkout(tmp_path, wheel_install):
    res = review.ReviewStage().run(_ctx(tmp_path, GOOD))
    assert res.verdict is not Verdict.BLOCK
    assert res.payload["missing_code_paths"] == [] and res.payload["missing_symbols"] == []
    assert res.payload["n_code_anchors"] == 2


def test_a_stale_symbol_still_blocks(tmp_path, wheel_install):
    body = GOOD + " {@code:src/cognos/stages/backtest.py#no_such_function_here}"
    res = review.ReviewStage().run(_ctx(tmp_path, body))
    assert res.verdict is Verdict.BLOCK
    assert res.payload["missing_symbols"] == ["src/cognos/stages/backtest.py#no_such_function_here"]


def test_a_missing_file_or_a_path_outside_the_package_blocks(tmp_path, wheel_install):
    body = "{@code:src/cognos/stages/gone.py} {@code:../../etc/passwd}"
    res = review.ReviewStage().run(_ctx(tmp_path, body))
    assert res.verdict is Verdict.BLOCK
    assert res.payload["missing_code_paths"] == ["src/cognos/stages/gone.py", "../../etc/passwd"]


def test_the_result_names_no_machine(tmp_path):
    res = review.ReviewStage().run(_ctx(tmp_path, GOOD))
    assert "repo_root" not in res.payload
    text = json.dumps(res.payload)
    assert str(tmp_path) not in text and str(Path(cognos.__file__).parent) not in text
    for v in res.payload.values():
        assert not (isinstance(v, str) and Path(v).is_absolute())


def test_old_and_package_relative_anchor_forms_resolve():
    pkg = review.package_dir()
    want = pkg / "stages" / "backtest.py"
    for form in ("src/cognos/stages/backtest.py", "cognos/stages/backtest.py",
                 "stages/backtest.py", "./src/cognos/stages/backtest.py"):
        assert review.resolve_anchor(form) == want.resolve()
    assert review.resolve_anchor("/etc/passwd") is None
    assert review.resolve_anchor("src/cognos/../../pyproject.toml") is None
