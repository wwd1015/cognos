#!/usr/bin/env python3
"""COGNOS eval harness.

Runs each case in ``cases.yaml`` through the COGNOS synthetic demo
(``python -m cognos.cli demo --task <task> --runs-dir <tmp>``), parses the
produced ``runs/<run_id>/summary.txt`` token block, and asserts the case's
``expected.*`` fields. Prints a per-case table, writes an optional JSON report,
and exits 0 only if every case passed.

Usage:
    python evals/score.py [--cases evals/cases.yaml] [--filter <case-id>]
                          [--report <out.json>] [--keep-runs]

Exit codes:
    0  every case passed
    1  one or more cases failed (or a harness error)

The harness is dependency-light: stdlib only, with PyYAML used when importable
and a tiny hand-rolled parser as a fallback (matching deputy's approach). It
runs the pipeline a few times, so a full pass takes ~30-60s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]

    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "evals" / "cases.yaml"
DEFAULT_REPORT = REPO_ROOT / "evals" / "report.json"

FINAL_VERDICT_RE = re.compile(r"^final_verdict:\s*(\S+)\s*$", re.MULTILINE)
STAGES_RUN_RE = re.compile(r"^stages_run:\s*(.*)$", re.MULTILINE)
STAGE_VERDICT_RE = re.compile(r"^verdict\.([A-Za-z_]+):\s*(\S+)\s*$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Python interpreter resolution
# --------------------------------------------------------------------------- #
def resolve_python() -> str:
    """Return the interpreter to run the COGNOS CLI with.

    Prefer the project venv at ``.venv/bin/python`` (it has COGNOS's deps
    installed); fall back to the interpreter running this harness.
    """
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def cli_env() -> dict[str, str]:
    """Environment for the CLI subprocess.

    The COGNOS package may not be pip-installed into the venv; prepend ``src``
    to PYTHONPATH so ``python -m cognos.cli`` imports from the source tree.
    This is a no-op if the package is already installed.
    """
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    return env


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #
def load_cases(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if _HAVE_YAML:
        data = yaml.safe_load(text)
    else:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"{path}: missing top-level 'cases' key")
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError(f"{path}: 'cases' must be a list")
    return cases


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML parser for the predictable shape of cases.yaml.

    Handles: a top-level ``cases:`` list of mappings, each with scalar keys, an
    ``expected:`` sub-mapping, inline ``[a, b]`` lists, and a nested
    ``stage_verdicts:`` mapping. No anchors, no multi-line scalars.
    """
    lines = text.splitlines()
    root: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    # Track the most recent mapping opened at a given indent so deeper keys
    # attach to the right parent.
    expected: dict[str, Any] | None = None
    nested: dict[str, Any] | None = None  # e.g. stage_verdicts
    in_cases = False

    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0 and stripped.startswith("cases:"):
            in_cases = True
            root["cases"] = cases
            continue
        if not in_cases:
            continue

        if indent == 2 and stripped.startswith("- "):
            current = {}
            cases.append(current)
            expected = None
            nested = None
            kv = stripped[2:]
            key, _, val = kv.partition(":")
            current[key.strip()] = _coerce(val.strip())
            continue

        if current is None:
            continue

        if indent == 4 and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                current[key] = {}
                expected = current[key]
                nested = None
            else:
                current[key] = _coerce(val)
                expected = None
                nested = None
            continue

        if indent == 6 and expected is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                expected[key] = {}
                nested = expected[key]
            else:
                expected[key] = _coerce(val)
                nested = None
            continue

        if indent == 8 and nested is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            nested[key.strip()] = _coerce(val.strip())
            continue

    return root


def _coerce(val: str) -> Any:
    if val == "":
        return None
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_coerce(p.strip()) for p in inner.split(",") if p.strip()]
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    try:
        return int(val) if "." not in val else float(val)
    except ValueError:
        return val


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    task: str
    passed: bool
    final_verdict: str | None
    stages_ran: int | None
    wall_clock_sec: float
    failed_assertions: list[str] = field(default_factory=list)
    error: str | None = None
    stdout_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "passed": self.passed,
            "final_verdict": self.final_verdict,
            "stages_ran": self.stages_ran,
            "wall_clock_sec": round(self.wall_clock_sec, 2),
            "failed_assertions": list(self.failed_assertions),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Summary parsing
# --------------------------------------------------------------------------- #
def find_summary(runs_dir: Path) -> Path | None:
    """Return the newest runs/<run_id>/summary.txt under ``runs_dir``."""
    candidates = sorted(
        runs_dir.glob("*/summary.txt"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def parse_summary(text: str) -> tuple[str | None, int | None, dict[str, str]]:
    """Parse a COGNOS summary token block.

    Returns ``(final_verdict, stages_ran, {stage: verdict})``.
    """
    fv = FINAL_VERDICT_RE.search(text)
    final_verdict = fv.group(1) if fv else None

    sr = STAGES_RUN_RE.search(text)
    stages_ran = len(sr.group(1).split()) if sr and sr.group(1).strip() else None

    stage_verdicts = {m.group(1): m.group(2) for m in STAGE_VERDICT_RE.finditer(text)}
    return final_verdict, stages_ran, stage_verdicts


# --------------------------------------------------------------------------- #
# Case execution
# --------------------------------------------------------------------------- #
def run_case(case: dict[str, Any], python: str, keep_runs: bool) -> CaseResult:
    case_id = case["id"]
    task = case["task"]
    expected = case.get("expected", {}) or {}

    tmp = Path(tempfile.mkdtemp(prefix=f"cognos-eval-{case_id}-"))
    runs_dir = tmp / "runs"
    cmd = [
        python,
        "-m",
        "cognos.cli",
        "demo",
        "--task",
        task,
        "--runs-dir",
        str(runs_dir),
    ]

    start = time.monotonic()
    failed: list[str] = []
    final_verdict: str | None = None
    stages_ran: int | None = None
    stage_verdicts: dict[str, str] = {}
    error: str | None = None
    combined = ""

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT),
            env=cli_env(),
            check=False,
        )
        elapsed = time.monotonic() - start
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # Prefer the on-disk summary.txt (canonical); fall back to stdout.
        summary_path = find_summary(runs_dir)
        summary_text = summary_path.read_text() if summary_path else combined
        final_verdict, stages_ran, stage_verdicts = parse_summary(summary_text)

        if final_verdict is None:
            failed.append("no summary token block found (final_verdict missing)")
            if proc.returncode not in (0, 2):
                error = f"cli exit_code={proc.returncode}"

        # --- assertions ---------------------------------------------------- #
        exp_fv = expected.get("final_verdict")
        if exp_fv is not None and final_verdict != exp_fv:
            failed.append(f"final_verdict: expected={exp_fv} observed={final_verdict!r}")

        exp_fv_in = expected.get("final_verdict_in")
        if exp_fv_in is not None and final_verdict not in exp_fv_in:
            failed.append(
                f"final_verdict_in: observed={final_verdict!r} not in {exp_fv_in}"
            )

        exp_stages = expected.get("stages_ran")
        if exp_stages is not None and stages_ran != exp_stages:
            failed.append(
                f"stages_ran: expected={exp_stages} observed={stages_ran!r}"
            )

        exp_sv = expected.get("stage_verdicts") or {}
        for stage, want in exp_sv.items():
            got = stage_verdicts.get(stage)
            if got != want:
                failed.append(
                    f"verdict.{stage}: expected={want} observed={got!r}"
                )

        exp_sv_in = expected.get("stage_verdicts_in") or {}
        for stage, allowed in exp_sv_in.items():
            got = stage_verdicts.get(stage)
            if got not in allowed:
                failed.append(
                    f"verdict.{stage}: observed={got!r} not in {allowed}"
                )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        error = "timeout"
        failed.append("cli timed out (>600s)")
    finally:
        if not keep_runs:
            shutil.rmtree(tmp, ignore_errors=True)

    return CaseResult(
        case_id=case_id,
        task=task,
        passed=not failed and error is None,
        final_verdict=final_verdict,
        stages_ran=stages_ran,
        wall_clock_sec=elapsed,
        failed_assertions=failed,
        error=error,
        stdout_tail="\n".join(combined.splitlines()[-20:]),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render_table(results: list[CaseResult]) -> str:
    headers = ("CASE", "TASK", "RESULT", "FINAL", "STAGES", "WALL(s)", "FAILURES")
    rows: list[tuple[str, ...]] = [headers]
    for r in results:
        rows.append(
            (
                r.case_id,
                r.task,
                "PASS" if r.passed else "FAIL",
                r.final_verdict or "-",
                str(r.stages_ran) if r.stages_ran is not None else "-",
                f"{r.wall_clock_sec:.1f}",
                "; ".join(r.failed_assertions) if r.failed_assertions
                else (r.error or ""),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    out: list[str] = []
    for i, row in enumerate(rows):
        out.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(row)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def write_report(path: Path, results: list[CaseResult]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "cases": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run COGNOS eval cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--filter", default=None, help="run only this case id")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="don't delete the per-case temp runs dirs (for debugging)",
    )
    args = parser.parse_args(argv)

    cases_path = Path(args.cases).resolve()
    if not cases_path.exists():
        print(f"error: cases file not found: {cases_path}", file=sys.stderr)
        return 1

    try:
        cases = load_cases(cases_path)
    except (ValueError, OSError) as exc:
        print(f"error: failed to load cases: {exc}", file=sys.stderr)
        return 1

    if args.filter:
        cases = [c for c in cases if c.get("id") == args.filter]
        if not cases:
            print(f"error: no case matched --filter {args.filter!r}", file=sys.stderr)
            return 1

    python = resolve_python()
    print(f"interpreter: {python}")
    print(f"cases: {len(cases)} (this runs the pipeline a few times; ~30-60s)\n")

    results: list[CaseResult] = []
    for case in cases:
        print(f"-> running {case['id']} (task={case['task']}) ...", flush=True)
        result = run_case(case, python, keep_runs=args.keep_runs)
        status = "PASS" if result.passed else "FAIL"
        print(f"   {status}  final={result.final_verdict} stages={result.stages_ran} "
              f"({result.wall_clock_sec:.1f}s)", flush=True)
        if not result.passed:
            for fa in result.failed_assertions:
                print(f"     - {fa}", flush=True)
            if result.error:
                print(f"     - error: {result.error}", flush=True)
                if result.stdout_tail:
                    print("     --- last output ---", flush=True)
                    for ln in result.stdout_tail.splitlines():
                        print(f"     | {ln}", flush=True)
        results.append(result)

    print()
    print(render_table(results))
    print()

    report_path = Path(args.report).resolve()
    write_report(report_path, results)
    print(f"report: {report_path}")

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    print(f"\n{n_passed}/{len(results)} cases passed.")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
