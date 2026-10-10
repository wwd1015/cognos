"""IMPACT as a validation tool — a placeholder.

IMPACT's test interface is not available yet. Until it is, COGNOS registers the tool it will
answer to so that the gap is on the record: every validation lists it as *not run* with the
reason, instead of silently not mentioning it. When IMPACT ships its tests, either fill in
``_run`` here or register a tool named ``impact_test_suite`` from a plugin: a later registration
under the same name replaces this one, and the Independent Validator can request it with no
other change to COGNOS.

Scoring through IMPACT (``impact_adapter.py``) is separate and already works when IMPACT is
installed.
"""

from __future__ import annotations

from typing import Any

NAME = "impact_test_suite"
NOT_READY = ("IMPACT's test interface is not available yet (placeholder). Register a tool named "
             f"'{NAME}' from a plugin to supply it.")


def _available() -> tuple[bool, str]:
    return False, NOT_READY


def _run(df, params: dict[str, str], env: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(NOT_READY)


def register(registry) -> None:
    from ..plugins import AnalysisTool

    registry.add_tool(AnalysisTool(
        name=NAME,
        description="Run the institution's custom model tests through IMPACT on the champion "
                    "and the sealed holdout.",
        fn=_run, params={"suite": "The IMPACT test suite to run; empty for the default suite"},
        stages=("validate",), needs=("model", "holdout", "results"), available=_available))
