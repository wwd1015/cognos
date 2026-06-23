"""Stage package. Importing it registers all eight stage agents in the registry.

Each stage module calls ``@register_stage`` at import time; ``load_all`` imports every module so the
orchestrator/CLI can resolve any stage by name.
"""

from __future__ import annotations

from .base import STAGE_REGISTRY, Stage, all_stage_names, make_stage, register_stage

_STAGE_MODULES = (
    "explore",
    "ideate",
    "model",
    "backtest",
    "validate",
    "comply",
    "document",
    "review",
)


def load_all() -> None:
    """Import every stage module so it registers itself (idempotent)."""
    import importlib

    for mod in _STAGE_MODULES:
        importlib.import_module(f"{__name__}.{mod}")


load_all()

__all__ = [
    "STAGE_REGISTRY",
    "Stage",
    "make_stage",
    "register_stage",
    "all_stage_names",
    "load_all",
]
