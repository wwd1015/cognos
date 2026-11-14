"""Integrations with external systems: IMPACT (feature tables)."""

from __future__ import annotations

from .impact_adapter import (
    ImpactRunResult,
    build_entity_config,
    impact_available,
    score_with_impact,
)

__all__ = [
    "ImpactRunResult",
    "build_entity_config",
    "impact_available",
    "score_with_impact",
]
