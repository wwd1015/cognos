"""Brain factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Brain, HeuristicBrain, ScriptedBrain

if TYPE_CHECKING:
    from ..config import BrainConfig

__all__ = ["Brain", "HeuristicBrain", "ScriptedBrain", "make_brain"]


def make_brain(cfg: BrainConfig | None = None) -> Brain:
    """Build a brain from config; gracefully degrades to HeuristicBrain when the LLM is unavailable."""
    from ..config import BrainConfig, BrainKind

    cfg = cfg or BrainConfig()
    if cfg.kind == BrainKind.LLM:
        from .llm import LLMBrain

        brain = LLMBrain(model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        if brain.available:
            return brain
        # No SDK / key — degrade silently so offline runs still work.
    return HeuristicBrain()
