"""The pluggable *brain* abstraction.

Each stage has solid deterministic logic of its own; the brain is an *optional* LLM augmentation
for the parts that genuinely benefit from open-ended reasoning (idea narratives, judgment calls,
prose). Stages branch on ``ctx.brain.available``:

    if ctx.brain.available:
        text = ctx.brain.generate(prompt)        # Claude-backed
    else:
        text = self._heuristic(...)              # deterministic fallback

This single seam is what makes COGNOS fully testable offline (HeuristicBrain, no API key) while
still being a real multi-agent AI system in production (LLMBrain).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class Brain(ABC):
    kind: str = "base"
    available: bool = False

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        """Return free-form text."""

    def judge(self, prompt: str, *, system: str | None = None) -> dict[str, Any]:
        """Return a structured JSON object (best-effort parse of the model's reply)."""
        raw = self.generate(prompt, system=system)
        return _extract_json(raw)


class HeuristicBrain(Brain):
    """No LLM. ``available`` is False so stages take their deterministic path."""

    kind = "heuristic"
    available = False

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        raise RuntimeError(
            "HeuristicBrain has no LLM. Stages must check `ctx.brain.available` before calling "
            "generate(); when False, use the deterministic code path."
        )


class ScriptedBrain(Brain):
    """A deterministic test double: ``available`` is True and ``generate`` replays queued responses.

    Lets tests exercise the LLM-driven ideation/search paths offline and reproducibly, without an API
    key — the reasoning layer's stand-in for CI (the production brain is LLMBrain).
    """

    kind = "scripted"
    available = True

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        if not self._responses:
            return ""
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (tolerant of code fences / prose)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        brace = re.search(r"(\{.*\})", text, re.DOTALL)
        candidate = brace.group(1) if brace else None
    if candidate is None:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}
