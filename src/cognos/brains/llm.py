"""Claude-backed brain. Optional — only active when `anthropic` is installed and an API key is set.

Falls back to *unavailable* (so stages use their deterministic path) whenever the SDK or key is
missing, rather than raising — COGNOS must always be runnable offline.
"""

from __future__ import annotations

import os

from .base import Brain


class LLMBrain(Brain):
    kind = "llm"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self.available = self._try_connect()

    def _try_connect(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        if not self.available or self._client is None:
            raise RuntimeError("LLMBrain is not available (missing anthropic SDK or ANTHROPIC_API_KEY).")
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
