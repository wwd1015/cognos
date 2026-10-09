"""Provider registry: which backend runs the agents (agents/providers.yaml).

Resolution order for a run: ``COGNOS_PROVIDER`` env → ``agents.provider`` in the profile. The value
``auto`` picks the first available LLM provider (claude_cli, anthropic, then the OpenAI-compatible
ones) and falls back to ``heuristic`` — so a machine with no key still runs, deterministically.
An explicitly requested provider that is unavailable is an error that names the available ones.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

AUTO_ORDER = ["claude_cli", "anthropic", "openai", "xai", "openrouter", "ollama"]


class ProviderUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("providers.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def all_providers() -> dict[str, dict[str, Any]]:
    out = {}
    for pid, p in _registry().items():
        p = {"id": pid, **p}
        env_model = os.environ.get(f"COGNOS_{pid.upper()}_MODEL")
        if env_model:
            p["model"] = env_model
        out[pid] = p
    return out


def get(pid: str, model: str | None = None) -> dict[str, Any]:
    provs = all_providers()
    if pid not in provs:
        raise ProviderUnavailable(f"unknown provider {pid!r}; configured: {sorted(provs)}")
    p = dict(provs[pid])
    if model:
        p["model"] = model
    prices = (p.get("model_prices") or {}).get(p.get("model"))
    if prices:
        p["prices"] = prices
    return p


def is_available(p: dict[str, Any]) -> bool:
    kind = p["kind"]
    if kind in ("heuristic", "replay"):
        return True
    if kind == "claude_cli":
        return shutil.which(os.environ.get("COGNOS_CLAUDE_BIN", "claude")) is not None
    key_env = p.get("api_key_env")
    return bool(key_env and os.environ.get(key_env))


def available_ids() -> list[str]:
    return [pid for pid, p in all_providers().items() if is_available(p)]


def resolve(requested: str | None, model: str | None = None) -> dict[str, Any]:
    """Resolve a provider id (or ``auto``) to a provider dict, honouring COGNOS_PROVIDER."""
    pid = os.environ.get("COGNOS_PROVIDER") or requested or "auto"
    if pid == "auto":
        for cand in AUTO_ORDER:
            p = all_providers().get(cand)
            if p and is_available(p):
                return get(cand, model)
        return get("heuristic")
    p = get(pid, model)
    if not is_available(p):
        raise ProviderUnavailable(
            f"provider {pid!r} is not available here (no key, CLI, or endpoint). Available: "
            f"{', '.join(available_ids())}")
    return p


def public_list() -> list[dict[str, Any]]:
    """What the UI and `cognos providers` show — never keys, only whether one is configured."""
    return [{"id": pid, "label": p.get("label", pid), "kind": p["kind"], "model": p.get("model"),
             "available": is_available(p)} for pid, p in all_providers().items()]


def cost(p: dict[str, Any], input_tokens: int, output_tokens: int,
         cache_write: int = 0, cache_read: int = 0) -> float:
    pr = p.get("prices")
    if not pr:
        return 0.0
    return (input_tokens * pr["input"] + cache_write * pr["input"] * 1.25
            + cache_read * pr["input"] * 0.1 + output_tokens * pr["output"]) / 1e6
