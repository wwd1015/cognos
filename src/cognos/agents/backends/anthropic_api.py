"""Anthropic Messages API backend.

One streamed request per attempt: the role prompt is the system prompt, the answer is constrained to
the contract with structured outputs (``output_config.format``; the runner's schema is already
cleaned of unsupported keywords, and Pydantic re-validates), adaptive thinking with an explicit
effort, and server-side refusal fallbacks (``fallbacks: "default"``) enabled by default so a policy
decline re-runs on a fallback model instead of failing the step. If the account/model rejects
fallbacks, the request is retried once without them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .. import providers
from .claude_cli import extract_json

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")


def _client(provider: dict[str, Any]):
    import os

    import anthropic

    if provider.get("base_url"):
        return anthropic.Anthropic(base_url=provider["base_url"],
                                   api_key=os.environ.get(provider.get("api_key_env", ""), ""))
    return anthropic.Anthropic()


def _stream(client, params: dict[str, Any], use_fallbacks: bool, deadline: float | None):
    import anthropic

    if use_fallbacks:
        try:
            with client.beta.messages.stream(**params, betas=[FALLBACK_BETA],
                                             fallbacks="default") as stream:
                return _finish(stream, deadline)
        except anthropic.BadRequestError as exc:
            if "fallback" not in str(exc).lower():
                raise
    with client.messages.stream(**params) as stream:
        return _finish(stream, deadline)


def _finish(stream, deadline: float | None):
    if deadline:
        for _ in stream:
            if time.time() >= deadline:
                raise TimeoutError("the time budget ran out mid-request")
    return stream.get_final_message()


def call(system: str, prompt: str, schema: dict[str, Any], provider: dict[str, Any],
         deadline: float | None, log_path: Path, client=None) -> tuple[dict, dict]:
    model = provider["model"]
    client = client or _client(provider)
    params: dict[str, Any] = dict(
        model=model,
        max_tokens=int(provider.get("max_tokens", 32000)),
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": provider.get("effort", "high"),
                       "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    timeout = max(10.0, deadline - time.time()) if deadline else 600.0
    timed = client.with_options(timeout=timeout) if hasattr(client, "with_options") else client
    response = _stream(timed, params, model.startswith(FALLBACK_MODELS), deadline)
    dump = response.to_dict() if hasattr(response, "to_dict") else {}
    log_path.write_text(json.dumps(dump, indent=1, default=str), encoding="utf-8")
    if response.stop_reason == "refusal":
        raise RuntimeError(f"the model declined this request ({getattr(response, 'stop_details', None)})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the answer was cut off (max_tokens)")
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    answer = extract_json(text)
    if answer is None:
        raise RuntimeError(f"no JSON in the answer: {text[:300]}")
    u = response.usage
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    meta = {"model": getattr(response, "model", model), "turns": 1, "input_tokens": inp + cw + cr,
            "output_tokens": out, "cost_usd": round(providers.cost(provider, inp, out, cw, cr), 6)}
    return answer, meta
