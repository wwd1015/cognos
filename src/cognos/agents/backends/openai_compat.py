"""OpenAI-compatible Chat Completions backend (OpenAI, xAI, OpenRouter, Ollama, …).

Portable by design: the answer arrives as a call to a ``submit_answer`` function whose parameters
are the contract schema ($ref-inlined — some providers reject $ref), because function calling is
far more widely supported than JSON-schema response formats. A text reply that parses as JSON is
accepted too; otherwise the model is reminded (at most twice).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .. import providers
from .claude_cli import extract_json

SUBMIT = "submit_answer"


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            if "$ref" in o and o["$ref"].startswith("#/$defs/"):
                return walk(defs[o["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in o.items() if k != "$defs"}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return walk(schema)


def _client(p: dict[str, Any]):
    from openai import OpenAI

    key = os.environ.get(p.get("api_key_env") or "", "") or ("none" if p.get("no_key_needed") else "")
    return OpenAI(base_url=p["base_url"], api_key=key, max_retries=2)


def call(system: str, prompt: str, schema: dict[str, Any], contract, provider: dict[str, Any],
         deadline: float | None, log_path: Path, client=None) -> tuple[dict, dict]:
    client = client or _client(provider)
    tool = {"type": "function", "function": {
        "name": SUBMIT, "description": "Submit your complete final answer (call exactly once).",
        "parameters": inline_refs(schema)}}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + f"\n\nCall the `{SUBMIT}` tool with your complete "
                                              "answer as its arguments."},
    ]
    log = []
    cost = 0.0
    in_tok = out_tok = 0
    for turn in range(1, 4):
        kwargs: dict[str, Any] = dict(model=provider["model"], messages=messages, tools=[tool],
                                      tool_choice="auto")
        if provider.get("max_tokens"):
            kwargs["max_tokens"] = provider["max_tokens"]
        timeout = max(10.0, deadline - time.time()) if deadline else 600.0
        c = client.with_options(timeout=timeout) if hasattr(client, "with_options") else client
        resp = c.chat.completions.create(**kwargs)
        log.append(resp.model_dump() if hasattr(resp, "model_dump") else str(resp))
        u = getattr(resp, "usage", None)
        if u is not None:
            pt, ct = getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
            in_tok, out_tok = in_tok + pt, out_tok + ct
            cost += providers.cost(provider, pt, ct)
        msg = resp.choices[0].message
        answer = None
        for tc in msg.tool_calls or []:
            if tc.function.name == SUBMIT:
                try:
                    answer = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    answer = None
        if answer is None:
            answer = extract_json(msg.content or "")
        if isinstance(answer, dict):
            log_path.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
            return answer, {"model": provider["model"], "turns": turn, "cost_usd": round(cost, 6),
                            "input_tokens": in_tok, "output_tokens": out_tok}
        messages.append({"role": "assistant", "content": msg.content or ""})
        messages.append({"role": "user", "content": f"Please call `{SUBMIT}` with your final answer now."})
    log_path.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    raise RuntimeError(f"no {SUBMIT} call after {turn} turns")
