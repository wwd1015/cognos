"""`claude -p` backend (the local Claude Code CLI; uses your Claude Code login).

Isolation: the agent runs in an empty temp cwd with ``--setting-sources ""`` (no user/project
CLAUDE.md, hooks or plugins), ``--strict-mcp-config`` (no MCP servers), ``--tools ""`` (no built-in
tools — the slice is the complete context), and COGNOS's role prompt *replaces* the default system
prompt. The prompt goes in on stdin and the system prompt as a file, so large contexts never hit the
Windows command-line limit. ``--json-schema`` makes the CLI return ``structured_output``.
Verified against claude CLI 2.1.x.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def claude_bin() -> str:
    return os.environ.get("COGNOS_CLAUDE_BIN", "claude")


def argv(system_file: str, schema: dict[str, Any], provider: dict[str, Any]) -> list[str]:
    return [
        claude_bin(), "-p",
        "--system-prompt-file", system_file,
        "--output-format", "json",
        "--json-schema", json.dumps(schema, separators=(",", ":")),
        "--model", str(provider.get("model") or "opus"),
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--max-turns", str(provider.get("max_turns", 6)),
    ]


def extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def call(system: str, prompt: str, schema: dict[str, Any], provider: dict[str, Any],
         deadline: float | None, log_path: Path) -> tuple[dict, dict]:
    workdir = tempfile.mkdtemp(prefix="cognos-agent-")
    system_file = Path(workdir) / "system.md"
    system_file.write_text(system, encoding="utf-8")
    cmd = argv(str(system_file), schema, provider)
    timeout = max(10.0, deadline - time.time()) if deadline else 600.0
    try:
        proc = subprocess.run(cmd, input=prompt, cwd=workdir, capture_output=True, text=True,
                              encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"claude -p ran out of time after {timeout:.0f}s") from exc
    log_path.write_text(json.dumps({"argv": cmd[:1] + cmd[2:], "returncode": proc.returncode,
                                    "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-4000:]},
                                   indent=1), encoding="utf-8")
    result = extract_json(proc.stdout)
    if result is None:
        raise RuntimeError(f"claude -p exited {proc.returncode} without a JSON result: "
                           f"{(proc.stderr or proc.stdout)[-400:]}")
    if result.get("is_error"):
        raise RuntimeError(f"claude -p reported an error ({result.get('subtype')}): "
                           f"{str(result.get('result'))[:400]}")
    meta = {"model": provider.get("model"), "turns": result.get("num_turns"),
            "cost_usd": result.get("total_cost_usd")}
    usage = result.get("usage") or {}
    meta["input_tokens"] = usage.get("input_tokens")
    meta["output_tokens"] = usage.get("output_tokens")
    if isinstance(result.get("structured_output"), dict):
        return result["structured_output"], meta
    parsed = extract_json(result.get("result") or "")
    if parsed is None:
        raise RuntimeError(f"no JSON in the answer: {str(result.get('result'))[:300]}")
    return parsed, meta
