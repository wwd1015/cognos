#!/usr/bin/env bash
# hook-common.sh — shared helpers for COGNOS PreToolUse hooks.
#
# Sourced by sibling hook scripts (protect-runs.sh, secret-scan.sh). Pure bash,
# no external state. Each helper is small and independently testable. Copied
# from the deputy reference system and kept deliberately identical so the two
# agent layers stay easy to reason about side-by-side.

# read_hook_payload — drain stdin into the variable named by $1.
# Usage:
#   read_hook_payload payload
#   echo "$payload"
#
# Claude Code delivers a JSON object on stdin once per hook invocation.
# We capture it whole; callers parse fields with jq.
read_hook_payload() {
  local __varname="${1:-}"
  if [[ -z "$__varname" ]]; then
    echo "[cognos] read_hook_payload: missing variable name" >&2
    return 1
  fi
  local __content
  __content="$(cat)"
  printf -v "$__varname" '%s' "$__content"
}

# extract_command — print the tool_input.command field from a payload on stdin.
# Empty string if the field is absent or null. Never errors on bad JSON; jq's
# `// empty` masks that, and we want hooks to fail open on parse weirdness
# rather than block every Bash call.
extract_command() {
  jq -r '.tool_input.command // empty' 2>/dev/null || true
}

# repo_root — print the absolute path to the current git toplevel.
# Returns nonzero (and prints nothing on stdout) if not inside a git repo.
repo_root() {
  git rev-parse --show-toplevel 2>/dev/null
}

# block — print a multi-line message to stderr and exit 1.
# Exit 1 is the universally-blocking signal for Claude Code PreToolUse hooks
# (nonzero blocks the tool call). We use 1 rather than 2 because some shells
# treat 2 specially; 1 is unambiguous "this hook said no".
#
# Usage: block "first line" "second line" "third line"
block() {
  local line
  for line in "$@"; do
    printf '%s\n' "$line" >&2
  done
  exit 1
}
