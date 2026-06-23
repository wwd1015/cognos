#!/usr/bin/env bash
# secret-scan.sh — COGNOS pre-commit / pre-push secret scanner.
#
# Fires on PreToolUse for Bash. If the command is a `git commit`, `git push`,
# or `gh pr create`, it scans the staged + unstaged diff for credential
# patterns and hard-blocks on any hit. Modeled on the deputy reference hook.
#
# This is intentionally simple regex matching — not a replacement for a real
# secrets-scanning service. It catches the obvious mistakes (a key pasted into
# a config file, an .env file accidentally added) before they reach a remote.
#
# Exit codes:
#   0  — allow (not our concern, or no secrets found).
#   1  — block (printed which pattern + file matched).
#
# Fail-open: parse weirdness or no git repo allows the command.

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/hook-common.sh
source "$HOOK_DIR/lib/hook-common.sh"

read_hook_payload PAYLOAD
COMMAND="$(printf '%s' "$PAYLOAD" | extract_command)"

# Only inspect commit/publish-class commands. Everything else passes through.
if [[ "$COMMAND" != *"git commit"* && "$COMMAND" != *"git push"* && "$COMMAND" != *"gh pr create"* ]]; then
  exit 0
fi

ROOT="$(repo_root || true)"
if [[ -z "$ROOT" ]]; then
  exit 0
fi

# Combine staged + working-tree diff. Either may carry the secret depending on
# whether the agent staged it yet. We check both rather than predicting which
# the upcoming commit/push will include.
DIFF="$(
  {
    git -C "$ROOT" diff --staged 2>/dev/null || true
    git -C "$ROOT" diff 2>/dev/null || true
  }
)"

if [[ -z "$DIFF" ]]; then
  printf '[cognos] secret-scan hook: no diff to scan, allowing\n' >&2
  exit 0
fi

# Pattern table. Each entry is "label|extended-regex". Patterns chosen for
# high-precision tokens (AWS access keys, Anthropic keys, GitHub PATs, PEM
# private keys) plus a loose "password =" heuristic. This hook errs toward
# blocking.
PATTERNS=(
  'AWS access key|AKIA[0-9A-Z]{16}'
  'AWS secret access key (named)|aws_secret_access_key'
  'Anthropic API key|sk-ant-[A-Za-z0-9_-]{20,}'
  'GitHub personal access token|ghp_[A-Za-z0-9]{36}'
  'GitHub OAuth token|gho_[A-Za-z0-9]{36}'
  'GitHub server-to-server token|ghs_[A-Za-z0-9]{36}'
  'GitHub user-to-server token|ghu_[A-Za-z0-9]{36}'
  'PEM private key header|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----'
  'Hardcoded password literal|password[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{6,}'
)

HITS=()

# Pre-filter: only added lines, excluding diff headers (`+++ b/path`).
ADDED_LINES="$(printf '%s\n' "$DIFF" | grep -nE '^\+' | grep -vE '^[0-9]+:\+\+\+ ' || true)"

for entry in "${PATTERNS[@]}"; do
  label="${entry%%|*}"
  regex="${entry#*|}"
  # `-e` is required: some patterns (PEM headers) start with `-` and would
  # otherwise be parsed as a grep option.
  match="$(printf '%s\n' "$ADDED_LINES" | grep -E -e "$regex" || true)"
  if [[ -n "$match" ]]; then
    HITS+=("$label")
    HITS+=("$match")
    HITS+=("")
  fi
done

# Also catch new .env file additions specifically — these often carry secrets
# even when the contents don't match a known token format.
ENV_HIT="$(printf '%s\n' "$DIFF" | grep -nE '^\+\+\+ b/(.*/)?\.env(\..+)?$' || true)"
if [[ -n "$ENV_HIT" ]]; then
  HITS+=(".env file added to diff")
  HITS+=("$ENV_HIT")
  HITS+=("")
fi

if [[ ${#HITS[@]} -gt 0 ]]; then
  block \
    "[cognos] secret-scan hook: BLOCKED" \
    "" \
    "Reason: potential secret(s) found in the diff." \
    "Command: $COMMAND" \
    "" \
    "Matches:" \
    "${HITS[@]}" \
    "Fix: remove the secret from the diff (and rotate it if it ever existed in" \
    "      this repo's history). Add the file to .gitignore if appropriate." \
    "" \
    "This is a COGNOS safety hook (.claude/hooks/secret-scan.sh)."
fi

printf '[cognos] secret-scan hook: clean diff, allowing\n' >&2
exit 0
