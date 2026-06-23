#!/usr/bin/env bash
# protect-runs.sh — COGNOS run-artifact safety backstop.
#
# Fires on PreToolUse for Bash. The COGNOS pipeline writes its load-bearing
# artifacts under `runs/<run_id>/` (StageResult JSON, summaries, OKF doc
# bundles, fitted models). Those are the determinism-on-disk that lets a stage
# resume in a fresh process and that the eval harness parses. This hook is a
# light backstop against two easy ways to destroy or leak them:
#
#   1. A destructive `rm -rf` (or `rm -r`/`git clean`) aimed at `runs/`.
#   2. A `git push` that would publish run artifacts (run dirs are generated
#      output and do not belong on a remote).
#
# It is intentionally simple substring/regex matching — a guardrail, not a
# sandbox. Agent prompts also forbid editing runs/ outside the CLI; we do not
# rely on that. Determinism here, judgment there.
#
# Exit codes:
#   0  — allow (not our concern, or the command is safe).
#   1  — block (printed reason to stderr).
#
# Fail-open: any parse weirdness allows the command rather than blocking every
# Bash call.

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/hook-common.sh
source "$HOOK_DIR/lib/hook-common.sh"

read_hook_payload PAYLOAD
COMMAND="$(printf '%s' "$PAYLOAD" | extract_command)"

# Nothing to inspect → allow (fail-open).
if [[ -z "$COMMAND" ]]; then
  exit 0
fi

# 1. Destructive deletion aimed at runs/. We look for an rm with a recursive
#    flag (-r / -rf / -fr / -R ...) or a `git clean` that names `runs`. The
#    `runs` token may appear as `runs`, `runs/`, `./runs`, `*/runs/...`, etc.
if printf '%s' "$COMMAND" | grep -Eq 'rm[[:space:]]+(-[A-Za-z]*r[A-Za-z]*[[:space:]]+)+[^|;&]*runs(/|[[:space:]]|$)'; then
  block \
    "[cognos] protect-runs hook: BLOCKED" \
    "" \
    "Reason: recursive delete targeting run artifacts under runs/." \
    "Command: $COMMAND" \
    "" \
    "runs/<run_id>/ holds load-bearing pipeline artifacts (StageResult JSON," \
    "summaries, OKF doc bundles, fitted models). Deleting them destroys the" \
    "determinism-on-disk the stages and eval harness depend on." \
    "" \
    "Fix: remove a single run dir by explicit path if you must, or use a" \
    "      throwaway --runs-dir for experiments. Don't rm -rf runs/." \
    "" \
    "This is a COGNOS safety hook (.claude/hooks/protect-runs.sh)."
fi

if printf '%s' "$COMMAND" | grep -Eq 'git[[:space:]]+clean[[:space:]]+[^|;&]*runs(/|[[:space:]]|$)'; then
  block \
    "[cognos] protect-runs hook: BLOCKED" \
    "" \
    "Reason: 'git clean' would remove untracked run artifacts under runs/." \
    "Command: $COMMAND" \
    "" \
    "This is a COGNOS safety hook (.claude/hooks/protect-runs.sh)."
fi

# 2. Pushing run artifacts to a remote. If the command is a `git push` AND the
#    runs/ tree is staged or committed in the range about to go out, block.
#    We keep it simple: if `git push` is present and there are tracked files
#    under runs/, warn off. runs/ should be gitignored generated output.
if printf '%s' "$COMMAND" | grep -Eq 'git[[:space:]]+push'; then
  ROOT="$(repo_root || true)"
  if [[ -n "$ROOT" ]]; then
    TRACKED_RUNS="$(git -C "$ROOT" ls-files -- 'runs/' 2>/dev/null | head -n 1 || true)"
    STAGED_RUNS="$(git -C "$ROOT" diff --cached --name-only -- 'runs/' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$TRACKED_RUNS" || -n "$STAGED_RUNS" ]]; then
      block \
        "[cognos] protect-runs hook: BLOCKED" \
        "" \
        "Reason: this push would publish run artifacts under runs/." \
        "Command: $COMMAND" \
        "First offending path: ${TRACKED_RUNS:-$STAGED_RUNS}" \
        "" \
        "runs/ is generated output and should not live on a remote. Add runs/" \
        "to .gitignore and 'git rm -r --cached runs/' before pushing." \
        "" \
        "This is a COGNOS safety hook (.claude/hooks/protect-runs.sh)."
    fi
  fi
fi

# Nothing matched → allow.
exit 0
