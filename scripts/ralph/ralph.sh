#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/ralph/ralph.sh validate
  ./scripts/ralph/ralph.sh status
  ./scripts/ralph/ralph.sh next

This repository is Codex-first. Ralph is used manually through:

  .agents/skills/ralph-sdd/SKILL.md
  scripts/ralph/prd.json
  scripts/ralph/progress.txt

Automatic Codex CLI execution is intentionally not implemented yet. Add it only
after confirming the local `codex` command and its non-interactive invocation.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
SKILL_FILE="$REPO_ROOT/.agents/skills/ralph-sdd/SKILL.md"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' was not found." >&2
    exit 127
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Error: required file '$1' was not found." >&2
    exit 1
  fi
}

validate() {
  require_command jq
  require_file "$PRD_FILE"
  require_file "$PROGRESS_FILE"
  require_file "$SKILL_FILE"

  jq empty "$PRD_FILE" >/dev/null

  jq -e '
    has("project")
    and has("branchName")
    and has("userStories")
    and (.userStories | type == "array")
    and ([.userStories[] | has("id") and has("title") and has("priority") and has("passes")] | all)
  ' "$PRD_FILE" >/dev/null

  echo "Ralph scaffold is valid."
}

status() {
  validate >/dev/null
  jq -r '
    "Project: \(.project)",
    "Branch: \(.branchName)",
    "Stories:",
    (.userStories[]
      | "- \(.id) [priority \(.priority)] passes=\(.passes): \(.title)")
  ' "$PRD_FILE"
}

next_story() {
  validate >/dev/null
  NEXT="$(
    jq -r '
      [.userStories[] | select(.passes == false)]
      | sort_by(.priority)
      | .[0]
      | if . == null then "COMPLETE" else "\(.id): \(.title)" end
    ' "$PRD_FILE"
  )"
  echo "$NEXT"
}

COMMAND="${1:-}"

case "$COMMAND" in
  validate)
    validate
    ;;
  status)
    status
    ;;
  next)
    next_story
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "Error: unknown command '$COMMAND'" >&2
    usage >&2
    exit 2
    ;;
esac
