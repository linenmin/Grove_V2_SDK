#!/usr/bin/env bash
set -euo pipefail

PLAN_DIR="${1:-plan}"

if [[ ! -d "$PLAN_DIR" ]]; then
  echo "Plan directory not found: $PLAN_DIR" >&2
  exit 1
fi

latest=$(
  find "$PLAN_DIR" -maxdepth 1 -type f -name "*.md" -printf "%T@ %p\n" \
    | sort -nr \
    | head -n1 \
    | cut -d" " -f2-
)

if [[ -z "${latest:-}" ]]; then
  echo "No markdown plan files found in: $PLAN_DIR" >&2
  exit 1
fi

echo "$latest"
