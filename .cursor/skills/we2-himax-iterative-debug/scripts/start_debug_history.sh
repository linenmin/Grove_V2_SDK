#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_debug_history.sh --topic <topic> [--latest-plan <path>] [--output <path>]

Examples:
  start_debug_history.sh --topic "himax-jpeg-source"
  start_debug_history.sh --topic "uart-html-preview" --latest-plan plan/plan-006.md
EOF
}

topic=""
latest_plan=""
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      topic="$2"
      shift 2
      ;;
    --latest-plan)
      latest_plan="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$topic" ]]; then
  echo "--topic is required." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$latest_plan" ]]; then
  latest_plan="$("$script_dir/find_latest_plan.sh")"
fi

if [[ ! -f "$latest_plan" ]]; then
  echo "Latest plan file not found: $latest_plan" >&2
  exit 1
fi

slug="$(echo "$topic" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
if [[ -z "$slug" ]]; then
  echo "Topic produced empty slug. Use alphanumeric topic text." >&2
  exit 1
fi

date_stamp="$(date +%Y%m%d)"
if [[ -z "$output" ]]; then
  output="plan/plan-debug-history-${date_stamp}-${slug}.md"
fi

if [[ -e "$output" ]]; then
  echo "History file already exists: $output" >&2
  exit 1
fi

cat > "$output" <<EOF
# Debug History: $topic

## Scope

- Topic: \`$topic\`
- Latest plan summary file: \`$latest_plan\`
- Created at: \`$(date -Iseconds)\`

## Run Log

Append one run per section. Keep each section compact:

- hypothesis
- minimal change
- exact verification command
- key evidence lines
- conclusion and next action

### Run 1

- Hypothesis:
  - TODO
- Change:
  - TODO
- Verify:
  - \`TODO\`
- Evidence:
  - TODO
- Conclusion:
  - TODO

---

EOF

echo "$output"
