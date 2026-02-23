#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  append_plan_entry.sh \
    --target <markdown> \
    --title <section-title> \
    --goal <text> \
    --change <text> [--change <text> ...] \
    --verify <command> \
    --result <text> [--result <text> ...] \
    --conclusion <text> \
    [--run-id <id>] \
    [--log <path>]

Notes:
  - Use repeatable --change and --result options for multiple bullets.
  - This script appends one new section at the end of the markdown file.
EOF
}

target=""
title=""
goal=""
verify_cmd=""
conclusion=""
run_id=""
log_path=""
declare -a changes
declare -a results

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="$2"
      shift 2
      ;;
    --title)
      title="$2"
      shift 2
      ;;
    --goal)
      goal="$2"
      shift 2
      ;;
    --change)
      changes+=("$2")
      shift 2
      ;;
    --verify)
      verify_cmd="$2"
      shift 2
      ;;
    --result)
      results+=("$2")
      shift 2
      ;;
    --conclusion)
      conclusion="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --log)
      log_path="$2"
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

if [[ -z "$target" || -z "$title" || -z "$goal" || -z "$verify_cmd" || -z "$conclusion" ]]; then
  echo "Missing required argument." >&2
  usage
  exit 1
fi

if [[ ${#changes[@]} -eq 0 ]]; then
  changes=("TODO")
fi

if [[ ${#results[@]} -eq 0 ]]; then
  results=("TODO")
fi

mkdir -p "$(dirname "$target")"
touch "$target"

{
  echo
  echo "### $title"
  echo
  echo "- **Goal**"
  echo "  - $goal"
  echo
  echo "- **Changes**"
  for item in "${changes[@]}"; do
    echo "  - $item"
  done
  echo
  echo "- **Verification Command**"
  echo "  - \`$verify_cmd\`"
  echo
  echo "- **Key Output**"
  for item in "${results[@]}"; do
    echo "  - $item"
  done
  echo
  echo "- **Conclusion**"
  echo "  - $conclusion"
  if [[ -n "$run_id" ]]; then
    echo
    echo "- **Run ID**"
    echo "  - \`$run_id\`"
  fi
  if [[ -n "$log_path" ]]; then
    echo
    echo "- **Log Path**"
    echo "  - \`$log_path\`"
  fi
  echo
  echo "---"
} >> "$target"

echo "appended_to=$target"
