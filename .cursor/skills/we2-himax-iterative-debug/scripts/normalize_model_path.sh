#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  normalize_model_path.sh --path <model_path>

Description:
  Resolve a model path for WSL workflows.
  - Accepts Linux paths directly.
  - Accepts Windows paths like D:\foo\bar.tflite or D:/foo/bar.tflite.
  - Prints key=value lines for downstream scripts.
EOF
}

input_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      input_path="$2"
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

if [[ -z "$input_path" ]]; then
  echo "--path is required." >&2
  usage
  exit 1
fi

to_wsl_path() {
  local raw="$1"
  if [[ "$raw" =~ ^([A-Za-z]):[\\/](.*)$ ]]; then
    local drive="${BASH_REMATCH[1],,}"
    local rest="${BASH_REMATCH[2]}"
    rest="${rest//\\//}"
    echo "/mnt/${drive}/${rest}"
    return 0
  fi
  return 1
}

resolved_path=""
converted_wsl_path=""

if [[ -f "$input_path" ]]; then
  resolved_path="$(realpath "$input_path")"
fi

if [[ -z "$resolved_path" ]]; then
  if converted_wsl_path="$(to_wsl_path "$input_path")"; then
    if [[ -f "$converted_wsl_path" ]]; then
      resolved_path="$(realpath "$converted_wsl_path")"
    fi
  fi
fi

if [[ -z "$resolved_path" ]]; then
  echo "Model file not found." >&2
  echo "input_path=$input_path" >&2
  if [[ -n "$converted_wsl_path" ]]; then
    echo "converted_wsl_path=$converted_wsl_path" >&2
  fi
  exit 1
fi

echo "input_path=$input_path"
if [[ -n "$converted_wsl_path" ]]; then
  echo "converted_wsl_path=$converted_wsl_path"
fi
echo "resolved_path=$resolved_path"

