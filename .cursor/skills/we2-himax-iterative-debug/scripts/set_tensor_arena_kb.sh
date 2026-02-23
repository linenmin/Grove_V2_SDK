#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  set_tensor_arena_kb.sh --kb <integer> [--app optical_cam_oflow] [--file <cvapp_cpp>]

Examples:
  set_tensor_arena_kb.sh --kb 1624 --app optical_cam_oflow
  set_tensor_arena_kb.sh --kb 1700 \
    --file EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_yolov8n_ob.cpp
EOF
}

kb=""
app_name="optical_cam_oflow"
target_file=""

resolve_default_file() {
  case "$1" in
    optical_cam_oflow)
      echo "EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_yolov8n_ob.cpp"
      ;;
    *)
      echo "Unknown --app: $1" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kb)
      kb="$2"
      shift 2
      ;;
    --app)
      app_name="$2"
      shift 2
      ;;
    --file)
      target_file="$2"
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

if [[ -z "$kb" ]]; then
  echo "--kb is required." >&2
  usage
  exit 1
fi

if ! [[ "$kb" =~ ^[0-9]+$ ]]; then
  echo "--kb must be an integer: $kb" >&2
  exit 1
fi

if [[ -z "$target_file" ]]; then
  target_file="$(resolve_default_file "$app_name")"
fi

if [[ ! -f "$target_file" ]]; then
  echo "Target file not found: $target_file" >&2
  exit 1
fi

old_line="$(rg -n "constexpr int tensor_arena_size = [0-9]+ \\* 1024;" "$target_file" | head -n1 || true)"
if [[ -z "$old_line" ]]; then
  echo "tensor_arena_size pattern not found in: $target_file" >&2
  exit 1
fi

sed -E -i "s/constexpr int tensor_arena_size = [0-9]+ \\* 1024;/constexpr int tensor_arena_size = ${kb} * 1024;/" "$target_file"

new_line="$(rg -n "constexpr int tensor_arena_size = [0-9]+ \\* 1024;" "$target_file" | head -n1 || true)"

echo "file=$target_file"
echo "old=$old_line"
echo "new=$new_line"
