#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  get_model_slot.sh [--app <scenario_app_name>] [--config <common_config.h>] \
    [--macro <FLASH_ADDR_MACRO>] [--model-file <path.tflite>] \
    [--flash-alias-base <hex>] [--list-macros]

Examples:
  get_model_slot.sh --app optical_cam_oflow
  get_model_slot.sh --app optical_cam_oflow --list-macros
  get_model_slot.sh --app tflm_yolo11_od --macro YOLO11_OBJECT_DETECTION_FLASH_ADDR
  get_model_slot.sh --config EPII_CM55M_APP_S/app/scenario_app/tflm_fd_fm/common_config.h \
    --macro FACE_MESH_FLASH_ADDR
EOF
}

app_name="optical_cam_oflow"
config_file=""
macro_name=""
model_file=""
flash_alias_base=0x3A000000
list_macros=0

resolve_default_config() {
  local app="$1"
  local cfg1="EPII_CM55M_APP_S/app/scenario_app/${app}/config/common_config.h"
  local cfg2="EPII_CM55M_APP_S/app/scenario_app/${app}/common_config.h"
  if [[ -f "$cfg1" ]]; then
    echo "$cfg1"
    return
  fi
  if [[ -f "$cfg2" ]]; then
    echo "$cfg2"
    return
  fi
  echo "Default common_config.h not found for --app ${app}" >&2
  echo "Tried: ${cfg1} and ${cfg2}" >&2
  exit 1
}

strip_comments() {
  echo "$1" | sed -E 's@//.*$@@' | sed -E 's@/\*.*\*/@@g'
}

normalize_expr() {
  local expr="$1"
  expr="$(echo "$expr" | tr -d '[:space:]')"
  while [[ "$expr" =~ ^\(.+\)$ ]]; do
    expr="${expr:1:${#expr}-2}"
  done
  echo "$expr"
}

parse_abs_addr() {
  local expr="$1"
  local base_dec="$2"
  local expr_u
  expr_u="$(echo "$expr" | tr '[:lower:]' '[:upper:]')"

  if [[ "$expr_u" =~ ^0X[0-9A-F]+$ ]]; then
    echo "$((expr))"
    return
  fi

  if [[ "$expr_u" =~ ^[0-9]+$ ]]; then
    echo "$expr_u"
    return
  fi

  if [[ "$expr_u" =~ ^BASE_ADDR_FLASH[0-9A-Z_]*_ALIAS\+(0X[0-9A-F]+|[0-9]+)$ ]]; then
    local off="${BASH_REMATCH[1]}"
    echo "$((base_dec + off))"
    return
  fi

  if [[ "$expr_u" =~ ^BASE_ADDR_FLASH[0-9A-Z_]*_ALIAS$ ]]; then
    echo "$base_dec"
    return
  fi

  echo "Unsupported FLASH_ADDR expression: ${expr}" >&2
  exit 1
}

extract_macro_lines() {
  rg -n "^[[:space:]]*#define[[:space:]]+[A-Za-z0-9_]*FLASH_ADDR\\b" "$config_file" || true
}

print_macro_row() {
  local macro_line="$1"
  local base_dec="$2"
  local line_no="${macro_line%%:*}"
  local def_line="${macro_line#*:}"
  local macro_token expr_raw expr_norm abs_dec abs_hex slot_dec slot_hex

  macro_token="$(echo "$def_line" | awk '{print $2}')"
  expr_raw="$(echo "$def_line" | sed -E 's/^[[:space:]]*#define[[:space:]]+[A-Za-z0-9_]+[[:space:]]+//')"
  expr_raw="$(strip_comments "$expr_raw")"
  expr_norm="$(normalize_expr "$expr_raw")"
  abs_dec="$(parse_abs_addr "$expr_norm" "$base_dec")"
  abs_hex="$(printf "0x%X" "$abs_dec")"
  slot_dec=$((abs_dec - base_dec))
  slot_hex="$(printf "0x%X" "$slot_dec")"

  echo "line=${line_no} macro=${macro_token} expr=${expr_norm} model_flash_abs=${abs_hex} model_flash_slot=${slot_hex}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      app_name="$2"
      shift 2
      ;;
    --config)
      config_file="$2"
      shift 2
      ;;
    --macro)
      macro_name="$2"
      shift 2
      ;;
    --model-file)
      model_file="$2"
      shift 2
      ;;
    --flash-alias-base)
      flash_alias_base="$2"
      shift 2
      ;;
    --list-macros)
      list_macros=1
      shift
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

if [[ -z "$config_file" ]]; then
  config_file="$(resolve_default_config "$app_name")"
fi

if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi

if ! [[ "$flash_alias_base" =~ ^0x[0-9A-Fa-f]+$ ]]; then
  echo "--flash-alias-base must be hex (e.g. 0x3A000000): ${flash_alias_base}" >&2
  exit 1
fi

base_dec=$((flash_alias_base))
mapfile -t macro_lines < <(extract_macro_lines)

if [[ "$list_macros" == "1" ]]; then
  if [[ ${#macro_lines[@]} -eq 0 ]]; then
    echo "No *FLASH_ADDR macro found in: $config_file" >&2
    exit 1
  fi
  echo "config=$config_file"
  echo "flash_alias_base=$flash_alias_base"
  for row in "${macro_lines[@]}"; do
    print_macro_row "$row" "$base_dec"
  done
  exit 0
fi

if [[ -n "$macro_name" ]]; then
  macro_line="$(rg -n "^[[:space:]]*#define[[:space:]]+${macro_name}\\b" "$config_file" | head -n1 || true)"
  if [[ -z "$macro_line" ]]; then
    echo "Macro ${macro_name} not found in: $config_file" >&2
    exit 1
  fi
else
  if [[ ${#macro_lines[@]} -eq 0 ]]; then
    echo "No *FLASH_ADDR macro found in: $config_file" >&2
    exit 1
  fi
  if [[ ${#macro_lines[@]} -gt 1 ]]; then
    echo "Multiple *FLASH_ADDR macros found in: $config_file" >&2
    echo "Use --macro <NAME> or inspect all choices via --list-macros." >&2
    for row in "${macro_lines[@]}"; do
      echo "  ${row#*:}" >&2
    done
    exit 1
  fi
  macro_line="${macro_lines[0]}"
fi

def_line="${macro_line#*:}"
macro_token="$(echo "$def_line" | awk '{print $2}')"
expr_raw="$(echo "$def_line" | sed -E 's/^[[:space:]]*#define[[:space:]]+[A-Za-z0-9_]+[[:space:]]+//')"
expr_raw="$(strip_comments "$expr_raw")"
expr_norm="$(normalize_expr "$expr_raw")"
abs_dec="$(parse_abs_addr "$expr_norm" "$base_dec")"
abs_hex="$(printf "0x%X" "$abs_dec")"

if (( abs_dec < base_dec )); then
  echo "Parsed address is below flash alias base: $abs_hex < $flash_alias_base" >&2
  exit 1
fi

slot_dec=$((abs_dec - base_dec))
slot_hex="$(printf "0x%X" "$slot_dec")"

echo "config=$config_file"
echo "macro=$macro_token"
echo "macro_line=$macro_line"
echo "macro_expr=$expr_norm"
echo "model_flash_abs=$abs_hex"
echo "model_flash_slot=$slot_hex"
echo "model_flash_offset=0x0"

if [[ -n "$model_file" ]]; then
  echo "model_file=$model_file"
  echo "model_arg=\"$model_file $slot_hex 0x00000\""
fi
