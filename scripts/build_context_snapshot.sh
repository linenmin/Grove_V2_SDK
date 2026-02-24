#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAN_DIR="${REPO_ROOT}/plan"
PIPELINE_DIR="${REPO_ROOT}/logs/pipeline"
CONTEXT_DIR="${REPO_ROOT}/logs/context"

MD_OUTPUT=""
JSON_OUTPUT=""
UPDATE_LATEST="1"

usage() {
  cat <<'USAGE'
Usage:
  build_context_snapshot.sh [options]

Options:
  --md-output PATH        Write markdown snapshot to PATH.
  --json-output PATH      Also write json snapshot to PATH.
  --no-latest             Do not refresh context_snapshot_latest.*
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --md-output)
      MD_OUTPUT="$2"
      shift 2
      ;;
    --json-output)
      JSON_OUTPUT="$2"
      shift 2
      ;;
    --no-latest)
      UPDATE_LATEST="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${CONTEXT_DIR}"

stamp="$(date +%Y%m%d_%H%M%S)"
now_iso="$(date --iso-8601=seconds)"

if [[ -z "${MD_OUTPUT}" ]]; then
  MD_OUTPUT="${CONTEXT_DIR}/context_snapshot_${stamp}.md"
fi

read_kv() {
  local file="$1"
  local key="$2"
  [[ -f "${file}" ]] || return 0
  awk -F= -v k="${key}" '$1==k {print substr($0, index($0, $2)); exit}' "${file}"
}

normalize_path() {
  local p="$1"
  if [[ -z "${p}" || "${p}" == "N/A" ]]; then
    printf '%s' "${p}"
    return
  fi
  if [[ "${p}" = /* ]]; then
    printf '%s' "${p}"
  else
    printf '%s' "${REPO_ROOT}/${p}"
  fi
}

extract_define() {
  local file="$1"
  local macro="$2"
  [[ -f "${file}" ]] || {
    printf '%s' "N/A"
    return
  }
  awk -v macro="${macro}" '
    $1=="#define" && $2==macro {
      sub(/^[[:space:]]*#define[[:space:]]+[^[:space:]]+[[:space:]]+/, "", $0)
      print $0
      found=1
      exit
    }
    END {
      if (!found) {
        print "N/A"
      }
    }
  ' "${file}"
}

extract_assignment() {
  local file="$1"
  local token="$2"
  [[ -f "${file}" ]] || {
    printf '%s' "N/A"
    return
  }
  awk -v token="${token}" '
    $0 ~ token"[[:space:]]*=" {
      line=$0
      sub(/.*=/, "", line)
      sub(/;.*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      print line
      found=1
      exit
    }
    END {
      if (!found) {
        print "N/A"
      }
    }
  ' "${file}"
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/}"
  printf '%s' "${s}"
}

context_index="${PLAN_DIR}/plan-000-context-index.md"
latest_debug_plan="$(ls -1t "${PLAN_DIR}"/plan-*.md 2>/dev/null | grep -v '/plan-000-context-index.md$' | head -n1 || true)"
latest_summary="$(ls -1t "${PIPELINE_DIR}"/*.key.summary.txt 2>/dev/null | head -n1 || true)"
latest_pipeline_log="$(ls -1t "${PIPELINE_DIR}"/pipeline_*.log 2>/dev/null | grep -v '\\.key\\.log$' | head -n1 || true)"

log_file=""
key_log=""
generated_at=""
invoke_count=""
name_count=""
model_count=""
jpeg_skip_count=""
camera_frame_capture_fail_count=""
wd3_raw_zero_count=""
first_initial_done=""
first_invoke=""
first_model=""
first_jpeg_skip=""
resolution=""

if [[ -n "${latest_summary}" ]]; then
  log_file="$(read_kv "${latest_summary}" "log_file" | tr -d '\r')"
  key_log="$(read_kv "${latest_summary}" "key_log" | tr -d '\r')"
  generated_at="$(read_kv "${latest_summary}" "generated_at" | tr -d '\r')"
  invoke_count="$(read_kv "${latest_summary}" "invoke_count" | tr -d '\r')"
  name_count="$(read_kv "${latest_summary}" "name_count" | tr -d '\r')"
  model_count="$(read_kv "${latest_summary}" "model_count" | tr -d '\r')"
  jpeg_skip_count="$(read_kv "${latest_summary}" "jpeg_skip_count" | tr -d '\r')"
  camera_frame_capture_fail_count="$(read_kv "${latest_summary}" "camera_frame_capture_fail_count" | tr -d '\r')"
  wd3_raw_zero_count="$(read_kv "${latest_summary}" "wd3_raw_zero_count" | tr -d '\r')"
  first_initial_done="$(read_kv "${latest_summary}" "first_initial_done" | tr -d '\r')"
  first_invoke="$(read_kv "${latest_summary}" "first_invoke" | tr -d '\r')"
  first_model="$(read_kv "${latest_summary}" "first_model" | tr -d '\r')"
  first_jpeg_skip="$(read_kv "${latest_summary}" "first_jpeg_skip" | tr -d '\r')"

  log_file="$(normalize_path "${log_file}")"
  key_log="$(normalize_path "${key_log}")"

  if [[ -n "${log_file}" ]]; then
    latest_pipeline_log="${log_file}"
  fi

  resolution="$(printf '%s\n' "${first_invoke}" | sed -n 's/.*"resolution"[[:space:]]*:[[:space:]]*\(\[[^]]*\]\).*/\1/p' | head -n1)"
fi

latest_loop_line="N/A"
if [[ -n "${key_log}" && -f "${key_log}" ]]; then
  latest_loop_line="$(tr -d '\r' < "${key_log}" | grep -E '\[loop=' | tail -n1 || true)"
  if [[ -z "${latest_loop_line}" ]]; then
    latest_loop_line="N/A"
  fi
fi

common_cfg="${REPO_ROOT}/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/common_config.h"
cvapp_cpp="${REPO_ROOT}/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_yolov8n_ob.cpp"
flow_render_cpp="${REPO_ROOT}/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/viz/flow_render.cpp"
cam_input_cpp="${REPO_ROOT}/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/io/camera/cam_input.cpp"

flash_addr="$(extract_define "${common_cfg}" "YOLOV8_OBJECT_DETECTION_FLASH_ADDR")"
tensor_arena_size="$(extract_assignment "${cvapp_cpp}" "tensor_arena_size")"
flow_dbg_freeze_pair="$(extract_define "${cvapp_cpp}" "FLOW_DBG_FREEZE_PAIR")"
flow_dbg_synth_inject="$(extract_define "${cvapp_cpp}" "FLOW_DBG_SYNTH_INJECT")"
flow_dbg_synth_curr_const="$(extract_define "${cvapp_cpp}" "FLOW_DBG_SYNTH_CURR_CONST")"
flow_dbg_synth_prev_const="$(extract_define "${cvapp_cpp}" "FLOW_DBG_SYNTH_PREV_CONST")"
flow_dbg_perturb_enable="$(extract_define "${cvapp_cpp}" "FLOW_DBG_PERTURB_ENABLE")"
flow_dbg_perturb_target="$(extract_define "${cvapp_cpp}" "FLOW_DBG_PERTURB_TARGET")"
flow_dbg_perturb_alt_every_other="$(extract_define "${cvapp_cpp}" "FLOW_DBG_PERTURB_ALT_EVERY_OTHER")"
flow_dbg_perturb_stride="$(extract_define "${cvapp_cpp}" "FLOW_DBG_PERTURB_STRIDE")"
flow_dbg_perturb_delta="$(extract_define "${cvapp_cpp}" "FLOW_DBG_PERTURB_DELTA")"
flow_viz_fixed_scale="$(extract_define "${flow_render_cpp}" "FLOW_VIZ_FIXED_SCALE")"
flow_viz_test_pattern="$(extract_define "${flow_render_cpp}" "FLOW_VIZ_TEST_PATTERN")"
flow_viz_light_smooth="$(extract_define "${flow_render_cpp}" "FLOW_VIZ_LIGHT_SMOOTH")"
flow_viz_remove_row_bias="$(extract_define "${flow_render_cpp}" "FLOW_VIZ_REMOVE_ROW_BIAS")"
cam_input_use_bgr="$(extract_define "${cam_input_cpp}" "CAM_INPUT_USE_BGR")"
cam_input_use_helium_resize="$(extract_define "${cam_input_cpp}" "CAM_INPUT_USE_HELIUM_RESIZE")"

cat > "${MD_OUTPUT}" <<EOF_MD
# Context Snapshot

- generated_at: ${now_iso}
- repo_root: ${REPO_ROOT}
- context_index: ${context_index:-N/A}
- latest_debug_plan: ${latest_debug_plan:-N/A}
- latest_key_summary: ${latest_summary:-N/A}
- latest_key_log: ${key_log:-N/A}
- latest_pipeline_log: ${latest_pipeline_log:-N/A}

## Runtime Summary

- key_summary_generated_at: ${generated_at:-N/A}
- invoke_count: ${invoke_count:-N/A}
- name_count: ${name_count:-N/A}
- model_count: ${model_count:-N/A}
- jpeg_skip_count: ${jpeg_skip_count:-N/A}
- camera_frame_capture_fail_count: ${camera_frame_capture_fail_count:-N/A}
- wd3_raw_zero_count: ${wd3_raw_zero_count:-N/A}
- first_initial_done: ${first_initial_done:-N/A}
- first_model: ${first_model:-N/A}
- first_jpeg_skip: ${first_jpeg_skip:-N/A}
- first_invoke_resolution: ${resolution:-N/A}

## Source Knobs

- tensor_arena_size: ${tensor_arena_size}
- YOLOV8_OBJECT_DETECTION_FLASH_ADDR: ${flash_addr}
- FLOW_DBG_FREEZE_PAIR: ${flow_dbg_freeze_pair}
- FLOW_DBG_SYNTH_INJECT: ${flow_dbg_synth_inject}
- FLOW_DBG_SYNTH_CURR_CONST: ${flow_dbg_synth_curr_const}
- FLOW_DBG_SYNTH_PREV_CONST: ${flow_dbg_synth_prev_const}
- FLOW_DBG_PERTURB_ENABLE: ${flow_dbg_perturb_enable}
- FLOW_DBG_PERTURB_TARGET: ${flow_dbg_perturb_target}
- FLOW_DBG_PERTURB_ALT_EVERY_OTHER: ${flow_dbg_perturb_alt_every_other}
- FLOW_DBG_PERTURB_STRIDE: ${flow_dbg_perturb_stride}
- FLOW_DBG_PERTURB_DELTA: ${flow_dbg_perturb_delta}
- FLOW_VIZ_FIXED_SCALE: ${flow_viz_fixed_scale}
- FLOW_VIZ_TEST_PATTERN: ${flow_viz_test_pattern}
- FLOW_VIZ_LIGHT_SMOOTH: ${flow_viz_light_smooth}
- FLOW_VIZ_REMOVE_ROW_BIAS: ${flow_viz_remove_row_bias}
- CAM_INPUT_USE_BGR: ${cam_input_use_bgr}
- CAM_INPUT_USE_HELIUM_RESIZE: ${cam_input_use_helium_resize}

## Latest Loop Stats

\`\`\`
${latest_loop_line}
\`\`\`
EOF_MD

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  cp "${MD_OUTPUT}" "${CONTEXT_DIR}/context_snapshot_latest.md"
fi

if [[ -n "${JSON_OUTPUT}" ]]; then
  mkdir -p "$(dirname "${JSON_OUTPUT}")"
  cat > "${JSON_OUTPUT}" <<EOF_JSON
{
  "generated_at": "$(json_escape "${now_iso}")",
  "repo_root": "$(json_escape "${REPO_ROOT}")",
  "context_index": "$(json_escape "${context_index:-N/A}")",
  "latest_debug_plan": "$(json_escape "${latest_debug_plan:-N/A}")",
  "latest_key_summary": "$(json_escape "${latest_summary:-N/A}")",
  "latest_key_log": "$(json_escape "${key_log:-N/A}")",
  "latest_pipeline_log": "$(json_escape "${latest_pipeline_log:-N/A}")",
  "runtime": {
    "key_summary_generated_at": "$(json_escape "${generated_at:-N/A}")",
    "invoke_count": "$(json_escape "${invoke_count:-N/A}")",
    "name_count": "$(json_escape "${name_count:-N/A}")",
    "model_count": "$(json_escape "${model_count:-N/A}")",
    "jpeg_skip_count": "$(json_escape "${jpeg_skip_count:-N/A}")",
    "camera_frame_capture_fail_count": "$(json_escape "${camera_frame_capture_fail_count:-N/A}")",
    "wd3_raw_zero_count": "$(json_escape "${wd3_raw_zero_count:-N/A}")",
    "first_initial_done": "$(json_escape "${first_initial_done:-N/A}")",
    "first_model": "$(json_escape "${first_model:-N/A}")",
    "first_jpeg_skip": "$(json_escape "${first_jpeg_skip:-N/A}")",
    "first_invoke_resolution": "$(json_escape "${resolution:-N/A}")"
  },
  "source_knobs": {
    "tensor_arena_size": "$(json_escape "${tensor_arena_size}")",
    "YOLOV8_OBJECT_DETECTION_FLASH_ADDR": "$(json_escape "${flash_addr}")",
    "FLOW_DBG_FREEZE_PAIR": "$(json_escape "${flow_dbg_freeze_pair}")",
    "FLOW_DBG_SYNTH_INJECT": "$(json_escape "${flow_dbg_synth_inject}")",
    "FLOW_DBG_SYNTH_CURR_CONST": "$(json_escape "${flow_dbg_synth_curr_const}")",
    "FLOW_DBG_SYNTH_PREV_CONST": "$(json_escape "${flow_dbg_synth_prev_const}")",
    "FLOW_DBG_PERTURB_ENABLE": "$(json_escape "${flow_dbg_perturb_enable}")",
    "FLOW_DBG_PERTURB_TARGET": "$(json_escape "${flow_dbg_perturb_target}")",
    "FLOW_DBG_PERTURB_ALT_EVERY_OTHER": "$(json_escape "${flow_dbg_perturb_alt_every_other}")",
    "FLOW_DBG_PERTURB_STRIDE": "$(json_escape "${flow_dbg_perturb_stride}")",
    "FLOW_DBG_PERTURB_DELTA": "$(json_escape "${flow_dbg_perturb_delta}")",
    "FLOW_VIZ_FIXED_SCALE": "$(json_escape "${flow_viz_fixed_scale}")",
    "FLOW_VIZ_TEST_PATTERN": "$(json_escape "${flow_viz_test_pattern}")",
    "FLOW_VIZ_LIGHT_SMOOTH": "$(json_escape "${flow_viz_light_smooth}")",
    "FLOW_VIZ_REMOVE_ROW_BIAS": "$(json_escape "${flow_viz_remove_row_bias}")",
    "CAM_INPUT_USE_BGR": "$(json_escape "${cam_input_use_bgr}")",
    "CAM_INPUT_USE_HELIUM_RESIZE": "$(json_escape "${cam_input_use_helium_resize}")"
  },
  "latest_loop_line": "$(json_escape "${latest_loop_line}")"
}
EOF_JSON

  if [[ "${UPDATE_LATEST}" == "1" ]]; then
    cp "${JSON_OUTPUT}" "${CONTEXT_DIR}/context_snapshot_latest.json"
  fi
fi

echo "[done] context snapshot generated"
echo "[info] markdown=${MD_OUTPUT}"
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  echo "[info] latest_markdown=${CONTEXT_DIR}/context_snapshot_latest.md"
fi
if [[ -n "${JSON_OUTPUT}" ]]; then
  echo "[info] json=${JSON_OUTPUT}"
  if [[ "${UPDATE_LATEST}" == "1" ]]; then
    echo "[info] latest_json=${CONTEXT_DIR}/context_snapshot_latest.json"
  fi
fi
