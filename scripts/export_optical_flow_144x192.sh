#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_DIR="${REPO_ROOT}/tools/model_export/optical_flow_144x192"

PYTHON_BIN="${PYTHON_BIN:-/home/enmin/miniconda3/envs/vela/bin/python}"
CHECKPOINT_PREFIX="${OPTICAL_FLOW_CHECKPOINT_PREFIX:-${TOOL_DIR}/assets/checkpoints/best.ckpt}"
CALIBRATION_DIR="${OPTICAL_FLOW_CALIBRATION_DIR:-${TOOL_DIR}/assets/calibration}"
VARIANT="${OPTICAL_FLOW_EXPORT_VARIANT:-mainline}"
OUTPUT_DIR="${OPTICAL_FLOW_EXPORT_OUTPUT_DIR:-}"
PUBLISHED_MODEL="${OPTICAL_FLOW_PUBLISHED_MODEL:-}"

CMD=(
  "${PYTHON_BIN}" "${TOOL_DIR}/run_export.py"
  --checkpoint-prefix "${CHECKPOINT_PREFIX}"
  --calibration-dir "${CALIBRATION_DIR}"
  --variant "${VARIANT}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
elif [[ "${VARIANT}" == "mainline" ]]; then
  CMD+=(--output-dir "${TOOL_DIR}/output")
fi

if [[ -n "${PUBLISHED_MODEL}" ]]; then
  CMD+=(--published-model "${PUBLISHED_MODEL}")
elif [[ "${VARIANT}" == "mainline" ]]; then
  CMD+=(--published-model "${REPO_ROOT}/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite")
fi

"${CMD[@]}" "$@"
