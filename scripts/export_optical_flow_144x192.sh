#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_DIR="${REPO_ROOT}/tools/model_export/optical_flow_144x192"

PYTHON_BIN="${PYTHON_BIN:-/home/enmin/miniconda3/envs/vela/bin/python}"
CHECKPOINT_PREFIX="${OPTICAL_FLOW_CHECKPOINT_PREFIX:-${TOOL_DIR}/assets/checkpoints/best.ckpt}"
CALIBRATION_DIR="${OPTICAL_FLOW_CALIBRATION_DIR:-${TOOL_DIR}/assets/calibration}"
OUTPUT_DIR="${OPTICAL_FLOW_EXPORT_OUTPUT_DIR:-${TOOL_DIR}/output}"
PUBLISHED_MODEL="${OPTICAL_FLOW_PUBLISHED_MODEL:-${REPO_ROOT}/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite}"

"${PYTHON_BIN}" "${TOOL_DIR}/run_export.py" \
  --checkpoint-prefix "${CHECKPOINT_PREFIX}" \
  --calibration-dir "${CALIBRATION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --published-model "${PUBLISHED_MODEL}" \
  "$@"
