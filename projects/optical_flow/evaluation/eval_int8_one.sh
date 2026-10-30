#!/usr/bin/env bash
# Eval one INT8 tflite on Sintel (Final, test_sintel mode by default).
set -euo pipefail
PY=${PYTHON_BIN:-python3}
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/int8_sintel_eval.py"
TFLITE=${1:?tflite path required}
REPORT=${REPORT:-/tmp/int8_eval_report.json}
LIST=${LIST:?Set LIST to the evaluation pair list}
ROOT=${SINTEL_ROOT:?Set SINTEL_ROOT to the dataset directory}
LIMIT=${LIMIT:-0}
REF_MODE=${REF_MODE:-test_sintel}
CLIP_VAL=${CLIP_VAL:-50}
PATCH_H=${PATCH_H:-416}
PATCH_W=${PATCH_W:-1024}
FLOW_SCALE=${FLOW_SCALE:-1.0}
exec "$PY" "$EVAL" \
  --tflite "$TFLITE" \
  --list "$LIST" \
  --sintel-root "$ROOT" \
  --limit "$LIMIT" \
  --ref-mode "$REF_MODE" \
  --clip-val "$CLIP_VAL" \
  --patch-h "$PATCH_H" \
  --patch-w "$PATCH_W" \
  --flow-scale "$FLOW_SCALE" \
  --report "$REPORT" \
  --threads 4
