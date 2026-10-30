#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON_BIN:-python3}
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_export.py"
: "${EFNAS_ROOT:?Set EFNAS_ROOT to the external EdgeFlowNAS checkout}"
MODEL=${MODEL:?MODEL env var required (v3_acc|v3_efn_fps|v3_light)}
HEIGHT=${HEIGHT:-157}
WIDTH=${WIDTH:-203}
OUT_DIR=${OUT_DIR:-}
RETRAIN_DIR=${RETRAIN_DIR:-}
EXTRA=()
[ -n "$OUT_DIR" ] && EXTRA+=(--output-dir "$OUT_DIR")
[ -n "$RETRAIN_DIR" ] && EXTRA+=(--retrain-dir "$RETRAIN_DIR")
exec "$PY" "$SCRIPT" --model-name "$MODEL" --height "$HEIGHT" --width "$WIDTH" "${EXTRA[@]}" "$@"
