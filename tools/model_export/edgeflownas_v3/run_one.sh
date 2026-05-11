#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/enmin/miniconda3/envs/vela/bin:${PATH:-}
PY=/home/enmin/miniconda3/envs/vela/bin/python
SCRIPT=/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/edgeflownas_v3/run_export.py
MODEL=${MODEL:?MODEL env var required (v3_acc|v3_efn_fps|v3_light)}
HEIGHT=${HEIGHT:-157}
WIDTH=${WIDTH:-203}
OUT_DIR=${OUT_DIR:-}
EXTRA=()
[ -n "$OUT_DIR" ] && EXTRA+=(--output-dir "$OUT_DIR")
exec "$PY" "$SCRIPT" --model-name "$MODEL" --height "$HEIGHT" --width "$WIDTH" "${EXTRA[@]}" "$@"
