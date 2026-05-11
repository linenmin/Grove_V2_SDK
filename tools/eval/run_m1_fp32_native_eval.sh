#!/usr/bin/env bash
set -euo pipefail
PY=/home/enmin/miniconda3/envs/vela/bin/python
EVAL=/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/eval/fp32_sintel_eval.py
LIST=/mnt/d/Dataset/MCUFlowNet/EdgeFlowNet/code/dataset_paths/MPI_Sintel_train_clean.txt
ROOT=/mnt/g/AI_thesis/datasets/MPI-Sintel-complete
REPORT=${REPORT:-/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean_native.json}
LIMIT=${LIMIT:-0}
export TF_CPP_MIN_LOG_LEVEL=2
exec "$PY" "$EVAL" \
  --list "$LIST" \
  --sintel-root "$ROOT" \
  --limit "$LIMIT" \
  --report "$REPORT"
