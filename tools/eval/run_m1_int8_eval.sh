#!/usr/bin/env bash
# Run INT8 Sintel EPE on the deployed M1 mainline tflite.
set -euo pipefail

PY=/home/enmin/miniconda3/envs/vela/bin/python
EVAL=/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/eval/int8_sintel_eval.py
TFLITE=/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite
LIST=/mnt/d/Dataset/MCUFlowNet/EdgeFlowNet/code/dataset_paths/MPI_Sintel_train_clean.txt
ROOT=/mnt/g/AI_thesis/datasets/MPI-Sintel-complete
REPORT=${REPORT:-/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/m1_int8_sintel_clean.json}
LIMIT=${LIMIT:-0}
EVAL_GRID=${EVAL_GRID:-native}

exec "$PY" "$EVAL" \
  --tflite "$TFLITE" \
  --list "$LIST" \
  --sintel-root "$ROOT" \
  --limit "$LIMIT" \
  --eval-grid "$EVAL_GRID" \
  --report "$REPORT" \
  --threads 4
