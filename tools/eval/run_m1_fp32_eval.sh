#!/usr/bin/env bash
# Run FP32 Sintel EPE on M1 mainline best.ckpt using the original
# EdgeFlowNet test_sintel.py (TF1 graph). CPU only (no CUDA in vela env).
set -euo pipefail

PY=/home/enmin/miniconda3/envs/vela/bin/python
CODE=/mnt/d/Dataset/MCUFlowNet/EdgeFlowNet/code
CKPT=/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/assets/checkpoints/best.ckpt
LIST=/mnt/d/Dataset/MCUFlowNet/EdgeFlowNet/code/dataset_paths/MPI_Sintel_train_clean.txt
SINTEL_SRC=/mnt/g/AI_thesis/datasets/MPI-Sintel-complete
WORK=/tmp/m1_fp32_eval
LOG=${LOG:-/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean.log}

# Stage the relative-path data root that test_sintel.py expects:
#   list lines reference Datasets/Sintel/training/...
mkdir -p "$WORK/Datasets"
ln -sfn "$SINTEL_SRC" "$WORK/Datasets/Sintel"

cd "$WORK"
export PYTHONPATH="$CODE":"${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2

exec "$PY" "$CODE/test_sintel.py" \
  --checkpoint "$CKPT" \
  --data_list "$LIST" \
  --gpu_device -1 \
  --patch_dim_0 416 \
  --patch_dim_1 1024 \
  --patch_channels 3 \
  --uncertainity
