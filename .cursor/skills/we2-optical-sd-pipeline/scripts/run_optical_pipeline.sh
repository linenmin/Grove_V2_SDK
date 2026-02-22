#!/usr/bin/env bash
set -euo pipefail

MODE="nomodel"
APP_TYPE="optical_sd"
PORT="/dev/ttyACM0"
BAUD="921600"
CAPTURE_SECONDS="12"
MODEL_ARG=""
NO_CLEAN="0"
SKIP_BUILD="0"
KEYWORDS=("initial done")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/pipeline"

usage() {
  cat <<'EOF'
Usage:
  run_optical_pipeline.sh [options]

Options:
  --mode nomodel|with-model      Flash mode. Default: nomodel
  --app-type NAME                APP_TYPE for make. Default: optical_sd
  --port DEVICE                  Serial port. Default: /dev/ttyACM0
  --baudrate N                   Serial baudrate. Default: 921600
  --capture-seconds N            UART capture duration. Default: 12
  --keyword TEXT                 Add expected UART keyword (repeatable)
  --model-arg "FILE ADDR OFF"    Required when mode=with-model
  --skip-build                   Skip make + image generation
  --no-clean                     Do not run make clean
  -h, --help                     Show this help

Examples:
  # Fast path: code-only changes (no model flash)
  scripts/run_optical_pipeline.sh --mode nomodel --app-type optical_sd_poc_model

  # Full path: app + model
  scripts/run_optical_pipeline.sh \
    --mode with-model \
    --app-type optical_sd_poc_model \
    --model-arg "model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite 0xB7B000 0x00000"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --app-type) APP_TYPE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --baudrate) BAUD="$2"; shift 2 ;;
    --capture-seconds) CAPTURE_SECONDS="$2"; shift 2 ;;
    --keyword) KEYWORDS+=("$2"); shift 2 ;;
    --model-arg) MODEL_ARG="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD="1"; shift ;;
    --no-clean) NO_CLEAN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "nomodel" && "${MODE}" != "with-model" ]]; then
  echo "Invalid --mode: ${MODE}. Use nomodel|with-model." >&2
  exit 2
fi

if [[ "${MODE}" == "with-model" && -z "${MODEL_ARG}" ]]; then
  echo "mode=with-model requires --model-arg \"FILE ADDR OFF\"." >&2
  exit 2
fi

echo "[preflight] checking python modules..."
python3 - <<'PY'
import importlib.util, sys
mods = ["serial", "xmodem"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("[error] missing python modules:", ", ".join(missing))
    print("Install with: pip3 install -r xmodem/requirements.txt")
    sys.exit(3)
print("[ok] python modules ready")
PY

echo "[preflight] checking serial port ${PORT}..."
if ! python3 - <<PY
import serial, sys
port = "${PORT}"
baud = int("${BAUD}")
try:
    s = serial.Serial(port, baud, timeout=1)
    s.close()
    print("[ok] serial open:", port)
except Exception as e:
    print("[error] serial open failed:", e)
    print("[hint] reconnect USB to WSL:")
    print("  Windows(admin): usbipd list")
    print("  Windows(admin): usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>")
    print("  WSL: ls -l /dev/ttyACM*")
    sys.exit(4)
PY
then
  exit 4
fi

IMG_PATH="${REPO_ROOT}/we2_image_gen_local/output_case1_sec_wlcsp/output.img"
ELF_PATH="${REPO_ROOT}/EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/pipeline_${MODE}_${APP_TYPE}_${STAMP}.log"

mkdir -p "${LOG_DIR}"

if [[ "${SKIP_BUILD}" == "0" ]]; then
  echo "[1/4] build app (${APP_TYPE})..."
  pushd "${REPO_ROOT}/EPII_CM55M_APP_S" >/dev/null
  if [[ "${NO_CLEAN}" == "0" ]]; then
    make clean APP_TYPE="${APP_TYPE}"
  fi
  make -s --no-print-directory -j4 APP_TYPE="${APP_TYPE}"
  popd >/dev/null

  echo "[2/4] generate image..."
  pushd "${REPO_ROOT}/we2_image_gen_local" >/dev/null
  cp "${ELF_PATH}" "input_case1_secboot/"
  ./we2_local_image_gen project_case1_blp_wlcsp.json
  popd >/dev/null
else
  echo "[1/4] skip build and image generation"
fi

if [[ ! -f "${IMG_PATH}" ]]; then
  echo "[error] image not found: ${IMG_PATH}" >&2
  exit 5
fi

echo "[3/4] flash image (${MODE})..."
FLASH_CMD=(python3 "${REPO_ROOT}/xmodem/xmodem_send.py"
  --port="${PORT}"
  --baudrate="${BAUD}"
  --protocol=xmodem
  --file="${IMG_PATH}")

if [[ "${MODE}" == "with-model" ]]; then
  FLASH_CMD+=(--model="${MODEL_ARG}")
fi

"${FLASH_CMD[@]}"

echo "[4/4] capture uart and verify keywords..."
CAPTURE_CMD=(python3 "${REPO_ROOT}/xmodem/serReadLoop.py"
  --port="${PORT}"
  --baudrate="${BAUD}"
  --timeout=1
  --duration="${CAPTURE_SECONDS}"
  --log-file="${LOG_PATH}")

for kw in "${KEYWORDS[@]}"; do
  CAPTURE_CMD+=(--keyword="${kw}")
done

"${CAPTURE_CMD[@]}"

echo
echo "[done] pipeline success"
echo "[info] mode=${MODE} app_type=${APP_TYPE} port=${PORT}"
echo "[info] log=${LOG_PATH}"
