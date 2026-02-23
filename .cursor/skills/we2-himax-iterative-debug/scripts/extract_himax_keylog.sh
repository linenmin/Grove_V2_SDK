#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  extract_himax_keylog.sh --log <raw_log> [--out <key_log>] [--summary <summary_file>] [--pattern-file <file>] [--no-redact]

Examples:
  extract_himax_keylog.sh --log logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_223357.log
  extract_himax_keylog.sh --log run.log --out run.key.log --summary run.summary.txt
EOF
}

log_file=""
out_file=""
summary_file=""
pattern_file=""
redact_base64="yes"

default_pattern='initial done|WD1\[|WD2_J\[|WD3_RAW\[|JPAuto\[|wait first camera frame timeout|wait new camera frame timeout|camera frame capture fail|cv_yolov8n_ob_run fail|"name": "NAME\?"|"name": "VER\?"|"name": "ID\?"|"name": "INFO\?"|"name": "MODEL\?"|"name": "INVOKE"|viz skip invalid jpeg|\[loop=|\[SUMMARY\]|\[done\]|pipeline success'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      log_file="$2"
      shift 2
      ;;
    --out)
      out_file="$2"
      shift 2
      ;;
    --summary)
      summary_file="$2"
      shift 2
      ;;
    --pattern-file)
      pattern_file="$2"
      shift 2
      ;;
    --no-redact)
      redact_base64="no"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$log_file" ]]; then
  echo "--log is required." >&2
  usage
  exit 1
fi

if [[ ! -f "$log_file" ]]; then
  echo "Log file not found: $log_file" >&2
  exit 1
fi

if [[ -z "$out_file" ]]; then
  if [[ "$log_file" == *.log ]]; then
    out_file="${log_file%.log}.key.log"
  else
    out_file="${log_file}.key.log"
  fi
fi

if [[ -z "$summary_file" ]]; then
  if [[ "$out_file" == *.log ]]; then
    summary_file="${out_file%.log}.summary.txt"
  else
    summary_file="${out_file}.summary.txt"
  fi
fi

pattern="$default_pattern"
if [[ -n "$pattern_file" ]]; then
  if [[ ! -f "$pattern_file" ]]; then
    echo "Pattern file not found: $pattern_file" >&2
    exit 1
  fi
  pattern="$(cat "$pattern_file")"
fi

rg -a -n "$pattern" "$log_file" > "$out_file" || true

if [[ "$redact_base64" == "yes" ]]; then
  sed -E -i 's/"image": "[^"]+"/"image":"<base64-redacted>"/g' "$out_file"
fi

count_or_zero() {
  local p="$1"
  local n
  n="$(rg -a -c "$p" "$log_file" 2>/dev/null || true)"
  if [[ -z "$n" ]]; then
    echo "0"
  else
    echo "$n"
  fi
}

first_hit_or_na() {
  local p="$1"
  local line
  line="$(grep -E -m1 "$p" "$out_file" 2>/dev/null || true)"
  if [[ -z "$line" ]]; then
    echo "N/A"
  else
    echo "$line"
  fi
}

invoke_count="$(count_or_zero '"name": "INVOKE"')"
name_count="$(count_or_zero '"name": "NAME\?"')"
model_count="$(count_or_zero '"name": "MODEL\?"')"
jpeg_skip_count="$(count_or_zero 'viz skip invalid jpeg')"
cam_fail_count="$(count_or_zero 'camera frame capture fail')"
wd3_zero_count="$(count_or_zero 'WD3_RAW\[0\]')"

cat > "$summary_file" <<EOF
log_file=$log_file
key_log=$out_file
generated_at=$(date -Iseconds)
invoke_count=$invoke_count
name_count=$name_count
model_count=$model_count
jpeg_skip_count=$jpeg_skip_count
camera_frame_capture_fail_count=$cam_fail_count
wd3_raw_zero_count=$wd3_zero_count
first_initial_done=$(first_hit_or_na 'initial done')
first_invoke=$(first_hit_or_na '"name": "INVOKE"')
first_model=$(first_hit_or_na '"name": "MODEL\?"')
first_jpeg_skip=$(first_hit_or_na 'viz skip invalid jpeg')
EOF

echo "key_log=$out_file"
echo "summary=$summary_file"
