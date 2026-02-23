#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  inspect_tflite_model.sh --model <path.tflite>

Description:
  Print compact model facts for debug and flash planning:
  - resolved model path
  - file size and sha256
  - input/output tensor shapes and dtypes
  - op histogram and Ethos-U hint
EOF
}

model_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model_path="$2"
      shift 2
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

if [[ -z "$model_path" ]]; then
  echo "--model is required." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
path_info="$("${script_dir}/normalize_model_path.sh" --path "$model_path")"
resolved_path=""
while IFS='=' read -r key value; do
  if [[ "$key" == "resolved_path" ]]; then
    resolved_path="$value"
  fi
done <<< "$path_info"

if [[ -z "$resolved_path" ]]; then
  echo "Failed to resolve model path: $model_path" >&2
  exit 1
fi

python3 - "$resolved_path" <<'PY'
import collections
import hashlib
import sys

model_path = sys.argv[1]

backend = None
Interpreter = None

try:
    from tflite_runtime.interpreter import Interpreter as RtInterpreter
    Interpreter = RtInterpreter
    backend = "tflite_runtime"
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        backend = "tensorflow"
    except Exception as exc:
        print("error=missing_tflite_interpreter")
        print("hint=install_tflite_runtime_or_tensorflow")
        print(f"detail={exc}")
        sys.exit(2)

with open(model_path, "rb") as f:
    blob = f.read()

sha256 = hashlib.sha256(blob).hexdigest()
size_bytes = len(blob)

interpreter = Interpreter(model_path=model_path)
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

try:
    ops = interpreter._get_ops_details()
except Exception:
    ops = []

def shape_to_text(shape):
    return "x".join(str(int(v)) for v in shape)

print(f"backend={backend}")
print(f"model_file={model_path}")
print(f"model_size_bytes={size_bytes}")
print(f"model_sha256={sha256}")
print(f"input_count={len(inputs)}")
for i, d in enumerate(inputs):
    print(f"input[{i}].name={d.get('name', '')}")
    print(f"input[{i}].shape={shape_to_text(d.get('shape', []))}")
    print(f"input[{i}].dtype={d.get('dtype')}")

print(f"output_count={len(outputs)}")
for i, d in enumerate(outputs):
    print(f"output[{i}].name={d.get('name', '')}")
    print(f"output[{i}].shape={shape_to_text(d.get('shape', []))}")
    print(f"output[{i}].dtype={d.get('dtype')}")

op_names = [str(op.get("op_name", "UNKNOWN")) for op in ops]
hist = collections.Counter(op_names)
print(f"op_count={len(op_names)}")
for name in sorted(hist):
    print(f"op[{name}]={hist[name]}")

ethosu = any("ETHOS" in name.upper() for name in op_names)
print(f"has_ethosu_op={1 if ethosu else 0}")
PY
