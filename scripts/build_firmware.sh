#!/usr/bin/env bash
# Build only. USB stays with the host OS. SDK paths remain unchanged for upstream compatibility.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="" OUT="" JOBS=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    *) echo "Usage: $0 --app SCENARIO --out NEW_DIRECTORY [--jobs N]" >&2; exit 2;;
  esac
done
[[ "$APP" =~ ^[a-zA-Z0-9_]+$ && -d "$ROOT/EPII_CM55M_APP_S/app/scenario_app/$APP" ]] || { echo 'Select an existing app' >&2; exit 2; }
[[ -n "$OUT" && "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'Output directory and positive jobs required' >&2; exit 2; }
for tool in make arm-none-eabi-gcc flock; do
  command -v "$tool" >/dev/null || { echo "Missing build dependency: $tool (install in Linux/WSL and add to PATH)" >&2; exit 2; }
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
[[ ! -e "$OUT/firmware.img" ]] || { echo 'Refusing to overwrite firmware output' >&2; exit 2; }
# The upstream SDK shares object/image directories between apps.
exec 9>"$ROOT/.build.lock"
flock -n 9 || { echo 'Another build owns this SDK checkout' >&2; exit 2; }
{
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --short
  arm-none-eabi-gcc --version
  make --version
  echo "APP_TYPE=$APP JOBS=$JOBS"
} > "$OUT/build-context.txt"
git -C "$ROOT" diff --binary HEAD > "$OUT/source.patch"
make -C "$ROOT/EPII_CM55M_APP_S" clean APP_TYPE="$APP"
make -C "$ROOT/EPII_CM55M_APP_S" -j"$JOBS" APP_TYPE="$APP"
ELF="$ROOT/EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
cp "$ELF" "$ROOT/we2_image_gen_local/input_case1_secboot/"
cd "$ROOT/we2_image_gen_local"
rm -f output_case1_sec_wlcsp/output.img
./we2_local_image_gen project_case1_blp_wlcsp.json
cp output_case1_sec_wlcsp/output.img "$OUT/firmware.img"
cp "$ELF" "$OUT/firmware.elf"
sha256sum "$OUT/firmware.img" "$OUT/firmware.elf" > "$OUT/SHA256SUMS"
echo "Build complete: $OUT/firmware.img (not flashed)"
