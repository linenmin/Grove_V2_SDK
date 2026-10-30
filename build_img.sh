#!/usr/bin/env bash
# Compatibility entry: explicit app/output are now required; never copies to a personal drive.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/build_firmware.sh" "$@"
