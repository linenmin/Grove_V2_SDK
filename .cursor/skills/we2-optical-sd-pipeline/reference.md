# Reference: WE2 Optical SD Pipeline

## Script

- Path: `.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh`

## Key Parameters

- `--mode nomodel|with-model`
- `--app-type <name>` (default `optical_sd`)
- `--port /dev/ttyACM0`
- `--model-arg "FILE ADDR OFF"` (required for `with-model`)
- `--keyword "<text>"` (repeatable)
- `--capture-seconds <N>`
- `--skip-build` (reuse existing image)

## Recommended Mode Selection

- Use `nomodel` when:
  - only C/C++ app code changed
  - model binary and model flash address are unchanged
- Use `with-model` when:
  - model file changed
  - model flash address changed
  - device model area may have been erased or overwritten

## Serial and USB Recovery

If `/dev/ttyACM0` is missing or open fails:

1. Windows (Administrator PowerShell):
   - `usbipd list`
   - `usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>`
2. WSL:
   - `ls -l /dev/ttyACM*`
   - `python3 -c "import serial; s=serial.Serial('/dev/ttyACM0',921600,timeout=1); print('open ok'); s.close()"`

## Validation Signals

Successful run should include:

- `xmodem_send bin file result = True`
- `Firmware upgrade completed, restart WE2 ...`
- UART keyword hits (`[KEYWORD_HIT] ...`)
- final summary: `[SUMMARY] all_keywords_hit`

## Typical Failure Patterns

- Serial open error:
  - likely USB detach or port occupied
  - fix USB mapping first, then retry
- Missing keyword:
  - app may not reach expected stage
  - inspect saved UART log in `logs/pipeline/`
- Build failure:
  - check current `APP_TYPE`
  - inspect compiler errors before retrying
