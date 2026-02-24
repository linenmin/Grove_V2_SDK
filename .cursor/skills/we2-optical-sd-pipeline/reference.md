# Reference: WE2 Optical SD Pipeline

## Script

- Path: `.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh`
- Context snapshot script: `scripts/build_context_snapshot.sh`

## Context Bootstrap Order

For low-context continuation, open files in order:

1. `plan/plan-000-context-index.md`
2. `logs/context/context_snapshot_latest.md`
3. Latest debug plan from snapshot

Only backtrack older plans when current evidence is inconsistent.

## Key Parameters

- `--mode nomodel|with-model`
- `--app-type <name>` (default `optical_sd`, use `optical_cam_oflow` for camera/flow app)
- `--port /dev/ttyACM0`
- `--model-arg "FILE ADDR OFF"` (required for `with-model`)
- `--keyword "<text>"` (repeatable)
- `--capture-seconds <N>`
- `--skip-build` (reuse existing image)
- `--viz-camera` (build with `VIZ_CAMERA=1`, force camera JPEG instead of flow)
- `--extract-frames` (after capture, extract INVOKE images to `logs/flow_frames/latest/`)
- `--max-frames N` (max frames to extract, default 10)

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

## Agent Visible Visualization (plan-008)

Extract script: `scripts/extract_invoke_frames_from_log.py`

Standalone extraction (no flash):
```bash
python3 scripts/extract_invoke_frames_from_log.py \
  --log logs/pipeline/pipeline_xxx.log \
  --output-dir logs/flow_frames/latest \
  --max-frames 5
```

Output: `logs/flow_frames/latest/frame_001.png`, `frame_002.png`, ...

After extraction and reporting, refresh snapshot:

```bash
bash scripts/build_context_snapshot.sh
```

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
