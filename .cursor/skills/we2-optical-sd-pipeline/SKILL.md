---
name: we2-optical-sd-pipeline
description: Runs the Grove Vision AI V2 optical_sd firmware iteration pipeline in WSL2 with dual flash modes (nomodel and with-model), including build, image generation, xmodem flash, UART keyword verification, and USB re-attach guidance. Use when user mentions optical_sd, cvapp_yolov8n_ob.cpp, flash_img_opticalSD, flash_img_opticalSD_noModel, xmodem, usbipd, COM3, 不烧模型, 烧录模型, 或 pipeline 打通.
---

# WE2 Optical SD Pipeline

## Purpose

Provide a repeatable workflow for:

1. Build firmware (`APP_TYPE` configurable)
2. Generate `output.img`
3. Flash via xmodem (`nomodel` or `with-model`)
4. Capture UART and verify keywords

This skill is optimized for the current WSL2 + usbipd workflow.

## Trigger Keywords

Use this skill when requests include terms such as:

- `optical_sd`
- `cvapp_yolov8n_ob.cpp`
- `flash_img_opticalSD`
- `flash_img_opticalSD_noModel`
- `xmodem`
- `usbipd attach`
- `COM3`
- `不烧模型`
- `烧录模型`
- `pipeline`

## Execution Steps

1. Confirm serial access:
   - Check `/dev/ttyACM0` exists.
   - If missing, ask user to run:
     - `usbipd list`
     - `usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>`
2. Choose flash mode:
   - `nomodel`: for code-only updates (fast path)
   - `with-model`: when model file or flash address changed
3. Run pipeline script:
   - `./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh ...`
4. Report:
   - Flash success/failure
   - Keyword verification result
   - Log path

## Commands

### Fast Path (no model flash)

```bash
./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode nomodel \
  --app-type optical_sd \
  --port /dev/ttyACM0 \
  --keyword "initial done"
```

### Full Path (with model flash)

```bash
./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode with-model \
  --app-type optical_sd \
  --port /dev/ttyACM0 \
  --model-arg "model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite 0xB7B000 0x00000" \
  --keyword "initial done"
```

## Notes

- Prefer `nomodel` when model is unchanged; transfer time is much shorter.
- If serial open fails, do not bypass; debug USB mapping first.
- Do not assume Windows `powershell.exe/cmd.exe` interop works from WSL in this environment.
- Detailed operations and troubleshooting: see [reference.md](reference.md).
