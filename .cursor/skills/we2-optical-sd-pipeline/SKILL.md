---
name: we2-optical-sd-pipeline
description: Runs the Grove Vision AI V2 optical_sd/optical_cam_oflow firmware iteration pipeline in WSL2 with dual flash modes (nomodel and with-model), including build, image generation, xmodem flash, UART keyword verification, and USB re-attach guidance. Supports agent-visible visualization: --viz-camera + --extract-frames to extract INVOKE images for agent to read. Use when user mentions optical_sd, optical_cam_oflow, cvapp_yolov8n_ob.cpp, flash_img_opticalSD, xmodem, usbipd, agent 可见, 可视化调试, extract-frames, pipeline.
---

# WE2 Optical SD Pipeline

## Purpose

Provide a repeatable workflow for:

1. Build firmware (`APP_TYPE` configurable)
2. Generate `output.img`
3. Flash via xmodem (`nomodel` or `with-model`)
4. Capture UART and verify keywords
5. Refresh debug context snapshot for next turn

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
- `agent 可见` / `可视化调试` / `extract-frames`

## Context Discipline (Mandatory)

Before deep analysis, read in this order:

1. `plan/plan-000-context-index.md`
2. `logs/context/context_snapshot_latest.md`
3. Latest debug plan file from the snapshot

Do not default to full reads across all `plan/` files.

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
5. Refresh context snapshot:
   - `bash scripts/build_context_snapshot.sh`

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

## Agent Visible Visualization Debugging (plan-008)

当 Agent 需要**自主验证**设备可视化输出（摄像头画面或光流图）时，使用以下流程，无需人工在 Windows Himax 页面观察：

1. **烧录 + 抓 log + 提取帧**（一条命令）：
   ```bash
   ./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
     --mode nomodel \
     --app-type optical_cam_oflow \
     --viz-camera \
     --extract-frames \
     --max-frames 5 \
     --capture-seconds 20 \
     --keyword "initial done" \
     --keyword '"name": "INVOKE"'
   ```

2. **参数说明**：
   - `--viz-camera`：强制发送摄像头画面（非光流），用于验证 pipeline 与提取链路
   - `--extract-frames`：抓取后自动从 log 解析 INVOKE 的 base64 图像，保存为 PNG
   - `--max-frames N`：最多提取帧数，默认 10

3. **输出位置**：`logs/flow_frames/latest/frame_001.png` 等

4. **Agent 验证**：用 `Read` 工具读取 `logs/flow_frames/latest/frame_002.png`，根据图像内容（人脸、光流、条纹、纯白等）判断下一步

5. **仅从已有 log 提取**（不烧录）：
   ```bash
   python3 scripts/extract_invoke_frames_from_log.py \
     --log logs/pipeline/pipeline_xxx.log \
     --output-dir logs/flow_frames/latest \
     --max-frames 5
   ```

6. **更新会话快照**（进入下一轮前）：
   ```bash
   bash scripts/build_context_snapshot.sh
   ```

## Notes

- Prefer `nomodel` when model is unchanged; transfer time is much shorter.
- If serial open fails, do not bypass; debug USB mapping first.
- Do not assume Windows `powershell.exe/cmd.exe` interop works from WSL in this environment.
- For agent-visible flow: use `--viz-camera` to get camera frames first; then switch to flow output for optical flow debugging.
- Always refresh `context_snapshot_latest` after each pipeline run before writing the next debug entry.
- Detailed operations and troubleshooting: see [reference.md](reference.md).
