---
name: we2-himax-iterative-debug
description: Iterative debugging playbook for Grove Vision AI V2 (WE2) and Himax AI Web Toolkit across WSL2 and Windows serial handoff. Use when tasks involve build/flash/UART verification, HTML preview issues, compact log extraction, and incremental debug history updates across changing plan markdown files.
---

# WE2 Himax Iterative Debug

## Purpose

Run reproducible WE2 debug cycles with small context cost:

1. One hypothesis per run
2. One pipeline execution per run
3. One compact evidence artifact per run
4. One incremental markdown update before the next run

## Mandatory Loop

Follow this loop in order. Do not skip step 5.

1. Define one hypothesis and exact expected signals.
2. Apply the smallest possible code/config change.
3. Run build/flash/capture pipeline once.
4. Extract compact evidence from the pipeline log.
5. Append incremental markdown history for this run.
6. Start the next attempt.

## Plan And History Routing

Use these routing rules to avoid binding to a specific plan file.

1. For short fixes likely solved in a few iterations, append history directly in the latest plan markdown.
2. For issues that require repeated iterations, create a dedicated debug history markdown and append every run there.
3. After final success, write only the key successful attempt(s) back into the latest plan markdown.
4. Keep old attempts in the dedicated history page; do not duplicate large logs in the latest plan.

Use scripts:

- `scripts/find_latest_plan.sh`
- `scripts/start_debug_history.sh`
- `scripts/append_plan_entry.sh`
- `scripts/set_tensor_arena_kb.sh`
- `scripts/get_model_slot.sh`
- `scripts/normalize_model_path.sh`
- `scripts/inspect_tflite_model.sh`

Detailed routing guidance: `references/plan-routing.md`.

## Compact Logging

Never paste full UART logs into conversation or plan files.

1. Run `scripts/extract_himax_keylog.sh --log <pipeline.log>`.
2. Use the generated key log and summary counts.
3. Store raw log path, not raw log body.
4. Redact base64 image blobs in extracted evidence.

Pattern sets and diagnostic mapping: `references/log-patterns.md`.

## Suggested Commands

Find latest plan:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/find_latest_plan.sh
```

Create dedicated history page:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/start_debug_history.sh \
  --topic "himax-jpeg-source" \
  --latest-plan plan/plan-006-current.md
```

Extract compact evidence from a pipeline log:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/extract_himax_keylog.sh \
  --log logs/pipeline/pipeline_nomodel_optical_cam_oflow_YYYYMMDD_HHMMSS.log
```

Append one run entry:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/append_plan_entry.sh \
  --target plan/plan-006-current.md \
  --title "3.4 Stage Four: fallback INVOKE validation" \
  --goal "Verify HTML preview update with fallback JPEG path." \
  --change "Added fallback JPEG payload when real JPEG is invalid." \
  --verify "bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode nomodel --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 18 --keyword '\"name\": \"INVOKE\"' --no-clean" \
  --result "Hit INVOKE/NAME?/MODEL?." \
  --result "Real JPEG still invalid: cisdp size remains 0." \
  --conclusion "HTML path is validated; continue with real JPEG source recovery." \
  --run-id R20260222_223357 \
  --log logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_223357.log
```

Set tensor arena quickly:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/set_tensor_arena_kb.sh \
  --app optical_cam_oflow --kb 1700
```

Get optical_cam_oflow model slot (flash absolute + xmodem slot):

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/get_model_slot.sh \
  --app optical_cam_oflow \
  --model-file model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite
```

List all model flash macros in an app config (recommended for generic app work):

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/get_model_slot.sh \
  --app tflm_fd_fm \
  --list-macros
```

Normalize Windows/WSL model path:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/normalize_model_path.sh \
  --path "D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\deployment\model\sram_test_modified_vela.tflite"
```

Inspect model I/O + op histogram:

```bash
./.cursor/skills/we2-himax-iterative-debug/scripts/inspect_tflite_model.sh \
  --model "/mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/model/sram_test_modified_vela.tflite"
```

## Generic Model Import Workflow (Any App)

Use this workflow for any WE2 scenario app, not only `optical_cam_oflow`.

1. Resolve model path for WSL:
   - `normalize_model_path.sh --path <linux-or-windows-model-path>`
2. Collect model facts before flashing:
   - `inspect_tflite_model.sh --model <resolved_path>`
   - If interpreter backend is missing, keep using Vela report and record model `size/hash/path` manually.
3. Select the correct flash slot macro:
   - `get_model_slot.sh --app <scenario_app_name> --list-macros`
   - If multiple macros exist, always pass `--macro <MACRO_NAME>`.
4. Build model arg from the selected slot:
   - `get_model_slot.sh --app <scenario_app_name> --macro <MACRO_NAME> --model-file <resolved_path>`
   - Use output `model_arg="<model> <slot> 0x00000"` directly in pipeline script.
5. Run one with-model flash/capture cycle:
   - `run_optical_pipeline.sh --mode with-model --app-type <scenario_app_name> --model-arg "<...>" ...`
6. Write one incremental plan/history entry before the next trial.

Guardrails:

- Never assume the first `*FLASH_ADDR` macro is correct when multiple macros exist.
- Model import success does not prove visualization path correctness; verify `INVOKE` content source separately.
- Keep model swap and app code changes in separate runs so failures are attributable.

## Arena Sweep Rule

For tensor arena tuning:

1. Change only `tensor_arena_size` in one run.
2. Keep app mode `nomodel` until pipeline is stable.
3. Pass criteria for each run:
   - `[SUMMARY] all_keywords_hit`
   - `invoke_count > 0`
   - `jpeg_skip_count = 0`
4. Write one plan entry after each run before starting next run.
5. After finding the ceiling, keep at least 16KB safety margin for with-model tests.

## Windows Interaction Rule

When a run requires user action in Windows:

1. Stop after writing the current run entry.
2. Notify user through `discord-notify-wsl2` only if requested.
3. Resume only after receiving explicit Windows-side feedback.
