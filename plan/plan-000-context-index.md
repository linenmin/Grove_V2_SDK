# Plan 000：上下文索引（单入口）

## 1. 目的

减少调试会话中的重复读文件开销；新会话默认只读本文件 + 最新快照 + 最新调试 plan。

## 2. 会话加载顺序（强约束）

1. 先读：`plan/plan-000-context-index.md`
2. 再读：`logs/context/context_snapshot_latest.md`
3. 最后读：`plan/plan-009-flow-visualization-agent-debug.md`（仅最新增量区段）
4. 仅在有证据缺口时，按需回看 `plan-008` 及更早计划

## 3. 当前活动上下文（2026-02-24）

- 当前主计划：`plan/plan-010-vela-input-channel-issue.md`
- 当前问题：Vela 编译后模型输入通道映射问题， 导致第二帧未被有效消费
- 最新关键日志：`logs/pipeline/pipeline_with-model_optical_cam_oflow_20260224_122631.log`
- 最新关键摘要：`logs/pipeline/pipeline_with-model_optical_cam_oflow_20260224_122631.key.summary.txt`
- 最新帧输出目录：`logs/flow_frames/latest/`
- 快照生成脚本：`scripts/build_context_snapshot.sh`

## 4. 关键配置锚点（代码事实）

- `pipeline/cvapp_yolov8n_ob.cpp`
- `tensor_arena_size = 1408 * 1024`
- `FLOW_DBG_FREEZE_PAIR = 0`
- `FLOW_DBG_SYNTH_INJECT = 0`

- `viz/flow_render.cpp`
- `FLOW_VIZ_FIXED_SCALE = 0`
- `FLOW_VIZ_TEST_PATTERN = 0`
- `FLOW_VIZ_LIGHT_SMOOTH = 1`
- `FLOW_VIZ_REMOVE_ROW_BIAS = 1`

- `io/camera/cam_input.cpp`
- `CAM_INPUT_USE_BGR = 1`
- `CAM_INPUT_USE_HELIUM_RESIZE = 1`

- `config/common_config.h`
- `YOLOV8_OBJECT_DETECTION_FLASH_ADDR = 0x3AB7B000`

## 5. 标准命令（先快照，后调试）

```bash
# 1) 生成上下文快照（每轮调试后执行）
bash scripts/build_context_snapshot.sh

# 2) 常用抓取（不改固件，直抓 UART）
python3 xmodem/serReadLoop.py \
  --port /dev/ttyACM0 \
  --baudrate 921600 \
  --timeout 1 \
  --duration 18 \
  --log-file logs/pipeline/pipeline_capture_manual.log \
  --keyword "initial done" \
  --keyword '"name": "INVOKE"'

# 3) 从日志提取 INVOKE 图像
python3 scripts/extract_invoke_frames_from_log.py \
  --log logs/pipeline/pipeline_capture_manual.log \
  --output-dir logs/flow_frames/latest \
  --max-frames 10
```

## 6. 更新协议（增量）

1. 每次实测后，只在最新 plan 追加一条 `Rxx`，不回写历史 R 段。
2. 每次实测后，执行 `scripts/build_context_snapshot.sh` 刷新 `logs/context/context_snapshot_latest.md`。
3. 本文件只维护“入口/指针/流程”，不记录细粒度实验细节。
4. 若“当前主计划”切换（例如 plan-010），只改本文件第 3 节指针与日期。

## 7. 触发条件（何时回读旧计划）

- 仅当出现以下任一情况才回读 `plan-001~008`：
- 当前日志与历史结论矛盾
- 关键宏与历史记录不一致且无法定位
- 需要追溯某个 patch 的首次引入原因

