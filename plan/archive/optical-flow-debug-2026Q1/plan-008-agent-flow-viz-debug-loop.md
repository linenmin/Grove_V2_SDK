> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 008：Agent 可见的光流可视化调试闭环

## 1. 问题背景

当前调试流程依赖人工在 Windows 端用 Himax 页面观察 Preview，Agent 无法直接看到设备输出。每次改动需：

1. Agent 烧录
2. Discord 通知用户
3. 用户切到 Windows、detach 串口、打开 Himax、观察
4. 用户反馈给 Agent

**瓶颈**：Agent 无法自主验证可视化效果，调试迭代慢。

---

## 2. 目标

建立「Agent 可见」的调试循环：Agent 在 WSL 完成烧录 + 抓串口后，能**自动解析出 flow 帧图像**并落盘，Agent 可直接读取图像做判断。

---

## 3. 数据流现状

| 环节 | 说明 |
|------|------|
| 设备 | 通过 UART 发送 JSON：`{"type":1,"name":"INVOKE","data":{"image":"<base64_jpeg>",...}}` |
| serReadLoop | 抓取 UART 原始字节，写入 `logs/pipeline/*.log` |
| Himax 页面 | 通过 Web Serial 连接串口，解析 JSON，解码 base64，渲染到 Preview |

**关键结论**：
- UART 数据已包含完整 JPEG（base64 编码在 JSON 的 `image` 字段）
- pipeline 日志已落盘
- Agent 可通过解析 log 中的 INVOKE JSON 并解码 base64 得到 JPEG 文件

---

## 4. 实现方案

### 4.1 新增脚本：`scripts/extract_invoke_frames_from_log.py`

**功能**：
- 输入：`logs/pipeline/pipeline_*.log`
- 解析：按行或按块扫描，识别 `"name":"INVOKE"` 的 JSON，提取 `data.image` 的 base64
- 输出：解码为 PNG，保存到 `logs/flow_frames/<timestamp>/`

**用法**：
```bash
python3 scripts/extract_invoke_frames_from_log.py \
  --log logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_xxx.log \
  --output-dir logs/flow_frames/latest \
  --max-frames 10
```

### 4.2 集成到 Pipeline

在 `run_optical_pipeline.sh` 新增参数：
- `--extract-frames`：调用 `extract_invoke_frames_from_log.py`
- `--max-frames`：最大提取帧数

### 4.3 Agent 调试循环

1. Agent 修改代码
2. 执行 `run_optical_pipeline.sh ... --extract-frames --max-frames 5`
3. 读取 `logs/flow_frames/latest/frame_001.png` 等
4. Agent 根据图像内容决定下一步

---

## 5. 实现要点

### 5.1 JSON 解析

- UART 输出可能分段：一个 JSON 可能跨多行，或一行多个 JSON
- 策略：按 `\r`/`\n` 分块，或用正则匹配 `"name":"INVOKE"` 起止，再提取 `"image":"..."`
- base64 可能含换行，需去除后再解码

### 5.2 日志格式

- serReadLoop 写入的是原始字节，decode 为 utf-8（errors=ignore）
- JSON 与调试打印（如 `[loop=...]`）混合，需能容错解析

### 5.3 输出格式

- 保存为 PNG 便于 Agent 读取
- 命名：`frame_001.png`, `frame_002.png`，或带时间戳

---

## 6. 验收标准

- [x] 给定 pipeline log，能提取至少 1 帧有效 PNG
- [x] Agent 能通过 Read 工具读取提取出的图像
- [x] 一次 pipeline 执行后，Agent 可根据图像内容做出判断

---

## 7. 实施记录（已完成）

### 7.1 已实现

- **extract_invoke_frames_from_log.py**：从 pipeline log 解析 INVOKE JSON，提取 base64 图像，保存为 PNG
- **FORCE_VIZ_CAMERA_JPEG**：`make VIZ_CAMERA=1` 时强制发送摄像头画面（非光流）
- **run_optical_pipeline.sh**：新增 `--viz-camera`、`--extract-frames`、`--max-frames`

### 7.2 验证命令（摄像头画面 + 提取）

```bash
bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode nomodel \
  --app-type optical_cam_oflow \
  --viz-camera \
  --extract-frames \
  --max-frames 5 \
  --capture-seconds 20 \
  --keyword "initial done" \
  --keyword '"name": "INVOKE"'
```

### 7.3 验证结果

- 提取 5 帧至 `logs/flow_frames/latest/`
- Agent 读取 frame_002.png 可分辨出摄像头画面（暗光、人脸/头部轮廓）
- **调试闭环已打通**：Agent 可自主烧录 → 抓 log → 提取帧 → 读取图像验证

---

## 8. 使用注意事项

- `--viz-camera` 与默认 flow 模式切换时，**不要使用 `--no-clean`**。否则可能复用旧目标文件，导致 `FORCE_VIZ_CAMERA_JPEG` 残留，INVOKE 仍发摄像头图（常见表现：`resolution=[320,240]`）。
- 若要确认当前确实在 flow 渲染分支，优先看 keylog 的 `resolution`：
  - flow 分支通常是模型输出尺寸（当前为 `208x160`）
  - camera 分支通常是传感器分辨率（如 `320x240`）

