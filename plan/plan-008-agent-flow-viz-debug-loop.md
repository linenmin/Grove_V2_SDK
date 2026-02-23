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

## 3. 可行性分析

### 3.1 数据流现状

| 环节 | 说明 |
|------|------|
| 设备 | 通过 UART 发送 JSON：`{"type":1,"name":"INVOKE","data":{"image":"<base64_jpeg>",...}}` |
| serReadLoop | 抓取 UART 原始字节，写入 `logs/pipeline/*.log` |
| Himax 页面 | 通过 Web Serial 连接串口，解析 JSON，解码 base64，渲染到 Preview |

### 3.2 关键结论

- **UART 数据已包含完整 JPEG**：base64 编码在 JSON 的 `image` 字段
- **pipeline 日志已落盘**：`run_optical_pipeline.sh` 的 `--log-file` 即原始 UART 流
- **Agent 可读**：若能从 log 中解析出 INVOKE JSON 并解码 base64，即可得到 JPEG 文件

---

## 4. 方案：从 Pipeline 日志提取 Flow 帧

### 4.1 新增脚本：`scripts/extract_flow_frames_from_log.py`

**功能**：
- 输入：`logs/pipeline/pipeline_*.log`
- 解析：按行或按块扫描，识别 `"name":"INVOKE"` 的 JSON，提取 `data.image` 的 base64
- 输出：解码为 JPEG/PNG，保存到 `logs/pipeline/frames/` 或 `logs/flow_frames/<timestamp>/`

**用法**：
```bash
python3 scripts/extract_flow_frames_from_log.py \
  --log logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_xxx.log \
  --output-dir logs/flow_frames/latest \
  --max-frames 10
```

### 4.2 集成到 Pipeline

在 `run_optical_pipeline.sh` 末尾（或作为可选步骤）：
- 若 `--extract-frames` 或 `--save-frames`，则调用 `extract_flow_frames_from_log.py`
- 输出路径写入环境变量或固定路径，供 Agent 读取

### 4.3 Agent 调试循环

1. Agent 修改代码（如 flow_render 的 kFixedScale）
2. 执行 `run_optical_pipeline.sh ... --extract-frames --max-frames 5`
3. 读取 `logs/flow_frames/latest/frame_001.png` 等
4. Agent 根据图像内容（纯白 / 条纹 / 有对比度）决定下一步

---

## 5. 实现要点

### 5.1 JSON 解析

- UART 输出可能**分段**：一个 JSON 可能跨多行，或一行多个 JSON
- 策略：按 `\r`/`\n` 分块，或用正则匹配 `"name":"INVOKE"` 起止，再提取 `"image":"..."`
- base64 可能含换行，需去除后再解码

### 5.2 日志格式

- serReadLoop 写入的是**原始字节**，decode 为 utf-8（errors=ignore）
- JSON 与调试打印（如 `[loop=...]`）混合，需能容错解析

### 5.3 输出格式

- 保存为 PNG 便于 Agent 读取（JPEG 亦可，Agent 支持）
- 命名：`frame_001.png`, `frame_002.png`，或带时间戳

---

## 6. 备选方案（若主方案不可行）

| 方案 | 说明 | 复杂度 |
|------|------|--------|
| A. 日志解析 | 上述主方案 | 低 |
| B. 设备端写 SD 卡 | 若 WE2 有 SD，可把 JPEG 写入 SD，WSL 挂载读取 | 中，需改固件 |
| C. 设备端发简化协议 | 如 raw 灰度 + 尺寸头，减少解析难度 | 中，需改协议 |
| D. Windows 自动化 | Puppeteer/Playwright 控制 Himax 页面截图 | 高，依赖 Windows 环境 |

---

## 7. 建议实施顺序

1. **实现 `extract_flow_frames_from_log.py`**（独立脚本，可先手动测试）
2. **验证**：用已有 pipeline log 跑一次，确认能提取出有效图像
3. **集成到 pipeline**：可选 `--extract-frames`
4. **更新 Skill**：在 `we2-optical-sd-pipeline` 或 `we2-himax-iterative-debug` 中增加「提取帧供 Agent 分析」的说明

---

## 8. 验收标准

- [ ] 给定 pipeline log，能提取至少 1 帧有效 JPEG/PNG
- [ ] Agent 能通过 `Read` 工具读取提取出的图像
- [ ] 一次 pipeline 执行后，Agent 可根据图像内容做出「继续调 kFixedScale / 换模型 / 其他」的判断，无需人工反馈
