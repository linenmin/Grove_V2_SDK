# Plan 006：optical_cam_oflow 光流可视化输出执行计划

## 1. 当前状态（2026-02-23）

- **with-model 烧录链路已打通**：
  - 模型：`/mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/model/sram_test_modified_vela.tflite`
  - 最近一次验证：`logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_002248.log`
  - 关键字：`initial done` / `NAME?` / `MODEL?` / `INVOKE` 均命中。
- **Windows Himax 页面仍是普通摄像头画面**：
  - 不是期望的光流估计可视化图。
  - 主观帧率低于 2 FPS。
  - 画面偏暗。
- **当前结论**：这不是"模型没烧进去"的问题，而是"可视化发布内容仍以 camera JPEG 为主，尚未切到 flow tensor 渲染图"的问题。

---

## 2. 问题定位假设

1. `INVOKE` 虽然持续发送，但 `image` 字段承载的是相机帧 JPEG，而不是模型输出 `out_data` 的渲染结果。
2. 低帧率主要由模型推理耗时 + UART/JPEG发送节流共同决定。
3. 偏暗主要来自 sensor 曝光/增益与当前场景光照。

---

## 3. 执行阶段规划（建议顺序：C → D → B）

**依赖关系**：
- 阶段 B 的 `flow_viz_jpeg` 模式需要 D 的渲染路径才能产出真实光流图
- 阶段 C 的 flow tensor 统计是 D 渲染的前提

### 阶段 A：性能与亮度优化（前置）

- 先保持 `camera_jpeg` 输出，完成 sensor 可视基线确认
- 调整 `INVOKE` 发送节流（例如每 2/3 帧发送一次）
- 检查 sensor 曝光/增益配置

### 阶段 C：确认 flow tensor 在运行时有效

- 在 app 内增加低频统计：`out_data` 的 min/max/mean、非零比例、绝对值分布
- 验收：串口能看到稳定非零的 flow 统计

### 阶段 D：实现 flow -> 可视化图的渲染路径

- 在 `optical_cam_oflow` 增加渲染函数：
  - 输入：模型输出 `H x W x 2`（dx, dy）
  - 输出：`RGB888` 伪彩色图（建议 HSV: angle->Hue, magnitude->Value）
- 将渲染图编码成 JPEG 后走现有 `viz_uart_send_invoke_jpeg()` 通道
- 验收：Himax 页面画面从"普通相机"切换为"随运动变化的光流伪彩图"

### 阶段 B：保留调试开关与回退

- 增加可切换发布源：`camera_jpeg` / `flow_viz_jpeg` / `fallback_test_pattern`
- 验收：三种模式可独立验证

---

## 4. 阶段 A 执行记录

### A1：低频性能/亮度观测基线

**改动**：VIZ_UART_MODE 下 g_ctx.log_print_interval 从 1000000 调整为 20，启用低频 [loop] 统计

**结果**：
- 30.30s 内 INVOKE=47 -> 约 1.55 FPS（网页可见帧率）
- loop 统计：total_ms_avg=202.924，infer_ms_avg=163.619；推理主循环约 4.0 loop/s
- 亮度统计（in2）: 平均约 12.63/255，整体偏暗

**日志**：`logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_104837.log`

### A2：INVOKE 节流改为每帧发送

**改动**：publish_viz_payload() 将 need_uart_invoke 从 loop_cnt%3==0 改为每帧发送

**结果**：
- invoke_count=126（30.14s），可见发送帧率约 4.18 FPS，较 A1 的 1.55 FPS 明显提升
- infer_ms_avg=163.620，total_ms_avg=202.926（推理主环路基本不变）

**日志**：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260223_105258.log`

**结论**：低帧率主因是发送节流配置，不是NPU推理耗时

### A3：OV5647 AE 目标窗口提升

**改动**：在 cis_ov5647/cisdp_sensor.c 新增 OV5647_ae_boost_setting（3a0f/10/1b/1e/11/1f）

**结果**：
- 亮度统计（in2）由 A2 约 13.32/255 上升到约 86.04/255；in2 max 范围提升到 197..255
- infer_ms_avg=163.619，total_ms_avg=202.922（性能无明显回退）

**日志**：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260223_105536.log`

### A5：FPS 瓶颈定量分析

**基于 A3 日志做阶段耗时拆分**：
- sd/capture=34.585ms
- preproc=4.718ms
- infer=163.619ms
- total=202.922ms

**结论**：当前瓶颈不在 sensor 原始帧率；capture 阶段 34.6ms 对应约 28.9 FPS 上限，远高于端到端 4~5 FPS

**algo_tick 修复**：将 UART INVOKE 的 algo_tick 从 microsecond 值改为 400MHz cycle 值（algo_tick_cycles=total_us*400）

---

## 5. 阶段 D 执行记录

### D1：flow -> 灰度图 -> JPEG -> INVOKE

**代码改动**：
- 新增 `viz/flow_render.cpp`：flow_render_to_gray（magnitude -> 亮度）、flow_render_gray_to_jpeg
- 新增 `viz/flow_render.h`
- `tflm_yolov8_od.mk` 增加 JPEGENC 库
- `publish_viz_payload` 增加 flow 参数；有 flow 时优先渲染 flow -> JPEG -> viz_uart_send_invoke_jpeg
- 静态缓冲：g_flow_viz_gray[256*256]、g_flow_viz_jpeg[32KB]

**用户反馈**：
- FPS 4.93 正常
- 画面：一条条黑白纹路，有轻微动态，与拍摄画面无关
- boxes: [] 持续输出

**分析**：
- 「与拍摄画面无关」是预期行为：光流图含义是亮=运动大，暗=运动小
- 黑白条纹可能成因：
  - A. JPEG 块状伪影（使用 JPEG_Q_LOW）
  - B. 静态场景下 flow 接近零，归一化后多为噪声
  - C. 模型输出的 int8 量化可能产生行列相关结构

**后续改进**：
- 改用 `JPEG_Q_HIGH` 或 `JPEG_Q_BEST` 减轻块状伪影
- 在明显运动场景下复测，确认亮区与运动区域对应

---

## 6. 条纹问题破案（转 plan-007）

**用户反馈**：挥手时黑线随挥手频率闪动，延迟低 → flow 对运动有响应，链路正常

**结论**：条纹来自模型输出本身，非 C++ 端和 Web 端渲染错误

**根因（详见 plan-007）**：
- `run_sram_test.py` 的量化校准使用了 `np.random.uniform(0.0, 1.0)` 白噪声，且范围错误（应为 0~255）
- 导致 `MultiScaleResNet` 内高低分辨率特征残差图的值域剧烈偏移
- 形成了"8 像素周期循环"的强垂直条纹

**修复**：采用真实 Sintel 图片进行 0~255 校准后，棋盘格完全消除

---

## 7. 调试记录策略

每次尝试严格执行：
1. 一个假设
2. 一次最小改动
3. 一次 pipeline 验证
4. 先写入 plan/history 再进行下一次

记录路由：
- 若预计 2-3 次内可解：直接增量写入最新 plan
- 若进入多轮深挖：新建独立 history markdown，完成后把关键成功尝试回写本文件

