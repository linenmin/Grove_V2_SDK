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
- **当前结论**：
  - 这不是“模型没烧进去”的问题，而是“可视化发布内容仍以 camera JPEG 为主，尚未切到 flow tensor 渲染图”的问题。

---

## 2. 问题定位假设

1. `INVOKE` 虽然持续发送，但 `image` 字段承载的是相机帧 JPEG，而不是模型输出 `out_data` 的渲染结果。
2. 低帧率主要由模型推理耗时 + UART/JPEG发送节流共同决定（不是网页显示层单点问题）。
3. 偏暗主要来自 sensor 曝光/增益与当前场景光照，不是 flow 算法本身。

---

## 3. 下一步执行计划

### 3.1 阶段 A：性能与亮度优化（前置）

- 先保持 `camera_jpeg` 输出，完成 sensor 可视基线确认：
  - 调整 `INVOKE` 发送节流（例如每 2/3 帧发送一次）并记录 `algo_tick` 与端到端帧率。
  - 优先检查 sensor 曝光/增益配置；必要时新增可调参数。
- 验收标准：
  - 在 Windows 侧普通摄像头预览下，亮度与可见细节达到可用。
  - 帧率提升到可接受范围后再切换到光流可视化。

### 3.2 阶段 B/C/D 顺序分析与建议

**当前 plan 顺序**：B → C → D

**依赖关系**：
- 阶段 B 的 `flow_viz_jpeg` 模式需要 D 的渲染路径才能产出真实光流图；否则无法“独立验证”flow 模式。
- 阶段 C 的 flow tensor 统计是 D 渲染的前提：若 out_data 全零，D 的渲染结果无意义，且无法区分是“模型问题”还是“渲染问题”。
- 阶段 B 的验收“三种模式可独立验证”要求 flow_viz_jpeg 有实际内容，故 B 依赖 D。

**建议顺序**：**C → D → B**

1. **C 先**：确认 flow tensor 在运行时有效（min/max/mean/ nonzero ratio）
   - 若 C 发现全零/常量，可先修模型/推理，再推进 D。
   - 串口已有 `ob_log_infer_line` 输出 out0/out1 min/max，可先人工确认是否非零；若不足，再增加低频统计。
2. **D 再**：实现 flow -> 渲染图 -> JPEG 路径
   - 在 `viz/` 或 `pipeline/` 增加 `flow_to_rgb888()` 或伪彩渲染，再 `viz_uart_send_invoke_jpeg()`。
3. **B 最后**：增加模式切换（camera_jpeg / flow_viz_jpeg / fallback_test_pattern）
   - 在 D 的基础上接入模式切换，便于调试时快速排除“模型输出 vs 渲染/发送”问题。

### 3.3 阶段 C：确认 flow tensor 在运行时有效

- 在 app 内增加低频统计（每 N 帧）：
  - `out_data` 的 min/max/mean
  - 非零比例
  - 绝对值分布（用于判定是否全零或极小值）
- 验收标准：
  - 串口能看到稳定非零的 flow 统计，不是全零/常量。

### 3.4 阶段 D：实现 flow -> 可视化图 的渲染路径

- 在 `optical_cam_oflow` 增加渲染函数：
  - 输入：模型输出 `H x W x 2`（dx, dy）
  - 输出：`RGB888` 伪彩色图（建议 HSV: angle->Hue, magnitude->Value）
- 将渲染图编码成 JPEG 后走现有 `viz_uart_send_invoke_jpeg()` 通道。
- 验收标准：
  - Himax 页面画面从“普通相机”切换为“随运动变化的光流伪彩图”。

### 3.5 阶段 B：保留调试开关与回退

- 增加可切换发布源：
  - `camera_jpeg`
  - `flow_viz_jpeg`
  - `fallback_test_pattern`
- 验收标准：
  - 三种模式可独立验证，便于快速排除是“模型输出问题”还是“渲染/发送问题”。

---

## 4. 调试记录策略（本轮起执行）

- 每次尝试严格执行：
  1. 一个假设
  2. 一次最小改动
  3. 一次 pipeline 验证
  4. 先写入 plan/history 再进行下一次
- 记录路由：
  - 若预计 2-3 次内可解：直接增量写入最新 plan（即本文件）。
  - 若进入多轮深挖：新建独立 history markdown，完成后把关键成功尝试回写本文件。

---

## 5. 当前优先级

1. **先在 `camera_jpeg` 模式完成性能与亮度基线**（避免切换光流后失去 sensor 观测窗口）。
2. 再按 **C → D → B** 顺序完成 flow 可视化：
   - C：确认 flow tensor 有效
   - D：实现 flow -> 渲染图 -> JPEG 路径
   - B：增加模式切换便于调试

### 6.1 阶段A-尝试1：开启低频性能/亮度观测基线

- **Goal**
  - 在不改变功能行为的前提下获取阶段A量化基线（FPS、algo_tick、亮度）

- **Changes**
  - 将 VIZ_UART_MODE 下 g_ctx.log_print_interval 从 1000000 调整为 20，启用低频 [loop] 统计

- **Verification Command**
  - `bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode with-model --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 30 --keyword 'initial done' --keyword '"name": "INVOKE"' --keyword '"name": "MODEL?"' --keyword '"name": "NAME?"' --keyword '[loop=' --model-arg '/mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/model/sram_test_modified_vela.tflite 0xB7B000 0x00000' --no-clean`

- **Key Output**
  - SUMMARY all_keywords_hit；invoke_count=47，camera_frame_capture_fail_count=0，jpeg_skip_count=0
  - 30.30s 内 INVOKE=47 -> 约 1.55 FPS（网页可见帧率）
  - loop 统计显示 total_ms_avg=202.924，infer_ms_avg=163.619；推理主循环约 4.0 loop/s
  - 亮度统计（in2）: 平均约 12.63/255，min 范围 1..9，max 范围 16..21，整体偏暗

- **Conclusion**
  - 阶段A基线已量化：低帧率主因是 UART INVOKE 当前每3帧发送一次（而非推理算力极限）；下一轮先只调整 INVOKE 发送步长验证 FPS 提升，再单独处理 sensor 曝光。

- **Run ID**
  - `R20260223_104837_A1`

- **Log Path**
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_104837.log`

---

### 6.2 阶段A-尝试2：INVOKE节流改为每帧发送

- **Goal**
  - 验证网页可见FPS瓶颈是否由每3帧发送一次INVOKE导致

- **Changes**
  - publish_viz_payload() 将 need_uart_invoke 从 loop_cnt%3==0 改为 send_uart 即每帧发送

- **Verification Command**
  - `bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode nomodel --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 30 --keyword 'initial done' --keyword '"name": "INVOKE"' --keyword '"name": "MODEL?"' --keyword '"name": "NAME?"' --keyword '[loop=' --no-clean`

- **Key Output**
  - SUMMARY all_keywords_hit；invoke_count=126（30.14s），camera_frame_capture_fail_count=0
  - 可见发送帧率约 4.18 FPS（126/30.14），较尝试1的 1.55 FPS（47/30.30）明显提升
  - infer_ms_avg=163.620，total_ms_avg=202.926（推理主环路基本不变）

- **Conclusion**
  - 阶段A性能子目标成立：低帧率主因是发送节流配置，不是NPU推理耗时；保留每帧发送作为当前基线。

- **Run ID**
  - `R20260223_105258_A2`

- **Log Path**
  - `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260223_105258.log`

---

### 6.3 阶段A-尝试3：OV5647 AE目标窗口提升

- **Goal**
  - 在保持自动AEC/AGC的前提下提升暗场亮度

- **Changes**
  - 在 cis_ov5647/cisdp_sensor.c 新增 OV5647_ae_boost_setting（3a0f/10/1b/1e/11/1f）并在 cisdp_sensor_init() 应用

- **Verification Command**
  - `bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode nomodel --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 30 --keyword 'initial done' --keyword '"name": "INVOKE"' --keyword '"name": "MODEL?"' --keyword '"name": "NAME?"' --keyword '[loop=' --no-clean`

- **Key Output**
  - SUMMARY all_keywords_hit；invoke_count=116（30.01s），camera_frame_capture_fail_count=0
  - 亮度统计（in2）由尝试2约 13.32/255 上升到约 86.04/255；in2 max 范围提升到 197..255
  - infer_ms_avg=163.619，total_ms_avg=202.922（性能无明显回退）

- **Conclusion**
  - 阶段A亮度调优在设备侧统计上显著生效；下一步需Windows/Himax人工观感确认（是否过曝、色偏、细节丢失）。

- **Run ID**
  - `R20260223_105536_A3`

- **Log Path**
  - `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260223_105536.log`

---

### 6.4 阶段A-尝试4：Windows观感确认请求（A2+A3后）

- **Goal**
  - 在继续切换flow可视化前，先完成人工观感验收（FPS与曝光）

- **Changes**
  - 基于尝试2保留每帧INVOKE发送，基于尝试3保留OV5647 AE boost配置
  - 按用户要求通过discord-notify-wsl2发送Windows侧操作通知

- **Verification Command**
  - `bash -i -c 'curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST -H "Content-Type: application/json" --data-binary @/tmp/temp_discord_msg.json "$DISCORD_WEBHOOK_URL"'`

- **Key Output**
  - Discord webhook返回 HTTP_CODE:204，通知发送成功
  - 当前状态进入人工验收等待：需Windows/Himax页面确认FPS和亮度主观表现

- **Conclusion**
  - 阶段A设备侧指标已满足前置条件，等待用户Windows观感反馈后再推进下一次改动。

- **Run ID**
  - `R20260223_StageA_A4_WinCheck`

---

### 6.5 阶段A-尝试5：Windows反馈回写 + FPS瓶颈定量分析

- **Goal**
  - 基于Windows人工观感确认阶段A效果，并定位当前4~5FPS瓶颈是否来自sensor

- **Changes**
  - 回写用户反馈：亮度明显提升；主观帧率约4FPS；网页左上角FPS显示约1971明显异常
  - 基于A3日志做阶段耗时拆分：sd/capture=34.585ms，preproc=4.718ms，infer=163.619ms，total=202.922ms
  - 在 publish_viz_payload() 中将 UART INVOKE 的 algo_tick 从 microsecond 值改为 400MHz cycle 值（algo_tick_cycles=total_us*400）以匹配Himax HTML计算口径

- **Verification Command**
  - `bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode nomodel --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 20 --keyword 'initial done' --keyword '"name": "INVOKE"' --keyword '"name": "MODEL?"' --keyword '"name": "NAME?"' --keyword '[loop=' --no-clean`

- **Key Output**
  - 用户Windows侧反馈：亮度改善、主观FPS约4，符合设备侧A2/A3统计趋势
  - 总耗时202.922ms对应理论上限约4.93FPS；其中infer占比约80.6%，sd/capture占比约17.0%，preproc占比约2.3%
  - 结论：当前瓶颈不在sensor原始帧率；若仅看capture阶段34.6ms，对应约28.9FPS上限，远高于端到端4~5FPS
  - 复测命令当前失败：/dev/ttyACM0 缺失（串口暂未挂载到WSL），待重新attach后验证新algo_tick显示

- **Conclusion**
  - 阶段A核心问题已收敛：亮度已提升、可见帧率已到4FPS量级；下一步优先验证algo_tick修复并继续通过降模型负载/降输入分辨率提升FPS。

- **Run ID**
  - `R20260223_StageA_A5_UserFeedback_Bottleneck`

- **Log Path**
  - `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260223_105536.log`

---

### 6.6 阶段 D-尝试1：flow -> 灰度图 -> JPEG -> INVOKE

- **Goal**
  - 实现光流 tensor 渲染为 JPEG 并通过 INVOKE 发送，Himax 页面显示光流图

- **Changes**
  - 新增 `viz/flow_render.cpp`：flow_render_to_gray（magnitude -> 亮度）、flow_render_gray_to_jpeg
  - 新增 `viz/flow_render.h`
  - `tflm_yolov8_od.mk` 增加 JPEGENC 库
  - `publish_viz_payload` 增加 flow 参数；有 flow 时优先渲染 flow -> JPEG -> viz_uart_send_invoke_jpeg
  - 静态缓冲：g_flow_viz_gray[256*256]、g_flow_viz_jpeg[32KB]

- **Verification Command**
  - `bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode with-model --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 30 --keyword 'initial done' --keyword '"name": "INVOKE"' --keyword '"name": "MODEL?"' --keyword '"name": "NAME?"' --keyword '[loop=' --model-arg '/mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/model/sram_test_modified_vela.tflite 0xB7B000 0x00000' --no-clean`

- **Key Output**
  - 待用户确认：Windows 侧 Himax 页面应显示光流灰度图（亮=运动），而非普通相机画面

- **Conclusion**
  - 阶段 D 实现完成；需 Windows 侧 detach 串口后打开 Himax 页面验证

- **Run ID**
  - `R20260223_StageD_D1_FlowViz`

---

### 6.7 阶段 D-尝试1 用户反馈：黑白条纹画面分析

- **Goal**
  - 分析 Windows Himax 页面显示「黑白条纹、轻微动态、与拍摄画面无关」的原因

- **用户反馈**
  - FPS 4.93 正常
  - 画面：一条条黑白纹路，有轻微动态，与拍摄画面无关
  - boxes: [] 持续输出

- **分析结论**

  1. **「与拍摄画面无关」是预期行为**
     - 阶段 D 已将 INVOKE 的 image 从 camera JPEG 切换为 **flow magnitude 灰度图**
     - 光流图含义：亮=运动大，暗=运动小；不是相机原图，故与拍摄内容无直接对应

  2. **黑白条纹的可能成因**
     - **A. JPEG 块状伪影**：当前使用 `JPEG_Q_LOW`，压缩强，8x8 DCT 块边界易显，形成条纹感
     - **B. 静态场景下的 flow 模式**：场景静止时 flow 接近零，归一化后多为噪声；模型输出的 int8 量化可能产生行列相关结构
     - **C. 轻微动态**：说明 flow 数据在更新，渲染链路基本正常

  3. **boxes: [] 的说明**
     - Himax 页面按 YOLOv8 OD 格式解析，期望 boxes 字段；当前发送的是 flow 图，无检测框，故 boxes 为空属正常

- **待验证假设**
  - 若条纹主要由 JPEG 质量引起：改用 `JPEG_Q_HIGH` 或 `JPEG_Q_MED` 可减轻块状伪影
  - 若条纹来自 flow 本身：需在场景中制造明显运动（如挥手、移动物体），观察亮区是否对应运动区域

- **建议下一步**
  1. **阶段 B**：增加模式切换（camera_jpeg / flow_viz_jpeg / fallback），便于对比 camera 与 flow 画面
  2. 尝试提高 flow JPEG 质量（JPEG_Q_MED）验证条纹是否减弱
  3. 在明显运动场景下复测，确认亮区与运动区域对应

- **已做改动（待验证）**
  - `flow_render.cpp`：flow JPEG 质量从 `JPEG_Q_LOW` 改为 `JPEG_Q_HIGH`，减轻块状伪影

- **Run ID**
  - `R20260223_StageD_D1_UserFeedback_Stripes`

- **后续**
  - 用户反馈：挥手时黑线随挥手频率闪动，延迟低 → flow 对运动有响应，链路正常
  - 新建 **plan-007-flow-stripe-debug.md** 用于条纹根因迭代调试

---
