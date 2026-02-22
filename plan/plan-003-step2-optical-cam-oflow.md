# Plan 003: Step2 `optical_cam_oflow`（CSI 摄像头输入光流）可行性与实施方案

## 0. 目标

基于 `plan-002` 的 Step2，规划一个新 app：

- 名称建议：`optical_cam_oflow`
- 目标：把当前 `optical_sd_clean` 的 **SD 双帧输入**替换为 **CSI 摄像头双帧输入**
- 保留：EdgeFlowNet 推理主干、DWT 计时、串口日志与可视化输出能力

---

## 1. 可行性结论（先说结论）

### 1.1 可行

- 从代码架构看，输入层（SD 读帧）与推理层（`cv_yolov8n_ob_*`）已经相对可分离。
- 通过新增“摄像头帧采集模块”并替换 `ob_sd_load_frame` 路径，可实现 camera pipeline。

### 1.2 可以做到“在线实时输入与输出”，但要明确实时等级

用现有日志样本（37 条）做了快速统计：

- `infer_ms_avg ≈ 196.896ms`
- `total_ms_avg ≈ 339.114ms`
- `total_ms_min ≈ 316.951ms`

推断：

- 当前总时延对应约 `2.9 ~ 3.2 FPS`
- 若改成摄像头输入，理论上可去掉多数 SD I/O 耗时（均值约 135ms），总体可望提升到约 `4~5 FPS` 量级
- 结论：**可在线连续运行，但不是高帧率实时（例如 15/30 FPS）**

---

## 2. 参考与借鉴（README + 现有 app）

### 2.1 摄像头接入与传感器配置参考

根 `README.md` 提供：

- `How to add support for raspberry pi camera?`
- 通过 `.mk` 的 `CIS_SUPPORT_INAPP_MODEL` 切换 `cis_ov5647 / cis_imx219 / cis_imx477`

### 2.2 输出可视化 pipeline 参考（重点）

`tflm_yolov8_od/README.md` 有可直接借鉴的 “Send image and meta data by UART”：

- 使用 Himax AI web toolkit（`index.html`）连接串口预览
- app 侧通过 `send_result.cpp` + 串口协议输出图像与元数据

对应代码实现可借鉴：

- `tflm_yolov8_od/tflm_yolov8_od.c`：
  - `hx_drv_spi_mst_protocol_write_sp(..., DATA_TYPE_JPG)`
  - `hx_drv_spi_mst_protocol_write_sp(..., DATA_TYPE_META_...)`
- `tflm_yolov8_od/send_result.cpp`：
  - JSON + base64 图像封装
  - `event_reply(...)` / `send_bytes(...)`

结论：

- **有现成可视化链路可复用**，建议先复用再最小改造。

---

## 3. Step2 需要改哪些代码（明确修改点）

## 3.1 新 app 建议目录

新增：

- `EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/`

建议结构（与 `optical_sd_clean` 对齐）：

- `app/`：入口与状态机
- `pipeline/`：推理与双帧编排
- `io/camera/`：摄像头采集与帧缓存
- `perf/`：DWT 计时
- `debug/`：日志与统计
- `viz/`：可视化输出（可选，先 UART 文本后图像）
- `config/`：分辨率、路径、sensor 选择
- `port/`：memory/fatfs/hardfault 等

## 3.2 输入路径替换

从：

- `ob_sd_init / ob_sd_load_frame / ob_sd_next_frame_idx`

到：

- `cam_input_init()`
- `cam_get_frame_pair(uint8_t* frame_t, uint8_t* frame_t1)`

实现建议：

- 先做最小双缓冲：每轮拿到两帧 RGB888（或 YUV 转 RGB）
- 保持 `prepare_optical_flow_input(...)` 不变，先保证可跑

## 3.3 入口与调度

当前 `optical_sd_clean/app/tflm_yolov8_od.c` 是 SD-only 无限循环。  
camera 版建议：

- 初始化 sensor + datapath（参考 `tflm_yolov8_od.c`）
- 进入循环：采集双帧 -> 推理 -> 输出
- 先不引入复杂 event 分支，避免第一版过重

## 3.4 输出定义（建议分阶段）

阶段 A（先打通）：

- UART 文本输出（中心点流场、均值流场、耗时）

阶段 B（可视化）：

- 复用 `send_result.cpp` 输出 JPEG + metadata
- 对接 Himax AI web toolkit 预览

---

## 4. 输入 / 输出规范（第一版建议）

### 输入

- 来源：OV5647 CSI 连续帧
- 有效尺寸：先固定为与模型一致输入前尺寸（建议预处理到 `240x180`）
- 帧对：`(t, t+1)` 连续时间帧

### 输出

- 必选（调试）：串口文本
  - `loop`, `dx/dy`, `mean_dx/mean_dy`, `sd/preproc/infer/total`
- 可选（可视化）：
  - `JPEG frame + meta`（借鉴 yolov8 UART web toolkit 方案）

---

## 5. 实施步骤（建议 3 个小里程碑）

### M1：camera 输入打通（无可视化）

目标：

- `optical_cam_oflow` 可启动摄像头并连续推理

验收：

- 串口有稳定 loop 输出
- 无 `CIS Init fail` / `DATAPATH Init fail`

### M2：可视化输出接入

目标：

- 接入 `send_result` 风格输出
- web toolkit 可看到帧与结果

验收：

- 串口/网页均可观测输出

### M3：性能与稳定性

目标：

- 稳定运行 5~10 分钟
- 统计 FPS 和关键阶段耗时

验收：

- 无内存异常
- 日志连续

---

## 6. 风险与调试优先级

1) 摄像头出流失败  
- 先验证 `allon_sensor_tflm`（硬件链路）  
- 再回到 `optical_cam_oflow`

2) 帧格式不匹配  
- 明确 camera 输出格式与模型输入格式转换路径

3) “看起来卡顿”但其实是推理瓶颈  
- 用 DWT 分项计时判断瓶颈（采集/预处理/推理）

4) 一次改太多导致不可定位  
- 严格按 M1->M2->M3 渐进

---

## 7. 下一步你该先做什么（建议）

先做 M1 的最小试验：

1. 从 `optical_sd_clean` 复制为 `optical_cam_oflow`  
2. 保留推理主干，替换输入模块为 camera 双帧  
3. 保留 DWT 计时和当前日志格式  
4. 先跑通串口文本输出，不急着上 web 可视化

等 M1 跑通后，再开始 M2（接入 `send_result` + web toolkit）。
