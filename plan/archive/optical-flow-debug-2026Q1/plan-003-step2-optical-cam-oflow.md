> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 003: Step2 `optical_cam_oflow`（CSI 摄像头输入光流）可行性与实施方案

## 0. 目标

基于 `plan-002` 的 Step2，规划一个新 app：

- 名称：`optical_cam_oflow`
- 目标：把当前 `optical_sd_clean` 的 **SD 双帧输入**替换为 **CSI 摄像头双帧输入**
- 保留：EdgeFlowNet 推理主干、DWT 计时、串口日志与可视化输出能力

---

## 1. 可行性结论

### 1.1 可行

- 从代码架构看，输入层（SD 读帧）与推理层（`cv_yolov8n_ob_*`）已经相对可分离。
- 通过新增"摄像头帧采集模块"并替换 `ob_sd_load_frame` 路径，可实现 camera pipeline。

### 1.2 实时等级

用现有日志样本（37 条）统计：
- `infer_ms_avg ≈ 196.896ms`
- `total_ms_avg ≈ 339.114ms`
- `total_ms_min ≈ 316.951ms`

推断：当前总时延对应约 `2.9 ~ 3.2 FPS`。若改成摄像头输入，理论上可去掉多数 SD I/O 耗时（均值约 135ms），可望提升到约 `4~5 FPS` 量级。

**结论：可在线连续运行，但不是高帧率实时（例如 15/30 FPS）**

---

## 2. 参考与借鉴

### 2.1 摄像头接入与传感器配置参考

根 `README.md` 提供：`How to add support for raspberry pi camera?`
通过 `.mk` 的 `CIS_SUPPORT_INAPP_MODEL` 切换 `cis_ov5647 / cis_imx219 / cis_imx477`

### 2.2 输出可视化 pipeline 参考

`tflm_yolov8_od/README.md` 有可直接借鉴的 "Send image and meta data by UART"：
- 使用 Himax AI web toolkit（`index.html`）连接串口预览
- app 侧通过 `send_result.cpp` + 串口协议输出图像与元数据

对应代码实现可借鉴：
- `tflm_yolov8_od/tflm_yolov8_od.c`：
  - `hx_drv_spi_mst_protocol_write_sp(..., DATA_TYPE_JPG)`
  - `hx_drv_spi_mst_protocol_write_sp(..., DATA_TYPE_META_...)`
- `tflm_yolov8_od/send_result.cpp`：JSON + base64 图像封装，`event_reply(...)` / `send_bytes(...)`

---

## 3. 代码修改点

### 3.1 新 app 目录

新增：`EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/`

建议结构（与 `optical_sd_clean` 对齐）：
- `app/`：入口与状态机
- `pipeline/`：推理与双帧编排
- `io/camera/`：摄像头采集与帧缓存
- `perf/`：DWT 计时
- `debug/`：日志与统计
- `viz/`：可视化输出（可选，先 UART 文本后图像）
- `config/`：分辨率、路径、sensor 选择
- `port/`：memory/fatfs/hardfault 等

### 3.2 输入路径替换

从：`ob_sd_init / ob_sd_load_frame / ob_sd_next_frame_idx`

到：`cam_input_init()`、`cam_input_get_frame_pair(uint8_t* frame_t, uint8_t* frame_t1)`

新增文件：`io/camera/cam_input.cpp`、`io/camera/cam_input.h`

在 `pipeline/cvapp_yolov8n_ob.cpp` 做最小改造：保留 `prepare_optical_flow_input(...)`、DWT 计时与日志逻辑不变，把 SD 分支替换为 camera 双帧获取分支。

### 3.3 入口与调度

当前 `optical_sd_clean/app/tflm_yolov8_od.c` 是 SD-only 无限循环。camera 版建议：
- 初始化 sensor + datapath（参考 `tflm_yolov8_od.c`）
- 进入循环：采集双帧 -> 推理 -> 输出
- 先不引入复杂 event 分支

文件级修改：
- `app/tflm_yolov8_od.c`：增加 `cisdp_sensor_init()`、`cisdp_dp_init(...)`
- `tflm_yolov8_od.mk`：恢复 camera 所需依赖，首版固定 `CIS_SUPPORT_INAPP_MODEL = cis_ov5647`

### 3.4 输出定义

**阶段 A（先打通）**：UART 文本输出（中心点流场、均值流场、耗时）

**阶段 B（可视化）**：复用 `send_result.cpp` 输出 JPEG + metadata，对接 Himax AI web toolkit 预览

---

## 4. 实施里程碑

### M1：camera 输入打通（无可视化）

**验收清单**：
- [ ] `optical_cam_oflow` 可编译通过
- [ ] 设备启动后无 `CIS Init fail` / `DATAPATH Init fail`
- [ ] 串口稳定输出 `initial done` 与 `loop` 日志
- [ ] 能连续运行 2~3 分钟无崩溃
- [ ] DWT 分项耗时仍可见（`sd/cam`, `preproc`, `infer`, `total`）

### M2：可视化输出接入

**验收**：串口/网页均可观测输出

### M3：性能与稳定性

**验收**：稳定运行 5~10 分钟，无内存异常，日志连续

---

## 5. M1 执行记录（已完成，2026-02-22）

### 5.1 代码修改

- 新建 `optical_cam_oflow`（基于 `optical_sd_clean`）
- `.mk` 恢复 camera 依赖（`event_handler`、`CIS_SUPPORT_INAPP`、`cis_ov5647`）
- 新增 `io/camera/cam_input.{h,cpp}`：
  - `cam_input_init()`：完成 sensor/datapath 启动
  - `cam_input_get_frame_pair()`：连续双帧采集
  - 320x240 RGB planar -> 240x180 RGB888 中心裁剪
- `pipeline/cvapp_yolov8n_ob.cpp` 输入路径从 SD 切到 camera
- `app/tflm_yolov8_od.c` 入口命名改为 camera 版本

### 5.2 首轮失败根因

**现象**：`initial done` 可出现，但后续大量 `wait new camera frame timeout` / `camera frame pair fail`

**结论**：camera 已启动，但"新帧检测逻辑"不稳定，导致误判为无新帧

**关键证据**：
- 曾出现 `I2C err_code:-60`（后确认是摄像头连接方式问题）
- 连接修正后，串口出现：`sensor_id=0x5647`、`OV5647 Init Stream by app`、`camera input init done`

### 5.3 代码修正

**文件**：`EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/io/camera/cam_input.cpp`

**修正点**：
1. 放弃"仅依赖 WDMA2 next idx 变化"作为新帧判据
2. 改为先等 `hx_drv_xdma_get_WDMA2FirstFrameCapflag()==1`（首帧已到）
3. 后续按固定帧间隔（`33ms`）节拍采样双帧

### 5.4 最终回归结果

**验证日志**：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_192244.log`

**命中关键词**：`initial done`、`[loop=`

**未再出现**：`wait first camera frame timeout`、`camera frame pair fail`、`CIS Init fail`、`DATAPATH Init fail`

---

## 6. M2 可视化执行记录（已完成，2026-02-22）

### 6.1 接入可视化发送链路

在 `optical_cam_oflow` 接入可视化发送链路（`JPEG + META_YOLOV8_OB_DATA`）：
- 初始化阶段打开 SPI master 通道（`SPI_SEN_PIC_CLK`）
- 推理循环中发送当前 JPEG 与 meta payload

**串口观测**：`viz tx ok loop=... jpeg=...`

**验收日志**：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_193302.log`

### 6.2 Himax 页面无输出问题修复

**现象**：Connect 成功，Preview / Device log 无任何输出

**根因与修改**：

1. **握手不完整**：官方 app 会发 NAME? / VER? / ID? / **INFO?** / **MODEL?** 五条；页面可能等齐再解析。
   - 在 `viz/viz_uart.cpp` 中补发 INFO?、MODEL?（与官方格式一致）。

2. **主机命令未响应**：官方流程中，主机点击「uart send」会发 `0xFF`，设备据此切到 UART 并回送握手。
   - 在 `viz_uart.cpp` 中增加 `viz_uart_poll_host_cmd()`：非阻塞读 1 字节，若为 0xFF/0xFE/0xFD 则立即回送完整五条握手。

**验证**：WSL 下 no-model 烧录 + 串口抓取，关键字 `"name": "NAME?"`、`"name": "INVOKE"`、`"name": "MODEL?"` 均命中。

---

## 7. 双帧采样策略可选方案

### 方案 A：固定间隔采样（当前实现）

- 首帧到达后，每轮按固定延迟（当前 `33ms`）取下一帧
- **优点**：实现最简单、稳定性高、易调试
- **缺点**：不是"真正按新帧到达驱动"，可能拿重复帧或跳帧
- **适用**：M1/M2 阶段优先"先稳定跑通"

### 方案 B：帧到达事件/索引驱动采样（推荐长期）

- 以 datapath/WDMA 的"新帧事件"或可靠帧计数变化作为触发
- **优点**：时间一致性更好，真正按新帧推进
- **缺点**：对底层事件链路依赖高，调试复杂
- **适用**：M3 阶段追求"更优时序质量"

### 方案 C：自适应间隔采样（折中）

- 基于 `total/infer` 运行时耗时动态调整采样间隔
- **优点**：兼顾稳定性与吞吐
- **缺点**：参数需要调优

### 推荐执行顺序

1. 短期：保留方案 A，先推进 M2 可视化
2. 中期：做方案 B 的小实验分支（仅替换采样触发，不动推理主干）
3. 保底：若 B 不稳定，转方案 C 做工程折中

