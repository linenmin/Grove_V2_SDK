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

建议直接新增文件（第一版）：

- `io/camera/cam_input.cpp`
- `io/camera/cam_input.h`

并在 `pipeline/cvapp_yolov8n_ob.cpp` 做最小改造：

1. 保留 `prepare_optical_flow_input(...)`、DWT 计时与日志逻辑不变  
2. 把 SD 分支替换为 camera 双帧获取分支  
3. 首版只保证稳定拿到连续两帧，不引入复杂异步队列

实现建议：

- 先做最小双缓冲：每轮拿到两帧 RGB888（或 YUV 转 RGB）
- 保持 `prepare_optical_flow_input(...)` 不变，先保证可跑

## 3.3 入口与调度

当前 `optical_sd_clean/app/tflm_yolov8_od.c` 是 SD-only 无限循环。  
camera 版建议：

- 初始化 sensor + datapath（参考 `tflm_yolov8_od.c`）
- 进入循环：采集双帧 -> 推理 -> 输出
- 先不引入复杂 event 分支，避免第一版过重

文件级修改建议：

- `app/tflm_yolov8_od.c`（新 app 内）：
  - 增加 `cisdp_sensor_init()`、`cisdp_dp_init(...)`
  - 在主循环中调用 `cv_yolov8n_ob_run(...)`
- `tflm_yolov8_od.mk`（新 app 内）：
  - 恢复 camera 所需依赖（`CIS_SUPPORT_INAPP`、`event_handler`、sensor 子目录）
  - 首版固定 `CIS_SUPPORT_INAPP_MODEL = cis_ov5647`

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

可视化输出建议（可直接复用现有链路）：

- 参考 `tflm_yolov8_od/README.md` 的 `Send image and meta data by UART`
- 复用 `send_result.cpp` 的 JSON + base64 编码与 `event_reply(...)`
- 复用 `hx_drv_spi_mst_protocol_write_sp(..., DATA_TYPE_JPG / DATA_TYPE_META_...)`

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

5) “实时”定义不清导致预期偏差  
- 当前目标定义为“在线连续处理（低帧率实时）”，不是 15/30 FPS 视频级实时

---

## 7. 下一步你该先做什么（建议）

先做 M1 的最小试验：

1. 从 `optical_sd_clean` 复制为 `optical_cam_oflow`  
2. 保留推理主干，替换输入模块为 camera 双帧  
3. 保留 DWT 计时和当前日志格式  
4. 先跑通串口文本输出，不急着上 web 可视化

等 M1 跑通后，再开始 M2（接入 `send_result` + web toolkit）。

---

## 8. M1 验收清单（你执行时可直接照抄）

- [ ] `optical_cam_oflow` 可编译通过
- [ ] 设备启动后无 `CIS Init fail` / `DATAPATH Init fail`
- [ ] 串口稳定输出 `initial done` 与 `loop` 日志
- [ ] 能连续运行 2~3 分钟无崩溃
- [ ] DWT 分项耗时仍可见（`sd/cam`, `preproc`, `infer`, `total`）

---

## 9. M1 执行进度（代码侧，2026-02-22）

已完成（代码修改）：

- 新建 `optical_cam_oflow`（基于 `optical_sd_clean`）
- `.mk` 恢复 camera 依赖（`event_handler`、`CIS_SUPPORT_INAPP`、`cis_ov5647`）
- 新增 `io/camera/cam_input.{h,cpp}`：
  - `cam_input_init()`：完成 sensor/datapath 启动
  - `cam_input_get_frame_pair()`：连续双帧采集
  - 320x240 RGB planar -> 240x180 RGB888 中心裁剪
- `pipeline/cvapp_yolov8n_ob.cpp` 输入路径从 SD 切到 camera
- `app/tflm_yolov8_od.c` 文案与入口命名改为 camera 版本
- `README.md` 更新为 `optical_cam_oflow` 说明

待你手动验证（按你的流程执行）：

- 编译、烧录、串口验收
- 若失败优先做日志调试（定位 `CIS Init` / `DATAPATH` / 帧获取超时位置）

---

## 10. M1 调试闭环结果（增量，2026-02-22 晚）

本轮按“先调试不绕过”执行了完整闭环：`定位 -> 打点 -> 复现 -> 修正 -> 回归 -> 清理临时调试代码`。

### 10.1 阶段 A：首轮失败根因

- 现象：`initial done` 可出现，但后续大量 `wait new camera frame timeout` / `camera frame pair fail`
- 结论：camera 已启动，但“新帧检测逻辑”不稳定，导致误判为无新帧

### 10.2 阶段 B：排查过程（关键证据）

- 初期曾出现 `I2C err_code:-60`（后确认是摄像头连接方式问题）
- 连接修正后，串口出现：
  - `sensor_id=0x5647`
  - `OV5647 Init Stream by app`
  - `OV5647 on by app done`
  - `camera input init done`
- 说明：`cisdp_sensor_init()` 与 `cisdp_dp_init()` 已正常，问题收敛到“帧等待策略”

### 10.3 阶段 C：代码修正（已落地）

- 文件：`EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/io/camera/cam_input.cpp`
- 修正点：
  1. 放弃“仅依赖 WDMA2 next idx 变化”作为新帧判据  
  2. 改为先等 `hx_drv_xdma_get_WDMA2FirstFrameCapflag()==1`（首帧已到）  
  3. 后续按固定帧间隔（`33ms`）节拍采样双帧（当前稳定基线，不是唯一方案）
- 同时清理了临时调试打印：
  - `EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/cis_sensor/cis_ov5647/cisdp_sensor.c` 中临时 `[DBG][cis]` 行已删除

### 10.4 阶段 D：最终回归结果（通过）

- 通过日志关键词：
  - `initial done`
  - `[loop=`
- 未再出现：
  - `wait first camera frame timeout`
  - `camera frame pair fail`
  - `CIS Init fail`
  - `DATAPATH Init fail`
- 验证日志：
  - `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_192244.log`

---

## 11. 下一步建议（M2 / M3）

### M2（建议先做，可视化链路）

1. 在 `optical_cam_oflow` 接入 `send_result` 风格输出（JPEG + meta）  
2. 用 Himax web toolkit 联调可视化  
3. 串口验收新增关键词：`DATA_TYPE_JPG` / `DATA_TYPE_META_*`（或你定义的发送成功标志）

### M3（稳定性与性能）

1. 连续运行 10~30 分钟，统计失败率（目标：0）  
2. 记录 `sd/cam`, `preproc`, `infer`, `total` 均值与 P95  
3. 若要提速，再分阶段评估：
   - 输入分辨率/裁剪策略
   - 帧间隔（33ms）是否可按负载自适应
   - 日志打印频率对吞吐的影响

### M2 当前执行进度（增量）

- 已在 `optical_cam_oflow` 接入可视化发送链路（`JPEG + META_YOLOV8_OB_DATA`）：
  - 初始化阶段打开 SPI master 通道（`SPI_SEN_PIC_CLK`）
  - 推理循环中发送当前 JPEG 与 meta payload
- 串口观测到稳定发送标志：
  - `viz tx ok loop=... jpeg=...`
- 最新验收日志：
  - `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_193302.log`
  - 关键词命中：`initial done`、`[loop=`、`viz tx ok`

---

## 12. 双帧采样策略可选方案（供选择）

问题：首帧等待是必要的；但首帧之后是否必须固定间隔采样？

结论：**不必须**。固定间隔只是当前最稳的基线策略。下面给出可选项。

### 方案 A：固定间隔采样（当前实现）

定义：

- 首帧到达后，每轮按固定延迟（当前 `33ms`）取下一帧

优点：

- 实现最简单、稳定性高、易调试  
- 对底层帧索引不稳定场景容错好  
- 串口日志节奏可预期

缺点：

- 不是“真正按新帧到达驱动”，可能拿到重复帧或跳帧  
- 端到端时延受固定 delay 影响  
- 在低光/曝光变化时，时间一致性不一定最优

适用：

- M1/M2 阶段优先“先稳定跑通”

### 方案 B：帧到达事件/索引驱动采样（推荐的长期方案）

定义：

- 以 datapath/WDMA 的“新帧事件”或可靠帧计数变化作为触发，取 `(t, t+1)`

优点：

- 时间一致性更好，真正按新帧推进  
- 可减少重复帧，提高有效信息密度  
- 更利于后续性能优化与可视化同步

缺点：

- 对底层事件链路依赖高，调试复杂  
- 需要确认 `evt_datapath`/XDMA 状态在本 app 下的可靠性

适用：

- M3 阶段追求“更优时序质量”与更高上限

### 方案 C：自适应间隔采样（折中）

定义：

- 基于 `total/infer` 运行时耗时动态调整采样间隔

优点：

- 兼顾稳定性与吞吐，易逐步引入  
- 在负载变化时可自动避免拥塞

缺点：

- 参数需要调优（上下限、调节步长）  
- 仍不如纯事件驱动精确

适用：

- 当方案 A 稳定、方案 B 实施成本高时的中间态

### 推荐执行顺序

1. **短期（现在）**：保留方案 A，先推进 M2 可视化  
2. **中期**：做方案 B 的小实验分支（仅替换采样触发，不动推理主干）  
3. **保底**：若 B 不稳定，转方案 C 做工程折中

### 你现在可以直接二选一

- 选项 1：继续当前方案 A（最快推进 M2）  
- 选项 2：先做方案 B PoC（优先时序正确性）

我建议：**先选项 1，把可视化链路打通；并行准备选项 2 的小实验分支。**
