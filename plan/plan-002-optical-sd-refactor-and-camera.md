# Plan 002: optical_sd 精简与摄像头输入改造路线

## 0. 当前状态（本次已处理）

- 已删除 pipeline 试验复制 app：
  - `EPII_CM55M_APP_S/app/scenario_app/optical_sd_poc_auto`
  - `EPII_CM55M_APP_S/app/scenario_app/optical_sd_poc_model`
- 已恢复 `EPII_CM55M_APP_S/makefile`：
  - `APP_TYPE = optical_sd`

---

## 1. 现状诊断（基于 optical_sd 代码）

目标代码：
- `EPII_CM55M_APP_S/app/scenario_app/optical_sd/cvapp_yolov8n_ob.cpp`
- `EPII_CM55M_APP_S/app/scenario_app/optical_sd/tflm_yolov8_od.c`
- `EPII_CM55M_APP_S/app/scenario_app/optical_sd/tflm_yolov8_od.mk`

关键事实：

1) 当前主流程是 **离线 SD 双帧读入 + 光流模型推理**  
- `tflm_yolov8_od.c` 中已改为 `while(1){ cv_yolov8n_ob_run(...) }` 的离线循环。  
- `cvapp_yolov8n_ob.cpp` 中 `RAW_DIR/RAW_FMT` 指向 SD 卡路径，逐帧读取并构建 2 帧输入。

2) 模型来自 Flash 固定地址  
- `cv_yolov8n_ob_init(..., model_addr)` 调 `tflite::GetModel((const void*)model_addr)`。  
- `common_config.h` 定义 `YOLOV8_OBJECT_DETECTION_FLASH_ADDR = 0x3AB7B000`。

3) 工程里仍保留了大量“摄像头/事件流”代码路径  
- `tflm_yolov8_od.c` 仍包含 `cisdp_dp_init` / `event_handler_*` 等路径（当前离线分支不走）。
- `tflm_yolov8_od.mk` 仍启用 `CIS_SUPPORT_INAPP_MODEL = cis_ov5647` 与相关 sensor 配置。

这意味着：你现在“能跑 SD 光流”，但代码形态仍带有原始 yolov8 app 的历史包袱。

---

## 2. 问题 1：是否可以删减代码改善“非模型部分”内存占用？

结论：**可以，而且有明确收益点。**

优先级从高到低：

### A. 去掉当前离线流程不需要的摄像头/事件流路径（高收益）

- 目前 `tflm_yolov8_od.c` 与 `.mk` 保留了摄像头相关初始化和模块依赖。
- 即便部分代码运行时不走，也可能引入额外代码段、静态对象和驱动依赖。

建议：
- 对“SD-only 版本”单独保留最小依赖（FatFS + TFLM + SPI/串口必要模块）。
- 不编译不使用的 `event_handler` / `cisdp_sensor` 流程。

### B. 精简 `cvapp_yolov8n_ob.cpp` 中未使用 include 与接口（中收益）

- 目前 include 较多，部分为历史遗留（例如 camera/datapath 关联头），应逐项核对是否实际使用。
- 可减少编译耦合与潜在链接引入。

### C. 评估并下调 `tensor_arena_size`（高风险但收益可能大）

- 当前 `tensor_arena_size = 1650 * 1024`。
- 这部分主要是模型运行内存（不属于“非模型”），但总体 RAM 占用最大头之一。
- 若你只想优化“非模型部分”，先不动它；若追求总 RAM，再做逐步压测（每次减 32KB/64KB，验证 `AllocateTensors` 成败）。

### D. 栈与缓冲复用（中收益）

- 目前双帧缓存约 `RAW_FRAME_BYTES * 2 = 259200 bytes`（约 253KB），这是数据面必须成本。
- 可继续保持双缓冲复用策略（你已做得很好），避免新增中间拷贝。

---

## 3. 问题 2：建议“新建干净 app”还是“在现有 app 上删减”？

结论：**建议新建干净 app（推荐），再择机回迁。**

原因：

1) 现有 `optical_sd` 已承载较多历史改造，直接删减容易误伤。  
2) 你后续计划加入“摄像头输入 -> 光流推理”，和当前“SD 离线输入”是两种 I/O 架构。  
3) 分离 app 可以清晰管理：
- `optical_sd_oflow`：专注 SD 输入
- `optical_cam_oflow`：专注 CSI 摄像头输入

建议策略：

- 阶段 1：从 `optical_sd` 复制出 `optical_sd_clean`，做“只保留 SD 推理链路”的净化。
- 阶段 2：从 `optical_sd_clean` 再复制 `optical_cam_oflow`，只替换输入源（SD -> camera），模型后处理保持一致。
- 阶段 3：稳定后再决定是否合并为单 app 多模式。

---

## 4. 问题 3：README 是否有 OV5647-62 FOV 摄像头可用验证教程？

结论：**有相关指引，但不是“专门的 OV5647 完整验证手册”。**

已存在的可用信息：

1) 根 `README.md` 有 “How to add support for raspberry pi camera?”  
- 明确给出通过 `CIS_SUPPORT_INAPP_MODEL` 切换传感器（包含 `cis_ov5647` / `cis_imx219` / `cis_imx477`）的方式。  

2) `optical_sd/tflm_yolov8_od.mk` 当前本就配置：
- `CIS_SUPPORT_INAPP_MODEL = cis_ov5647`

可执行的最小验证路径（手动）：

1. 先用一个“已知摄像头路径能跑”的官方 app（如 `allon_sensor_tflm`）验证 OV5647 是否正常出流。  
2. 观察串口中是否出现 datapath/sensor init success（无 `DATAPATH Init fail` / `sensor init fail`）。  
3. 若该 app 正常，再迁移到光流 camera app，减少“模型问题”和“传感器问题”耦合调试成本。

---

## 5. 问题 4：一个 app 只能有一种运行脚本吗？改成摄像头输入需要新建 app 吗？

结论：

- 技术上：**一个 app 可以支持多运行模式**（例如 SD 模式 / Camera 模式），通过编译宏或运行时参数切换。  
- 工程上：当前阶段**更建议新建 app**，原因是可维护性和调试效率更高。

建议：

- 短期：分 app（`optical_sd_clean` 与 `optical_cam_oflow`）。  
- 中期：若两条链路稳定，再抽象公共推理模块，按需合并为一个 app 的双模式入口。

---

## 6. 下一步你应该尝试什么（建议顺序）

### Step 1（今天可做）：建立“干净 SD app”

目标：
- 复制 `optical_sd` -> `optical_sd_clean`
- 去掉 camera/event 相关依赖，仅保留 SD->预处理->推理->日志

验收：
- 烧录（nomodel）成功
- 关键字命中：`initial done` + 推理 loop 输出

### Step 2（接着做）：建立“camera app 骨架”

目标：
- 复制 `optical_sd_clean` -> `optical_cam_oflow`
- 输入接口替换为 camera 帧（先单帧/双帧缓存，暂不追求性能）

验收：
- 摄像头初始化成功
- 可获得连续帧并喂入模型

### Step 3（稳定化）：内存与结构优化

目标：
- 清理未使用 include/模块
- 再做 arena 逐步缩减测试（可选）

验收：
- 功能不回归
- RAM/镜像体积有可量化改善

---

## 7. 风险与调试优先项

1) USB 串口偶发断连（WSL2 + usbipd）  
- 现象：`/dev/ttyACM0` 消失、烧录失败  
- 处理：先 `usbipd attach` 恢复，再继续

2) Camera bring-up 与模型推理同时改动导致定位困难  
- 处理：先在官方 camera app 验证硬件，再迁移模型链路

3) 过早压缩 tensor arena 导致误判  
- 处理：先保证功能路径稳定，再做内存极限优化

