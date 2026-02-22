## optical_sd_clean（SD 双帧输入 + EdgeFlowNet 光流推理）

`optical_sd_clean` 是从 `optical_sd` 提炼出来的 SD-only 版本，目标是：

- 代码干净、可维护
- 路径单一（只保留 SD 输入链路）
- 关键计时能力保留（DWT）

---

## 1. 功能边界

### 保留

- SD 卡读取连续 raw 帧
- 两帧拼接成 6 通道输入
- TFLM + Ethos-U 推理
- `ob_perf.cpp` 中 DWT 高精度计时
- 关键日志与关键词验证

### 不包含

- 摄像头 datapath/event handler 流程
- 运行时模型切换逻辑

---

## 2. 输入、模型与内存

### 输入

- 帧格式：`RGB888`
- 尺寸：`240 x 180`
- 单帧字节：`240 * 180 * 3 = 129600`
- 模板：`frame_%04d_rgb888.raw`

### 模型

- 从固定 Flash 地址加载：`YOLOV8_OBJECT_DETECTION_FLASH_ADDR`
- 默认用于“模型不变，快速迭代 app 代码”场景

### 内存（当前实测结论）

- `tensor_arena_size` 当前设置：`1670 * 1024`
- 大跨度验证结果：
  - `1700 * 1024`：失败（`alloc raw buffer fail`）
  - `1800 * 1024`：失败（`alloc raw buffer fail`）

说明：当前瓶颈是“模型 arena 与非模型运行内存（双帧 buffer 等）”的总内存平衡，而不是仅看 `AllocateTensors`。

---

## 3. 构建与运行

### 3.1 构建

```bash
cd EPII_CM55M_APP_S
make clean APP_TYPE=optical_sd_clean
make -s --no-print-directory -j4 APP_TYPE=optical_sd_clean
```

### 3.2 生成镜像

```bash
cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json
```

### 3.3 烧录（不改模型推荐）

```bash
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img
```

### 3.4 串口验收

```bash
python3 xmodem/serReadLoop.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --timeout=1 \
  --duration=12 \
  --keyword="initial done"
```

---

## 4. 当前目录结构（已完成分层）

当前 `optical_sd_clean` 已按职责拆分为子目录：

```text
optical_sd_clean/
  app/
    tflm_yolov8_od.c              # app 入口与生命周期
  pipeline/
    cvapp_yolov8n_ob.cpp          # 推理主流程（init/run）
    cvapp_yolov8n_ob.h
  io/
    ob_sd_frame.cpp               # SD 读帧
    ob_sd_frame.h
  perf/
    ob_perf.cpp                   # DWT 计时
    ob_perf.h
  debug/
    ob_debug_stats.cpp            # 统计与日志输出
    ob_debug_stats.h
  core/
    ob_runtime_ctx.h              # 运行时上下文
  config/
    common_config.h
  port/
    memory_manage.c
    memory_manage.h
    ffconf.h
    hardfault_handler.c
  tflm_yolov8_od.mk
  TFLM_yolov8_od_S_only.ld
  TFLM_yolov8_od_S_only.sct
  README.md
```

说明：

- `tflm_yolov8_od.mk` 已将上述子目录加入 `SCENARIO_APP_SUPPORT_LIST`，可正常被构建系统收集源码。
- 构建与运行行为与重构前保持一致（已通过 pipeline 关键词回归验证）。

后续维护原则：

1. 只做“搬家 + include 路径调整”，不改业务逻辑  
2. 每次搬一个模块后立即编译验证  
3. 保持 DWT 计时代码原样  
4. 保持串口关键字验收（`initial done`）作为回归门槛

---

## 5. 常见问题

### Q1: 为什么不把摄像头也塞进这个 app？

为了保持单一职责。`optical_sd_clean` 专注 SD 输入链路；摄像头输入建议放到 `optical_cam_oflow`，避免两个 I/O 流程耦合。

### Q2: arena 还能继续增大吗？

可以继续测，但建议按“较大步长 + 实机验证”方式，不做细颗粒度穷举，避免浪费迭代时间。

