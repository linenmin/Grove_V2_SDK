## optical_cam_oflow（CSI 摄像头双帧输入 + EdgeFlowNet 光流推理）

`optical_cam_oflow` 是 `plan-003` 的 M1 实现版本，目标是：

- 复用 `optical_sd_clean` 的可维护结构
- 将输入从 SD 双帧替换为 CSI 摄像头连续双帧
- 保留 TFLM + Ethos-U + DWT 计时 + 串口文本日志

---

## 1. 功能边界

### 保留

- OV5647 CSI 连续采集（RGB 平面数据）
- 两帧拼接成 6 通道输入
- TFLM + Ethos-U 推理
- `ob_perf.cpp` 中 DWT 高精度计时
- 关键日志与关键词验证

### 不包含

- 图像可视化（JPG/meta UART 打包）
- 运行时模型切换逻辑

---

## 2. 输入、模型与内存

### 输入

- 源帧格式：`RGB planar`（来自 `app_get_raw_addr()`）
- 处理中：中心裁剪并转换为 `RGB888`
- 尺寸：`240 x 180`
- 单帧字节：`240 * 180 * 3 = 129600`
- 每轮输入：连续两帧拼接为 6 通道

### 模型

- 从固定 Flash 地址加载：`OPTICAL_FLOW_MODEL_FLASH_ADDR`
- 默认用于“模型不变，快速迭代 app 代码”场景

### 内存（当前实测结论）

- `tensor_arena_size` 当前主线设置：`1432 * 1024`
- 当前 `144x192` 基线已验证可工作。
- 更大 arena 的历史尝试保留在归档计划中，不再作为当前主线建议。

说明：当前瓶颈仍然是“模型 arena 与非模型运行内存（双帧 buffer 等）”的总内存平衡，而不是仅看 `AllocateTensors`。

---

## 3. 构建与运行

### 3.1 构建

```bash
cd EPII_CM55M_APP_S
make clean APP_TYPE=optical_cam_oflow
make -s --no-print-directory -j4 APP_TYPE=optical_cam_oflow
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

## 4. 当前目录结构

当前 `optical_cam_oflow` 与 `optical_sd_clean` 保持同一分层：

```text
optical_cam_oflow/
  app/
    optical_flow_app.c              # app 入口与生命周期
  pipeline/
    cvapp_optical_flow.cpp          # 推理主流程（init/run）
    cvapp_optical_flow.h
  io/
    camera/
      cam_input.cpp               # 摄像头双帧采集与格式转换
      cam_input.h
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
  optical_flow_app.mk
  optical_flow_app_S_only.ld
  optical_flow_app_S_only.sct
  README.md
```

M1 关键点：

1. `cam_input_init()` 完成 sensor/datapath 启动  
2. `cam_input_get_frame_pair()` 每轮输出连续双帧  
3. pipeline 主流程无需关心摄像头底层细节
