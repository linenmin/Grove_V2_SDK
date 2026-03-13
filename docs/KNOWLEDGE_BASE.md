# Optical Flow Project: Knowledge Base (Current Baseline)

## 1. 硬件限制与规格 (Hardware Specs)

- **芯片**: Himax WE2 (CM55M + Ethos-U55)。
- **SRAM**: 总计 **2 MiB**。由于引导加载程序和栈空间，实际应用可控空间约 **1.9 MiB**。
- **Flash 地址**: 当前光流模型槽位宏为 `OPTICAL_FLOW_MODEL_FLASH_ADDR = 0x3AB7B000`。
- **命名状态**: `optical_cam_oflow` 主线入口已切到光流命名；协议头已提供 `struct_optical_flow_algoResult` / `DATA_TYPE_META_OPTICAL_FLOW_DATA` 别名，旧 `yolo` 名字仅作为兼容层保留。
- **配置管理**: 模型分辨率和 Arena 大小统一在 `common_config.h` 中管理。

---

## 2. NPU 与模型细节 (NPU & Model)

- **输入要求**: 6-channel **NHWC** (Interleaved)。
    - 格式：`[R0, G0, B0, R1, G1, B1, R0, G0, B0, R1, G1, B1, ...]` 其中 0=prev, 1=curr。
    - **避坑**: 必须在 C++ 中手动交错拼装。
- **当前有效模型基线**: `157x203 -> 160x208x2`。
- **输出特征**: **NHWC** (Planar=0)。
    - 量化：`scale ≈ 0.407547`, `zp = -4`。
    - 对应 `int8` [-128, 127] 映射到约 [-64, +64] 像素偏移。
- **Tensor Arena**: 推荐至少 **1432 KiB**。
    - Vela 报告峰值: 1430 KiB (`157x203` 到 `160x208` 这一档都相同)。
- **bilinear 实验热点**: 当前 `172x224 -> 176x224` bilinear 模型的 Vela SRAM peak 位于 decoder 末段 `ResizeBilinear_1`，峰值 `1386.00 KiB`。
- **bilinear 失败对照**: `172x228 -> 176x240` 的 Vela 峰值为 `1485.00 KiB`，板端 `AllocateTensors()` 请求 `1520720 B`，与 Vela 峰值只差 `80 B` 左右。
- **当前默认主线**: 只承认 `157x203` 为有效部署基线。
- **运行期边界**: `157x203` 已验证可稳定运行；`158x202` 与 `155x206` 虽然仍通过 Vela 编译，但会在板端 `alloc prev buffer fail`。
- **实验状态说明**: `150x200 -> 160x208` 模型虽然可推理，但当前会导致可视化 fallback，不应作为默认部署模型。

---

## 3. 可视化规则 (Visualization)

- **渲染系数**: `mag * 0.05`。
    - **原因**: 默认 `mag * 2.0` 会导致 0.5 像素以上的变动全部饱和（鲜艳纯色），无法区分运动强度。
- **颜色映射**: HSV 模式。H=方向, S=1.0, V=强度。
- **当前有效输出尺寸**: `INVOKE resolution = [208, 160]`。
- **失效特征**: 如果 `INVOKE resolution` 变成 `320x240` 相机图，通常说明光流渲染没有命中，而是退回了 fallback 相机 JPEG。
- **SPI 协议**: 连接 Himax 预览屏幕时，需确保传输 Buffer 对齐。

---

## 4. 摄像头配置 (Camera)

- **原始采集**: $320 \times 240$ Planar (B/G/R 独立平面)。
- **内存占用**: 约 **225 KiB** ($320 \times 240 \times 3 \div 1024$)。
- **下采样模式**: `SUBSAMPLE_2X`。使用 Center Crop 比直接 Resize 对光流更友好。

---

## 5. 调试工具与脚本

- **流水线抓取**: `scripts/run_optical_pipeline.sh` (集成 UART 读取、图像提取与预览)。
- **当前部署说明**: 见 `docs/DEPLOYMENT.md`。
- **模型导出**: 当前推荐入口是 `scripts/export_optical_flow_144x192.sh`。
- **导出逻辑位置**: `tools/model_export/optical_flow_144x192/`。
- **默认发布模型**: `model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`。
- **外部依赖边界**: checkpoint 与 calibration 数据已复制进仓库；当前剩余前提主要是本机 Python/TensorFlow/OpenCV/Vela 环境，具体见 `docs/MODEL_EXPORT.md`。
