# Optical Flow Project: Knowledge Base (High Efficiency)

## 1. 硬件限制与规格 (Hardware Specs)

- **芯片**: Himax WE2 (CM55M + Ethos-U55)。
- **SRAM**: 总计 **2 MiB**。由于引导加载程序和栈空间，实际应用可控空间约 **1.9 MiB**。
- **Flash 地址**: `YOLOV8_OBJECT_DETECTION_FLASH_ADDR = 0x3AB7B000`。
- **配置管理**: 模型分辨率和 Arena 大小现在统一在 `common_config.h` 中管理。

---

## 2. NPU 与模型细节 (NPU & Model)

- **输入要求**: 6-channel **NHWC** (Interleaved)。
    - 格式：`[R0, G0, B0, R1, G1, B1, R0, G0, B0, R1, G1, B1, ...]` 其中 0=prev, 1=curr。
    - **避坑**: 必须在 C++ 中手动交错拼装。
- **输出特征**: **NHWC** (Planar=0)。
    - 量化：`scale ≈ 0.5`, `zp = -1`。
    - 对应 `int8` [-128, 127] 映射到约 [-64, +64] 像素偏移。
- **Tensor Arena**: 推荐至少 **1432 KiB**。
    - $144 \times 192$ 峰值占用: 1188 KiB。
    - $150 \times 200$ 峰值占用: 1430 KiB。

---

## 3. 可视化规则 (Visualization)

- **渲染系数**: `mag * 0.05`。
    - **原因**: 默认 `mag * 2.0` 会导致 0.5 像素以上的变动全部饱和（鲜艳纯色），无法区分运动强度。
- **颜色映射**: HSV 模式。H=方向, S=1.0, V=强度。
- **SPI 协议**: 连接 Himax 预览屏幕时，需确保传输 Buffer 对齐。

---

## 4. 摄像头配置 (Camera)

- **原始采集**: $320 \times 240$ Planar (B/G/R 独立平面)。
- **内存占用**: 约 **225 KiB** ($320 \times 240 \times 3 \div 1024$)。
- **下采样模式**: `SUBSAMPLE_2X`。使用 Center Crop 比直接 Resize 对光流更友好。

---

## 5. 调试工具与脚本

- **流水线抓取**: `scripts/run_optical_pipeline.sh` (集成 UART 读取、图像提取与预览)。
- **模型导出**: `EdgeFlowNet/sramTest/run_sram_test.py`。
