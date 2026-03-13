> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 007：光流条纹问题迭代调试计划

## 1. 问题背景（已解决）

- **已确认**：flow 数据对运动有响应（挥手时出现黑线，随挥手频率闪动，延迟低）
- **待解决（已解决）**：画面为垂直黑白条纹，而非期望的「亮=运动区域」灰度图
- **JPEG_Q_HIGH / JPEG_Q_BEST** 均已尝试，条纹未消失，说明非 JPEG 压缩质量导致

---

## 2. 条纹根因分析（已破案）

### 2.1 初步假设与验证

曾考虑的假设：
- H1：flow tensor 空间布局/周期性与条纹对齐
- H2：per-frame max 归一化导致静态噪声被放大
- H3：flow tensor 内存布局与读取 stride 不匹配
- H4：JPEG addMCU 的 (x,y) 坐标语义
- H5：条纹周期与 JPEG MCU（8x8）对齐

**已排除**：H4（JPEG MCU 拼接正确）、H5（JPEG 质量无关）

### 2.2 关键发现：模型输出本身带条纹

**测试方法**：
1. 植入纯几何测试图（`FLOW_VIZ_TEST_PATTERN=1`）隔离 Web 端与数据源
2. 修改采样方式为 `step=1` 的连续密集列采样（之前 step=8 完美踩中 8 像素周期，掩盖了问题）

**串口证据**（取自 `sram_test_modified_vela.tflite`）：
```
[col_mean_mag] step=1 c0=3899 c1=4115 c2=3334 c3=3334 c4=2075 c5=2410 c6=2472 c7=2446 c8=3915 c9=4155 c10=3334 c11=3334 c12=2402 c13=2494 c14=2394 c15=2422
```

**逐列分析（明显的 8 像素周期）**：
- `c0, c1` / `c8, c9`：~4000（高）
- `c2, c3` / `c10, c11`：绝对恒定 3334
- `c4, c5, c6, c7` / `c12, c13, c14, c15`：~2000-2400（低）

### 2.3 根本原因

**导出脚本问题**：`MCUFlowNet/EdgeFlowNet/sramTest/run_sram_test.py` 的量化校准（Calibration）存在致命失误：

```python
# 错误代码
representative_dataset = np.random.uniform(0.0, 1.0)  # 白噪声，范围错误
```

**连锁反应**：
1. 白噪声与极端缩小的比例尺（应为 0~255）导致 `MultiScaleResNet` 内大量 `ConvTranspose(3,3)` 与 `stride(2,2)` 生成的特征残差图值域剧烈偏移
2. 最终输出只有被错误量化放大的 `1/4` 分支网络残留物
3. 底层分辨率带有典型的 2 像素棋盘格，被上采样放大 4 倍后形成"8 像素周期循环"的强垂直条纹

### 2.4 修复方案

**修复**：使用真实 Sintel 子集照片进行 0~255 量化校准

**验证**：采用真实图片校准后，模型顺利提取出真实激活极值，棋盘格完全消除，边缘设备的流水线视觉质量与 PC 浮点一致

---

## 3. 调试经验总结

### 3.1 采样陷阱

**问题**：`step=8` 的采样步长正好完美踩中了 8 像素的波动周期，导致每次采样都在同一相位，从而完美掩盖了剧烈的高频振荡

**教训**：做周期性诊断时，采样步长应避免与疑似周期对齐

### 3.2 测试图隔离法

**方法**：植入 `FLOW_VIZ_TEST_PATTERN=1`，强制生成固定渐变图作为输入替身

**作用**：
- 若测试图显示正常，说明后链路（JPEG+串口+Web）完美无瑕
- 若测试图也有条纹，说明后链路有问题

### 3.3 关键代码路径

**测试图开关**：`viz/flow_render.cpp` 中 `FLOW_VIZ_TEST_PATTERN`

**列采样诊断**：`debug/ob_debug_stats.cpp` 中 `ob_log_col_mean_mag_sample()`

---

## 4. 后续问题：BGR vs RGB 通道顺序

### 4.1 发现

**通道顺序不匹配**：
| 环节 | 通道顺序 | 来源 |
|------|----------|------|
| `run_sram_test.py` 校准 | **BGR** | `cv2.imread()` 默认 BGR |
| `cam_input.cpp` 输出 | **RGB** | `plane_r, plane_g, plane_b` 顺序 |

**影响**：模型量化时用 BGR 校准，推理时收到 RGB，通道错位会导致激活分布异常

### 4.2 验证结果

**BGR 修改后反馈**：显示从灰白条纹变成了黑白条纹，依旧没有光流输出

**结论**：BGR 修改改变了条纹对比度/色调，但垂直条纹模式仍存在，说明问题主要在模型输出本身

---

## 5. 单尺度模型排查

### 5.1 背景

**假设**：AccumPreds（resize_bilinear + add）在 INT8/Vela 下可能引入 8 像素周期伪影

**动作**：新增 `run_sram_test_singlescale.py`，导出单尺度（network_outputs[-1]）模型

### 5.2 结果

**单尺度模型**：条纹消失，但 Preview 为纯白，能看到一点点动态

**结论**：单尺度模型消除了 8 像素条纹，说明 AccumPreds 确为条纹元凶

### 5.3 纯白问题分析

**现象**：单尺度模型输出接近纯白

**原因**：flow magnitude 分布极均匀或 per-frame max 归一化将多数像素映射到高亮

**解决**：改用固定 scale 并降低倍数（如 `kFixedScale=40` 或更低）

---

## 6. 关键文件与配置

**导出脚本**：
- 多尺度：`MCUFlowNet/EdgeFlowNet/sramTest/run_sram_test.py`
- 单尺度：`MCUFlowNet/EdgeFlowNet/sramTest/run_sram_test_singlescale.py`

**校准数据路径**：
- 错误：`np.random.uniform(0.0, 1.0)`
- 正确：`/mnt/d/Dataset/MCUFlowNet/EdgeFlowNet/Datasets/calibration`（真实 Sintel 子集）

**板端代码**：
- `cam_input.cpp`：`CAM_INPUT_USE_BGR` 开关
- `flow_render.cpp`：`FLOW_VIZ_FIXED_SCALE`、`kFixedScale`

