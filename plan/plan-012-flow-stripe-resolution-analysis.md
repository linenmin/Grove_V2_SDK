# Plan 012：光流条纹问题与分辨率分析

> **状态**: 分析中 | **关键发现**: 输入输出分辨率不一致

---

## 1. 问题摘要

**现象**: 彩色/灰度光流都看不出棋盘格形状，回到条纹输出

**根因**: 模型输入 (200×150) 与输出 (208×160) 分辨率不匹配

---

## 2. 分辨率链路分析

| 阶段 | 分辨率 | 处理 | 问题 |
|------|--------|------|------|
| 摄像头原始 | 640×480 | MIPI | - |
| 模型输入 | **200×150×6** | Helium resize | ⚠️ 非 4:3 |
| 模型输出 | **208×160×2** | NPU 推理 | ⚠️ 与输入不同 |
| 光流可视化 | 208×160 | 直接使用 | - |

**关键问题**:
- 输入 200×150 ≠ 输出 208×160
- 差值: W+8, H+10 (约 4-7% 放大)

---

## 3. 根因分析

### 3.1 网络架构问题

`MultiScaleResNet` 多尺度输出结构：
```
输入 200×150
  ↓ Encoder (下采样)
  ↓ Decoder (上采样 with ConvTranspose)
  → 输出1: ~200×150 (strides=1,1)
  → 输出2: ~204×154 (ConvTransposeBNReLUBlock 默认 strides=2,2)
  → 输出3: ~208×160 (再次上采样)
  ↓ AccumPreds resize_bilinear 累加
最终输出: 208×160 (取最大尺度)
```

**问题代码** (`BaseLayers.py` L48-58):
```python
def ConvTransposeBNReLUBlock(..., strides=None, ...):
    if strides is None:
        strides = self.strides  # 默认 (2,2)!
    ...
```

### 3.2 AccumPreds 累加

`misc/utils.py` L45-58:
```python
def AccumPreds(prVals):
    for prVali in prVals:
        prValAccum = tf.image.resize_bilinear(prValAccum, [prVali.shape[1], prVali.shape[2]])
        prValAccum += prVali
    return prValAccum  # 尺寸 = 最大输出的尺寸 = 208×160
```

---

## 4. 解决方案

### 方案 A: 修复网络架构 (推荐)

修改 `MultiScaleResNet.py` 确保所有 ConvTranspose 使用 `strides=(1,1)`:
```python
# L127: 添加 strides=(1,1)
Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters, 
                                     kernel_size=(5,5), strides=(1,1))
# L135: 添加 strides=(1,1)  
Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters,
                                     kernel_size=(7,7), strides=(1,1))
```

### 方案 B: 降低分辨率到 192×144 (4:3)

**优势**:
- 保持 4:3 比例 (与摄像头 640×480 一致)
- 减少内存: 200×150=30KPix → 192×144=27.6KPix (-8%)
- prev buffer: 90KB → 82.8KB (-7.2KB)

**计算**:
```
192×144×3×2 = 165,888 bytes ≈ 162KB (两帧 RGB)
vs
200×150×3×2 = 180,000 bytes = 175.8KB (当前)
节省: ~14KB
```

### 方案 C: 单尺度输出

绕过 AccumPreds，直接使用最高分辨率输出：
```python
# run_sram_test.py
final_output = network_outputs[0][..., 0:2]  # 直接取第一个输出
```

---

## 5. Plan 011 关键调试记录 (压缩)

### 内存约束
- SRAM 总量: 1,920 KB
- tensor_arena: 1,432 KB (多尺度 vela)
- prev buffer: 90 KB
- curr buffer: 90 KB
- JPEG buffer: 24 KB (灰度) / 48 KB (RGB)
- 余量: ~158 KB

### RGB 光流实现
- **分块渲染**: 8 行一块，buffer 仅 5KB
- **函数**: `flow_render_rgb_to_jpeg_block()`
- **颜色编码**: Hue=方向, Value=幅度

### RGB JPEG buffer 问题
- **方案 A 失败**: 48KB buffer 导致 prev buffer 分配失败
- **方案 B 采用**: RGB 失败时回退灰度
- **效果**: 静止/小运动=彩色, 快速运动=灰度

### 关键配置
```c
// cvapp_yolov8n_ob.cpp
FLOW_VIZ_RGB_OUTPUT = 1
FLOW_TENSOR_ARENA_KB = 1432
kFlowVizGrayJpegBufSize = 24576  // 24KB
kFlowVizRgbBlockSize = 5376      // 5KB (8行×224×3)
```

---

## 6. 执行结果

**R1: 192×144 模型生成成功**
- 分辨率: 192×144 (4:3 比例)
- SRAM 占用: 1.160 MB (之前 200×150 约 1.43 MB)
- 推理时间: 139.84 ms
- 内存节省: ~270 KB

**日志确认**:
```
[out_hist] ch0 top=-87(3.2%) second=-90(3.2%) near_max=106(0.3%)
resolution: [192, 144]
```

**输出直方图分析**:
- 之前 (200×150): top=-114(28.5%) - 集中在常量值
- 现在 (192×144): top=-87(3.2%) - 分布更分散

**问题状态**: 
- ✅ 分辨率统一为 192×144
- ⚠️ 光流输出仍看不出棋盘格形状
- 可能原因: 模型本身的问题，而非分辨率

**Windows HTML 验证**:
- 请刷新页面观察 192×144 分辨率下的光流输出
- 检查是否有彩色光流显示

---

## 7. 验证命令

```bash
# 生成新模型
cd /home/enmin/MCUFlowNet/EdgeFlowNet/sramTest
python run_sram_test.py

# 烧录测试
cd /home/enmin/Seeed_Grove_Vision_AI_Module_V2
./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode with-model \
  --app-type optical_cam_oflow \
  --port /dev/ttyACM0 \
  --model-arg "/home/enmin/MCUFlowNet/EdgeFlowNet/sramTest/output/sram_test_modified_vela.tflite 0xB7B000 0x00000" \
  --capture-seconds 30 \
  --extract-frames --max-frames 8
```
