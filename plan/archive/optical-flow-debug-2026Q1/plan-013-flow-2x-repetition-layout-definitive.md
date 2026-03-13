> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 013：光流 2x 水平重复 — 输出布局终极判定

> **状态**: 待执行 | **前置**: Plan-012 D8.2 Attempt 3 已将 Planar 模式开启，修复了"双头"问题

---

## 1. 问题描述

**现象**: 设置 `FLOW_VIZ_OUT_PLANAR=1` 后，静态空间结构正确（不再多头），但**手/头从左到右移动时，运动响应在画面中出现至少 2 次**。相当于原本 3 头变成了 2 次重复，距离被拉长但未消除。

---

## 2. 根因分析

### 2.1 关键事实

| 事实              | 值                               | 来源                |
| ----------------- | -------------------------------- | ------------------- |
| 输出 tensor bytes | 55,296                           | D8.2 Attempt 1 日志 |
| 期望值 H×W×C      | 144 × 192 × 2 = 55,296           | 无 padding ✅        |
| Vela Op Resolver  | `AddTranspose()` + `AddEthosU()` | cvapp L1103         |
| 当前读取模式      | `FLOW_VIZ_OUT_PLANAR=1`          | flow_render.cpp L44 |

### 2.2 Vela Transpose 的含义

Vela 编译模型时，会在 NPU 子图的**输出端**插入一个 `Transpose` op，由 **CPU 执行**。这个 Transpose 的作用是：
- 将 NPU 内部优化的 tensor 排列**转换回标准 NHWC**
- TFLM 报告的 dims `[1, 144, 192, 2]` 是 **Transpose 之后**的逻辑形状

**因此，`output(0)->data.int8` 指向的数据应该已经是 NHWC 交错格式**：
```
[dx(0,0), dy(0,0), dx(0,1), dy(0,1), ..., dx(143,191), dy(143,191)]
```

### 2.3 为什么 Planar 读取看起来"修复了双头"但产生 2x 重复

**假设数据实际是 NHWC**，设 `C=2`：

**NHWC 读取（stride=2，原来的代码）**：
```
dx(pixel_i) = data[i*2 + 0]  → 正确读取 dx
dy(pixel_i) = data[i*2 + 1]  → 正确读取 dy
```
如果这个读取是正确的，那之前看到"双头"的原因不是读取格式错，而是其他原因（如输入撕裂，已被 D8.2 Attempt 2 修复）。

**Planar 读取（当前代码）**：
```
dx(pixel_i) = data[i]              → 读到的是交替的 dx/dy 值
dy(pixel_i) = data[27648 + i]      → 读到的是后半段的交替 dx/dy 值
```
- 像素 0 读 byte 0 = 真实 dx(0,0) ✓
- 像素 1 读 byte 1 = 真实 **dy(0,0)** ✗（应该是 dx(0,1)）
- 像素 2 读 byte 2 = 真实 dx(0,1) ✓
- 像素 3 读 byte 3 = 真实 **dy(0,1)** ✗

**结果**: 每隔一个像素就读错了（dx 和 dy 交替），有效空间分辨率减半 → **水平运动出现 2x 重复**，但静态结构"看起来"正确（因为 dx 和 dy 在边缘处都有响应）。

### 2.4 核心结论

> **H9**: 数据实际上是 **NHWC 交错格式**，不是 Planar。
> - 原来的"双头"是输入撕裂（DMA 竞态）导致的，不是输出布局问题
> - 当 DMA 撕裂和 planar 误读同时存在时，效果叠加看起来像"多头"
> - 修了 DMA 撕裂 + 开了 Planar 读取后，只剩 Planar 误读造成的 2x 重复

---

## 3. 验证方案：D9 终极布局判定

### D9.1: 回到 NHWC 模式（逆转 Planar 修改）

既然 DMA 撕裂已经通过 100ms 延迟修复，现在**只把 Planar 关掉**，回到 NHWC 交错读取：

**修改 `flow_render.cpp`**:
```cpp
// L44: 回到 NHWC 读取
#define FLOW_VIZ_OUT_PLANAR 0
```

**保持不变**:
- `kInterFrameDelayMs = 100`（cam_input.cpp，防止 DMA 撕裂）
- `FLOW_VIZ_GRAY_COMPONENT` 设为 `0`（使用幅值模式，更直观）
- `FLOW_VIZ_REMOVE_GLOBAL_MOTION 1`（恢复全局运动去除）
- `FLOW_VIZ_LIGHT_SMOOTH 1`（恢复平滑）
- `FLOW_VIZ_REMOVE_ROW_BIAS 0`（暂时关闭行偏置）

**烧录命令**（nomodel）:
```bash
cd /home/enmin/Seeed_Grove_Vision_AI_Module_V2
./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode nomodel \
  --app-type optical_cam_oflow \
  --port /dev/ttyACM0 \
  --capture-seconds 30 \
  --extract-frames --max-frames 8
```

### D9.1 执行结果 (2026-02-25)

- **现象**: 手/头移动时，画面中仍出现 **2-3 次** 重复（棋盘格晃动时明显）。
- **清晰度**: 相比之前的 Planar 模式，图像变得**不清晰**了。
- **结论**: 简单切回 NHWC 并没有解决问题，反而清晰度下降。这暗示 NPU 输出的布局可能不是标准的 NHWC，或者在 NHWC 模式下我们读取的步长/偏移仍然不对。

### D9.2 执行结果 (2026-02-25)

- **NHWC 组 (Burn A)**: 
  - **现象**: 画面中直接看到**多个**棋盘格。
  - **清晰度**: 输出**不清晰**，光流估计结果模糊。
- **Planar 组 (Burn B)**: 
  - **现象**: 画面中亮斑更清晰，能清楚看到五官的光流形状。
  - **重复**: 尽管更清晰，但移动棋盘格时仍能看到 **2 个**（距离较远），说明重复问题依然存在。
  - **间距**: 相比 NHWC，重复项之间的物理间距在画面中显得更大。

### 核心结论与后续思考

1. **布局判定**: **Planar 模式在空间对准上明显优于 NHWC**（五官清晰可见）。这证实了之前的布局推断：NPU 输出是分量隔离的，或者 TFLM 后的 Transpose 并没有按照我们预想的交错排列。
2. **残留问题**: 2x 水平重复（或更复杂的倍数重复）不是简单的“交错/平面”误读引起的，而是更深层的 Stride、Padding 或 Vela 编译时的 Tensor 重组问题。
3. **下一步方向**: 需要根据“Planar 更清晰但有重复”这一事实，重新分析 144x192 布局在内存中的真实 Stride，可能存在 $192 \times 2$ 或类似的水平填充/截断。

---

## 4. 操作清单

- [ ] D9.1: `flow_render.cpp` 将 `FLOW_VIZ_OUT_PLANAR` 改回 `0`
- [ ] D9.1: 确认 `kInterFrameDelayMs` 仍为 100ms（cam_input.cpp）
- [ ] D9.1: 恢复渲染参数（GRAY_COMPONENT=0, REMOVE_GLOBAL_MOTION=1, LIGHT_SMOOTH=1）
- [ ] D9.1: nomodel 烧录，**Windows HTML 肉眼观察**手部移动是否仍 2x 重复
- [ ] D9.1: 在 plan-013 记录观察结果
- [ ] （条件）D9.2: NHWC vs Planar 各烧一次，肉眼对比哪个运动只出现 1 次

---

## 5. 关键文件

| 文件                          | 修改内容                        |
| ----------------------------- | ------------------------------- |
| `viz/flow_render.cpp` L44     | `FLOW_VIZ_OUT_PLANAR` → 0       |
| `io/camera/cam_input.cpp` L19 | 确认 `kInterFrameDelayMs = 100` |
