# Model Design Idea Feasibility And Order

## Goal

在当前 WE2 + Ethos-U55 + Vela 约束下，筛出**优先值得验证**的 bilinear encoder-decoder 结构改造方向，并明确它们对 `SRAM peak` 与 `FPS` 的预期风险。

## Current Architecture Constraints

- 当前 bilinear 主体是纯 encoder-decoder 结构。
- decoder 逐级使用 `ResizeConv`，当前没有显式的 encoder-to-decoder 长跳连。
- 当前 `SRAM peak` hotspot 位于 decoder 尾段 `ResizeBilinear_1`，不是最慢卷积本身。

代码锚点：

- bilinear 网络：
  [MultiScaleResNet_bilinear.py](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear.py)
- Vela 支持算子表：
  [SUPPORTED_OPS.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md)

## Operator Feasibility Summary

基于
[SUPPORTED_OPS.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md)
中的 Ethos-U55/U65 约束，和当前 TFLite/NHWC 路线，以下判断成立：

- `Additive skip`：
  **可行**
  依赖 `CONV_2D` + `ADD`，都在支持列表中。
- `Concat skip`：
  **算子层面可行**
  依赖 `CONCATENATION`，支持；但运行期 peak 风险更高。
- `Lite ASPP / atrous context`：
  **可行**
  `CONV_2D` 支持 dilation，3x3 + dilation 2/4 都远在约束范围内。
- `ECA / 轻量通道注意力`：
  **大概率可行**
  依赖 `MEAN` + `RESHAPE` + `LOGISTIC` + `MUL`，都在支持列表中。
  但是否完全留在 NPU，还要看导出后的 TFLite lowering。
- `Spatial attention / CBAM full`：
  **算子上未必完全不可行，但不推荐作为第一轮**
  主要问题不是算子表，而是高分辨率特征上的额外 buffer 与 latency 风险。

## Recommended Order

### 1. Two-Stage Additive Skip

**定义**

- 只在低/中分辨率两级加入 skip。
- encoder feature 先做 `1x1 conv` 对齐通道，再与 decoder 做 `add`。
- 不做最后一级高分辨率 skip。

**为什么先做**

- 当前网络缺的就是长跳连。
- 对 encoder-decoder dense prediction，skip 往往是最直接的精度增益项。
- 用 `add` 代替 `concat`，是当前内存预算下最稳的版本。

**优点**

- 最有希望带来明显 acc 提升。
- 实现简单，改动可控。
- 相比 concat，SRAM 增量小很多。

**缺点**

- 需要保留 encoder 中间特征，运行期仍会增加一部分 feature map 生命周期。
- 若 skip 接到高分辨率 stage，peak 可能明显上升。

**算子可行性**

- `CONV_2D`: supported
- `ADD`: supported

**实验优先级**

- `P0`

### 2. Bottleneck Lite ASPP

**定义**

- 只在 bottleneck 插 2 到 3 个 dilation branch。
- 推荐从 dilation `1/2/4` 开始。
- branch 输出再用 `1x1 conv` 融合。

**为什么第二个做**

- 光流对上下文和感受野敏感。
- bottleneck 分辨率最低，在这里加 context 模块对 SRAM 最友好。
- 比在高分辨率 decoder 加复杂模块更稳。

**优点**

- 有望改善大位移、弱纹理、歧义区域。
- 分支开销主要发生在低分辨率层。
- 不直接触碰当前 `ResizeBilinear_1` 热点位置。

**缺点**

- 分支太多会抬高 latency。
- 若导出图构图不理想，Vela 调度可能不如纯串行块稳定。

**算子可行性**

- `CONV_2D` with dilation: supported
- `CONCATENATION` or `ADD` for fusion: supported

**实验优先级**

- `P1`

### 3. ECA-Style Channel Attention

**定义**

- 只在 bottleneck 和 decoder 前两级插轻量 channel attention。
- 不在最终高分辨率输出级插 attention。

**为什么第三个做**

- 它是低风险增益项。
- 比 full SE/CBAM 更轻，更适合当前 SRAM/FPS 优先的实验目标。

**优点**

- 参数量和额外激活较小。
- 对 `SRAM peak` 风险最小。
- 可以和前两类结构配合。

**缺点**

- 单独带来的 acc 提升通常不如 skip / context 明显。
- TFLite lowering 细节要实测，不能只凭算子表下结论。

**算子可行性**

- `MEAN`: supported
- `RESHAPE`: supported
- `LOGISTIC`: supported
- `MUL`: supported

**实验优先级**

- `P2`

## Not Recommended First

- 全尺度 `concat` skip
- 最后一级高分辨率 spatial attention
- full CBAM
- transformer / self-attention
- 任何会在 `ResizeBilinear_1` 附近再复制一份大 feature map 的设计

## Validation Rule

每个 idea 都必须按这个顺序验证：

1. 导出模型
2. 读 `Vela summary`
3. 读 `per-layer`
4. 读 `detailed allocation`
5. 上板
6. 只记录：
   - `SRAM peak`
   - `infer ms`
   - `total ms`
   - `algorithm FPS`
7. 暂不评估准确率
