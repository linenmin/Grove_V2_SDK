# Plan 000：项目全局地图与上下文索引

## 1. 目的

本文件为项目的“单入口”指南。旨在保留所有调试历史细节的同时，提供清晰的当前技术状态地图，确保新会话能快速定位到核心结论。

## 2. 核心里程碑与查阅指南（按需回溯）

| 阶段              | 核心计划                                                                                                                           | 关键结论 / 遗留细节                                                                          |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **A. 基础构建**   | [plan-001~003](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/archive/optical-flow-debug-2026Q1/plan-001-optical-sd-pipeline.md)                            | 确立了 6-channel NHWC 输入结构与 SD 卡存储逻辑。                                             |
| **B. 可视化突破** | [plan-004~006](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/archive/optical-flow-debug-2026Q1/plan-006-optical-cam-oflow-flow-visualization-execution.md) | 解决了 Himax 屏幕驱动与 SPI 协议对齐问题，实现实时预览。                                     |
| **C. 深度分析**   | [plan-012](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/archive/optical-flow-debug-2026Q1/plan-012-flow-stripe-resolution-analysis.md)                    | **[重要]** 确立了 1.9MB SRAM 限制、Scale/Quantization 对齐以及 $144 \times 192$ 分辨率决策。 |
| **D. 根因修复**   | [plan-015](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/archive/optical-flow-debug-2026Q1/plan-015-reverse-flow-fix-and-cleanup.md)                       | 修复了反向光流 Bug（输入交错缺失）、增益设置、以及 Planar vs NHWC 的终极判定。               |
| **E. 优化分析**   | [plan-016](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/archive/optical-flow-debug-2026Q1/plan-016-memory-latency-cleanup.md)                             | **[最新]** 全链路内存地图与优化选项 (A-E)、延迟优化 (L1-L4)、调试遗留清理。                  |
| **F. 项目整理**   | [plan-018](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-018-optical-flow-project-reorganization.md)                | **[当前活动计划]** 当前默认主线已更新为 `157x203 -> 160x208`，继续整理命名、导出脚本、文档入口与历史计划归档。   |

---

## 3. 当前技术快照 (2026-03-13)

### 3.1 核心状态
- **当前主线分辨率**: 输入 $157 \times 203$，输出 $160 \times 208$。
- **实验分辨率状态**: `158x202` 与 `155x206` 虽然通过 Vela 编译，但会在板端 `alloc prev buffer fail`；`150x200 -> 160x208` 会触发可视化 fallback，导致 `INVOKE.image` 回退为相机图。
- **内存占用**: `tensor_arena` = 1432 KiB，Vela 报告峰值 1430 KiB；实际运行上限还受 `prev buffer` 等额外 SRAM 开销约束。
- **运行期内存分账**: 已改为 `frame_buffers -> sensor+other -> viz_buffers -> arena`，其中 `viz` 现在按模型输出尺寸动态分配，属于必需预算项。
- **实机验证**: bilinear `172x224 -> 176x224` 在新预算下可启动，并输出 `INVOKE resolution = [224, 176]` 的光流图。
- **bilinear 峰值热点**: 当前 `172x224` bilinear Vela SRAM peak 位于 decoder 尾段 `ResizeBilinear_1`，峰值 `1386.00 KiB`。
- **bilinear 上机失败对照**: `172x228` 的 Vela 峰值升到 `1485.00 KiB`，板端 `AllocateTensors()` 请求 `1520720 B`，与 Vela 峰值仅差 `80 B` 级别。
- **R1 addskip 结论**: two-stage additive skip 在 `172x224 -> 176x224` 上可完整通过 `Vela + 板端`，但 `SRAM peak` 仍为 `1386.00 KiB`，板端 `infer` 从 `178.513 ms` 变为 `182.055 ms`，算法 FPS 从 `4.846` 降到 `4.765`。
- **R1 addskip 原因归纳**: 变慢主要来自新增 `CONV + PAD + ADD`，不是主 hotspot `ResizeBilinear_1` 的 `Util%` 恶化；该 hotspot 仍约 `6.08%`。
- **168x224 分辨率复验**: `168x224 -> 176x224` 仍保持 `1386.00 KiB` Vela 峰值，baseline 板端 `infer ≈ 177.562 ms`、算法 FPS `≈ 4.876`，略快于 `172x224` baseline。
- **168x224 addskip 结论**: 它并没有消掉 skip padding；Vela 仍保留 `skip_4x_pad` 与 `skip_8x_pad`，板端 `infer ≈ 182.055 ms`、算法 FPS `≈ 4.772`，因此不能作为 addskip 的补救分辨率。
- **输入布局**: **NHWC** (Interleaved)。必须手动交错 `prev` 和 `curr` 帧。
- **输出布局**: **NHWC** (Planar=0)。当前主线量化参数 `scale ≈ 0.407547, zp = -4`。
- **可视化增益**: `mag * 0.05` (避免饱和)。
- **仓库内导出状态**: `scripts/export_optical_flow_144x192.sh` 已切到当前主线默认导出，可直接生成并发布 `157x203` 模型。

### 3.2 关键文件索引
- **核心逻辑**: `pipeline/cvapp_optical_flow.cpp`。
- **渲染算法**: `viz/flow_render.cpp` (控制颜色扩展与幅值计算)。
- **模型导出**: `scripts/export_optical_flow_144x192.sh`。
- **导出逻辑**: `tools/model_export/optical_flow_144x192/`。
- **默认发布模型**: `model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`。

---

## 4. 文档管理方法（方法论）

为了防止信息淹没，我们采用 **“三层结构”**：

1.  **Level 1: 索引档 (本文件)** —— 仅记录“主指针”和“核心结论”。每当一个 Plan 完成，在此更新结论。
2.  **Level 2: 知识库 (KNOWLEDGE_BASE.md)** —— 提取跨 Plan 的通用事实（如寄存器地址、SRAM 布局、硬件坑）。
3.  **Level 3: 实验日志 (plan-0XX.md)** —— 保留所有失败的尝试、原始日志和推导过程。**永远不修改旧日志**，只创建新编号计划。

---

## 5. 接下来操作

- 0-context 接管入口：`docs/START_HERE.md`
- 最小部署路线：`docs/MINIMAL_DEPLOYMENT.md`
- 模型导出入口：`docs/MODEL_EXPORT.md`
- 优先读取：`plan-000-context-index.md` (本文件)
- 当前部署主线：`docs/DEPLOYMENT.md`
- 当前活动计划：`plan-018-optical-flow-project-reorganization.md`
- 当前 bilinear 实验工作区：`plan/experiments/optical-flow-bilinear-sram-fps-20260313/README.md`
- 当前 bilinear file-based 计划区：`plan/optical-flow-bilinear-sram-fps-pi-20260313/README.md`
- 当前模型设计计划区：`plan/model_design/README.md`
- 当前 `R1 addskip` 结果：`plan/model_design/findings.md`
- 历史计划归档索引：`plan/archive/optical-flow-debug-2026Q1/README.md`
- 最近技术背景：`plan/archive/optical-flow-debug-2026Q1/plan-016-memory-latency-cleanup.md`
- 硬件事实查询：`docs/KNOWLEDGE_BASE.md` (Plan 中提取的精华)
