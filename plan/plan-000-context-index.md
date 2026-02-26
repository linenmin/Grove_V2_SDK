# Plan 000：项目全局地图与上下文索引

## 1. 目的

本文件为项目的“单入口”指南。旨在保留所有调试历史细节的同时，提供清晰的当前技术状态地图，确保新会话能快速定位到核心结论。

## 2. 核心里程碑与查阅指南（按需回溯）

| 阶段              | 核心计划                                                                                                                           | 关键结论 / 遗留细节                                                                          |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **A. 基础构建**   | [plan-001~003](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-001-optical-sd-pipeline.md)                            | 确立了 6-channel NHWC 输入结构与 SD 卡存储逻辑。                                             |
| **B. 可视化突破** | [plan-004~006](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-006-optical-cam-oflow-flow-visualization-execution.md) | 解决了 Himax 屏幕驱动与 SPI 协议对齐问题，实现实时预览。                                     |
| **C. 深度分析**   | [plan-012](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-012-flow-stripe-resolution-analysis.md)                    | **[重要]** 确立了 1.9MB SRAM 限制、Scale/Quantization 对齐以及 $144 \times 192$ 分辨率决策。 |
| **D. 根因修复**   | [plan-015](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-015-reverse-flow-fix-and-cleanup.md)                       | 修复了反向光流 Bug（输入交错缺失）、增益设置、以及 Planar vs NHWC 的终极判定。               |
| **E. 优化分析**   | [plan-016](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-016-memory-latency-cleanup.md)                             | **[最新]** 全链路内存地图与优化选项 (A-E)、延迟优化 (L1-L4)、调试遗留清理。                  |

---

## 3. 当前技术快照 (2026-02-26)

### 3.1 核心状态
- **当前分辨率**: $144 \times 192$ (4:3) —— 正在测试 $150 \times 200$。
- **内存占用**: `tensor_arena` = 1432 KiB (1188 KiB peak)。
- **输入布局**: **NHWC** (Interleaved)。必须手动交错 `prev` 和 `curr` 帧。
- **输出布局**: **NHWC** (Planar=0)。量化参数 `scale ≈ 0.5, zp = -1`。
- **可视化增益**: `mag * 0.05` (避免饱和)。

### 3.2 关键文件索引
- **核心逻辑**: `pipeline/cvapp_yolov8n_ob.cpp` (包含了 `interleave` 修复)。
- **渲染算法**: `viz/flow_render.cpp` (控制颜色扩展与幅值计算)。
- **模型导出**: `EdgeFlowNet/sramTest/run_sram_test.py` (包含代表性数据集生成)。

---

## 4. 文档管理方法（方法论）

为了防止信息淹没，我们采用 **“三层结构”**：

1.  **Level 1: 索引档 (本文件)** —— 仅记录“主指针”和“核心结论”。每当一个 Plan 完成，在此更新结论。
2.  **Level 2: 知识库 (KNOWLEDGE_BASE.md)** —— 提取跨 Plan 的通用事实（如寄存器地址、SRAM 布局、硬件坑）。
3.  **Level 3: 实验日志 (plan-0XX.md)** —— 保留所有失败的尝试、原始日志和推导过程。**永远不修改旧日志**，只创建新编号计划。

---

## 5. 接下来操作

- 优先读取：`plan-000-context-index.md` (本文件)
- 最新进展：`plan-016-memory-latency-cleanup.md`
- 硬件事实查询：`docs/KNOWLEDGE_BASE.md` (Plan 中提取的精华)
