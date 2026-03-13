# Optical Flow Debug Archive 2026Q1

这个目录保存 `2026 Q1` 光流项目的历史调试计划。它们用于追溯推理、显示、量化、UART 和 Windows/WSL 交接过程，但**不应**作为当前事实入口。

在读取任何归档计划前，先看：

1. `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/START_HERE.md`
2. `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MINIMAL_DEPLOYMENT.md`
3. `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/DEPLOYMENT.md`
4. `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-018-optical-flow-project-reorganization.md`

## 状态索引

| 计划 | 状态 | 说明 |
| :--- | :--- | :--- |
| `plan-001` ~ `plan-006` | `reference-only` | 早期搭链和可视化突破，适合看过程，不适合当当前入口。 |
| `plan-007` ~ `plan-014` | `partially-invalid` | 保留了大量有用实验，但其中部分结论已被后续验证推翻。 |
| `plan-015` | `historical-valid` | 反向光流与输入交错问题的关键修复记录，仍有参考价值。 |
| `plan-016` | `partially-invalid` | 内存/延迟分析有价值，但部分“已删除/已完成”状态已过期。 |
| `plan-017` | `reference-only` | UART 瓶颈排查记录，属于局部问题分析。 |

## 使用规则

- 归档计划用于查“为什么当时这么做”。
- 当前部署、当前模型和当前入口，一律以 `docs/` 和 `plan-000` 为准。
- 如果归档内容与当前主线冲突，以 `docs/DEPLOYMENT.md` 和 `plan-018` 为准。
