> 历史材料：保留当时记录，当前部署按 docs/START_HERE.md；旧工具路径仅作追溯。

# Start Here

> 2026-09-24 Windows 接管：先读 [连接核查与协作方案](CONNECTION_AUDIT_2026-09.md)。下文 157×203 为历史模型基线，不能代表当前板端版本；Windows checkout 的 `.agent/skills` 链接可能不可用。连接检查可运行 `scripts/grove_probe.ps1`。新的构建/烧录工作流尚未验收。

如果你是第一次接手这个仓库，或者你是一个**0-context agent**，请按下面顺序读取：

1. [plan/README.md](../optical-flow/README.md)
2. [docs/MINIMAL_DEPLOYMENT.md](../../MINIMAL_DEPLOYMENT.md)
3. [docs/DEPLOYMENT.md](../../DEPLOYMENT.md)
4. [docs/MODEL_EXPORT.md](../../MODEL_EXPORT.md)
5. [docs/KNOWLEDGE_BASE.md](../../KNOWLEDGE_BASE.md)
6. [plan/plan-018-optical-flow-project-reorganization.md](../optical-flow/plan-018-optical-flow-project-reorganization.md)

## 当前只认这条主线

- 当前有效部署基线是 `157x203 -> 160x208` 光流模型。
- 当前有效可视化结果是 `INVOKE resolution = [208, 160]` 且输出为光流图。
- `158x202` 及更大输入在当前 `1432 KiB` arena 下仍可能通过 Vela 编译，但板端会在 `prev buffer` 分配阶段失败，不作为默认部署主线。
- `150x200 -> 160x208` 只保留为早期实验记录，不作为默认部署主线。

## 不要先做的事

- 不要先相信旧 `plan` 中所有结论都还有效。
- 不要把旧 `144x192` 或 `150x200` 模型当成默认模型。
- 不要先改根目录 `README.md`，它保持 Seeed 上游语义。

## 你要找什么文档

- 只想最快部署一次：
  看 [docs/MINIMAL_DEPLOYMENT.md](../../MINIMAL_DEPLOYMENT.md)
- 想知道当前主线到底是什么：
  看 [docs/DEPLOYMENT.md](../../DEPLOYMENT.md)
- 想重新导出当前主线模型：
  看 [docs/MODEL_EXPORT.md](../../MODEL_EXPORT.md)
- 想知道硬事实和坑：
  看 [docs/KNOWLEDGE_BASE.md](../../KNOWLEDGE_BASE.md)
- 想知道后续整理怎么推进：
  看 [plan/plan-018-optical-flow-project-reorganization.md](../optical-flow/plan-018-optical-flow-project-reorganization.md)
