# Plan Workspace README

这个目录用于承载当前项目的**治理入口、执行计划和历史调试归档**，不替代仓库根目录原始 `README.md`。

## 当前优先阅读顺序

如果你是 **0-context agent**，严格按这个顺序看：

1. [docs/START_HERE.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/START_HERE.md)
2. [docs/MINIMAL_DEPLOYMENT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MINIMAL_DEPLOYMENT.md)
3. [plan-000-context-index.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-000-context-index.md)
4. [docs/DEPLOYMENT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/DEPLOYMENT.md)
5. [docs/MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)
6. [plan-018-optical-flow-project-reorganization.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-018-optical-flow-project-reorganization.md)
7. [docs/KNOWLEDGE_BASE.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/KNOWLEDGE_BASE.md)

## 当前主线事实

- 当前有效部署基线是 `157x203 -> 160x208` 光流模型。
- `158x202` 及更大输入在当前 `1432 KiB` arena 下会撞上运行期内存边界，不作为默认主线。
- `144x192` 与 `150x200 -> 160x208` 仅保留为历史记录，不作为默认主线。
- 本目录中的计划文档负责区分“当前事实”和“历史调试经验”。

## 目录职责

- `plan-000-context-index.md`
  当前总索引与主线地图。
- `plan-018-optical-flow-project-reorganization.md`
  当前整理计划，定义命名、导出脚本内聚、历史归档等执行路线。
- `docs/DEPLOYMENT.md`
  当前唯一有效部署主线说明。
- `docs/MODEL_EXPORT.md`
  当前 `157x203` 模型导出入口与环境边界。
- `docs/MINIMAL_DEPLOYMENT.md`
  最短可工作部署路径。
- `archive/`
  已用于存放已完成或已过期的历史调试计划；索引见 `archive/optical-flow-debug-2026Q1/README.md`。

## 规则

- 根目录 `README.md` 保持原 Seeed 仓库语义，不在本轮整理中改写。
- 与当前事实冲突的旧计划已迁入归档区并标注为历史记录。
- 新会话若只需要快速接管项目，优先读本目录，不必先翻完整历史调试日志。
