---
name: project-governance
description: 项目治理与文档更新协议 (Index & Knowledge Base Update Policy). Use when creating new plans or after major technical milestones to ensure plan-000 and KNOWLEDGE_BASE are in sync.
---

# 项目治理协议 (Project Governance)

为了保持项目上下文的高度一致性，避免 AI Agent 在长期调试中迷失，必须严格执行以下文档更新流程。

## 1. 计划创建限制 (Plan Creation Policy)

- **禁止自主创建**: **禁止**在没有用户明确指示的情况下自行创建新的 `plan/plan-0XX.md` 文件。
- **用户授权**: 仅当用户明确要求“创建一个新计划”或“开启新篇章”时，才启动 Plan 创建流程。

## 2. 更新动作 (Mandatory Actions)

### A. 同步更新 (Synchronized Updates)
一旦根据用户指示创建了新的 Plan：
1.  **更新索引 ([plan-000-context-index.md](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-000-context-index.md))**: 立即将新 Plan 的编号、日期及预期分析目标加入“当前活动上下文”。
2.  **同步萃取 ([docs/KNOWLEDGE_BASE.md](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/KNOWLEDGE_BASE.md))**: 在 Plan 的实验段落（R1, R2...）结束后，必须检查是否有新的“硬事实”需要同步到知识库。

### B. 刷新快照
- **执行**: `bash scripts/build_context_snapshot.sh`
- **目的**: 确保下一轮会话能准确识别最新的文件结构。

## 3. 文档风格约束

- **简洁优先**: 禁止在大地图/知识库中贴入大段代码。
- **链接完整**: 引用文件时必须使用完整的 `file:///` 绝对路径链接。
- **结论导向**: 重点记录“什么是正确的”，而不是“为什么失败”（失败的过程由 Level 3 的 Plan 日志保留）。
