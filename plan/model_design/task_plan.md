# Task Plan: Bilinear Optical Flow SRAM/FPS Iteration

## Goal

基于当前 Ethos-U55/Vela 支持算子与 bilinear baseline，筛选并按优先级验证最值得做的模型结构改造，只比较 `Vela SRAM peak` 与板端 `推理 FPS`。

## Current Phase

Phase 5

## Phases

### Phase 1: Requirements & Discovery

- [x] 理解用户目标与约束
- [x] 读取项目入口文档与现有 bilinear 记录
- [x] 固定当前 baseline 与验证口径
- **Status:** complete

### Phase 2: Planning Workspace Setup

- [x] 安装 `pi-planning-with-files` skill 到本机与仓库
- [x] 创建基于该 skill 的文件化计划目录
- [x] 写入 `task_plan.md` / `findings.md` / `progress.md`
- [x] 把该目录接入项目索引
- **Status:** complete

### Phase 3: Baseline Verification

- [x] 固化当前 bilinear baseline 的 Vela 峰值结论
- [x] 固化当前 bilinear baseline 的板端 `infer ms` / `FPS`
- [x] 明确失败对照版本与失败原因
- **Status:** complete

### Phase 4: Supported-Op Feasibility Review

- [x] 复制 `SUPPORTED_OPS.md` 到当前仓库导出目录
- [x] 判断推荐的结构想法是否在算子层面可行
- [x] 输出推荐验证顺序
- **Status:** complete

### Phase 5: Iterative Idea Validation

- [x] `R1` two-stage additive skip
- [x] `R1` 先完成 Vela 侧验证
- [x] `R1` 再完成板端验证
- [x] `R1` 结果写入 `findings.md` 与 `progress.md`
- [x] `R2` bottleneck Lite ASPP
- [x] `R3` ECA channel attention
- **Status:** in_progress

### Phase 6: Comparative Decision Log

- [ ] 按轮次整理保留/放弃/待复验的想法
- [ ] 标出当前最佳 `SRAM peak` / `FPS` 候选
- [ ] 准备下一轮实验输入
- **Status:** pending

## Key Questions

1. 哪些推荐 idea 在当前 Ethos-U55 支持算子约束下可以直接进入实验？
2. 哪种 idea 最可能提升 acc，同时不显著抬高 decoder 末段 `SRAM peak`？
3. 第一轮验证是先做 `skip`、`Lite ASPP` 还是 `channel attention`？

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 计划文件放在 `plan/` 子目录而不是项目根目录 | 用户明确要求在 `plan/` 下创建子文件夹 |
| 继续保留旧 `plan/experiments/...` 目录 | 它已经承载 baseline 原始记录，可作为实验事实来源 |
| 当前实验先不看准确率 | 用户明确要求只比较 `SRAM peak` 与 `FPS` |
| 每轮实验只改一个主变量 | 避免结构改动之间相互污染结论 |
| 支持算子文档放到 `tools/model_export/optical_flow_144x192/vela/` | 这是最直接服务模型导出和结构设计判断的位置 |
| 第一轮 idea 顺序定为 `additive skip -> Lite ASPP -> ECA` | 在表达力提升与 SRAM 风险之间最平衡 |
| `R1 addskip` 不直接推进为当前最佳候选 | 虽然 SRAM peak 持平，但板端 infer/FPS 小幅变差，暂不符合当前优先级 |
| `168x224` 先作为 baseline 候选，不作为 addskip 挽救方案 | 它保持 `4:3` 且 baseline 略快，但仍不能消掉 addskip padding，也没有改善 addskip 的 SRAM/FPS |
| `R2 Lite ASPP` 标记为“可行但不保留” | 它保持 `SRAM peak` 不变，但会增加 off-chip flash 和瓶颈卷积成本，导致板端 FPS 继续下降 |
| `R3 ECA` 进入当前优先保留列表 | 它保持 `SRAM peak` 不变，off-chip flash 只小幅增加，板端 infer/total 仅比 baseline 略慢 |
| `globalgate4x` 升级为当前最佳候选 | 它利用跨层全局向量广播实现高层语义调制，在保持 `SRAM peak` 不变的同时，比 `R3 ECA` 更轻 |
| `globalgate2x` 不升级为最佳候选 | 它证明更高尺度门控仍然安全，但 `1/2` 门控的 `MUL` 明显更重，板端略差于 `globalgate4x` |
| `globalgate4x_eca` 标记为“组合可行但不保留” | 它保持 `SRAM peak` 不变，但在 `1/4` 阶段叠加双门控后，板端时延已明显劣化 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `pi-planning-with-files` 不在当前可用 skill 列表中 | 1 | 按用户要求先把 skill 复制到 `~/.codex/skills` 与仓库 `.cursor/skills`，再读取 `SKILL.md` |
| `R1 addskip` 在 `172x224` 上出现 `/4` skip shape mismatch | 1 | 在 skip 分支加 `PAD` 解决，避免改动主干上采样几何 |
| 用户希望 `168x224` 通过几何对齐消掉 addskip 的 `PAD` | 1 | 已完成 `168x224 baseline/addskip` 验证，确认 Vela 仍保留 `skip_4x_pad` 与 `skip_8x_pad` |
| `R2 Lite ASPP` 可能通过 dilation 提升表达力，但存在延迟风险 | 1 | 已完成 `172x224` 验证，确认主要代价来自 bottleneck dilated conv，而不是尾部 hotspot |
| `R3 ECA` 需要保证当前导出图不走 transpose-heavy channel mixing | 1 | 用 reshape-based 1D channel conv 实现，成功保留在当前 Vela/TFLite 路径内 |
| 跨层全局 gate 需要验证是否真的优于层内 `ECA` | 1 | 已完成 `globalgate4x` 验证，确认它在 `Vela` 和板端都略优于 `R3 ECA` |
| `globalgate2x` 需要确认更高尺度门控是否仍符合当前 `FPS` 约束 | 1 | 已完成验证，确认峰值不变，但 `1/2` `MUL` 成本偏高 |
| `globalgate4x_eca` 需要确认层内门控与跨层门控能否安全叠加 | 1 | 已完成验证，确认不会抬高峰值，但不值得保留 |

## Notes

- 当前 baseline 与 Vela 报告详见
  [00-current-bilinear-baseline.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/experiments/optical-flow-bilinear-sram-fps-20260313/00-current-bilinear-baseline.md)
- 当前 idea 可行性与顺序详见
  [idea-feasibility-and-order.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/idea-feasibility-and-order.md)
- `R1 addskip` 结果目录：
  [output_bilinear_addskip/172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_addskip/172x224)
- `R1 addskip` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_181740.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_181740.log)
- `168x224` baseline 结果目录：
  [168x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/168x224)
- `168x224` addskip 结果目录：
  [168x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_addskip/168x224)
- `168x224` baseline 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_182747.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_182747.log)
- `168x224` addskip 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_183003.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_183003.log)
- `R2 Lite ASPP` 结果目录：
  [172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_liteaspp/172x224)
- `R2 Lite ASPP` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_184001.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184001.log)
- `R3 ECA` 结果目录：
  [172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_eca/172x224)
- `R3 ECA` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_184842.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184842.log)
- `globalgate4x` 结果目录：
  [172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x/172x224)
- `globalgate4x` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_193049.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193049.log)
- `globalgate2x` 结果目录：
  [172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate2x/172x224)
- `globalgate2x` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_193850.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193850.log)
- `globalgate4x_eca` 结果目录：
  [172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_eca/172x224)
- `globalgate4x_eca` 上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_194450.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_194450.log)
- 若改动涉及导出逻辑，回看
  [MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)
- 每轮实验必须先写 Vela 侧结论，再写板端结论
