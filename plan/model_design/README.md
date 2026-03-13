# PI Planning Workspace: Bilinear SRAM/FPS

## Purpose

这个目录按 `pi-planning-with-files` 的结构组织，用于持续推进 bilinear 光流模型实验。

当前实验目标固定为：

- 先看 `Vela` 输出
- 再看上板部署输出
- 先不看准确率
- 只比较：
  - `SRAM peak`
  - `推理 FPS`

## Files

- [task_plan.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/task_plan.md)
- [findings.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/findings.md)
- [progress.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/progress.md)
- [idea-feasibility-and-order.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/idea-feasibility-and-order.md)
- [training-shortlist.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/training-shortlist.md)
- [fixed-arch-training-plan-20260314.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/model_design/fixed-arch-training-plan-20260314.md)

## Related Context

- 原始 bilinear baseline 记录：
  [00-current-bilinear-baseline.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/experiments/optical-flow-bilinear-sram-fps-20260313/00-current-bilinear-baseline.md)
- 固定验证流程：
  [01-validation-protocol.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/experiments/optical-flow-bilinear-sram-fps-20260313/01-validation-protocol.md)
- 当前项目总索引：
  [plan-000-context-index.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-000-context-index.md)
- 当前支持算子参考：
  [SUPPORTED_OPS.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md)

## Current Baseline

- 当前板上可运行 bilinear baseline:
  `172x224 -> 176x224`
- 当前 Vela peak hotspot:
  `ResizeBilinear_1`
- baseline Vela peak:
  `1386.00 KiB`
- baseline board infer:
  `~178.5 ms`
- baseline algorithm FPS:
  `~4.84`

## Latest Iteration

- `R1 = two-stage additive skip`
- Vela SRAM peak:
  `1386.00 KiB`，与 baseline 持平
- Vela peak hotspot:
  仍然是 `ResizeBilinear_1`
- board infer:
  `~182.1 ms`
- board algorithm FPS:
  `~4.77`
- current decision:
  结构可跑通，但在 `SRAM peak` 不变的前提下带来约 `2%` 推理变慢，先记为“可行但非优先保留”
