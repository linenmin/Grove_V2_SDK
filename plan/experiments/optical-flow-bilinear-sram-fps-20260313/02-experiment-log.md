# Experiment Log

## R0. Baseline capture

- 日期：2026-03-13
- 模型：bilinear `172x224 -> 176x224`
- Vela peak：`1386.00 KiB`
- Vela peak op：`ResizeBilinear_1`
- hotspot 组成：
  `Conv53` + `ResizeBilinear/add` + `ResizeBilinear_1/add_1` 并存
- 板端：可启动
- 输出：`resolution = [224, 176]`
- `infer ms`：约 `178.5`
- `total ms`：约 `206.3`
- `algo fps`：约 `4.84`
- 结论：
  当前可运行 bilinear baseline 可作为后续所有改造的比较基准

## R1. Next idea

- 想法：
- 结构改动：
- 是否涉及导出脚本：
- Vela peak：
- Vela peak op：
- 板端是否进入 `initial done`：
- `INVOKE resolution`：
- `infer ms`：
- `total ms`：
- `algo fps`：
- 结论：

## R2. Next idea

- 想法：
- 结构改动：
- 是否涉及导出脚本：
- Vela peak：
- Vela peak op：
- 板端是否进入 `initial done`：
- `INVOKE resolution`：
- `infer ms`：
- `total ms`：
- `algo fps`：
- 结论：
