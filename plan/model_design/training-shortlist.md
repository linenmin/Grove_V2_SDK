# Training Shortlist

## Current Rule

- 当前进入轻量训练 shortlist 的前提：
  - `SRAM peak` 不高于当前 bilinear baseline 的 `1386.00 KiB`
  - `Vela inference_time` 明显低于用户当前容忍线 `+20%`
  - 结构尽量沿着同一升级主线生长，避免训练分支过多

## Primary Shortlist

1. `globalgate4x`
   - 最接近 baseline 的语义增强版本
   - `Vela inference_time ≈ 174.358 ms`
2. `globalgate4x_bneckeca`
   - 在当前最佳点上补一个 bottleneck-only `ECA`
   - `Vela inference_time ≈ 174.657 ms`
3. `globalgate4x_bneckeca_skip8x`
   - 当前最划算的单 skip 升级
   - `Vela inference_time ≈ 175.778 ms`
4. `globalgate4x_bneckeca_skip8x4x`
   - 当前最像轻量 U-Net 的多尺度候选
   - `Vela inference_time ≈ 178.319 ms`

## Extended Candidate

5. `globalgate4x_bneckeca_skip8x4x2x`
   - 三尺度长跳跃同时存在的上限探索版
   - `Vela inference_time ≈ 182.915 ms`
   - 按之前更保守的口径它不进训练优先列表；按用户当前阈值，它可以保留做“多尺度上限”参照

## Export Note

- 这些 bilinear 变体现在都可以通过：
  - [run_export.py](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/run_export.py)
  - [export_optical_flow_144x192.sh](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh)
- 导出时必须提供和结构完全匹配的 checkpoint
