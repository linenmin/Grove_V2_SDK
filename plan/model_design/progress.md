# Progress Log

## Session: 2026-03-13

### Phase 1: Requirements & Discovery

- **Status:** complete
- **Started:** 2026-03-13
- Actions taken:
  - 按用户指定顺序阅读 `START_HERE.md`、`MINIMAL_DEPLOYMENT.md`、`plan-000-context-index.md`、`plan-018`
  - 读取项目 `AGENTS.md` 与 `project-governance` skill
  - 定位 bilinear 模型导出脚本、Vela summary、per-layer、detailed allocation、板端日志
  - 提取 `172x224` 与 `172x228` 的峰值和板端边界
- Files created/modified:
  - `plan/experiments/optical-flow-bilinear-sram-fps-20260313/*`
  - `plan/plan-000-context-index.md`
  - `docs/KNOWLEDGE_BASE.md`

### Phase 2: Planning Workspace Setup

- **Status:** complete
- Actions taken:
  - 将 `pi-planning-with-files` 从 Windows 路径复制到 `~/.codex/skills`
  - 将同一 skill 复制到仓库 `.cursor/skills`，供 `.agent/skills` 使用
  - 读取 `SKILL.md` 与模板文件
  - 在 `plan/` 下创建新的 file-based 计划目录
  - 写入 `README.md`、`task_plan.md`、`findings.md`、`progress.md`
- Files created/modified:
  - `/home/enmin/.codex/skills/pi-planning-with-files/*`
  - `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.cursor/skills/pi-planning-with-files/*`
  - `plan/optical-flow-bilinear-sram-fps-pi-20260313/README.md`
  - `plan/optical-flow-bilinear-sram-fps-pi-20260313/task_plan.md`
  - `plan/optical-flow-bilinear-sram-fps-pi-20260313/findings.md`
  - `plan/optical-flow-bilinear-sram-fps-pi-20260313/progress.md`

### Phase 3: Baseline Verification

- **Status:** complete
- Actions taken:
  - 确认 `172x224` 的 Vela peak op 是 `ResizeBilinear_1`
  - 确认 `172x228` 的板端失败值与 Vela 峰值高度一致
  - 确认 `172x224` 的板端算法 FPS 约 `4.84`
- Files created/modified:
  - `plan/experiments/optical-flow-bilinear-sram-fps-20260313/00-current-bilinear-baseline.md`

### Phase 4: Supported-Op Feasibility Review

- **Status:** complete
- Actions taken:
  - 从 `/home/enmin/MCUFlowNet/EdgeFlowNet/SUPPORTED_OPS.md` 读取 Ethos-U55/U65 支持算子与约束
  - 复制该文档到仓库导出目录 `tools/model_export/optical_flow_144x192/vela/`
  - 对 `additive skip`、`Lite ASPP`、`ECA` 三个方向做算子层面可行性判断
  - 写入模型设计优先级与验证顺序文档
- Files created/modified:
  - `tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md`
  - `tools/model_export/optical_flow_144x192/README.md`
  - `plan/model_design/idea-feasibility-and-order.md`
  - `plan/model_design/README.md`
  - `plan/model_design/task_plan.md`
  - `plan/model_design/findings.md`
  - `plan/model_design/progress.md`

### Phase 5: Iterative Idea Validation

- **Status:** in_progress
- Actions taken:
  - 为 `run_sram_test_bilinear.py` 增加 `--variant`，支持 `baseline` / `addskip`
  - 新增独立模型文件 `network/MultiScaleResNet_bilinear_addskip.py`
  - 在 `/8` 与 `/4` 两级接入 additive skip，使用 `1x1 conv + add`
  - 发现 `172x224` 的 `/4` skip 存在 `43x56 -> 44x56` 的奇数尺寸错位
  - 用 skip 分支 `PAD` 补齐一行，避免引入非整数 resize
  - 完成 `R1 addskip` 的 `Vela -> 板端` 全流程验证
  - 按用户新假设补做 `168x224` baseline 与 `168x224 addskip` 的 `Vela -> 板端` 对照
  - 确认 `168x224` 仍不能消掉 `skip_4x_pad` / `skip_8x_pad`
  - 确认 `168x224` baseline 略快于 `172x224` baseline，但 `168x224 addskip` 仍然慢于 baseline
  - 新增 `network/MultiScaleResNet_bilinear_liteaspp.py`，在 bottleneck 接入 `1x1 + dilation 2/4 + residual add`
  - 为 `run_sram_test_bilinear.py` 增加 `liteaspp` 变体映射
  - 完成 `R2 Lite ASPP` 的 `172x224` Vela 与板端验证
  - 确认 `R2 Lite ASPP` 不增加 `SRAM peak`，但会明显增加瓶颈分支成本与实际推理时延
  - 新增 `network/MultiScaleResNet_bilinear_eca.py`，实现 `MEAN + RESHAPE + LOGISTIC + MUL` 的 `ECA-style` channel attention
  - 为 `run_sram_test_bilinear.py` 增加 `eca` 变体映射
  - 完成 `R3 ECA` 的 `172x224` Vela 与板端验证
  - 确认 `R3 ECA` 保持 `SRAM peak` 不变，且实际时延增量远小于 `R1 addskip` 与 `R2 Lite ASPP`
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x.py`，实现 `bottleneck global vector -> decoder 1/4 gate`
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x` 变体映射
  - 完成 `globalgate4x` 的 `172x224` Vela 与板端验证
  - 确认跨层全局向量广播比 `R3 ECA` 还略轻，是当前最佳候选
  - 新增 `network/MultiScaleResNet_bilinear_globalgate2x.py`，实现 `bottleneck global vector -> decoder 1/2 gate`
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate2x` 变体映射
  - 完成 `globalgate2x` 的 `172x224` Vela 与板端验证
  - 确认更高尺度 `1/2` 全局门控仍可守住 `SRAM peak`，但比 `globalgate4x` 更慢
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_eca.py`，实现 `globalgate4x + ECA` 组合门控
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x_eca` 变体映射
  - 完成 `globalgate4x_eca` 的 `172x224` Vela 与板端验证
  - 确认 `1/4` 阶段叠加多重门控会拖慢推理，当前不保留组合方案
  - 新增 `network/MultiScaleResNet_bilinear_compressedskip2xadd.py`，实现 encoder `/2` 到 decoder `/2` 的压缩式高尺度 skip
  - 为 `run_sram_test_bilinear.py` 增加 `compressedskip2xadd` 变体映射
  - 完成 `compressedskip2xadd` 的 `172x224` Vela 与板端验证
  - 确认高尺度 `/2` skip 仍能守住 `SRAM peak`，但代价比 global gate 更集中在 skip 分支自身
  - 新增 `network/MultiScaleResNet_bilinear_shareddualgate4x2x.py`，实现一次共享 bottleneck `MEAN` 后分别门控 decoder `/4` 与 `/2`
  - 为 `run_sram_test_bilinear.py` 增加 `shareddualgate4x2x` 变体映射
  - 完成 `shareddualgate4x2x` 的 `172x224` Vela 与板端验证
  - 确认共享 `MEAN` 本身几乎不构成成本，主要额外代价仍是 `/4` 与 `/2` 的两次 `MUL`
  - 确认该方案板端明显慢于 `globalgate4x`，但仍优于 `globalgate4x_eca` 与 `compressedskip2xadd`
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca.py`，实现 bottleneck-only `ECA` 加 decoder `/4` global gate
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x_bneckeca` 变体映射
  - 完成 `globalgate4x_bneckeca` 的 `172x224` Vela 与板端验证
  - 确认 bottleneck `ECA` 本身很轻，新增代价远小于把 attention 推到 `/4` 或 `/2`
  - 确认该方案比 `shareddualgate4x2x`、`globalgate4x_eca` 和 `compressedskip2xadd` 更优，但仍略慢于 `globalgate4x`
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip4x.py`，实现 `globalgate4x_bneckeca` 基础上的压通道 `/4` skip
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x_bneckeca_skip4x` 变体映射
  - 完成 `globalgate4x_bneckeca_skip4x` 的 `172x224` Vela 与板端验证
  - 确认继续加 `/4` skip 仍不会碰到 `SRAM peak`，但已显著拉高 `/4` 阶段的代价
  - 确认这条“在最稳基座上继续加尺度”的升级线暂时没有转化成更好的轻量训练候选
  - 按用户新要求切换到 `Vela-only` 快速筛选，不再对每个新想法重复上板
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip2x.py`，实现 `globalgate4x_bneckeca` 基础上的压通道 `/2` skip
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x.py`，实现 `globalgate4x_bneckeca` 基础上的压通道 `/8` skip
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x_bneckeca_skip2x` 与 `globalgate4x_bneckeca_skip8x` 变体映射
  - 完成 `globalgate4x_bneckeca_skip2x` 与 `globalgate4x_bneckeca_skip8x` 的 `172x224` Vela 验证
  - 确认同一基座上的增量性价比排序为 `skip8x > skip4x > skip2x`
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x.py`，实现真正的 `/8 + /4` 双长跳跃版本
  - 新增 `network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x2x.py`，实现 `/8 + /4 + /2` 三长跳跃版本
  - 为 `run_sram_test_bilinear.py` 增加 `globalgate4x_bneckeca_skip8x4x` 与 `globalgate4x_bneckeca_skip8x4x2x` 变体映射
  - 完成 `globalgate4x_bneckeca_skip8x4x` 与 `globalgate4x_bneckeca_skip8x4x2x` 的 `172x224` Vela 验证
  - 确认真正多尺度同时存在时，`/2` 仍是最不划算的一层；`/8 + /4` 可保留观察，`/8 + /4 + /2` 当前不保留
- Files created/modified:
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_addskip.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_liteaspp.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_eca.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate2x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_eca.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_compressedskip2xadd.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip2x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip4x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x2x.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_shareddualgate4x2x.py`
  - `tools/model_export/optical_flow_144x192/run_sram_test_bilinear.py`
  - `tools/model_export/optical_flow_144x192/output_bilinear_addskip/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear/168x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_addskip/168x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_liteaspp/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_eca/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate2x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_eca/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_compressedskip2xadd/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip2x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip4x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x4x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x4x2x/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_shareddualgate4x2x/172x224/*`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_181740.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_182747.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_183003.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184001.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184842.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193850.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193049.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_194450.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_212019.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_213812.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_221625.log`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_212810.log`
  - `logs/flow_frames/latest/frame_001.png`
  - `logs/flow_frames/latest/frame_002.png`
  - `logs/flow_frames/latest/frame_003.png`

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Skill source exists | `ls /mnt/d/Dataset/.agents/skills/pi-planning-with-files` | 能看到 `SKILL.md` 与模板/脚本 | 存在且可读 | pass |
| Copy to local skills | `cp -a ... ~/.codex/skills/` | 本机 skill 可用 | 完成 | pass |
| Copy to repo skills | `cp -a ... .cursor/skills/` | 仓库内 skill 可用 | 完成 | pass |
| Context snapshot | `bash scripts/build_context_snapshot.sh` | 生成最新 snapshot | 已生成 `context_snapshot_20260313_174323.md` | pass |
| Supported ops copy | `cp SUPPORTED_OPS.md .../vela/` | 文档进入仓库导出目录 | 完成 | pass |
| R1 addskip Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant addskip` | 不显著抬高 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1` | pass |
| R1 addskip board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_addskip_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动并输出 `224x176` flow | `initial done` / `INVOKE` 全命中；`infer ≈ 182.055 ms` | pass |
| 168x224 baseline Vela | `run_sram_test_bilinear.py --height 168 --width 224 --optimise Size --variant baseline` | 保持与现有 baseline 同量级 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1` | pass |
| 168x224 addskip Vela | `run_sram_test_bilinear.py --height 168 --width 224 --optimise Size --variant addskip` | 验证是否可消掉 pad | 峰值仍为 `1386.00 KiB`，且仍有 `skip_4x_pad` / `skip_8x_pad` | pass |
| 168x224 baseline board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_168x224_vela.tflite 0xB7B000 0x00000'` | 可启动并输出 `224x176` flow | `initial done` / `INVOKE` 全命中；`infer ≈ 177.562 ms`，`FPS ≈ 4.876` | pass |
| 168x224 addskip board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_addskip_168x224_vela.tflite 0xB7B000 0x00000'` | 若 pad 消失则希望至少不慢于 baseline | 可启动，但 `infer ≈ 182.055 ms`，`FPS ≈ 4.772` | pass |
| R2 Lite ASPP Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant liteaspp` | 不显著抬高 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，但 `inference_time` 升到 `181.016 ms` | pass |
| R2 Lite ASPP board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_liteaspp_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动并输出 `224x176` flow | `initial done` / `INVOKE` 全命中；`infer ≈ 186.851 ms`，`total ≈ 214.657 ms`，`FPS ≈ 4.66` | pass |
| R3 ECA Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant eca` | 不显著抬高 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 175.231 ms` | pass |
| R3 ECA board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_eca_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动并尽量接近 baseline FPS | `initial done` / `INVOKE` 全命中；`infer ≈ 180.035 ms`，`total ≈ 207.842 ms` | pass |
| globalgate2x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate2x` | 验证更高尺度门控是否仍可守住 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 174.782 ms` | pass |
| globalgate2x board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_globalgate2x_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动并尽量贴近 `globalgate4x` | `initial done` / `INVOKE` 全命中；`infer ≈ 180.089 ms`，`total ≈ 207.893 ms` | pass |
| globalgate4x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x` | 不显著抬高 peak，且尽量比 `R3 ECA` 更轻 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 174.358 ms` | pass |
| globalgate4x board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_globalgate4x_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动并尽量贴近 baseline | `initial done` / `INVOKE` 全命中；`infer ≈ 179.675 ms`，`total ≈ 207.481 ms` | pass |
| globalgate4x_eca Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_eca` | 验证轻量层内门控和跨层门控能否叠加而不明显伤害时延 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，但 `inference_time ≈ 176.45 ms` | pass |
| globalgate4x_eca board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_globalgate4x_eca_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动，且若叠加有效则至少不明显慢于 `R3 ECA` | `initial done` / `INVOKE` 全命中；`infer ≈ 181.198 ms`，`total ≈ 209.005 ms` | pass |
| compressedskip2xadd Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant compressedskip2xadd` | 验证高尺度 `/2` skip 是否仍可守住 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 177.745 ms` | pass |
| compressedskip2xadd board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_compressedskip2xadd_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动，且在当前可接受延迟范围内 | `initial done` / `INVOKE` 全命中；`infer ≈ 184.194 ms`，`total ≈ 211.992 ms` | pass |
| shareddualgate4x2x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant shareddualgate4x2x` | 验证共享全局摘要后同时门控 `/4` 与 `/2` 是否仍可守住 peak | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 175.714 ms` | pass |
| shareddualgate4x2x board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_shareddualgate4x2x_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动，且代价应低于更重的 skip/组合门控方案 | `initial done` / `INVOKE` 全命中；`infer ≈ 181.065 ms`，`total ≈ 208.870 ms` | pass |
| globalgate4x_bneckeca Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca` | 验证 bottleneck-only `ECA` 能否在不推进高尺度代价的前提下补一点表达力 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 174.657 ms` | pass |
| globalgate4x_bneckeca board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_globalgate4x_bneckeca_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动，且应明显优于把 attention 推到 `/4`/`/2` 的更重方案 | `initial done` / `INVOKE` 全命中；`infer ≈ 179.884 ms`，`total ≈ 207.673 ms` | pass |
| globalgate4x_bneckeca_skip4x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca_skip4x` | 验证在当前最稳基座上继续补一个 `/4` skip 后是否仍适合作为轻量训练候选 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 177.196 ms` | pass |
| globalgate4x_bneckeca_skip4x board | `run_optical_pipeline.sh --mode with-model --skip-build --model-arg '...optical_flow_bilinear_globalgate4x_bneckeca_skip4x_172x224_vela.tflite 0xB7B000 0x00000'` | 可启动，并确认 `/4` skip 升级是否仍值得保留 | `initial done` / `INVOKE` 全命中；`infer ≈ 182.451 ms`，`total ≈ 210.249 ms` | pass |
| globalgate4x_bneckeca_skip2x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca_skip2x` | 只用 `Vela` 快速判断在当前基座上补 `/2` skip 是否划算 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 179.253 ms` | pass |
| globalgate4x_bneckeca_skip8x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca_skip8x` | 只用 `Vela` 快速判断在当前基座上补 `/8` skip 是否划算 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 175.778 ms` | pass |
| globalgate4x_bneckeca_skip8x4x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca_skip8x4x` | 验证真正 `/8 + /4` 多尺度长跳跃同时存在时是否仍值得训练 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 178.319 ms` | pass |
| globalgate4x_bneckeca_skip8x4x2x Vela | `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant globalgate4x_bneckeca_skip8x4x2x` | 验证真正 `/8 + /4 + /2` 多尺度长跳跃同时存在时是否仍值得训练 | 峰值仍为 `1386.00 KiB`，hotspot 仍为 `ResizeBilinear_1`，`inference_time ≈ 182.915 ms` | pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-13 | `pi-planning-with-files` 不在当前会话技能列表中 | 1 | 先复制 skill，再按本地文件读取使用 |
| 2026-03-13 | `.agent/skills` 是符号链接，不是独立目录 | 1 | 改复制到 `.cursor/skills` |
| 2026-03-13 | 用户给的 `MCUFlowNet/EdgeFlowNet/SUPPORTED_OPS.md` 相对路径在仓库内不存在 | 1 | 改为读取并列目录 `/home/enmin/MCUFlowNet/EdgeFlowNet/SUPPORTED_OPS.md` |
| 2026-03-13 | `R1 addskip` 导出时报 `skip_4x_add` 维度不匹配：`43x56` vs `44x56` | 1 | 在 `/4` skip 分支增加 `1-row PAD` 对齐静态 shape |
| 2026-03-13 | 用户猜测 `168x224` 可能消掉 skip padding | 1 | 已证伪；Vela 仍保留 `skip_4x_pad` 与 `skip_8x_pad` |
| 2026-03-13 | `R2 Lite ASPP` 在算子层面可编译，但模型体积与瓶颈卷积成本明显上升 | 1 | 完成板端验证后确认该方向当前不保留 |
| 2026-03-13 | `R3 ECA` 需要在当前 TF1/TFLite 路径下避免引入 transpose-heavy lowering | 1 | 用 `mean -> reshape to [N,1,C,1] -> Conv2D(1x3) -> reshape back` 实现，Vela 可稳定编译 |
| 2026-03-13 | 跨层全局 gate 需要验证长生命周期小 tensor 是否会扰动调度 | 1 | 已完成 `globalgate4x` 验证，发现主要额外成本仍集中在 `1/4` `MUL`，调度影响很小 |
| 2026-03-13 | 直接用系统默认 `python` 跑导出脚本缺少 TensorFlow | 1 | 切回 `conda activate vela` 环境，恢复与前几轮一致的导出流程 |
| 2026-03-13 | `globalgate2x` 可能因门控尺度更高而拖慢推理 | 1 | 已验证，主要额外成本确实落在 `1/2` `MUL`，因此不升级为最佳候选 |
| 2026-03-13 | `globalgate4x + ECA` 可能因双门控叠加而拖慢 `1/4` 阶段 | 1 | 已验证，组合版虽不抬高峰值，但板端时延明显变差，当前不保留 |
| 2026-03-13 | `compressedskip2xadd` 首次导出时报 `/2` skip 形状不匹配：`86x112` vs `88x112` | 1 | 在 `/2` skip 分支增加静态 `PAD` 对齐，恢复导出与编译 |
| 2026-03-13 | `shareddualgate4x2x` 需要确认“共享 `MEAN`”是否真的比堆叠双门控更结构高效 | 1 | 已验证；共享 `MEAN` 很轻，但真实成本仍主要是 `/4` 与 `/2` 的两次 `MUL`，整体处于 `globalgate4x` 与更重方案之间 |
| 2026-03-13 | `globalgate4x_bneckeca` 需要确认在 bottleneck 单独补一个 `ECA` 是否会引入值得担心的新热点 | 1 | 已验证；新增 bottleneck `ECA` 很轻，主要额外代价仍来自原有 `/4` global gate |
| 2026-03-13 | `globalgate4x_bneckeca_skip4x` 需要确认在当前最稳基座上继续补 `/4` skip 是否还能作为训练候选保留 | 1 | 已验证；它仍不碰峰值，但 `/4` skip 的 `ADD/PAD/Conv` 代价已经足够明显，当前不升级为训练优先候选 |
| 2026-03-13 | 用户指出既然还没碰峰值，就该继续看 `1/2` 与 `1/8` | 1 | 已按 `Vela-only` 快速筛选补做；结果显示 `skip8x` 明显优于 `skip4x` 和 `skip2x` |
| 2026-03-13 | 用户进一步要求验证真正同时存在 `1/2 + 1/4 + 1/8` 的多尺度长跳跃 | 1 | 已完成 `skip8x4x` 与 `skip8x4x2x` 对比；结果显示三层同时存在仍不碰峰值，但 `/2` 会把总时延明显拉高 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: `R1 addskip`、`168x224` 分辨率复验、`R2 Lite ASPP`、`R3 ECA`、`globalgate2x`、`globalgate4x`、`globalgate4x_eca`、`compressedskip2xadd`、`shareddualgate4x2x`、`globalgate4x_bneckeca`、`globalgate4x_bneckeca_skip4x` 都已完成，并额外补做了 `skip2x` / `skip8x` 和真正多尺度 `skip8x4x` / `skip8x4x2x` 的 `Vela-only` 对比 |
| Where am I going? | 从当前结果里收缩出最少量、但最值得训练的多尺度候选，而不是继续扩展所有可运行结构 |
| What's the goal? | 找到在不明显恶化 `SRAM peak` 与 `FPS` 的前提下更值得保留的改造 |
| What have I learned? | `globalgate4x` 仍是效率最优点，`globalgate4x_bneckeca` 是新的第二优点；真正多尺度同时存在时，`/8 + /4` 仍可接受，但把 `/2` 也加进来后收益明显变差 |
| What have I done? | 已完成 `R1 addskip`、`168x224` 复验、`R2 Lite ASPP`、`R3 ECA`、`globalgate2x`、`globalgate4x`、`globalgate4x_eca`、`compressedskip2xadd`、`shareddualgate4x2x`、`globalgate4x_bneckeca`、`globalgate4x_bneckeca_skip4x` 的导出、Vela 分析、上板验证，并额外完成 `skip2x` / `skip8x` 及 `skip8x4x` / `skip8x4x2x` 的 `Vela-only` 对比 |
