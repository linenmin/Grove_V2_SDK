# Progress Log

## Session: 2026-03-31

### Phase 12: Stem-Dilation Vela Probe

- **Status:** complete
- Actions taken:
  - 参考现有 `progress.md` 的 bilinear Vela 流程，继续复用 `run_sram_test_bilinear.py --height 172 --width 224 --optimise Size --variant ...`
  - 新增 `globalgate4x_bneckeca_stemdilate` 与 `globalgate4x_bneckeca_stempostdilate` 两个试验版，保持 `bottleneck ECA + /4 global gate` 不变，只改 `E0/E1`
  - 本轮受控结构假设固定为：
    - `E0`: `3x3 dilated conv(rate=3)`，再接 `3x3 stride-2 conv`
    - `E1`: `3x3 dilated conv(rate=2)`，再接 `3x3 stride-2 conv`
  - 第二个受控对照结构固定为：
    - `E0`: `3x3 stride-2 conv`，再接 `3x3 dilated conv(rate=3)`
    - `E1`: `3x3 stride-2 conv`，再接 `3x3 dilated conv(rate=2)`
  - 为 bilinear 导出脚本增加新 variant 映射，并先做 `py_compile`
  - 复跑原版 `globalgate4x_bneckeca` 的 `172x224` Vela 结果，确认当前环境下仍为：
    - `SRAM peak = 1386.00 KiB`
    - `inference_time ≈ 174.657 ms`
    - hotspot 仍为最终 `ResizeBilinear_1`
  - 第一次导出新试验版时，发现 `StemDilatedDownsampleBlock` 内误给 `ConvBNReLUBlock` 传入 `name` 参数，修正后重新导出
  - 新试验版 `globalgate4x_bneckeca_stemdilate` 已成功完成 `172x224` Vela 编译
  - 新试验版结果：
    - `SRAM peak = 1386.00 KiB`
    - `inference_time ≈ 198.706 ms`
    - `FPS ≈ 5.033`
    - hotspot 仍为最终 `ResizeBilinear_1`
    - `Off-chip Flash = 2822.53 KiB`
  - 回查 `detailed_performance.txt` 后确认新增主要代价集中在两层 stem dilated conv：
    - `E0_stemdilate_dilated`: `8,202,299 cycles`, `10.32% Network`
    - `E1_stemdilate_dilated`: `2,840,072 cycles`, `3.57% Network`
  - 同时确认尾部峰值点没有前移，说明这次变慢不是由最终 resize 恶化导致，而是 stem 本身引入了明显额外卷积成本
  - 新增对照版 `globalgate4x_bneckeca_stempostdilate` 也已成功完成 `172x224` Vela 编译
  - `stempostdilate` 结果：
    - `SRAM peak = 1386.00 KiB`
    - `inference_time ≈ 190.056 ms`
    - `FPS ≈ 5.261`
    - hotspot 仍为最终 `ResizeBilinear_1`
    - `Off-chip Flash = 2826.62 KiB`
  - 回查 `detailed_performance.txt` 后确认 `stempostdilate` 的主要代价也集中在两层 stem dilated conv：
    - `E0_stempostdilate_dilated`: `7,730,532 cycles`, `10.17% Network`
    - `E1_stempostdilate_dilated`: `1,421,960 cycles`, `1.87% Network`
  - 三者对比表明：
    - 原版 `globalgate4x_bneckeca`: `174.657 ms`
    - `stemdilate`（先空洞再下采样）: `198.706 ms`
    - `stempostdilate`（先下采样再空洞）: `190.056 ms`
  - 这说明把 dilation 放到下采样后确实更省，但当前实现仍慢于原版 `7x7/5x5` stem
  - 随后继续补做只改 `E0` 的 dense 两层对照版 `globalgate4x_bneckeca_e0twolayer`：
    - `E0`: `3x3 stride-2 conv`，再接 `3x3 stride-1 conv`
    - `E1`: 保持原版 `5x5 stride-2 conv`
  - `e0twolayer` 已成功完成 `172x224` Vela 编译，结果为：
    - `SRAM peak = 1386.00 KiB`
    - `inference_time ≈ 174.150 ms`
    - `FPS ≈ 5.742`
    - hotspot 仍为最终 `ResizeBilinear_1`
    - `Off-chip Flash = 2814.94 KiB`
  - 这版结果非常关键：它比原版 `globalgate4x_bneckeca (174.657 ms)` 还略快约 `0.507 ms`，差异虽小，但至少说明“E0 多插一层 dense 3x3”本身并不会像 dilation 那样明显拖慢
  - 回查 `detailed_performance.txt` 后，`E0` 两层的主要指标为：
    - `E0 first 3x3 stride-2`: `483,904 cycles`, `0.69% Network`, `53.74% Util`
    - `E0 second 3x3 stride-1`: `1,478,894 cycles`, `2.12% Network`, `93.79% Util`
  - 两层合计 `E0 Network% ≈ 2.81%`，反而略低于原版 `7x7 stride-2` 的 `3.10%`
  - 这一步基本把问题进一步收窄为：当前翻车的关键矛盾不只是“E0 上多一层卷积”，而更像是“E0 上的 dilation 路径不划算”
- Files created/modified:
  - `tools/model_export/optical_flow_144x192/run_sram_test_bilinear.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_stemdilate.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_stempostdilate.py`
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_globalgate4x_bneckeca_e0twolayer.py`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_stemdilate/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_stempostdilate/172x224/*`
  - `tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_e0twolayer/172x224/*`
  - `plan/model_design/stage-findings-summary-20260331.md`
  - `plan/model_design/task_plan.md`
  - `plan/model_design/findings.md`
  - `plan/model_design/progress.md`
- Vela 结果确认：
  - 原版 `globalgate4x_bneckeca`、`stemdilate`、`stempostdilate`、`e0twolayer` 的 `SRAM peak` 都是 `1386.00 KiB`
  - `stemdilate` 比原版慢约 `24.049 ms`，相对增幅约 `13.77%`
  - `stempostdilate` 比原版慢约 `15.399 ms`，相对增幅约 `8.82%`
  - `stempostdilate` 比 `stemdilate` 快约 `8.650 ms`
  - `e0twolayer` 比原版快约 `0.507 ms`，相对变化约 `-0.29%`
  - 当前可以确认：`先下采样再空洞` 比 `先空洞再下采样` 更合理，但两者都还没有优于原版 stem；而 `E0` 的 dense 两层 `3x3` 是一个值得保留的真实候选

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
| fixed-arch baseline Vela | `FixedArchModel(arch=0,2,1,1,0,0,0,0,0, variant=baseline)` + `172x224` + `from_session -> INT8 TFLite -> Vela(Size)` | 训练基线在最终训练输入分辨率上仍满足部署侧约束 | 输出 `176x224`；`SRAM peak = 1386.00 KiB`；`inference_time ≈ 166.179 ms`；`FPS ≈ 6.018` | pass |
| fixed-arch `globalgate4x_bneckeca` Vela | `FixedArchModel(arch=0,2,1,1,0,0,0,0,0, variant=globalgate4x_bneckeca)` + `172x224` + `from_session -> INT8 TFLite -> Vela(Size)` | 轻量消融版在最终训练输入分辨率上仍满足部署侧约束 | 输出 `176x224`；`SRAM peak = 1386.00 KiB`；`inference_time ≈ 167.551 ms`；`FPS ≈ 5.968` | pass |
| fixed-arch `globalgate4x_bneckeca_skip8x4x2x` Vela | `FixedArchModel(arch=0,2,1,1,0,0,0,0,0, variant=globalgate4x_bneckeca_skip8x4x2x)` + `172x224` + `from_session -> INT8 TFLite -> Vela(Size)` | 三尺度训练主力版在最终训练输入分辨率上仍满足部署侧约束 | 输出 `176x224`；`SRAM peak = 1386.00 KiB`；`inference_time ≈ 175.810 ms`；`FPS ≈ 5.688` | pass |

## Fixed-Arch Vela Precheck

- 在 `MCUFlowNet/EdgeFlowNAS` 中对 joint training 的三个候选做了结构级 `Vela` 预检，而不是等训练结束后再看部署约束。
- 输入分辨率固定为和前面实际部署保持一致的 `172x224`，因此输出仍是 `176x224`，和 bilinear 板端验证口径一致。
- 导出路径改成 `tf.compat.v1.lite.TFLiteConverter.from_session`，因为第一次走 `SavedModel` 会把 BN 的 train/infer 双分支带进 TFLite，出现 `OptionalFromValue / FusedBatchNormV3` 非 native op。
- 多尺度输出累加也改成了原始 `AccumPreds` 的逐级 `2x resize + add`；如果直接把 `/4` 结果一次 resize 到 full，会触发 Vela 对非 2x `ResizeBilinear` 路径的不支持。
- 三个 fixed-arch 候选在 `172x224` 上都仍然卡在同一个 `SRAM peak = 1386.00 KiB`，热点还是最终 `ResizeBilinear_1`。
- 这说明当前 joint training 的三模型设计和前面 bilinear 结构实验的部署结论是一致的：真正需要关注的是时延差异，而不是峰值被新结构打穿。
- fixed-arch `globalgate4x_bneckeca` 相比 fixed-arch baseline 只慢约 `0.83%`，完全可以作为 joint training 的轻量消融项。
- fixed-arch `globalgate4x_bneckeca_skip8x4x2x` 相比 fixed-arch baseline 慢约 `5.80%`，仍远低于用户当前可接受的 `20%` 时延阈值，因此可直接进入 first-round 上限训练。
- 用户在 `epoch 80` 的 fixed-arch 训练结果里观察到：`ablation` 只小幅优于 baseline，而 full 版暂时没有拉开差距，因此下一步补 joint-training 对应的 Sintel evaluator，用来验证“FC2 对这些结构是否过于简单”这个假设。
- 已在 `MCUFlowNet/EdgeFlowNAS` 中新增 fixed-arch joint training 的 Sintel evaluator，路径是 `wrappers/fixed_arch_compare/run_sintel_test.py`。
- 新 evaluator 支持：
  - 直接读取整个 `experiment_dir`
  - 自动发现 `model_baseline / model_ablation / model_full`
  - 从 `run_manifest.json` 解析每个模型对应的 `variant`
  - 按 `best` 或 `last` checkpoint 批量输出 `sintel_eval_<ckpt>.json/.csv`
- 这一步的目的不是替代 FC2 验证，而是给“FC2 是否过于简单”这个怀疑补一个跨数据集判据。
- 用户随后补充了真实 `Sintel` 结果：
  - `baseline`: `fc2_val_epe = 4.044281`，`sintel_epe = 6.001463`
  - `ablation/globalgate4x_bneckeca`: `fc2_val_epe = 3.985728`，`sintel_epe = 5.579553`
  - `full/globalgate4x_bneckeca_skip8x4x2x`: `fc2_val_epe = 3.920078`，`sintel_epe = 5.647002`
- 这轮结果确认：`FC2` 的确没有完全显露跨域排序；两个改造版在 `Sintel` 上都明显优于 baseline。
- 同时也确认了一个评估口径问题：当前 `--ckpt_name best` 实际表示“FC2-best checkpoint”，不是 “Sintel-best checkpoint”。
- 现阶段新的阶段性结论是：
  - `ablation` 是当前最稳的跨域赢家；
  - `full` 仍优于 baseline，但它是否真正优于 `ablation`，需要进一步看 `epoch` 选择而不是只看单个 `best.ckpt`。
- 用户进一步要求规划“六模型冲榜训练”，并明确不再重训 `baseline` 与 `ablation`。
- 本轮新增计划结论：
  - 训练分辨率继续固定 `172x224`，先保持和已有曲线完全一致；
  - 训练入口继续扩展现有 `fixed_arch_compare/run_train.py`，不新开平行脚本；
  - 后续单模型训练仍沿用同一入口，通过 `--model_variants`/`--model_names` 选择单个模型；
  - 新增文档 `fixed-arch-six-model-plan-20260315.md`，固化六模型冲榜候选和脚本组织建议。
- 在 `MCUFlowNet/EdgeFlowNAS/efnas/network/fixed_arch_models.py` 中实现了 6 个新变体：
  - `globalgate4x_bneckeca_skip8x4x`
  - `globalgate8x4x_bneckeca`
  - `globalgate8x4x_bneckeca_skip8x`
  - `globalgate4x_dual_eca8_bneckeca`
  - `globalgate8x4x_bneckeca_skip8x4x`
  - `skip8x4x`
- 新增：
  - `configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml`
  - `wrappers/fixed_arch_compare/run_vela_precheck.py`
- 本地验证已完成：
  - `py_compile` 通过
  - 6 个新变体合成 batch dry-run 通过

## Session: 2026-03-17

### Phase 11: Six-Model Training Review & Shortlist Refresh

- **Status:** in_progress
- Actions taken:
  - 收集六模型联合训练在 `115 / 135 / 225 / 300 epoch` 的 `FC2 + Sintel` 中期结果
  - 将六模型结果与既有 `baseline / globalgate4x_bneckeca / globalgate4x_bneckeca_skip8x4x2x` 的历史最佳点做 apples-to-apples 对比
  - 确认 `globalgate8x4x_bneckeca_skip8x4x` 的 `Sintel-best` 出现在 `epoch 220` (`4.885117`)，而 `FC2-best` 已漂移到 `epoch 290` (`5.001160`)
  - 确认即便按更严格的 `FC2-best` 口径，它也仍然超过既有 `ablation` 最佳 `5.050387 @ epoch 245` 与旧 `full` 最佳 `5.066905 @ epoch 240`
  - 确认“`/8 global gate + /4 global gate + /8+/4 skip`”在 searched backbone 上存在真实协同：组合版显著优于对应的 gate-only 和 skip-only 版本
  - 重新对齐排序后，确认 `globalgate4x_bneckeca_skip8x4x` 与 `globalgate8x4x_bneckeca_skip8x` 都是第二梯队，其中前者在当前 `FC2-best` 口径下略优
  - 回查 fixed-arch Vela 汇总，补齐主线冠军与稳健对照的部署代价：
    - `globalgate4x_bneckeca`: `167.551 ms`, `5.968 FPS`
    - `globalgate8x4x_bneckeca_skip8x4x`: `172.374 ms`, `5.801 FPS`
  - 确认 `globalgate4x_dual_eca8_bneckeca` 与 `skip8x4x_plain` 当前不值得再优先投入
  - 更新研究重点：从继续铺开新结构，转向围绕 `globalgate8x4x_bneckeca_skip8x4x` 做收敛
- Files created/modified:
  - `plan/model_design/findings.md`
  - `plan/model_design/task_plan.md`
  - `plan/model_design/progress.md`
  - `plan/model_design/fixed-arch-six-model-plan-20260315.md`
  - `172x224` 的 INT8 TFLite + Vela 预检通过
- Vela 结果确认：
  - 6 个新变体全部保持 `SRAM peak = 1386.00 KiB`
  - hotspot 仍全部是最终 `ResizeBilinear_1`
  - 最快的是 `globalgate4x_dual_eca8_bneckeca (168.445 ms)`
  - 最重的是 `globalgate8x4x_bneckeca_skip8x4x (172.374 ms)`

### Phase 11: English Stage Summary Draft

- **Status:** in_progress
- Actions taken:
  - 读取 `task_plan.md`、`findings.md`、`progress.md`、`fixed-arch-training-plan-20260314.md`、`fixed-arch-six-model-plan-20260315.md`
  - 按用户新要求重组阶段汇报结构，不按时间线写，而按“约束 -> 模块设计 -> 映射关系 -> 部署结果 -> 训练结果 -> 最终结论”组织
  - 新增英文汇总文档 `stage-summary-model-design-20260317.md`
  - 在文档中加入 4 张 `mermaid` 图，分别说明：
    - baseline 与尾部 peak opportunity
    - skip family
    - global-gating family
    - fixed-arch compare 与最终冠军结构
  - 加入模型名与模块组合映射表，避免导师汇报时只看到命名而看不出结构含义
  - 加入前期 `Vela + board` 对照表、fixed-arch 训练结果表、six-model leaderboard 结果表
  - 在 source notes 中区分 deployment artefacts 与 planning-summary-derived training numbers，避免把 planning 汇总误写成原始训练输出
- Files created/modified:
  - `plan/model_design/stage-summary-model-design-20260317.md`

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
| 2026-03-14 | fixed-arch 三模型第一次 `SavedModel -> TFLite` 导出把 BN train/infer 双分支一起带进图，出现 `OptionalFromValue / FusedBatchNormV3` | 1 | 改为 `from_session` 直接冻结 `input -> final_output` inference graph |
| 2026-03-14 | fixed-arch 三模型第一次直接把 `/4` 预测 resize 到 full 再累加，触发 Vela 对非 2x `ResizeBilinear` 路径不支持 | 1 | 改回原始 `AccumPreds`：逐级 `2x resize + add` |
| 2026-03-31 | `globalgate4x_bneckeca_stemdilate` 第一次导出时报 `ConvBNReLUBlock() got an unexpected keyword argument 'name'` | 1 | 去掉 `StemDilatedDownsampleBlock` 中对 `ConvBNReLUBlock` 的 `name` 传参后重跑，成功完成 TFLite + Vela |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | bilinear 结构筛选已基本收敛，并且刚完成 fixed-arch 三模型在最终训练输入 `172x224` 下的结构级 `Vela` 预检 |
| Where am I going? | 进入 HPC 联合训练前，先确认 baseline / 轻量消融 / 三尺度 full 版都满足当前部署侧约束 |
| What's the goal? | 用最少训练成本先看到 accuracy 上限，同时不把部署侧 `SRAM peak + FPS` 风险留到训练后才发现 |
| What have I learned? | fixed-arch 三模型在 `172x224` 下的 `SRAM peak` 仍全部锁在 `1386.00 KiB`；三尺度 full 版只比 fixed-arch baseline 慢约 `5.80%` |
| What have I done? | 已完成 bilinear 结构筛选，并额外完成 fixed-arch `baseline` / `globalgate4x_bneckeca` / `globalgate4x_bneckeca_skip8x4x2x` 的 `172x224` INT8 TFLite 导出与 Vela 预检 |
