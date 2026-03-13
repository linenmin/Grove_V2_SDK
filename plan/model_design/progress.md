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
- Files created/modified:
  - `tools/model_export/optical_flow_144x192/network/MultiScaleResNet_bilinear_addskip.py`
  - `tools/model_export/optical_flow_144x192/run_sram_test_bilinear.py`
  - `tools/model_export/optical_flow_144x192/output_bilinear_addskip/172x224/*`
  - `logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_181740.log`
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

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-13 | `pi-planning-with-files` 不在当前会话技能列表中 | 1 | 先复制 skill，再按本地文件读取使用 |
| 2026-03-13 | `.agent/skills` 是符号链接，不是独立目录 | 1 | 改复制到 `.cursor/skills` |
| 2026-03-13 | 用户给的 `MCUFlowNet/EdgeFlowNet/SUPPORTED_OPS.md` 相对路径在仓库内不存在 | 1 | 改为读取并列目录 `/home/enmin/MCUFlowNet/EdgeFlowNet/SUPPORTED_OPS.md` |
| 2026-03-13 | `R1 addskip` 导出时报 `skip_4x_add` 维度不匹配：`43x56` vs `44x56` | 1 | 在 `/4` skip 分支增加 `1-row PAD` 对齐静态 shape |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: `R1 addskip` 已完成，准备进入下一轮结构实验 |
| Where am I going? | 优先做 `Lite ASPP`，或者接收用户新的结构点子 |
| What's the goal? | 找到在不明显恶化 `SRAM peak` 与 `FPS` 的前提下更值得保留的改造 |
| What have I learned? | `additive skip` 在当前 bilinear 几何下可做，但不会改善 hotspot，且会带来约 `2%` infer 变慢 |
| What have I done? | 已完成 `addskip` 的导出、Vela 分析、上板验证与 Discord 通知 |
