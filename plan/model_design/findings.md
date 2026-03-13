# Findings & Decisions

## Requirements

- 先看 `START_HERE.md`、`MINIMAL_DEPLOYMENT.md`、`plan-000-context-index.md`、`plan-018-optical-flow-project-reorganization.md`
- 仔细分析当前 bilinear 版本的 Vela 报告
- 说明当前 `SRAM peak` 点在哪里
- 后续所有模型改造实验都固定按：
  - `Vela 输出`
  - `上板部署输出`
  - 暂不评估准确率
  - 只看 `SRAM peak` 和 `推理 FPS`
- 按 `pi-planning-with-files` 形式在 `plan/` 下建立新的计划子目录

## Research Findings

- 当前默认部署主线仍是 `157x203 -> 160x208`，这是项目对外主线，不等于当前 bilinear 实验主线。
- 当前板上可运行的 bilinear baseline 是 `172x224 -> 176x224`。
- `172x228 -> 176x240` 能过 Vela，但板端 `AllocateTensors()` 失败。
- 当前 bilinear 模型的 Vela SRAM peak 不在最慢卷积本身，而在 decoder 末段 `ResizeBilinear_1`。
- `172x224` 的 Vela 峰值是 `1386.00 KiB`。
- `172x228` 的 Vela 峰值是 `1485.00 KiB`。
- `172x228` 板端失败请求值是 `1520720 B`，和 Vela detailed allocation 峰值 `1520640 B` 只差 `80 B` 左右。
- `172x224` 板端成功日志显示 `infer ≈ 178.5 ms`，`total ≈ 206.3 ms`，算法 FPS 约 `4.84`。
- 当前 bilinear 网络没有显式 encoder-to-decoder 长跳连。
- Ethos-U55 支持 `CONV_2D`、`ADD`、`CONCATENATION`、`MEAN`、`MUL`、`LOGISTIC`、`RESIZE_BILINEAR`。
- `CONV_2D` 支持 dilation，因此 Lite ASPP 在算子层面可做。
- `MEAN + LOGISTIC + MUL` 组合在算子层面成立，因此轻量 channel attention 大概率可做。
- `R1 addskip` 已完成 `Vela -> 板端` 全流程验证。
- `R1 addskip` 的 Vela SRAM peak 仍是 `1419264 B = 1386.00 KiB`，与 baseline 持平。
- `R1 addskip` 的 Vela peak hotspot 仍在最终 `ResizeBilinear_1`，没有被 skip 分支前移。
- `R1 addskip` 新增的 `skip_8x_proj / skip_8x_add / skip_4x_proj / skip_4x_add` 都停留在中低分辨率，SRAM 占用较低。
- `172x224` 的奇数高度会让 `/4` 跳连出现 `43x56` 对 `44x56` 的静态 shape mismatch，因此 `R1 addskip` 额外引入了 `1-row PAD`。
- 这个 `PAD` 在 Vela 上可编译，但 `Util%` 很低：`skip_4x_pad_main = 1.35%`，`skip_4x_pad_bottom = 0.08%`。
- `R1 addskip` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`INVOKE resolution` 仍为 `[224, 176]`。
- `R1 addskip` 板端 `infer ≈ 182.055 ms`，`total ≈ 209.877 ms`，算法 FPS 约 `4.765`。
- 相比 baseline 的 `infer ≈ 178.513 ms` / `FPS ≈ 4.846`，`R1 addskip` 在不改善 `SRAM peak` 的前提下带来约 `+1.98%` infer 变慢与约 `-1.67%` FPS 下降。
- 当前更合理的解释是：变慢来自新增 `CONV + PAD + ADD` 开销，而不是 Vela 主 hotspot `ResizeBilinear_1` 的 `Util%` 恶化；该 hotspot 的 `Util%` 仍约 `6.08%`，与 baseline 基本一致。
- 用户提出的 `168x224` 输入分辨率假设已完成验证，目标是看它是否能在保持 `4:3` 宽高比的同时消掉 addskip 的 `PAD`。
- `168x224` baseline 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，热点仍是 decoder 尾段 `ResizeBilinear_1`。
- `168x224` addskip 的 Vela `SRAM peak` 也仍为 `1419264 B = 1386.00 KiB`，没有比 `172x224` 更低。
- `168x224` addskip 的 `detailed_performance.txt` 仍出现 `skip_4x_pad` 与 `skip_8x_pad`，说明该分辨率并没有消掉 skip 对齐 padding。
- `168x224` baseline 板端可正常启动，`model io` 为 `in(h=168,w=224,c=6) out(h=176,w=224,c=2)`，`infer ≈ 177.562 ms`，算法 FPS 约 `4.876`。
- `168x224` addskip 板端也可正常启动，但 `infer ≈ 182.055 ms`，算法 FPS 约 `4.772`。
- 相比 `168x224` baseline，`168x224` addskip 仍然更慢，且没有带来 `SRAM peak` 改善，因此不能作为“通过改分辨率挽救 addskip”的有效路径。
- `168x224` baseline 相比 `172x224` baseline 略快，说明它可以保留为一个可继续复验的 baseline 分辨率候选，但当前不改变 `R1 addskip` 的结论。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 新建 `plan/optical-flow-bilinear-sram-fps-pi-20260313/` | 这是用户要求的 `plan/` 子文件夹，同时保留 `pi-planning-with-files` 的三文件结构 |
| baseline 事实引用旧实验目录而不重复造一份原始记录 | 减少重复维护，避免两个 baseline 文档逐渐漂移 |
| 用 `algo_tick` 换算算法 FPS | 当前项目里 Web Toolkit 的 FPS 口径本来就是算法时间，不含串口传输 |
| `SUPPORTED_OPS.md` 复制到导出工具 `vela/` 子目录 | 方便模型设计与导出约束放在一起查 |
| 第一轮先做 `additive skip`，不做 `concat skip` | `add` 更省 SRAM，更适合当前 hotspot 在 decoder 尾段的场景 |
| 第二轮做 bottleneck `Lite ASPP` | 低分辨率增加上下文，比高分辨率模块更不容易推高 peak |
| 第三轮做 `ECA` 而不是 full SE/CBAM | 更轻，算子组合更简单，对 SRAM/FPS 风险更小 |
| `R1 addskip` 先记为“可行但非优先保留” | 它验证了 additive skip 在当前算子和 SRAM 约束下能跑通，但收益方向与当前阶段目标不一致 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 仓库会话可用 skills 中原本没有 `pi-planning-with-files` | 已从 `/mnt/d/Dataset/.agents/skills/pi-planning-with-files` 复制到 `~/.codex/skills` 和仓库 `.cursor/skills` |
| `.agent/skills` 实际是指向 `.cursor/skills` 的符号链接 | 直接复制到 `.cursor/skills`，仓库内即可通过 `.agent/skills` 访问 |

## Resources

- skill source:
  `/mnt/d/Dataset/.agents/skills/pi-planning-with-files`
- local skill copy:
  `/home/enmin/.codex/skills/pi-planning-with-files`
- repo skill copy:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.cursor/skills/pi-planning-with-files`
- bilinear baseline Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x224`
- bilinear fail Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x228`
- `R1 addskip` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_addskip/172x224`
- `R1 addskip` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_181740.log`
- `168x224` baseline Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/168x224`
- `168x224` addskip Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_addskip/168x224`
- `168x224` baseline board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_182747.log`
- `168x224` addskip board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_183003.log`
- supported ops copy:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md`

## Visual/Browser Findings

- 当前主要信息来自本地 markdown、Vela CSV、`detailed_performance.txt` 和板端 UART 日志。
- 关键文字事实已落入本文件与
  [00-current-bilinear-baseline.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/experiments/optical-flow-bilinear-sram-fps-20260313/00-current-bilinear-baseline.md)。
