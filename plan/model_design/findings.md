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
- `R2 Lite ASPP` 已在 `172x224` 上完成 `Vela -> 板端` 全流程验证，改造位置只在 bottleneck，采用 `1x1 + dilation rate 2/4 + residual add`。
- `R2 Lite ASPP` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，与 baseline 持平，peak hotspot 仍是最终 `ResizeBilinear_1`。
- `R2 Lite ASPP` 没有把 peak 从 decoder 尾段前移；最终 `ResizeBilinear_1` 的 `Util%` 仍约 `6.08%`，与 baseline 基本一致。
- `R2 Lite ASPP` 的新增瓶颈分支主要成本来自 `lite_aspp_rate4` 与 `lite_aspp_rate2` 两个 dilated conv，Vela `Network%` 分别约 `2.58%` 与 `1.38%`，不是尾部 resize hotspot 恶化。
- `R2 Lite ASPP` 的 Vela 预估推理时间从 baseline `173.149 ms` 升到 `181.016 ms`，约增加 `4.54%`。
- `R2 Lite ASPP` 的 off-chip flash 占用从 baseline `2776.391 KiB` 升到 `3159.953 KiB`，说明它在权重体积上也明显更重。
- `R2 Lite ASPP` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`INVOKE resolution` 仍为 `[224, 176]`。
- `R2 Lite ASPP` 板端 `infer ≈ 186.851 ms`，`total ≈ 214.657 ms`，按当前口径折算算法 FPS 约 `4.66`。
- 相比 baseline 的 `infer ≈ 178.513 ms` / `total ≈ 206.3 ms` / `FPS ≈ 4.846`，`R2 Lite ASPP` 在不改善 `SRAM peak` 的前提下进一步拉低 FPS，因此当前应标记为“可行但不保留”。
- `R3 ECA-style channel attention` 已在 `172x224` 上完成 `Vela -> 板端` 全流程验证，位置放在 bottleneck、`/8` decoder 和 `/4` decoder 三个低中分辨率阶段。
- `R3 ECA` 使用 `MEAN -> RESHAPE -> Conv1D-style Conv2D -> LOGISTIC -> MUL` 的轻量门控路径，Vela 可稳定编译，没有出现不支持的 lowering。
- `R3 ECA` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，与 baseline 持平，peak hotspot 仍是最终 `ResizeBilinear_1`。
- `R3 ECA` 没有把 peak 前移，也没有恶化最终 `ResizeBilinear_1` 的 `Util%`；该 hotspot 仍约 `6.08%`。
- `R3 ECA` 的新增 attention 分支本身很轻：三个 `Conv1D-style` channel mixing 节点的单层 `Network%` 都接近 `0.00%`，主要额外代价来自中分辨率 `MUL`，但仍明显小于 `Lite ASPP` 的 dilated conv 分支。
- `R3 ECA` 的 Vela 预估推理时间从 baseline `173.149 ms` 升到 `175.231 ms`，只增加约 `1.20%`。
- `R3 ECA` 的 off-chip flash 占用从 baseline `2776.391 KiB` 小幅升到 `2823.641 KiB`，增量远小于 `Lite ASPP`。
- `R3 ECA` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`INVOKE resolution` 仍为 `[224, 176]`，arena 仍保留 `32 B` 余量。
- `R3 ECA` 板端 `infer ≈ 180.035 ms`，`total ≈ 207.842 ms`。
- 相比 baseline 的 `infer ≈ 178.513 ms` / `total ≈ 206.3 ms`，`R3 ECA` 只带来很小的时延增量，同时保持 `SRAM peak` 不变，因此当前应标记为“可行且优先保留”。
- 截至 `R1/R2/R3` 三轮对比，`R3 ECA` 是目前最接近目标约束的改造：比 `addskip` 和 `Lite ASPP` 更能控制 `FPS` 损失，同时没有恶化 `SRAM peak`。
- 基于 `R3` 的稳定 attention 原语，新增了 `globalgate4x`：从 bottleneck 提取全局均值向量，经 `1x1 conv + sigmoid` 后跨层广播到 decoder `1/4` 特征图。
- `globalgate4x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，与 baseline、`R3 ECA` 持平，peak hotspot 仍是最终 `ResizeBilinear_1`。
- `globalgate4x` 的跨层上下文向量本身几乎不占空间；Vela 报告中真正的额外成本主要是 `global_gate_4x_scale` 这次 `1/4` 分辨率 `MUL`，而不是 `mean` 或 `proj`。
- `globalgate4x` 的 Vela 预估推理时间是 `174.358 ms`，比 `R3 ECA` 的 `175.231 ms` 还略低，只比 baseline `173.149 ms` 高约 `0.70%`。
- `globalgate4x` 的 off-chip flash 占用是 `2804.969 KiB`，同样低于 `R3 ECA` 的 `2823.641 KiB`。
- `globalgate4x` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，arena 仍保留 `32 B` 余量。
- `globalgate4x` 板端 `infer ≈ 179.675 ms`，`total ≈ 207.481 ms`。
- 相比 baseline 的 `infer ≈ 178.513 ms` / `total ≈ 206.3 ms`，`globalgate4x` 只带来极小的时延增量；相比 `R3 ECA` 的 `infer ≈ 180.035 ms` / `total ≈ 207.842 ms`，它还略优一些。
- 截至当前所有实验，`globalgate4x` 是最值得保留的版本：保持 `SRAM peak` 不变，板端性能最接近 baseline，同时比 `R3 ECA` 更轻。
- `globalgate2x` 已在 `172x224` 上完成 `Vela -> 板端` 全流程验证：它把同样的 bottleneck 全局向量广播到 decoder `1/2` 特征图。
- `globalgate2x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，没有触发新的内存峰值。
- `globalgate2x` 的主要额外成本集中在 `global_gate_2x_scale` 这次 `1/2` 分辨率 `MUL`；该节点在 Vela 报告里单层 `SRAM Usage = 473120 B`、`Network% ≈ 0.68`，明显重于 `globalgate4x` 的 `1/4` 门控。
- `globalgate2x` 的 Vela 预估推理时间约 `174.782 ms`，虽然仍接近 baseline，但已略差于 `globalgate4x` 的 `174.358 ms`。
- `globalgate2x` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`INVOKE resolution` 仍为 `[224, 176]`。
- `globalgate2x` 板端 `infer ≈ 180.089 ms`，`total ≈ 207.893 ms`，略差于 `globalgate4x`，也没有优于 `R3 ECA`。
- 结论上，`globalgate2x` 证明“跨层全局门控可以安全推到更高尺度”，但在当前约束下不值得替代 `globalgate4x`。
- 新增组合实验 `globalgate4x_eca`：保留 `R3 ECA` 的 bottleneck、`/8`、`/4` 层内轻门控，并额外叠加 `globalgate4x` 的跨层 `1/4` 全局门控。
- `globalgate4x_eca` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，说明组合并未突破当前峰值上限。
- `globalgate4x_eca` 的额外代价主要来自 `eca_decoder_4x_pre_scale` 与 `global_gate_4x_scale` 在同一 `1/4` 阶段叠加；其中前者单层 `Network% ≈ 0.34`，后者也约 `0.34`。
- `globalgate4x_eca` 的 Vela 预估推理时间升到 `176.45 ms`，已经明显高于 `globalgate4x` 与 `globalgate2x`。
- `globalgate4x_eca` 板端可正常启动，日志中 `initial done` 与 `INVOKE` 全命中，`model io` 仍保持 `172x224 -> 176x224`。
- `globalgate4x_eca` 板端 `infer ≈ 181.198 ms`，`total ≈ 209.005 ms`，劣于 `globalgate4x`、`globalgate2x` 和 `R3 ECA`。
- 组合实验的结论是：当前 `1/4` 阶段已经不适合继续叠加多个门控；若后续还想继续加表达力，应优先避免在同一高活跃阶段再叠 `MUL`。
- 新增高尺度 skip 实验 `compressedskip2xadd`：从 encoder `/2` 取高分辨率特征，在 decoder `/2` 阶段做 `1x1 squeeze -> BN/ReLU -> 1x1 expand -> BN -> add` 融合。
- `compressedskip2xadd` 首次导出就暴露了真实几何约束：encoder `/2` 是 `86x112`，decoder `/2` 是 `88x112`，所以必须像前面的 skip 实验一样加入静态 `PAD` 对齐。
- `compressedskip2xadd` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，说明即使把 skip 推到 `/2`，当前峰值仍未被打破。
- `compressedskip2xadd` 的主要额外代价集中在 `/2` skip 分支本身：`skip_2x_squeeze`、`skip_2x_expand`、`skip_2x_pad` 和 `skip_2x_add` 都明显出现在 Vela 报告中，其中 `skip_2x_add` 单层 `Network% ≈ 1.33`。
- `compressedskip2xadd` 的 Vela 预估推理时间是 `177.745 ms`；比 baseline 慢约 `2.65%`，比 `globalgate4x` 慢，但仍明显低于此前用户给出的 `20%` 容忍线。
- `compressedskip2xadd` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`INVOKE resolution` 仍为 `[224, 176]`，arena 仍保留 `32 B` 余量。
- `compressedskip2xadd` 板端 `infer ≈ 184.194 ms`，`total ≈ 211.992 ms`；同样慢于 `globalgate4x`，但仍在当前可接受区间内。
- 这轮实验说明：高尺度 `/2` skip 的确更“像精度向”的结构，但它带来的代价也比单纯 global gate 更直接、更集中；在现阶段，它应被标记为“可行且值得继续保留观察”，但还不是新的效率最优解。
- 新增联动门控实验 `shareddualgate4x2x`：从 bottleneck 只做一次共享 `MEAN` 得到全局上下文向量，再分别投影成 `/4` 与 `/2` 两层的门控信号。
- `shareddualgate4x2x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，说明双层门控仍未突破当前峰值边界。
- `shareddualgate4x2x` 验证了用户提出的“共享全局摘要”判断：共享 `MEAN` 本身确实极轻，`shared_dual_gate_mean` 对应 `MEAN` 只有约 `0.08%` `Network%`，真实代价仍主要落在 `/4` 与 `/2` 的两次 `MUL`。
- 两次门控中，`shared_dual_gate_4x_scale` 的 feature map 约 `157696 B`、单层 `Network% ≈ 0.34`；`shared_dual_gate_2x_scale` 约 `315392 B`、单层 `Network% ≈ 0.67`，和此前 `globalgate2x` 的观察一致：高尺度 `/2` 门控依旧是更重的那一半。
- `shareddualgate4x2x` 的 Vela 预估推理时间是 `175.714 ms`，比 `globalgate4x` 的 `174.358 ms` 更慢，但仍明显轻于 `globalgate4x_eca` 的 `176.45 ms` 与 `compressedskip2xadd` 的 `177.745 ms`。
- `shareddualgate4x2x` 的 off-chip flash 占用是 `2814.078 KiB`，略高于 `globalgate4x`，但仍低于 `R3 ECA` 和 `globalgate4x_eca`。
- `shareddualgate4x2x` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`arena_budget=1419872`，`remaining_after_arena=32`，与前几轮一致。
- 这次板端日志只有 7 次有效 loop 样本，但数值非常稳定：`infer ≈ 181.065 ms`，`total ≈ 208.870 ms`，`infer` 抖动仅约 `0.002 ms`。
- 相比单层版本，`shareddualgate4x2x` 明显比 `globalgate4x` 更慢，也略慢于 `globalgate2x`；但它仍优于 `globalgate4x_eca` 和 `compressedskip2xadd`，因此当前应标记为“可行且结构干净的中间候选”，不是新的最佳效率点。
- 新增轻量叠加实验 `globalgate4x_bneckeca`：只在 bottleneck 加一个 `ECA`，然后保留 `globalgate4x` 的 decoder `/4` 跨层门控，不再在 `/8` 或 `/4` 叠额外层内 attention。
- `globalgate4x_bneckeca` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，说明“bottleneck-only ECA + /4 global gate”仍完全守住当前峰值边界。
- 这轮验证了预期判断：bottleneck `ECA` 本身非常轻，`eca_bottleneck_scale` 约 `39424 B` 生命周期、单层 `Network% ≈ 0.08`；模型新增的主要高分辨率代价仍然是原来的 `global_gate_4x_scale`，不是新加的 bottleneck 注意力。
- `globalgate4x_bneckeca` 的 Vela 预估推理时间是 `174.657 ms`，只比 `globalgate4x` 的 `174.358 ms` 高约 `0.30 ms`，明显轻于 `R3 ECA`、`shareddualgate4x2x`、`globalgate4x_eca` 和 `compressedskip2xadd`。
- `globalgate4x_bneckeca` 的 off-chip flash 占用是 `2813.938 KiB`，和 `shareddualgate4x2x` 接近，但仍显著低于 `globalgate4x_eca`。
- `globalgate4x_bneckeca` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`arena_budget=1419872`，`remaining_after_arena=32`，没有引入新的运行时内存问题。
- 板端日志共提取到 9 次有效 loop 样本，数值几乎不抖：`infer ≈ 179.884 ms`，`total ≈ 207.673 ms`，`infer` 抖动约 `0.001 ms`。
- 相比 baseline 的 `178.513 / 206.3 ms`，`globalgate4x_bneckeca` 只带来很小的时延增量；相比 `globalgate4x` 的 `179.675 / 207.481 ms`，它略慢，但差距很小。
- 当前结论是：`globalgate4x_bneckeca` 成为新的“第二优候选”。如果你更偏向稳妥增加一点表达力，而不想把代价推进到 `/2` 或在 `/4` 叠多重门控，这一版比 `shareddualgate4x2x`、`globalgate4x_eca`、`compressedskip2xadd` 都更合理。
- 按“在当前最稳基座上继续升级”的思路，新增 `globalgate4x_bneckeca_skip4x`：在 `globalgate4x_bneckeca` 基础上，再给 decoder `/4` 增加一个压通道 `skip4x add`。
- `globalgate4x_bneckeca_skip4x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`；这说明即使开始叠加更像 U-Net 的 `/4` skip，当前真实峰值仍没有被打到。
- 这轮 `/4` skip 的新代价已经非常具体：`skip_4x_squeeze`、`skip_4x_expand`、`skip_4x_pad`、`skip_4x_add` 全部进入了 Vela 报告，其中 `skip_4x_add` 单层 `Network% ≈ 0.67`，与 `global_gate_4x_scale` 同量级。
- `/4` skip 仍然遇到之前就见过的几何问题：encoder `/4` 与 decoder `/4` 需要一行静态 `PAD` 对齐，所以这条路径不是“纯净无对齐代价”的 skip。
- `globalgate4x_bneckeca_skip4x` 的 Vela 预估推理时间是 `177.196 ms`，比 `globalgate4x_bneckeca` 的 `174.657 ms` 明显更慢，也高于 `shareddualgate4x2x` 的 `175.714 ms`。
- `globalgate4x_bneckeca_skip4x` 的 off-chip flash 占用是 `2823.984 KiB`，已经接近 `R3 ECA` 一档。
- `globalgate4x_bneckeca_skip4x` 板端可正常启动，`model io` 仍为 `in(h=172,w=224,c=6) out(h=176,w=224,c=2)`，`arena_budget=1419872`，`remaining_after_arena=32`，没有触发新的运行时内存错误。
- 板端日志共提取到 8 次有效 loop 样本，数值稳定：`infer ≈ 182.451 ms`，`total ≈ 210.249 ms`。
- 相比 `globalgate4x_bneckeca` 的 `179.884 / 207.673 ms`，这轮 `/4` skip 升级带来了约 `+2.567 ms infer` 和约 `+2.576 ms total`；相比 `globalgate4x`，差距已经更明显。
- 当前结论是：`globalgate4x_bneckeca_skip4x` 证明了“继续加尺度”依然不会立刻撞到峰值，但在当前实现方式下，它不是好的轻量训练候选；排序上它落后于 `globalgate4x`、`globalgate4x_bneckeca`、`R3 ECA`、`shareddualgate4x2x`。
- 用户已明确同意当前阶段改为 `Vela-only` 快速筛选，不再要求每个新想法都上板；板端和 `Vela` 当前可先按“稳定多约 5 ms”处理，最后只对 shortlist 上板复验。
- 按这个新口径，已补做同一基座上的另外两个尺度：`globalgate4x_bneckeca_skip2x` 和 `globalgate4x_bneckeca_skip8x`。
- `globalgate4x_bneckeca_skip2x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`，说明把 skip 推到 `/2` 也依然没有击穿当前峰值边界。
- 但 `globalgate4x_bneckeca_skip2x` 的代价明显更重：Vela `inference_time ≈ 179.253 ms`，已经劣于 `skip4x`、`skip8x` 和所有当前主力候选。
- `skip2x` 的详细代价和此前单独 `compressedskip2xadd` 的规律一致：`skip_2x_squeeze / expand / pad / add` 都很重，其中 `skip_2x_add` 单层 `Network% ≈ 1.32`，是当前这些 skip 里最昂贵的一档。
- `globalgate4x_bneckeca_skip8x` 的 Vela `SRAM peak` 同样仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`。
- `globalgate4x_bneckeca_skip8x` 的 Vela `inference_time ≈ 175.778 ms`，明显好于 `skip4x` 的 `177.196 ms` 和 `skip2x` 的 `179.253 ms`；在只看 `Vela` 的口径下，它是目前“多一个 skip”的最佳尺度。
- `skip8x` 的代价主要集中在中低分辨率 `skip_8x_squeeze / expand / add`，其中 `skip_8x_add` 单层 `Network% ≈ 0.34`，和 `global_gate_4x_scale` 基本同量级；它没有出现 `/4`、`/2` 那种明显放大的高尺度代价。
- 这轮多尺度对比的结论已经很清楚：在 `globalgate4x_bneckeca` 这个基座上，增量性价比排序是 `skip8x > skip4x > skip2x`。
- 如果目标是从当前主线里再挑少数可训练候选，那么 `skip8x` 值得保留观察；`skip4x` 和 `skip2x` 目前都不值得进入训练优先列表。
- 按用户要求，已进一步验证真正“同时存在多个尺度长跳跃”的版本，而不只是单尺度增量。
- `globalgate4x_bneckeca_skip8x4x` 代表一个更轻的 U-Net-like 多尺度版本：同时保留 `/8` 与 `/4` skip，并保留 `bottleneck ECA + /4 global gate`。
- `globalgate4x_bneckeca_skip8x4x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`；说明即使 `/8 + /4` 长跳跃同时存在，当前总峰值仍没有被打破。
- `globalgate4x_bneckeca_skip8x4x` 的 Vela `inference_time ≈ 178.319 ms`，明显高于单独 `skip8x` 的 `175.778 ms`，也高于单独 `skip4x` 的 `177.196 ms`，但仍明显好于包含 `/2` 的全量版。
- 这版的成本主要来自三处并存：`skip_8x_add`、`global_gate_4x_scale`、`skip_4x_add`。其中 `skip_8x_add` 单层 `Network% ≈ 0.33`，`global_gate_4x_scale ≈ 0.33`，`skip_4x_add ≈ 0.66`。
- `globalgate4x_bneckeca_skip8x4x2x` 代表更完整的 U-Net-like 多尺度版本：`/8 + /4 + /2` 三层长跳跃同时存在。
- `globalgate4x_bneckeca_skip8x4x2x` 的 Vela `SRAM peak` 仍为 `1419264 B = 1386.00 KiB`，peak hotspot 仍是最终 `ResizeBilinear_1`；即使三层 skip 同时存在，也依然没有越过当前尾部峰值。
- 但 `globalgate4x_bneckeca_skip8x4x2x` 的 Vela `inference_time ≈ 182.915 ms`，已经明显高于 `skip8x4x` 的 `178.319 ms`。主要拖累来自 `/2`：`skip_2x_add` 单层 `Network% ≈ 1.29`，是整个多尺度联合版里最贵的单点。
- 这轮“真正多尺度同时存在”的结论已经非常明确：在当前基座和实现方式下，`/2` 是联合版里最不划算的那一级；`/8 + /4` 可以保留为多尺度训练候选，但 `+ /2` 后代价上升已经过于明显。
- 当前如果只想挑少数值得训练的版本，我会把多尺度 shortlist 收缩成：
  - `globalgate4x`
  - `globalgate4x_bneckeca`
  - `globalgate4x_bneckeca_skip8x`
  - `globalgate4x_bneckeca_skip8x4x`
- 已对 fixed-arch joint training 的三个候选模型做结构级 `Vela` 预检，输入分辨率固定为和前面部署一致的 `172x224`，输出仍为 `176x224`。
- 这轮 `Vela` 预检使用的是 fixed subnet `0,2,1,1,0,0,0,0,0` 的随机初始化权重，只用于判断结构带来的 `SRAM peak` 与 `inference_time/FPS`，不代表训练后的精度。
- fixed-arch `baseline` 的 `Vela` 结果是：`SRAM peak = 1386.00 KiB`，`inference_time ≈ 166.179 ms`，`FPS ≈ 6.018`。
- fixed-arch `globalgate4x_bneckeca` 的 `Vela` 结果是：`SRAM peak = 1386.00 KiB`，`inference_time ≈ 167.551 ms`，`FPS ≈ 5.968`。
- fixed-arch `globalgate4x_bneckeca_skip8x4x2x` 的 `Vela` 结果是：`SRAM peak = 1386.00 KiB`，`inference_time ≈ 175.810 ms`，`FPS ≈ 5.688`。
- 这三个 fixed-arch 候选在 `172x224` 上都没有突破当前 `1386.00 KiB` 的峰值上限，说明训练 shortlist 与此前 bilinear 结构实验在部署侧约束上是对齐的。
- 三个 fixed-arch 候选的 `Vela` hotspot 仍都落在最终 `ResizeBilinear_1`，没有因为 `global gate` 或三尺度 skip 把峰值前移。
- `globalgate4x_bneckeca` 相比 fixed-arch `baseline` 只慢约 `0.83%`，代价非常小，仍适合作为稳妥消融项。
- `globalgate4x_bneckeca_skip8x4x2x` 相比 fixed-arch `baseline` 慢约 `5.80%`，仍明显低于用户当前“20% 内都可考虑”的阈值，因此它完全可以作为“先看 accuracy 上限”的训练主力版本。
- fixed-arch `172x224` joint training 跑到 `epoch 80` 后，三模型的差距并不大，这本身是一个重要信号：当前改造没有带来“立刻拉开”的精度收益。
- 从用户提供的 `comparison.csv` 看，`globalgate4x_bneckeca` 在 `16` 个评估点里有 `14` 次优于 baseline，最终 `epoch 80` 的 `EPE` 是 `4.1486`，baseline 是 `4.1606`；这说明 gate-only 版本大概率是“轻微有效”，但收益幅度很小。
- 同一份 `comparison.csv` 里，full 版 `globalgate4x_bneckeca_skip8x4x2x` 只在 `16` 个评估点里有 `3` 次优于 baseline，最终 `epoch 80` 的 `EPE = 4.2107`，反而比 baseline 差约 `0.0502`；这说明 full 版至少在当前训练阶段还没有兑现它的结构上限。
- 这里不能简单解读成“full 结构无效”，更合理的解释是“当前它比 baseline 更难优化”。因为这次训练只到 `80/400 epochs`，学习率仍在 `9.05e-5`，离初始 `1e-4` 很近，实际上还处在比较早的训练阶段。
- 另一个重要原因是当前 full 版的 `/4` 和 `/2` skip 在 `172x224` 下都包含静态 `PAD` 对齐；这对部署是安全的，但对学习未必理想。也就是说，这版 full 并不是一个“几何上很干净的 U-Net skip”，而是一个带边界补齐的工程实现。
- full 版还同时叠加了 `bottleneck ECA + /4 global gate + /8 /4 /2 additive skip`。这会增加优化耦合度；在完全沿用 baseline 学习率、batch size、loss 权重的情况下，更复杂的版本更容易表现为“前期收敛慢，而不是最终上限低”。
- 当前损失函数不是纯 EPE/L1，而是带 uncertainty 分支的 multiscale loss。更复杂的 full 结构可能先把容量用在 uncertainty 分支的重标定上，而不会立即转化成更低的 flow EPE；这也是“loss 在下降，但 EPE 优势没有同步拉开”的合理解释之一。
- 评估口径本身也值得标注：当前 `_evaluate_model()` 使用的是 `Train-Mode BN`，不是严格的 inference-mode BN。这个口径对三模型相互比较仍然有参考价值，但会放大 batch 统计带来的波动，因此不应该把当前 `±0.05 EPE` 量级的差距过度绝对化。
- 综合判断：到 `epoch 80` 为止，当前最稳的结论不是“full 失败”，而是：
  - `globalgate4x_bneckeca` 已经显示出小而稳定的正收益；
  - full 版目前更像“潜在上限更高，但需要更长训练或更合适超参”的模型；
  - 如果现在立刻停训，只凭 `epoch 80` 就否定三尺度 full，是不够严谨的。

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
- `R2 Lite ASPP` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_liteaspp/172x224`
- `R2 Lite ASPP` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184001.log`
- `R3 ECA` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_eca/172x224`
- `R3 ECA` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_184842.log`
- `globalgate4x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x/172x224`
- `globalgate4x` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193049.log`
- `globalgate2x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate2x/172x224`
- `globalgate2x` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_193850.log`
- `globalgate4x_eca` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_eca/172x224`
- `globalgate4x_eca` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_194450.log`
- `compressedskip2xadd` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_compressedskip2xadd/172x224`
- `compressedskip2xadd` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_212019.log`
- `shareddualgate4x2x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_shareddualgate4x2x/172x224`
- `shareddualgate4x2x` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_212810.log`
- `globalgate4x_bneckeca` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca/172x224`
- `globalgate4x_bneckeca` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_213812.log`
- `globalgate4x_bneckeca_skip4x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip4x/172x224`
- `globalgate4x_bneckeca_skip4x` board log:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_221625.log`
- `globalgate4x_bneckeca_skip2x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip2x/172x224`
- `globalgate4x_bneckeca_skip8x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x/172x224`
- `globalgate4x_bneckeca_skip8x4x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x4x/172x224`
- `globalgate4x_bneckeca_skip8x4x2x` Vela dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear_globalgate4x_bneckeca_skip8x4x2x/172x224`
- fixed-arch three-model Vela summary:
  `/home/enmin/MCUFlowNet/EdgeFlowNAS/outputs/fixed_arch_vela_compare/172x224/summary.json`
- fixed-arch baseline Vela dir:
  `/home/enmin/MCUFlowNet/EdgeFlowNAS/outputs/fixed_arch_vela_compare/172x224/baseline/vela`
- fixed-arch `globalgate4x_bneckeca` Vela dir:
  `/home/enmin/MCUFlowNet/EdgeFlowNAS/outputs/fixed_arch_vela_compare/172x224/globalgate4x_bneckeca/vela`
- fixed-arch `globalgate4x_bneckeca_skip8x4x2x` Vela dir:
  `/home/enmin/MCUFlowNet/EdgeFlowNAS/outputs/fixed_arch_vela_compare/172x224/globalgate4x_bneckeca_skip8x4x2x/vela`
- supported ops copy:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md`

## Visual/Browser Findings

- 当前主要信息来自本地 markdown、Vela CSV、`detailed_performance.txt` 和板端 UART 日志。
- 关键文字事实已落入本文件与
  [00-current-bilinear-baseline.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/experiments/optical-flow-bilinear-sram-fps-20260313/00-current-bilinear-baseline.md)。
