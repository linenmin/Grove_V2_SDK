# Fixed-Arch Training Plan (2026-03-14)

## Goal

基于已确认更优的 NAS 子网骨架 `0,2,1,1,0,0,0,0,0`，做一轮真正面向训练的固定架构对比，而不是继续只看 `Vela`。

本轮目标不是继续筛结构，而是回答：

1. 这个更优子网骨架本身，训练后能达到什么水平
2. 在这个骨架上叠加 `globalgate4x_bneckeca_skip8x4x2x` 后，是否能进一步拉开精度
3. 如果加一个中间消融 `globalgate4x_bneckeca`，能否把“gate 的收益”和“多尺度 skip 的收益”拆开

## Baseline And Target

### Backbone Code

- 固定架构码：`0,2,1,1,0,0,0,0,0`

### Model A: Fixed-Arch Baseline

- 含义：只保留这个 fixed subnet 的原始骨架
- 用途：作为本轮所有改造的干净基准

### Model B: Fixed-Arch + `globalgate4x_bneckeca_skip8x4x2x`

- 含义：在同一个 fixed subnet 骨架上，增加：
  - bottleneck-only `ECA`
  - decoder `/4` `global gate`
  - `/8 + /4 + /2` 三尺度 compressed additive skip
- 用途：优先看 accuracy 上限

### Optional Model C: Fixed-Arch + `globalgate4x_bneckeca`

- 含义：只加 bottleneck `ECA` 和 `/4 global gate`，不加三尺度 skip
- 用途：拆解“gate”与“多尺度 skip”的净收益

## Training Recommendation

### Recommended Run Order

1. `fixed baseline` + `fixed full multiscale`
2. 如果结果显示 full multiscale 明显更好，再补 `fixed + globalgate4x_bneckeca`

### Why This Order

- 你当前最想先看到的是上限，而不是最细的消融
- `fixed baseline` 对 `fixed full multiscale` 能最快回答“这条多尺度 skip 主线值不值得”
- `globalgate4x_bneckeca` 更适合作为第二轮解释性消融，而不是第一轮必要项

## Joint Training Or Separate Training

### Recommended: Joint Training In One Process

- 方案：单进程、同一 TF Session、共享同一 batch，同时训练两个或三个模型
- 原因：
  - 数据完全对齐，比较最公平
  - 当前 `standalone_trainer` 已验证这种逻辑稳定
  - 显存压力很低，HPC 上没必要拆成多个任务

### Pros

- 每个 epoch 的对比更直接
- 相同样本、相同数据顺序、相同增广，方差更小
- 日志和 checkpoint 管理更统一

### Cons

- 需要新 trainer 支持“fixed subnet baseline + custom augmented model”的混合图构建
- 不能直接复用当前只支持 `arch_codes` 的 `run_standalone_train.py`

### Separate Training

- 优点：实现最简单，脚本也更解耦
- 缺点：同样本顺序和随机性更难完全对齐，对比不如联合训练干净

### Conclusion

- 推荐先做 `joint training`
- 只有在 HPC 调度或代码复杂度成为阻塞时，才退回 `separate training`

## Implementation Plan

### New Training Entry

建议在 `MCUFlowNet/EdgeFlowNAS` 里新增一套 fixed-arch compare 入口，而不是硬改现有 `run_standalone_train.py`：

- `wrappers/run_fixed_arch_compare_train.py`
- `efnas/engine/fixed_arch_compare_trainer.py`
- `efnas/network/fixed_arch_models.py`
- `configs/fixed_arch_compare_fc2_180x240.yaml`

### Why Not Reuse `run_standalone_train.py` Directly

- 现有入口只能表达：`arch_code -> fixed subnet`
- 现在第二个模型已经不是纯 `arch_code` 子网，而是：
  - `fixed subnet backbone`
  - 加上额外的 `ECA / global gate / multi-scale skip`
- 继续把它塞进 `arch_codes` 接口会把现有 standalone retrain 语义搞乱

## Model Build Assumptions

下面这些是当前最合理、但仍建议显式确认的实现假设：

1. `skip_2x` 取自 `E0` 输出后的 `/2` 特征
2. `skip_4x` 取自 `EB0` 输出后的 `/4` 特征
3. `skip_8x` 取自 `EB1` 输出后的 `/8` 特征
4. bottleneck `ECA` 放在 `DB0` 之后
5. `/4 global gate` 放在 `Up2` 之后、`H0Out` 之前
6. `/8 skip` 放在 `Up1` 之后、`DB1` 之前
7. `/2 skip` 放在 `H1` 上采样之后、`H1Out` 之前

## Recommended Training Setups

### Option A: Two-Model Joint Training (Recommended)

- 模型：
  - `fixed_baseline`
  - `fixed_globalgate4x_bneckeca_skip8x4x2x`
- 优点：
  - 最快看到上限
  - 训练成本最低
  - 最符合你当前目标
- 缺点：
  - 不能拆出 gate-only 的贡献

### Option B: Three-Model Joint Training

- 模型：
  - `fixed_baseline`
  - `fixed_globalgate4x_bneckeca`
  - `fixed_globalgate4x_bneckeca_skip8x4x2x`
- 优点：
  - 能直接看 gate-only 和 full multiscale 的差值
  - 消融信息最完整
- 缺点：
  - 实现稍复杂
  - 单次训练时间更长
  - 第一轮不一定必须要这么全

### Option C: Separate Training

- 模型分开跑
- 优点：
  - 最好实现
- 缺点：
  - 对比不如联合训练干净
  - 管理更碎

## Recommendation Summary

按推荐程度排序：

1. `Option A`
2. `Option B`
3. `Option C`

## Suggested HPC First Run

当前实际已切到：

- 数据：`FC2 172x224`
- 训练方式：`joint training`
- 模型数：`3`
- 模型组合：
  - `fixed_baseline`
  - `fixed_globalgate4x_bneckeca`
  - `fixed_globalgate4x_bneckeca_skip8x4x2x`

原因：

- 用户已经确认 `172x224` 是当前部署对齐分辨率
- `globalgate4x_bneckeca` 的部署代价非常小，适合作为轻量消融
- full 版的部署代价也仍在可接受范围内，值得直接观察 accuracy 上限

## Observed Training Readout (epoch 80)

用户提供的 `comparison.csv` 截至 `epoch 80` 的读数：

- `baseline`: `EPE = 4.1606`
- `globalgate4x_bneckeca`: `EPE = 4.1486`
- `globalgate4x_bneckeca_skip8x4x2x`: `EPE = 4.2107`

### Direct Observations

- `globalgate4x_bneckeca` 在 `16` 个评估点中有 `14` 次优于 baseline，说明它的收益虽然小，但趋势是稳定偏正的。
- full 版只在 `16` 个评估点中有 `3` 次优于 baseline，当前并没有表现出比 baseline 更强的收敛。
- 但 full 版和 baseline 的差距仍然不大，最终只差约 `0.0502 EPE`；这不是一个足以立刻判定“结构失败”的量级。

### Likely Reasons

1. **训练还太早**

- 当前只跑到 `80/400 epochs`
- 学习率仍在 `9.05e-5`，离初始 `1e-4` 很近
- 对于更复杂的 full 版，这更像“尚未收敛充分”，而不是“最终上限已被证明更差”

2. **full 版的多尺度 skip 不是几何上完全干净的 U-Net**

- 在 `172x224` 下，`/4` 与 `/2` skip 都需要静态 `PAD`
- 这对部署安全，但会让 skip 学习到的边界信息不如理想 U-Net 那样干净
- 因此这版 full 更准确地说是“工程可部署的轻量多尺度版”，不是 textbook U-Net

3. **优化难度明显高于 baseline**

- full 版同时引入：
  - bottleneck `ECA`
  - decoder `/4` global gate
  - `/8 + /4 + /2` additive skip
- 这些模块一起上，会增加优化耦合度
- 如果仍沿用 baseline 同一套 `lr / batch_size / schedule`，更复杂模型在前中期收敛更慢是很常见的

4. **当前 loss 不是纯 EPE**

- 训练用的是带 uncertainty 分支的 multiscale loss
- 更复杂的 full 版可能先把容量花在 uncertainty 标定，而不会立刻转化成更低的 flow EPE
- 这会造成“训练 loss 正常下降，但验证 EPE 优势没有同步显现”的现象

5. **当前验证口径会压缩小差距的可信度**

- 当前 `_evaluate_model()` 使用的是 `Train-Mode BN`
- 这对三模型相互比较仍有参考价值，但会引入 batch-stat 噪声
- 因此像 `±0.05 EPE` 这种量级，不应该被过度绝对化

### Current Decision

- 到 `epoch 80` 为止：
  - `globalgate4x_bneckeca` 可以认为是“小幅稳定正收益”
  - full 版可以认为是“暂时没有兑现上限，但还不能判死刑”
- 这意味着：
  - 如果训练预算允许，full 版应该继续跑到更后期再判断
  - 如果训练预算很紧，当前最稳妥的赢家其实是 `globalgate4x_bneckeca`

### Recommended Next Judgment Point

建议不要在 `epoch 80` 就下最终结论，至少等到下面两个节点之一：

1. `epoch 200`
2. `cosine` 学习率明显下降后的后半程

如果到那时 full 版仍持续落后 baseline，再把结论升级成“这条三尺度实现方式不值得继续”会更严谨。

## Next Step: Sintel Evaluator

由于当前 FC2 上三模型差距很小，而且用户明确怀疑“FC2 对这些结构过于简单”，因此下一步不是立刻继续改结构，而是先补跨数据集验证入口。

新增目标：

- 为 fixed-arch joint training 补一个和 `run_standalone_test.py` 对应的 Sintel evaluator
- 能直接从 `outputs/fixed_arch_compare/<experiment>/` 读取：
  - `model_baseline`
  - `model_ablation`
  - `model_full`
- 自动根据 `run_manifest.json` 恢复各自的 `variant`
- 支持评估 `best` 或 `last` checkpoint
- 输出统一的 `json/csv` 汇总，避免手工逐个模型抄 EPE

判断意义：

- 如果在 FC2 上差距小，但在 Sintel 上能明显拉开，那么更合理的归因就是“FC2 对结构差异不够敏感”
- 如果在 Sintel 上仍然拉不开，才更应该反过来审视 full 结构本身或训练超参

## Observed Sintel Readout (best.ckpt)

用户补充的 `Sintel Final train` 结果如下：

- `baseline`: `fc2_val_epe = 4.044281`，`sintel_epe = 6.001463`
- `globalgate4x_bneckeca`: `fc2_val_epe = 3.985728`，`sintel_epe = 5.579553`
- `globalgate4x_bneckeca_skip8x4x2x`: `fc2_val_epe = 3.920078`，`sintel_epe = 5.647002`

### Direct Interpretation

- 这已经足够说明当前结构改造不是“只在 FC2 上有效”。两个改造版在 `Sintel` 上都明显优于 baseline。
- `globalgate4x_bneckeca` 是当前最稳的赢家：它不仅在 `FC2 val` 上优于 baseline，在 `Sintel` 上也拿到了三者中最好的结果。
- full 版在 `FC2 val` 上最好，但在 `Sintel` 上没有超过 `ablation`。这说明 full 更像一个高容量模型，而不是一个天然更稳的跨域模型。

### Why The Epoch Feels Unclear

这里最关键的是 checkpoint 语义：

- `run_sintel_test.py --ckpt_name best` 读取的是训练期间按 `FC2 val EPE` 最低时保存的 `best.ckpt`
- 不是按 `Sintel EPE` 选出来的 `best`

所以当前结果应该理解成：

- “每个模型在自己 `FC2-best` checkpoint 上的 Sintel 表现”

而不是：

- “每个模型在自己 `Sintel-best` checkpoint 上的表现”

这两者对复杂度不同的模型来说，很可能不是同一个 epoch。

### Engineering Interpretation

从工程角度，这组结果最像下面这个模式：

1. `globalgate4x_bneckeca` 是低风险、强泛化增强项

- 结构克制
- 额外部署代价很低
- 对 `FC2` 和 `Sintel` 都有稳定正收益

2. `globalgate4x_bneckeca_skip8x4x2x` 是更高容量的上限模型

- 它对 `FC2` 更有利，说明容量确实被利用上了
- 但它的 `FC2-best` checkpoint 还没有转换成当前最好的 `Sintel` 结果
- 更像“checkpoint selection 还没对上”，而不是“结构没有价值”

### Updated Decision

现阶段最合理的训练/评估结论是：

1. `baseline` 已完成基准使命
2. `globalgate4x_bneckeca` 是当前最值得优先保留和部署验证的版本
3. `globalgate4x_bneckeca_skip8x4x2x` 仍值得继续追踪，但下一步重点应该是验证它的 `Sintel-best epoch`，而不是只盯住当前的 `FC2-best` checkpoint

### Recommended Next Evaluation

如果你后面想把这条线做扎实，下一步最值得补的是：

1. 在训练过程中保留更多 `epoch checkpoint`
2. 对 `full` 和 `ablation` 做一次 epoch-sweep 的 Sintel 评估
3. 明确回答：
   - `full` 的 `Sintel-best epoch` 是否晚于/早于 `FC2-best epoch`
   - `ablation` 的 `FC2-best` 和 `Sintel-best` 是否更接近

## Pending Confirmation

当前最需要你拍板的只有两件事：

1. 第一轮是先跑 `2-model`，还是直接跑 `3-model`
2. fixed subnet 的 `/4` skip 是否接受我按 `EB0 output` 来实现

我当前推荐：

- 第一轮先 `2-model`
- `/4 skip` 先按 `EB0 output`
