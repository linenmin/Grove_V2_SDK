# Fixed-Arch Six-Model Leaderboard Plan (2026-03-15)

## Goal

在不重复训练已完成曲线的 `baseline` 与 `globalgate4x_bneckeca` 前提下，挑 6 个最有希望超过当前 `ablation` 的新模型，继续做 fixed-subnet 联合或单模型训练。

本轮重点不是再证明 `baseline` 是否弱，而是尽快找到**有机会在 Sintel 上超过 `globalgate4x_bneckeca`** 的结构。

---

## Resolution Decision

### Recommendation

本轮 **不要** 把训练分辨率从 `172x224` 改成 `176x224` 或其他 8/16 倍数。

### Why

1. **当前最重要的是和已有曲线做苹果对苹果比较**

- 你已经有：
  - `baseline @ 172x224`
  - `globalgate4x_bneckeca @ 172x224`
  - `globalgate4x_bneckeca_skip8x4x2x @ 172x224`
- 如果现在把输入改成 `176x224`，后续所有变化都会同时混入：
  - 结构变化
  - 输入几何变化
  - 数据裁剪/感受野变化
- 这样就很难判断“进步到底来自结构，还是来自分辨率变化”。

2. **当前部署预检也全部锁定在 `172x224`**

- 你已经完成了 `172x224` 的 `Vela`/部署侧约束验证
- 如果训练切到其他分辨率，等于要重新验证整套部署口径
- 这会把训练成本节省掉的一部分，又从部署回归里补回来

3. **`PAD` 的确可能影响精度，但当前不是主导矛盾**

- 现在最清晰的现象不是“模型都被 pad 害了”
- 而是：
  - `full` 在 `FC2` 上更强
  - 但在 `Sintel` 上没有稳定超过 `ablation`
- 这更像是结构复杂度/泛化的问题，而不是几何对齐本身主导了结果

### Practical Conclusion

- **这一轮六模型训练继续固定 `172x224`**
- 如果后面要专门做“去 pad 的几何清洁版”实验，再单独开一个 resolution branch
- 那时最合理的候选是 `176x224`，因为它离当前部署口径最近

---

## Script Strategy

### Recommendation

继续扩展现有：

- `wrappers/fixed_arch_compare/run_train.py`
- `efnas/network/fixed_arch_models.py`
- `configs/fixed_arch_compare_*.yaml`

**不要** 再新建一套平行训练入口。

### Why

1. 当前 trainer 本来就支持任意数量的模型联合训练
2. CLI 已经支持：
   - `--model_variants`
   - `--model_names`
   - `--backbone_arch_code`
3. 后续你还想单独训练某一个模型，这恰好适合沿用同一个入口：
   - 联合训练：传多个 variant
   - 单模型训练：只传一个 variant

### What To Add

建议在现有脚本上补强，而不是重写：

1. 扩展 `SUPPORTED_VARIANTS`
2. 在 `FixedArchModel` 中为新变体增加布尔开关/路径
3. 增加更明确的参数校验：
   - `model_variants` 和 `model_names` 数量一致
   - 名称不重复
4. 增加一份新的 6-model config：
   - `configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml`

### Single-Model Training

后续单模型训练不需要新脚本，直接支持：

- `--model_variants globalgate8x4x_bneckeca`
- `--model_names target`

这样可以保持：

- 同一训练逻辑
- 同一 checkpoint 规则
- 同一 best-by-FC2 选择口径

---

## Checkpoint Policy

### Recommendation

保持不变：

- `best.ckpt` 仍然按 `FC2 val EPE` 选择

### Why

- 如果用 Sintel 选 best，相当于把目标测试分布泄漏进训练选择
- 那会让后面的跨域结论不再干净
- 当前做法是合理的：训练选择只看源域验证，跨域只做外部观察

---

## Six Models To Train

下面 6 个模型按“最有可能超过当前 `ablation`”排序。

### 1. `globalgate4x_bneckeca_skip8x4x`

**定位：** 从当前 full 版中去掉最可疑的 `/2 skip`

**为什么最值得试：**

- `full` 当前最大嫌疑点就是 `/2 skip`
- `/8 + /4` 更像是保留多尺度信息的主收益，但避免最强的高分辨率细节灌回
- 这是当前最像“有机会直接赢 `ablation`”的版本

### 2. `globalgate8x4x_bneckeca`

**定位：** 在当前冠军 `ablation` 上，只额外增加一个 `/8 global gate`

**为什么值得试：**

- `/8 gate` 的代价应显著低于 `/2 gate`
- 更符合“尽早纠偏”的解码逻辑
- 比继续堆 skip 更克制

### 3. `globalgate8x4x_bneckeca_skip8x`

**定位：** 早期全局导航 + 最便宜的单级 skip

**为什么值得试：**

- 这是一个很平衡的组合：
  - gate 负责方向
  - `/8 skip` 负责低成本补信息
- 有机会比纯 gate 更强，但不会像 full 那样重

### 4. `globalgate4x_dual_eca8_bneckeca`

**定位：** 在 encoder `/8` 再加一个 ECA，配合现有 bottleneck ECA + `/4 global gate`

**为什么值得试：**

- 测“源头特征净化”是否真的对跨域泛化有帮助
- `/8` 是比 `/4` 更稳妥的 encoder 插入点
- 比直接到 decoder 多层堆 ECA 风险低

### 5. `globalgate8x4x_bneckeca_skip8x4x`

**定位：** 强冲榜版，双 gate + 双 skip，但仍刻意避开 `/2 skip`

**为什么值得试：**

- 如果这个版本还能保持泛化，就很可能是当前结构上限之一
- 它比 full 更克制，但表达力明显高于当前 `ablation`

### 6. `skip8x4x`

**定位：** 纯 skip 派诊断版本

**为什么要保留：**

- 它能帮助判断：
  - 是 skip 本身就值
  - 还是 gate/eca 与 skip 的耦合出了问题
- 即使它不是最终冠军，它也能给后续结构选择非常强的解释力

---

## What Not To Prioritize

当前不建议优先把名额投给：

- `globalgate2x4x8x + decoder2x4x8x ECA`
- 带 `/2 global gate` 的更满版本
- `encoder 1/4 ECA + decoder 多层 ECA + 多层 gate` 的全叠版

### Why

- 当前最强的是更克制的 `ablation`
- `full` 的问题更像 `/2 skip` 和高分辨率路径过强，而不是 attention 不够多
- 继续往 `/2` 和 `/4` 全面叠门控，更像增加耦合复杂度，而不是解决当前的核心问题

---

## Implementation Recommendation

### New Config

新增一份：

- `configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml`

默认包含上述 6 个变体。

### CLI Usage

保持当前接口设计：

- 多模型联合训练：`--model_variants a+b+c`
- 单模型训练：`--model_variants a`

不需要为“单模型模式”再发明另一套入口。

### Suggested Validation

本轮代码完成后，先做两步：

1. `py_compile`
2. graph dry-run（1 step synthetic batch）

先确认 6 个变体都能建图，再上 HPC 真训练。

---

## Expected Outcomes

### Best-Case

- `globalgate4x_bneckeca_skip8x4x` 或 `globalgate8x4x_bneckeca` 超过当前 `ablation`

### Medium-Case

- 没有模型超过 `ablation`
- 但能明确知道：
  - 是 `/2 skip` 的问题
  - 还是 gate/eca 与 skip 的耦合问题

### Worst-Case

- 六个都赢不了 `ablation`

若出现这种情况，当前结论基本可以收敛为：

- `globalgate4x_bneckeca` 就是当前 fixed-subnet 下最优的工程折中点
