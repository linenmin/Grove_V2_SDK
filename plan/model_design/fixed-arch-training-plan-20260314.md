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

第一轮建议：

- 数据：沿用 `FC2 180x240`
- 训练方式：`joint training`
- 模型数：`2`
- 模型组合：
  - `fixed_baseline`
  - `fixed_globalgate4x_bneckeca_skip8x4x2x`
- 目的：先看最关键的上限差值

如果第一轮结果显示 full multiscale 明显优于 baseline，再开第二轮：

- `fixed_baseline`
- `fixed_globalgate4x_bneckeca`
- `fixed_globalgate4x_bneckeca_skip8x4x2x`

## Pending Confirmation

当前最需要你拍板的只有两件事：

1. 第一轮是先跑 `2-model`，还是直接跑 `3-model`
2. fixed subnet 的 `/4` skip 是否接受我按 `EB0 output` 来实现

我当前推荐：

- 第一轮先 `2-model`
- `/4 skip` 先按 `EB0 output`
