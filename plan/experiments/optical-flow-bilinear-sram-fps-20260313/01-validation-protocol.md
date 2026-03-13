# Validation Protocol

## 1. 固定验证顺序

每个新想法都严格按下面顺序走：

1. 修改模型结构
2. 导出 TFLite / Vela
3. 先读 Vela summary / per-layer / detailed allocation
4. 记录：
   - `sram peak`
   - peak op 在哪里
   - 是否比 baseline 上升或下降
5. 上板部署
6. 记录：
   - 是否成功进入 `initial done`
   - 是否有稳定 `INVOKE`
   - `resolution` 是否仍是光流尺寸
   - `infer ms`
   - `total ms`
   - 换算算法 fps
7. 本轮先不评价准确率

## 2. 当前 baseline 对照

- baseline model:
  `172x224 -> 176x224`
- baseline Vela peak:
  `1386.00 KiB`
- baseline peak op:
  `ResizeBilinear_1`
- baseline board infer:
  `~178.5 ms`
- baseline algorithm fps:
  `~4.84 fps`

## 3. 判定标准

### 3.1 Vela 侧

- `sram peak` 下降：
  视为正向
- peak hotspot 从 decoder 尾段移开：
  视为有价值变化
- `sram peak` 上升但仍低于当前上机边界：
  可继续上板验证

### 3.2 板端侧

- 能到 `initial done`：
  说明 arena 分配至少过了初始化
- `INVOKE resolution` 保持模型输出尺寸：
  说明可视化没有 fallback
- `infer ms` 下降 / fps 上升：
  视为正向

## 4. 记录格式

每轮实验至少补这几项：

- 想法名称
- 改动点
- 模型 I/O
- Vela peak
- Vela peak op
- 板端是否启动
- `infer ms`
- `total ms`
- `algo fps`
- 结论：保留 / 放弃 / 待复验
