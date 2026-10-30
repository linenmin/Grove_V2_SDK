> 历史材料：保留当时记录，当前部署按 docs/START_HERE.md；旧工具路径仅作追溯。

# Optical Flow Bilinear SRAM/FPS Workspace

## 1. 目的

这个工作区用于承接后续一系列 **bilinear 光流模型改造实验**。

本轮实验的唯一优先级：

1. 先看 **Vela 输出**
2. 再看 **上板部署输出**
3. **先不看准确率**
4. 只记录：
   - `sram peak`
   - `推理 fps`

## 2. 当前入口

- 当前 bilinear baseline:
  [00-current-bilinear-baseline.md](00-current-bilinear-baseline.md)
- 固定验证顺序:
  [01-validation-protocol.md](01-validation-protocol.md)
- 实验记录:
  [02-experiment-log.md](02-experiment-log.md)
- 想法清单:
  [03-idea-backlog.md](03-idea-backlog.md)

## 3. 当前结论

- 当前板上可启动的 bilinear 边界是 `172x224 -> 176x224`。
- `172x228 -> 176x240` 已在板端 `AllocateTensors()` 失败。
- 当前 Vela SRAM peak 热点位于 decoder 尾段的 `ResizeBilinear_1`，不是最慢的大卷积本身。

## 4. 工作约定

- 每次只改一个主要想法，避免变量缠在一起。
- 每次都先留下 Vela summary / per-layer 结论，再追加板端日志结论。
- 若某个想法会改导出逻辑，再回看
  [docs/MODEL_EXPORT.md](../../../../MODEL_EXPORT.md)。
