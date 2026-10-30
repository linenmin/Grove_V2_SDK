> 历史材料：保留当时记录，当前部署按 docs/START_HERE.md；旧工具路径仅作追溯。

# Idea Backlog

## 待验证方向

- decoder 中某些尺度的 skip connection 改成跳跃式保留 / 删除
- 优先压 `ResizeBilinear_1` 前后的并存特征图
- 优先减少 `Conv53` 与末段 add 分支同时存活时间
- 若结构改动不够，再看是否需要改导出图构图方式

## 使用方式

- 你提出一个新点子后，先放到这里。
- 确认只改一个主变量后，再写入
  [02-experiment-log.md](02-experiment-log.md)
  的下一轮记录。
