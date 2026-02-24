# Plan 009：光流可视化 Agent 自动调试闭环（已归档）

> **状态**: 已归档至 plan-010
> **结论**: Vela 编译后模型在 Ethos-U NPU 上执行时，第二帧输入（curr）未被有效消费

---

## 1. 目标

使用 Agent 可见调试闭环，解决光流估计输出全白或条纹问题。

---

## 2. 调试闭环流程

```bash
# 1) 抓取 + 提取帧
run_optical_pipeline.sh --mode nomodel --app-type optical_cam_oflow --extract-frames

# 2) 从已有 log 提取
python3 scripts/extract_invoke_frames_from_log.py --log <pipeline.log> --output-dir logs/flow_frames/latest
```

---

## 3. 关键实验记录（R20-R28）

### 3.1 合成输入对照（R20）

| 配置 | shift | center dx/dy | 输出变化 |
|------|-------|--------------|----------|
| A组 | (0,0) | -12.925/0.847 | 基线 |
| B组 | (3,1) | -12.925/0.847 | **无变化** |

**发现**: 输出对可控输入位移变化不敏感

### 3.2 极端对照（R21-R22）

| 实验 | prev | curr | center dx/dy | 输出变化 |
|------|------|------|--------------|----------|
| R21 | 纹理 | 常量 | -12.925/0.847 | 无 |
| R22 | 常量 | 纹理 | 2.330/2.330 | **有** |

**结论**: 输出对 prev 敏感，对 curr 不敏感

### 3.3 扰动实验（R27-R28）

| 实验 | 扰动目标 | in1_sum 变化 | in2_sum 变化 | mean_dx 变化 |
|------|----------|--------------|--------------|--------------|
| R27 | curr | 1.0x | 2.08x | 1.48x |
| R28 | prev | 1.45x | 1.0x | 1.12x |

**结论**: 动态场景下两半区都有响应，但基线波动大

### 3.4 离线验证（R25-R26）

| 环境 | 模型 | center dx/dy | 说明 |
|------|------|--------------|------|
| TFLite CPU | singlescale non-vela | -0.011/-0.001 | 正确 |
| 板端 NPU | singlescale vela | -22.249/-22.037 | 异常 |

**关键发现**: 首层权重通道 0-2 与 3-5 范数比≈1.02（正常），排除权重塌缩

---

## 4. 最终结论

**Vela 编译后的模型在 Ethos-U NPU 上执行时，输入张量后半区（curr 帧）未被有效消费。**

证据链：
1. R21: curr 常量化 → 输出不变
2. R22: prev 常量化 → 输出变为常量场
3. R25: non-vela 模型首层权重正常
4. R26: TFLite CPU 对合成输入正确响应

---

## 5. 配置快照

```c
// 关键宏
tensor_arena_size = 1408 * 1024
FLOW_DBG_PERTURB_ENABLE = 0  // 调试后关闭
FLOW_DBG_SYNTH_INJECT = 0    // 使用真实输入
CAM_INPUT_USE_BGR = 1        // 与导出脚本一致
```

---

## 6. 续篇

详见 **plan-010-vela-input-channel-issue.md**，包含：
- Vela 编译配置调优计划
- 模型导出验证步骤
- 离线对比验证方法
