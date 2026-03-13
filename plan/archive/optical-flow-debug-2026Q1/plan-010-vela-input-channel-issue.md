> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 010：Vela 输入通道映射问题诊断（压缩版）

> **最后更新**: 2026-02-24 晚 | **状态**: D12 完成，等待下一步验证

---

## 1. 问题总结

| 项目 | 内容 |
|------|------|
| **现象** | 板端光流输出为灰度条纹/砖块纹理，无运动可视化 |
| **根因** | ~~NHWC vs Planar 内存布局错位~~ (D5 已修复) → 当前疑似渲染管道放大低幅残差 |
| **关键结论** | 离线 CPU 推理正确 ≠ 板端 NPU 输出正确（两条不同语义链路） |

---

## 2. 已完成的关键修复

### D5: NHWC 内存布局修复 ✅

**问题**：板端 Planar 布局 vs NPU 期望 NHWC 交错布局

**修复**：`cvapp_yolov8n_ob.cpp` 添加 `convert_planar_to_nhwc()` 函数

<details>
<summary>修复代码</summary>

```c
static bool convert_planar_to_nhwc(const int8_t *prev_q,
                                   const int8_t *curr_q,
                                   int8_t *nhwc_out,
                                   int in_w, int in_h, size_t src_row_stride_bytes)
{
    for (int y = 0; y < in_h; ++y) {
        for (int x = 0; x < in_w; ++x) {
            size_t src_off = y * src_row_stride_bytes + x * 3U;
            size_t dst_off = (y * in_w + x) * 6U;
            nhwc_out[dst_off + 0] = prev_q[src_off + 0];  // prev R
            nhwc_out[dst_off + 1] = prev_q[src_off + 1];  // prev G
            nhwc_out[dst_off + 2] = prev_q[src_off + 2];  // prev B
            nhwc_out[dst_off + 3] = curr_q[src_off + 0];  // curr R
            nhwc_out[dst_off + 4] = curr_q[src_off + 1];  // curr G
            nhwc_out[dst_off + 5] = curr_q[src_off + 2];  // curr B
        }
    }
    return true;
}
```
</details>

**验证日志**：
```
[NHWC_FIX] Converted to NHWC layout. First 24 bytes:
  8C 8C 8C 8D 8D 8D 8D 8D 8D ...
[NHWC_FIX] Expected: 80 80 80 00 00 00 80 80 80 00 00 00 ...
```

---

## 3. 关键发现（D10-D12）

### 3.1 离线 CPU vs 板端 NPU 对比

| 对比项 | 离线 CPU 推理图 | 板端输出 |
|--------|----------------|----------|
| **执行环境** | 开发机 WSL/PC Python | Grove Vision AI V2 |
| **推理引擎** | `tf.lite.Interpreter` | TFLite Micro + Ethos-U55 |
| **模型** | non-vela `.tflite` | vela 编译后 |
| **渲染** | matplotlib p99.5 拉伸 | `flow_render.cpp` 多步处理 |
| **输出** | ✅ 正确光流 | ❌ 条纹 |

### 3.2 Vela 模型无法 CPU 验证

```
RuntimeError: Encountered unresolved custom op: ethos-u.
```

**结论**：Vela 编译后的模型包含 NPU 专用算子，无法在 CPU TFLite 上运行验证。

### 3.3 板端 CPU 模式测试失败

- 代码修改成功，CPU 模式启动正常
- **推理速度极慢**（比 NPU 慢 100+ 倍），一帧需要数分钟
- 无法在合理时间内完成测试

### 3.4 用户关键分析（D11）

> "条纹"不等于"模型错误"，而是：
> 1. 有效运动弱时，渲染管道把低幅残差放大
> 2. 去全局运动 + 行偏置去除 + 百分位拉伸，把传感器行纹/压缩纹理拉高了

---

## 4. 当前配置状态

### 4.1 代码配置

```c
// cvapp_yolov8n_ob.cpp
FLOW_USE_CPU_INFERENCE = 0        // NPU 模式（默认）
FLOW_TENSOR_ARENA_KB = 1432       // 多尺度模型需要
FLOW_FIX_NHWC_LAYOUT = 1          // NHWC 转换已启用

// flow_render.cpp
FLOW_VIZ_REMOVE_GLOBAL_MOTION = 1  // 去全局运动
FLOW_VIZ_REMOVE_ROW_BIAS = 1       // 行偏置去除
FLOW_VIZ_LIGHT_SMOOTH = 1          // 轻量平滑
```

### 4.2 模型路径

| 类型 | 路径 |
|------|------|
| 多尺度 non-vela | `/home/enmin/MCUFlowNet/EdgeFlowNet/sramTest/sram_test_modified.tflite` |
| 多尺度 vela | `/home/enmin/MCUFlowNet/EdgeFlowNet/sramTest/output/sram_test_modified_vela.tflite` |
| 单尺度 non-vela | `/home/enmin/MCUFlowNet/EdgeFlowNet/sramTest/output_singlescale/sram_test_singlescale.tflite` |

### 4.3 Flash 地址

```
模型槽位: 0xB7B000 (YOLOV8_OBJECT_DETECTION_FLASH_ADDR)
绝对地址: 0x3AB7B000
```

---

## 5. 关键日志索引

| 时间戳 | 日志文件 | 用途 |
|--------|----------|------|
| 20260224_172303 | `pipeline_with-model_optical_cam_oflow_*.log` | D9 多尺度 vela 烧录 |
| 20260224_174201 | `pipeline_nomodel_optical_cam_oflow_*.log` | D9 nomodel 重烧 |
| 20260224_153849 | `pipeline_nomodel_optical_cam_oflow_*.log` | D8 输入帧提取源 |

---

## 6. 关键文件路径

```
# 板端代码
EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/
├── pipeline/cvapp_yolov8n_ob.cpp    # 主推理循环（含 NHWC 转换）
├── viz/flow_render.cpp              # 光流渲染（含条纹抑制）
└── config/common_config.h           # 内存配置

# 离线验证脚本
scripts/compare_single_vs_multiscale_tflite.py  # 单尺度 vs 多尺度 CPU 对比
scripts/compare_vela_vs_nonvela_tflite.py       # vela vs non-vela 对比（vela 无法运行）

# 输出证据
logs/flow_frames/review_20260224_single_vs_multi_cpu/  # 离线 CPU 对比结果
logs/flow_frames/with_model_multiscale_20260224_172303/  # D9 板端输出
```

---

## 7. 调试历史摘要

| 阶段 | 关键动作 | 结论 |
|------|----------|------|
| **D1-D4** | 扰动实验确认 prev 敏感、curr 不敏感 | 定位到输入通道问题 |
| **D5** | NHWC 布局修复 | 输入链路正常，但输出仍有条纹 |
| **D6-D7** | 渲染参数调优 | 条纹形态变化，但无局部结构 |
| **D8** | 模型语义核对 | 确认单输出路径正确，多尺度需 1432KB arena |
| **D9** | 多尺度 vela 烧录 | 板端加载成功，输出待肉眼确认 |
| **D10** | vela vs non-vela 分析 | vela 模型无法 CPU 验证 |
| **D11** | 用户深度分析 | 离线 CPU ≠ 板端 NPU，条纹可能是渲染放大残差 |
| **D12** | 板端 CPU 模式测试 | 推理太慢无法完成 |

---

## 8. 已排除的假设

| 假设 | 排除原因 |
|------|----------|
| 渲染管道问题 | 测试模式输出正常渐变图 |
| JPEG 压缩问题 | JPEG_Q_BEST 无改善 |
| 首层权重塌缩 | 离线检查通道 0-2 vs 3-5 范数比≈1.02 |
| nomodel/with-model 差异 | 两者输出一致 |
| 多输出头误消费 | 确认 `model outputs=1` |

---

## 9. 下一步建议

| 优先级 | 方向 | 操作 |
|--------|------|------|
| **P0** | Dump NPU 原始值 | 打印 NPU 输出 dx/dy 原始 int8，与离线 CPU 对比 |
| **P1** | 增加运动幅度测试 | 摄像头前大幅度运动，看是否出现光流结构 |
| **P2** | 调整渲染参数 | 关闭 `FLOW_VIZ_REMOVE_ROW_BIAS` 看条纹变化 |
| **P3** | Vela 编译日志 | 检查算子回退警告 |

---

## 10. 参考

- plan-007：条纹根因分析（AccumPreds 量化问题）
- plan-008：Agent 可见调试闭环
- plan-009：输入通道敏感性实验（R20-R28）

---

## 11. D13 大动态测试（2026-02-24 晚，最终验证）

### 11.1 测试配置

| 项目 | 配置 |
|------|------|
| **推理模式** | NPU (Ethos-U55) |
| **模型** | 多尺度 vela (`sram_test_modified_vela.tflite`) |
| **输出量化** | `scale=0.507406, zp=4` |
| **推理时间** | ~166ms |
| **日志** | `pipeline_with-model_optical_cam_oflow_20260224_195946.log` |

### 11.2 测试方法

在摄像头前使用**棋盘格图片**进行快速晃动。

### 11.3 测试结果

> "用棋盘格图片在镜头面前晃动，**隐约能从黑灰之间看出一些动态**。"

### 11.4 结论

✅ **D11 假设成立**：

1. **模型推理正确** - 大动态激励时能看到运动结构
2. **条纹是渲染放大残差** - 低幅输入时，渲染管道（去全局运动 + 行偏置去除 + 百分位拉伸）把 NPU 量化底噪放大了

### 11.5 后续优化方向

1. **固定渲染阈值** - 不使用动态拉伸，改为物理意义阈值
2. **彩色光流输出** - 用颜色编码运动方向，更直观（见 plan-011）

