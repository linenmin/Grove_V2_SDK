# Plan 015 v2: 反方向光流根因 — 输入 Tensor 未填充当前帧

## 实验结果

`FLOW_VIZ_REMOVE_GLOBAL_MOTION=0` 后烧录验证，**粉色反方向光流依旧存在**。
→ 均值减法不是根因。需要更深层分析。

---

## 关键发现：当前帧从未进入模型输入 Tensor ⚠️

> [!CAUTION]
> 这是一个**严重且致命的输入拼装 Bug**。

### 证据链

**模型期望的输入**：`[1, 144, 192, 6]` NHWC — 每个像素 6 通道 = `[prev_R, prev_G, prev_B, curr_R, curr_G, curr_B]`
- 总大小 = `144 × 192 × 6 = 165,888 bytes`

**代码实际做的**：`g_raw_frame_bytes = W × H × 3 = 82,944 bytes`（仅一帧的 3 通道）

| 代码位置          | 操作                                                         | 效果                                          |
| ----------------- | ------------------------------------------------------------ | --------------------------------------------- |
| 首帧 L1274        | `memcpy(input_ptr, curr_q, g_raw_frame_bytes)`               | 82,944 bytes → 填入 tensor 前半段             |
| 后续帧 L1254-1278 | 捕获新帧到 `curr_q`，但**从未 memcpy 到 input_ptr 的后半段** | channels 3-5 = 上次 Invoke 残留的脏数据       |
| Invoke 后 L1345   | `memcpy(input_ptr, g_curr_q_shadow, g_raw_frame_bytes)`      | 仅覆盖前 82,944 bytes（为下一帧的 prev 准备） |

### 结果

模型每次 Invoke 时：
- **Channels 0-2** (prev): ✅ 正确的上一帧数据
- **Channels 3-5** (curr): ❌ **从未被写入**，包含 Invoke 残留的中间计算垃圾

### 但为什么模型还能输出"有形状"的光流？

由于 NHWC 交错，`memcpy(input_ptr, prev_q, 82944)` 实际写入结果：

```
input_ptr[0] = prev(0,0).R → model 读为 pixel(0,0).ch0 = prev_R ✓
input_ptr[1] = prev(0,0).G → model 读为 pixel(0,0).ch1 = prev_G ✓
input_ptr[2] = prev(0,0).B → model 读为 pixel(0,0).ch2 = prev_B ✓
input_ptr[3] = prev(0,1).R → model 读为 pixel(0,0).ch3 = "curr_R" ✗✗✗
input_ptr[4] = prev(0,1).G → model 读为 pixel(0,0).ch4 = "curr_G" ✗✗✗
input_ptr[5] = prev(0,1).B → model 读为 pixel(0,0).ch5 = "curr_B" ✗✗✗
input_ptr[6] = prev(0,2).R → model 读为 pixel(0,1).ch0 = prev_R ✗
...
```

模型以为：
- pixel(0,0) 的 "prev" = 实际 prev 帧的 pixel(0,0)
- pixel(0,0) 的 "curr" = 实际 prev 帧的 **pixel(0,1)**（相邻像素！）

→ 模型看到的是**同一帧内相邻像素之间的差异**，不是真正的帧间运动！
→ 每条边缘产生 ±1 像素的"假"光流 → 输出像边缘检测器
→ 整张图被水平压缩 2 倍（每 2 个 prev 像素被挤进 1 个 model pixel）

### 这完美解释了所有现象

| 现象                 | 解释                                                                         |
| -------------------- | ---------------------------------------------------------------------------- |
| "双影残差"           | 模型看的不是 prev↔curr，而是 prev 内自身的相邻像素差，左右边缘各产生一次响应 |
| 反方向粉色光流       | 边缘的一侧是正向梯度差、另一侧是反向梯度差 → 颜色相反                        |
| "帧差式边缘检测"     | 正是因为 curr 帧完全缺失，模型只能提取 prev 帧内的自差                       |
| 输出对运动"有点响应" | prev 帧每次更新，所以不同帧的 prev 内容不同，间接反映了运动                  |

### D5 注释的佐证

代码 L113 和 L1150 的注释写道：
> `// D5: NHWC 转换临时缓冲区（存储 prev_q 用于交错转换）`

说明开发者**知道需要做 NHWC 交错**，分配了 `g_prev_q_buffer`，但**交错逻辑从未实现**。

---

## 修复方案

### 核心修复：在 Invoke 前将 prev 和 curr 正确交错到 input tensor

在 `cv_yolov8n_ob_run` 中，Invoke 前加入交错拼装函数：

```cpp
// 将 prev_q (H*W*3) 和 curr_q (H*W*3) 交错写入 input_ptr (H*W*6)
// NHWC 布局：每像素 [prev_R, prev_G, prev_B, curr_R, curr_G, curr_B]
static void interleave_prev_curr_nhwc(int8_t *dst_6ch,
                                       const int8_t *prev_q,
                                       const int8_t *curr_q,
                                       size_t pix_cnt)
{
    for (size_t i = 0; i < pix_cnt; ++i) {
        const size_t s3 = i * 3U;
        const size_t d6 = i * 6U;
        dst_6ch[d6 + 0] = prev_q[s3 + 0];
        dst_6ch[d6 + 1] = prev_q[s3 + 1];
        dst_6ch[d6 + 2] = prev_q[s3 + 2];
        dst_6ch[d6 + 3] = curr_q[s3 + 0];
        dst_6ch[d6 + 4] = curr_q[s3 + 1];
        dst_6ch[d6 + 5] = curr_q[s3 + 2];
    }
}
```

### 配套修改

1. **`g_prev_q_buffer` 恢复为无条件分配**（撤回我之前的 Level A 清理）— 现在它有正当用途：存储上一帧
2. 在每帧处理流程中：
   - 将 prev_q 存入 `g_prev_q_buffer`（在相机抓帧前保存）
   - 相机抓帧到 `curr_q`
   - 调用 `interleave_prev_curr_nhwc(input_ptr, g_prev_q_buffer, curr_q, pix_cnt)` 
   - 然后 Invoke
3. Invoke 后：将 `curr_q` 复制到 `g_prev_q_buffer`（为下一帧提供 prev）

### 修改文件清单

#### [MODIFY] [cvapp_yolov8n_ob.cpp](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_yolov8n_ob.cpp)

- 新增 `interleave_prev_curr_nhwc()` 函数
- 恢复 `g_prev_q_buffer` 的无条件分配
- 重写 `cv_yolov8n_ob_run` 的帧对拼装逻辑：prev 存到 buffer → 抓当前帧 → 交错 → Invoke → 更新 prev

---

## 验证方案

烧录后观察：
1. **反向光流应消失**：背景区域不再出现粉色反方向光流
2. **形状应更清晰**：模型终于能看到真正的两帧差异
3. **运动响应应更准确**：棋盘格移动方向和光流颜色应一致

```bash
./.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode nomodel \
  --app-type optical_cam_oflow \
  --port /dev/ttyACM0 \
  --capture-seconds 30 \
  --extract-frames --max-frames 8
```

---

## v3 增量更新 (2026-02-25 22:48)

### R1: NHWC 输入交错 + Planar 输出 (FLOW_VIZ_OUT_PLANAR=1)

**修改**：实现 `interleave_prev_curr_nhwc()`，`g_prev_q_buffer` 无条件分配。
**现象**：灰度网点抖动画面 + 偶尔出现少量彩色散点。
**分析**：NHWC 输入正确后，模型产生了真实光流。但 `FLOW_VIZ_OUT_PLANAR=1` 误读 NHWC 输出 → 每隔一像素读错通道 → 棋盘格抖动。
**日志确认**：`[NHWC] interleaved prev+curr into input tensor (27648 pixels)` ✅

### R2: NHWC 输入交错 + NHWC 输出 (FLOW_VIZ_OUT_PLANAR=0)

**修改**：`FLOW_VIZ_OUT_PLANAR` → 0。
**现象**：
- 彩色帧：大面积饱和色（红、品红、青、紫）覆盖画面，颜色混乱
- 灰度帧：出现实际摄像头画面内容（灰度，有六边形/多边形图案叠加）
- 两种模式间歇交替

**日志关键数据**：
```
model io: in(h=144,w=192,c=6) out(h=144,w=192,c=2)
[quant] in: type=9 scale=1.000000 zp=-128 | out: type=9 scale=0.499424 zp=-1
[NHWC] input[0..11]: -115 -115 -115 -115 -115 -115 | -115 -114 -114 -115 -115 -116
```
**Raw log**: `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260225_224328.log`

### R2 根因分析

#### 1. 为什么彩色帧全是饱和色？

`flow_render.cpp` L541:
```cpp
float val = mag * 2.0f;
if (val > 1.0f) val = 1.0f;
```

输出量化：`scale=0.499424, zp=-1`。int8 范围 [-128, 127] 反量化后：
- 最小位移：`(-128 - (-1)) * 0.499 = -63.4 像素`
- 最大位移：`(127 - (-1)) * 0.499 = +63.9 像素`

`mag = sqrt(dx² + dy²)` 的范围可达 **90 像素**。而 `val = mag * 2.0`，意味着只要 `mag > 0.5 像素`（即 dx 或 dy 的原始 int8 量化偏差超过 ±1），颜色就**完全饱和**。

→ **模型产生的真实光流幅度远大于 0.5 px**，几乎所有像素都饱和 → 整帧被鲜艳颜色覆盖。

#### 2. 为什么灰色帧显示摄像头画面？

RGB JPEG 颜色丰富 → 高熵 → 压缩后体积大 → 超出 24KB buffer → 回退到灰度模式。

灰度模式下 `FLOW_VIZ_GRAY_COMPONENT=0` 计算 `|flow|` 幅值。由于模型的光流幅度与图像纹理边缘相关（纹理区域有更多光流响应），灰度幅值图看起来像摄像头画面。

#### 3. 是否需要重新判断 Planar vs NHWC？

- R1 (Planar): 明显网点抖动 = Planar 读 NHWC 数据的经典症状
- R2 (NHWC): 彩色帧有连续色块（不是网点），说明 NHWC 读取是对的
- **结论**：**输出确实是 NHWC**，`FLOW_VIZ_OUT_PLANAR=0` 是正确的

### R3 计划：降低可视化增益

**假设**：模型输出是正确的真实光流，但可视化增益太高（`mag * 2.0`），导致所有像素饱和。

**修改内容**：

#### [MODIFY] flow_render.cpp

1. 降低 `flow_render_to_rgb` 和 `flow_render_rgb_to_jpeg_block` 中的增益：
```diff
-float val = mag * 2.0f;
+float val = mag * 0.05f;  // 降低 40 倍：mag=20px 时 val=1.0
```

2. 同时关闭 `FLOW_VIZ_RGB_OUTPUT` 暂时强制灰度，避免 RGB JPEG buffer 溢出干扰观察：
```diff
-#define FLOW_VIZ_RGB_OUTPUT 1
+#define FLOW_VIZ_RGB_OUTPUT 0
```

**验证**：灰度模式下应该看到清晰的光流幅值图——运动区域亮，静止区域暗。

### R3 结果 (2026-02-25 23:10) ✅

**修改**：`val = mag * 0.05f`（RGB/Gray 渲染），`FLOW_VIZ_RGB_OUTPUT=0`（强制灰度）。
**编译错误**：首次编译失败（缺失 `#endif` for `FLOW_VIZ_LIGHT_SMOOTH` 和 `FLOW_VIZ_FIXED_SCALE`），已修复。
**现象**：✅ **灰度光流输出正常！** 运动区域亮，静止背景暗，符合物理规律。
**结论**：
1. **NHWC 输入交错修复是正确的** — 模型终于收到了正确的两帧输入
2. **NHWC 输出读取是正确的** — `FLOW_VIZ_OUT_PLANAR=0` 是对的
3. **增益 mag*0.05 合理** — 灰度幅值图清晰可辨

### R4: 切换到彩色光流

**修改**：`FLOW_VIZ_RGB_OUTPUT` → `1`，增益保持 `mag * 0.05f`。

### R4 结果 (2026-02-25 23:21) ✅✅

**现象**：✅ **几乎完美的彩色光流输出！**
- 运动方向正确映射到 HSV 颜色
- 背景接近黑色，运动物体有清晰的彩色光流形状
- ⚡ **Zoom-in / FOV 问题也同时被解决**：之前 JPEG 和 RAW RGB 之间的视场差异消失了

**Log**: `logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260225_231117.log`
**Commit**: `18f5ab2` ("finish demo yes")

---

## 最终总结

### 解决的 Bugs

| Bug                       | 根因                                                                                            | 修复                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **当前帧未进入模型输入**  | `memcpy` 只写了 82,944 bytes (一帧) 到 165,888 bytes (两帧) 的 6 通道 tensor，curr 帧从未被写入 | 实现 `interleave_prev_curr_nhwc()` 像素级交错拼装 |
| **输出布局误判 (Planar)** | 之前判断为 Planar 是因为输入错误导致输出是噪声，Planar 误读碰巧"看起来好"                       | `FLOW_VIZ_OUT_PLANAR` → `0` (NHWC)                |
| **可视化饱和**            | `val = mag * 2.0f` 使 0.5px 以上的位移全部饱和                                                  | 增益降至 `mag * 0.05f`                            |
| **反方向粉色光流**        | 根因就是输入缺失 curr 帧                                                                        | 同上第一条修复                                    |
| **Zoom-in / FOV 差异**    | 同上，正确输入后 FOV 匹配自然恢复                                                               | 同上                                              |

### 修改的文件

| 文件                                                                                                                                                         | 关键修改                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| [cvapp_yolov8n_ob.cpp](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_yolov8n_ob.cpp) | 新增 `interleave_prev_curr_nhwc()`，`g_prev_q_buffer` 存前帧，Invoke 前交错拼装 |
| [flow_render.cpp](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/viz/flow_render.cpp)                | `PLANAR=0`, `REMOVE_GLOBAL_MOTION=0`, 增益 `mag*0.05f`                          |
