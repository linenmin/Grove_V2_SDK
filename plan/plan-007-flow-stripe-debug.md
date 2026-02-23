# Plan 007：光流条纹问题迭代调试计划

## 1. 当前现象（2026-02-23）

- **已确认**：flow 数据对运动有响应
  - 挥手时出现黑线，随挥手频率闪动
  - 延迟低，说明 flow → 渲染 → INVOKE 链路正常
- **待解决**：画面为垂直黑白条纹，而非期望的「亮=运动区域」灰度图
- **JPEG_Q_HIGH / JPEG_Q_BEST** 均已尝试，条纹未消失，故主要怀疑非 JPEG 压缩质量导致

---

## 2. 问题定位假设（按优先级）

### H1：flow tensor 空间布局/周期性与条纹对齐

- **假设**：模型输出的 flow 具有列向或行向周期性（如 8/16 像素），导致条纹
- **验证**：串口输出每帧 flow 的列均值曲线（或每隔 N 列采样），检查是否有周期
- **改动**：在 `ob_log_infer_line` 或新增低频调试中，输出 `col_mean[0..W-1]` 的简化版（如每 8 列一个值）

### H2：per-frame max_mag 归一化导致静态噪声被放大

- **假设**：静态时 max_mag 来自量化噪声，归一化后整幅图接近均匀灰；有运动时 max_mag 突增，相对关系变化，可能凸显模型固有的行列结构
- **验证**：改用固定 scale 归一化（如 mag 直接映射到 0–255，用固定阈值截断），观察条纹是否变化
- **改动**：`flow_render_to_gray` 增加模式：固定 scale（mag * k + offset）vs 当前 per-frame max 归一化

### H3：flow tensor 内存布局与读取 stride 不匹配

- **假设**：若模型输出为 NCHW（先整幅 dx，再整幅 dy），当前 stride=2 的 NHWC 读法会错位，产生条纹
- **验证**：查 TFLite 输出 dims 与 layout；或尝试 NCHW 读法对比
- **改动**：按 `out_dims` 判断 layout，分支读取

### H4：JPEG addMCU 的 (x,y) 坐标语义

- **假设**：`jpe.x`, `jpe.y` 若为 MCU 索引而非像素坐标，`gray[jpe.x + jpe.y*width]` 会错位
- **验证**：查阅 JPEGENC 源码，确认 addMCU 的 (x,y) 含义
- **改动**：按正确语义计算像素偏移

### H5：条纹周期与 JPEG MCU（8x8）对齐

- **假设**：条纹若为 8 像素周期，可能与 JPEG 8x8 块边界重合
- **验证**：输出 raw 灰度图（不 JPEG），或改用 PNG/其他格式；若条纹消失则与 JPEG 相关
- **改动**：临时绕过 JPEG，直接发送 raw 小图（需改协议或仅本地调试）

---

## 3. 调试执行策略

1. **一次一假设**：每次只改一个变量，便于归因
2. **最小改动**：优先加调试输出，再考虑改渲染逻辑
3. **验证命令**：`run_optical_pipeline.sh --mode with-model ... --capture-seconds 20 --no-clean`
4. **记录**：每轮写入本 plan 的「调试记录」小节

---

## 4. 建议首轮尝试（H2：固定 scale）

- **理由**：实现简单，可快速排除 per-frame 归一化影响
- **改动**：在 `flow_render_to_gray` 中，用固定 scale 代替 `mag/max_mag`：
  - 例如 `v = mag * 80.0f`（或根据 out_scale 调），`v = min(255, v)`
  - 观察条纹是否减弱、是否出现「挥手区域明显变亮」
- **实现**：增加 `FLOW_VIZ_FIXED_SCALE` 宏，为 1 时走固定 scale 路径

---

## 5. 调试记录

### R1：用户反馈（2026-02-23）

- **现象**：条纹仍存在；挥手时黑线随挥手频率闪动，延迟低
- **结论**：flow 对运动有响应，链路正常；条纹来源待定位
- **下一步**：优先尝试 H2（固定 scale）

### R2：H2 固定 scale 实现（2026-02-23）

- **改动**：`flow_render.cpp` 增加 `FLOW_VIZ_FIXED_SCALE=1`，用 `mag * 80` 代替 `mag/max_mag`
- **验证命令**：`run_optical_pipeline.sh --mode with-model --app-type optical_cam_oflow --port /dev/ttyACM0 --capture-seconds 20 --model-arg '...' --no-clean`
- **用户反馈**：条纹整体变亮，其他没变；条纹仍存在；有一点点动态反馈。

### R3：H1 列均值诊断实现（2026-02-23）

- **改动**：`ob_debug_stats` 增加 `ob_log_col_mean_mag_sample()`，每 8 列一个采样，输出该列 mean magnitude（×1000）
- **验证**：烧录后观察串口 `[col_mean_mag]` 行，若 c0,c8,c16,... 呈交替高低模式，则条纹来自 flow 数据周期；若较平则可能为 JPEG 或 H3/H4
- **结果**：列均值较平坦（3876–4105），无周期 → H1 排除，指向 H5/H4

### R4：H5 尝试1 - JPEG_Q_BEST（2026-02-23）

- **改动**：`flow_render.cpp` 将 `JPEG_Q_HIGH` 改为 `JPEG_Q_BEST`（最高质量）
- **验证**：烧录后 Windows 端观察条纹是否减轻
- **用户反馈**：还是条纹，无明显改善。
- **结论**：H5 尝试1 排除；条纹非 JPEG 压缩质量导致。下一步待定：H5 尝试2（raw 绕过 JPEG 验证）、H3（NCHW 布局）、H4（addMCU 坐标语义）。

---

## 6. 与 plan-006 的关系

- 本 plan 为 plan-006 阶段 D 的后续细化，聚焦「条纹为何存在」的根因定位
- 成功消除条纹后，将关键结论回写 plan-006
