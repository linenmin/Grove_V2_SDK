# Task Plan: MCUFlowNet — 三模型部署 + Sintel 评估 + QAT 决策

## Goal

在 Himax WE2 (Grove Vision AI V2, CM55M + Ethos-U55, 2 MiB SRAM) 上完成 3 个光流模型的板端部署 + 实时可视化 + Sintel EPE 评估，最终基于评估结果决定是否引入 QAT，并给出每个模型的 INT8 EPE 与 FP32 baseline 的差距数字。

## Current Phase

Phase 3（M2 EdgeFlowNAS retrain_v3 子网部署）— M1 Phase 1/2 已闭环，QAT 不需要

## Phases

### Phase 1: M1 (EdgeFlowNet 原版 transpose-conv) 部署 — **complete**
- [x] PTQ INT8 导出 (`optical_flow_157x203.tflite`)
- [x] Vela 编译 (`optical_flow_157x203_vela.tflite`, 1430 KiB SRAM peak)
- [x] xmodem 烧写到 Flash `0x3AB7B000`
- [x] 板端 NPU 推理跑通（`[NPU_MODE] Ethos-U55`, INVOKE resolution=[208,160]）
- [x] Windows flow_viewer.py 实时光流可视化（粗糙但可识别运动方向）
- **Status:** complete

### Phase 2: M1 Sintel EPE 评估 + QAT 决策 — **complete**
- [x] 写 INT8 TFLite EPE evaluator（`tools/eval/int8_sintel_eval.py`，支持 `--eval-grid {native,pred}`）
- [x] 写 FP32 evaluator（`tools/eval/fp32_sintel_eval.py`，同 pipeline 同 grid，apples-to-apples）
- [x] FP32 baseline (native grid)：6.7915 avg / 2.1971 median (1041 frames)
- [x] INT8 evaluation (native grid)：6.9238 avg / 2.3303 median (1041 frames)
- [x] **ΔEPE (INT8 − FP32) = +0.1323**，远低于 0.3 阈值 → **QAT 不立项**
- [x] Discord 推送结果（INT8 阶段已推；FP32 完成后再推一次）
- **Status:** complete

### Phase 3: M2 (EdgeFlowNAS retrain_v3 子网) 部署 — **in_progress**
- [x] 写 export 脚本 `tools/model_export/edgeflownas_v3/run_export.py`：FixedArchModelV3 graph + 输入归一化 `(x-127.5)/127.5` 烧进 graph + PTQ INT8 + Vela
- [x] 3 个候选子网 (v3_acc / v3_efn_fps / v3_light) 在 157×203 都成功 INT8+Vela 导出
- [x] Sintel Final EPE @ 157×203（同 mainline 方法学）：v3_acc 10.66 / v3_efn_fps 10.67 / v3_light 10.93 (vs mainline 7.79)
- [ ] **决策点**：v3 SRAM peak (1143 KiB) 比 mainline (1430 KiB) 少 287 KiB → 应放大 input 尺寸利用余量，再评估
  - [ ] 找到 Vela peak 接近 1432 KiB 的最大输入尺寸（试 172×224、200×256 等）
  - [ ] 重新跑 EPE → 看能否反超 mainline 7.79
- [ ] 板端烧录最佳 v3 子网 + flow_viewer 验证
- **Status:** in_progress

### Phase 4: M3 (第三个模型，TBD) 部署
- [ ] 模型类型与权重路径待定
- [ ] 同上流水线
- **Status:** pending

### Phase 5: 综合对比 + 论文/报告数据
- [ ] 三模型 (FP32, PTQ INT8, [QAT INT8?]) EPE 表
- [ ] 板端 latency / SRAM peak / Flash size 对比
- [ ] 可视化样例帧
- [ ] 是否需要 QAT 的最终结论
- **Status:** pending

## Key Questions

1. **PTQ 后 EPE 损失多大？** → 决定是否启动 QAT pipeline（QAT 需要重新搭训练 loop，成本不低）。
2. **retrain_v3 子网网络结构和 mainline 差多少？** → 决定 export 脚本需要多大改动（新加 variant module vs 复用现有）。
3. **M3 是什么？** → 占位，用户后续指定。
4. **Sintel 评估的输入分辨率是否对齐板端？** → 板上是 157x203 输入 / 160x208 输出；evaluator 是否要 letterbox 到 157x203 后比 160x208 上采样回原分辨率，还是直接评估 native res 下的 tflite？决定方案要和 `representative_dataset_gen` 一致（直接 resize 到 157x203）。

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 用 WSL2 vela conda env 跑评估 | TF 2.15.0 + cv2/skimage/scipy/tqdm/termcolor 全齐；export pipeline 也在这里跑过，量化参数复现性最稳 |
| 评估目标用 pre-Vela `optical_flow_157x203.tflite`（非 `_vela.tflite`） | Vela 只重排算子调度给 NPU，权重和量化 scale/zp 不变；标准 TFLite interpreter 跑这个即代表板上数值行为 |
| evaluator 输入流水线对齐 `representative_dataset_gen`：cv2.imread → resize(W=203,H=157) → concat(prev,curr,axis=2) → int8 量化 | 保证量化误差和板端一致；不引入额外 preprocessing 偏差 |
| Sintel 数据路径 `/mnt/g/AI_thesis/datasets/MPI-Sintel-complete/training/` | U 盘挂载，跑前先确认 mount 稳定 |
| M1 部署已完成不再回头改 | 现有 INVOKE resolution=[208,160] + HSV 渲染链路 OK，先把数字测出来 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `ModuleNotFoundError: xmodem` 在 system python | 1 | Pipeline 用 `python3` 走 `/usr/bin/python3`，无 xmodem。把 `/home/enmin/miniconda3/envs/vela/bin` 加到 PATH 头部，让 `python3` 解析到 vela env（已装 xmodem 0.4.7 + pyserial 3.5） |
| `flow_viewer.py` 收到大量 `[sync] Skipped N bytes` + `Corrupt JPEG` | 待 1 | 现象：UART 同步打滑，疑似板→PC 端字节流抖动。用户表示窗口能看到光流图（粗糙）。先记录，不在 Phase 2 解决；后续若评估通过再回头排查（可能是波特率边际 / 板端 JPEG block size 对齐 / PC 端 read buffer 太小） |
| Windows `cmd /c "conda activate gpu_env && python ..."` 输出为空 | 1 | conda hook 在非 Anaconda Prompt 下失败。直接调 `D:/Anaconda3/envs/gpu_env/python.exe -u flow_viewer.py COM3 921600`，绕开 activate |

## Notes

- 总线索引：[plan/README.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/README.md)（旧主线说明）
- 当前 MCUFlowNet 部署专用计划在本目录，**不进入** plan-NNN 编号体系，避免和老调试归档混
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
- 每个 Phase 完成后把数字 / 异常 / 决定写到 findings.md，进度时间线写到 progress.md
