# Task Plan: MCUFlowNet — 三模型部署 + Sintel 评估 + QAT 决策

## Goal

在 Himax WE2 (Grove Vision AI V2, CM55M + Ethos-U55, 2 MiB SRAM) 上完成 3 个光流模型的板端部署 + 实时可视化 + Sintel EPE 评估，最终基于评估结果决定是否引入 QAT，并给出每个模型的 INT8 EPE 与 FP32 baseline 的差距数字。

## Current Phase

Phase 2（M1 EPE 评估，QAT 决策铺垫）

## Phases

### Phase 1: M1 (EdgeFlowNet 原版 transpose-conv) 部署 — **complete**
- [x] PTQ INT8 导出 (`optical_flow_157x203.tflite`)
- [x] Vela 编译 (`optical_flow_157x203_vela.tflite`, 1430 KiB SRAM peak)
- [x] xmodem 烧写到 Flash `0x3AB7B000`
- [x] 板端 NPU 推理跑通（`[NPU_MODE] Ethos-U55`, INVOKE resolution=[208,160]）
- [x] Windows flow_viewer.py 实时光流可视化（粗糙但可识别运动方向）
- **Status:** complete

### Phase 2: M1 Sintel EPE 评估 + QAT 决策 — **in_progress**
- [ ] 写 INT8 TFLite EPE evaluator（基于 `EdgeFlowNet/code/test_sintel.py` 改写，加 tflite interpreter 路径）
- [ ] FP32 baseline：用原 `test_sintel.py` 跑 `best.ckpt` 在 Sintel clean training 上 → 记录 EPE
- [ ] INT8 evaluation：用新写的 evaluator 跑 `optical_flow_157x203.tflite` 在同样 Sintel 子集上 → 记录 EPE
- [ ] 对比 FP32 vs INT8 EPE 差值 → 决策是否需要 QAT（阈值待定，先看绝对差距）
- [ ] Discord 推送结果
- **Status:** in_progress

### Phase 3: M2 (EdgeFlowNAS retrain_v3 子网) 部署
- [ ] 从 HPC 下载的权重 (`D:\Dataset\MCUFlowNet\EdgeFlowNAS\outputs\retrain_v3_ft3d\retrain_v3_ft3d_run1`) 适配到 export 脚本
- [ ] 比对 retrain_v3 网络定义和当前 `run_export.py` 的 variant 表，确定是否需要新增 variant module
- [ ] PTQ INT8 导出 + Vela 编译 → 检查 SRAM peak 是否在 1432 KiB arena 内
- [ ] 板端部署 + 可视化验证
- [ ] Sintel EPE 评估（FP32 + INT8）
- [ ] （如 Phase 2 决定需要 QAT）做 QAT 后重测
- **Status:** pending

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
