# Progress Log: MCUFlowNet Deployment

按时间线追加，最新条目放最上面。

---

## Next Action

**用户确认排查结论后选路径**

排查结果（findings.md §7d/7e）：v3 的"HPC win"在 deploy res 萎缩主因是**训练 (480×640) / 部署 (157×203) input size mismatch**，不是量化。三条候选路径：

1. **零成本立刻试**：把 v3_efn_fps export 到 172×224 input，跑 EPE 看 Δ_downsample 是否缩小
2. **彻底解**：重训 v3 在 157×203 (或 172×224) input
3. **接受现状**：烧 v3_efn_fps INT8 @ 157×203 上板（EPE 7.34 < mainline 7.79）继续 M3

---

**原 Next Action：烧 v3_efn_fps 到板上，看实时光流是否正常**

修复 `flow_scale=12.5` bug 后，**v3_efn_fps EPE 7.34 < mainline 7.79**，且推理时间 165ms < 188ms。是当前最佳板端候选。

具体步骤：
1. **检查 viz_publish.cpp 渲染系数**
   - mainline output scale = 0.4265，渲染 `mag * 0.05` → 等效物理像素 `int8 * 0.4265 * 0.05`
   - v3_efn_fps 的 output tflite 量化参数要看一下（之前 v3_acc 是 scale 0.054, zp -1，再 ×12.5 才是物理像素）
   - 板端从 int8 直接读 → 渲染时要按 `int8 * scale * 12.5` 计算 magnitude，再决定 `* 0.05` 还是其他系数
   - 最保险：让固件能读取 tflite output scale & 12.5 div 然后动态计算渲染系数；或者临时把固件 `mag * 0.05` 改成 `mag * 0.668`（= 0.0534×12.5）然后烧两个版本
2. **烧 v3_efn_fps INT8 @ 157×203**
   ```
   bash .agent/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
     --mode with-model --app-type optical_cam_oflow \
     --port /dev/ttyACM0 --skip-build \
     --capture-seconds 10 --keyword 'initial done' --keyword 'INVOKE' \
     --extract-frames --max-frames 3 \
     --model-arg 'model_zoo/optical_flow/edgeflownas_v3/v3_efn_fps/157x203/edgeflownas_v3_efn_fps_157x203_vela.tflite 0xB7B000 0x00000'
   ```
3. flow_viewer.py 检查可视化是否正常（光流图，不是相机原图）
4. 如果可视化正常 → M2 部署完成；如果颜色偏暗/偏亮 → 调渲染系数

（可选）放大 input 至 172×224 等用 v3 SRAM 余量 + 重跑 EPE 看能否再降。

---

## 2026-05-11 — Session 6

**用户怀疑 v3 在 deploy res 优势消失"不对"，要求排查。结论**：

跑 FP32 v3 在 157×203 (`tools/eval/fp32_v3_sintel_eval.py`) 与 INT8 v3 在 157×203 对照，外加 v3_acc FP32 @ 416×1024 (5.0898) sanity check 验证我的 evaluator 没毛病：

| Model | FP32 @ 416×1024 | FP32 @ 157×203 | INT8 @ 157×203 | Δ_downsample | Δ_pure_quant |
|---|---:|---:|---:|---:|---:|
| Mainline | 6.31 | 7.71 | 7.79 | +1.40 | +0.08 |
| v3_acc | 5.09 | 8.27 | 8.40 | +3.18 | +0.13 |
| v3_efn_fps | 4.89 | 7.75 | 7.34 | +2.86 | −0.41 (noise) |
| v3_light | 5.58 | 10.60 | 12.73 | +5.02 | **+2.13 ⚠** |

**结论 1**：v3 架构对 input 降分辨率敏感 2-3.6× more than mainline。v3 训练在 480×640，强制 157×203 是 OOD。Mainline 网络结构（无 ECA / global gate / 大量 bilinear ResizeConv）对 input scale 更鲁棒。
**结论 2**：PTQ INT8 在 v3_acc / v3_efn_fps 几乎无损 (Δ ≤ 0.5)，**但在 v3_light 上 +2.13 EPE 异常**（最轻架构通道少，per-tensor INT8 把仅剩精度也丢了）。
**结论 3**：用户怀疑成立 — HPC v3_efn_fps win mainline +1.42 EPE，在 deploy res 萎缩到 +0.45。主因 train/deploy input size mismatch。

**完成**：
- 写 `tools/eval/fp32_v3_sintel_eval.py`，同 pipeline 跑 v3 FP32 比较
- v3_acc FP32 @ 416×1024 sanity: 我的 evaluator 跑出 5.0898 — 与 HPC's 完全一致 ✅
- 3 个 v3 子网 FP32 @ 157×203 完成
- findings.md §7d/7e 写诊断 + 路径修复 4 个候选
- progress.md R16-R19 新增

**待决策**（progress.md Next Action）：
1. 零成本试：v3 @ 172×224 看 Δ_downsample 是否缩
2. 彻底解：重训 v3 在 ~157×203 input（HPC 新一轮）
3. 接受现状：烧 v3_efn_fps @ 157×203 上板继续 M3

---

## 2026-05-11 — Session 5

**核心 bug 修复**：3 个 v3 子网 EPE 异常高的原因是 evaluator 忘乘 `flow_scale=12.5`（v3 训练时 GT 被 div by 12.5，inference 要乘回来）。修复后：

| Model | INT8 @ 157×203 (修复后) | HPC FP32 @ 416×1024 |
|---|---:|---:|
| Mainline | 7.79 | 6.31 |
| v3_acc | 8.40 | 5.09 |
| **v3_efn_fps** | **7.34** ← WIN | 4.89 |
| v3_light | 12.73 | 5.58 |

**v3_efn_fps INT8 @ 157×203 = 7.34，正面超过 mainline 7.79**（−0.45 EPE / −5.8%），且推理 165ms vs mainline 188ms。

**完成**：
- 写 `probe_v3_hpc_eval.py` 调 HPC 自己的 evaluator 验证 sintel_best.ckpt.meta.json 里的 5.09 (v3_acc)、4.89 (v3_efn_fps)、5.58 (v3_light) 都对得上
- v3_acc 在 416×1024 input PTQ INT8 EPE = 5.25 vs HPC FP32 5.09，**Δ_pure_quant = +0.16**，validate 量化几乎无损
- v3_acc/efn_fps/light 在 157×203 重新跑 flow_scale=12.5 拿到真数字
- findings.md §7b/7c 大幅重写；task_plan.md Phase 3 子任务勾选；progress 表 R8-R10 标注 invalid（bug），新增 R11-R15

**新发现：**
- v3 训练时 `(uint8/255)*2-1` 输入归一化和 `flow/12.5` GT 缩放是两件独立的事，缺一不可
- 越轻的 v3 子网（少 block / 小 kernel）越对 input 分辨率降级敏感：mainline +1.48 / v3_efn_fps +2.45 / v3_acc +3.31 / v3_light +7.15 (416×1024→157×203)
- v3_efn_fps 是当前唯一同时打过 mainline EPE 和 latency 的 NAS 子网

**当前阶段**：Phase 3 in_progress，下一步烧 v3_efn_fps 上板验证可视化

---

## 2026-05-11 — Session 4

**完成 Phase 3 第一轮（V3 子网 @ 157×203）：**
- 探索 `efnas.network.fixed_arch_models_v3.FixedArchModelV3` + `retrain_v3_candidates.csv`
- 写 `tools/model_export/edgeflownas_v3/run_export.py`：FixedArchModelV3 graph 构造 + 输入归一化烧进 graph + PTQ INT8 + Vela 一站式
- 重要踩坑：v3 训练用 `(uint8/255)*2 - 1` 归一化，不烧进 graph 直接喂原 uint8 → EPE 爆到 12+。修复后 v3_acc 5-frame smoke EPE 5.13
- 3 个子网 (v3_acc / v3_efn_fps / v3_light) 全部成功 Vela 编译，SRAM peak 都是 1143 KiB（比 mainline 1430 KiB 少 287 KiB）
- 全 Sintel Final EPE：v3_acc 10.66 / v3_efn_fps 10.67 / v3_light 10.93 — 都比 mainline 7.79 差 +2.87~+3.14

**新发现：**
- v3 训练在 480×640 (4:3)，强制 157×203 (1:1.3) 推理可能因 aspect ratio 不匹配损精度
- SRAM 余量 287 KiB 没被利用，v3 真正部署点应该是更大 input
- v3_light 推理 96 ms，比 mainline 188 ms 快近 2× — latency 优势已经显现

**遗留风险：**
- 是否能用更大 input 让 v3 反超 mainline 仍待验证

**当前阶段：** Phase 3 in_progress，下一步放大 input

---

## 2026-05-11 — Session 3

**完成：**
- 用户提醒 `wrappers/run_test.py` 默认数据集是 **Sintel Final**，不是 Clean → 之前 EPE 数字含糊不清的根因
- 用 test_sintel.py + Final 复现用户 6.31 baseline 成功（EPE=6.3117）
- 给 `int8_sintel_eval.py` 和 `fp32_sintel_eval.py` 加 `--ref-mode test_sintel`：复现 `test_sintel.py` 方法学（ResizeNearestCrop @ 416×1024 + clip_val=50 + flow vector 上采样到 patch grid）
- wrapper `run_m1_int8_eval.sh` 默认数据集切到 **Final**，默认 ref-mode = test_sintel
- 跑完三组对照：
  - INT8 test_sintel mode @ Final = **7.7911** (vs 用户 6.31 → +1.48)
  - FP32 test_sintel mode @ Final (157×203 input) = **7.7059**
  - **Δ_pure_quant = INT8 − FP32(同 input) = +0.085 (+1.1%)** ← 量化本身几乎无损
  - **Δ_downsample = FP32(157×203) − FP32(416×1024) = +1.39 (+22%)** ← 输入分辨率丢失
- **结论确认**：QAT 不立项；想降 EPE 必须动 input 分辨率 / 模型结构（这正是 M2 retrain_v3 的研究目标）

**新发现（已落 findings.md §12-§13）：**
- 项目默认评估集切到 Sintel Final
- 完整 Δ 矩阵 = 量化损失 +0.085 + 降分辨率损失 +1.39 = 总 +1.48 vs 论文/FP32 416×1024 baseline

---

## 2026-05-11 — Session 2

**完成：**
- 修 evaluator EPE 方法学 bug：加 `--eval-grid {native,pred}`，default native
- M1 INT8 重跑（native grid 1024×436）：**avg EPE 6.9238**（median 2.33，1041 帧，287s）
- 评估 GPU 适配性：硬件 OK（RTX 4060 Laptop），vela env TF 2.15 是 CUDA build 但缺 CUDA 12.2/cuDNN 8 lib；tf_work_hpc 不是 CUDA build。当前任务 (TFLite int8 + 小模型 FP32) 不需要 GPU
- git commit `d8bc67d`：feat(plan): add MCUFlowNet_Deployment plan + INT8 Sintel evaluator
- 跑 FP32 baseline（test_sintel.py 默认 416×1024）：EPE 5.4649（旁路对照，方法学不同）
- 写 `fp32_sintel_eval.py`（同 INT8 pipeline，apples-to-apples）
- 跑 FP32 native-grid：**avg EPE 6.7915 / median 2.1971**
- **ΔEPE (INT8 − FP32) = +0.1323（+1.9%）→ QAT 不立项**
- Phase 2 闭环；Phase 3 (M2) 进入开局

**新发现（已落 findings.md §10）：**
- 标准 OpFlow EPE 必须在 GT 原分辨率算，否则数字被同比例压缩
- PTQ INT8 几乎无损（Δ=+0.13），高 EPE 长尾来自模型本身（input 157×203 + INT8 ±64px 动态范围 + 多尺度累加结构），不是量化
- 想再降 EPE 必须动模型结构或输入分辨率，不是量化方式

---

## 2026-05-11 — Session 1

**完成：**
- Phase 1 整体跑通：
  - 确认 `optical_flow_157x203_vela.tflite` 已发布在 `model_zoo/optical_flow/157x203/`
  - usbipd attach board (busid 1-2) 到 WSL
  - PATH 注入 vela env 解决 `xmodem` 缺失，跑通 `run_optical_pipeline.sh --mode with-model`
  - 板端日志命中所有成功判据：`[NPU_MODE]`, `[out_tensor=0] dims=[1,160,208,2] scale=0.426478 zp=-4`, `initial done`, `INVOKE ... "resolution": [208, 160]`, `[viz] out=208x160`
  - usbipd detach 后 Windows 端 `flow_viewer.py COM3 921600` 成功握手，看到光流图（粗糙但能识别运动方向）

**新发现（已落 findings.md）：**
- vela env 是当前唯一带 xmodem + pyserial + TF 2.15 + cv2 + skimage 的 env
- 仓库自带 calibration data 只有 PERTURBED_market_3 + PERTURBED_shaman_1 共 100 帧对
- Sintel 数据集在 `/mnt/g/AI_thesis/datasets/MPI-Sintel-complete/training/`（U 盘）
- `test_sintel.py` 不支持 INT8 tflite，要重写
- flow_viewer 有 JPEG sync 打滑问题，画面能看但延迟/丢帧明显

**遗留风险：**
- UART JPEG sync 打滑 — 不影响 EPE 评估（不走串口），延后处理
- U 盘挂载稳定性 — 评估前 sanity check

**当前阶段：** Phase 1 complete, Phase 2 in_progress

---

## Test Results Log

待 Phase 2 evaluator 跑完后追加：

| Run | Model | Quant | Sintel split | #pairs | EPE (avg / med) | Notes |
|-----|-------|-------|--------------|--------|-----------------|-------|
| 2026-05-11 R1 | M1 mainline | PTQ INT8 | train clean | 1041 | 1.8472 / 0.6168 | **METHOD BUG**: pred-grid EPE，单位被压缩 ~3.66×；保留为 debug 参考，不可用 |
| 2026-05-11 R2 | M1 mainline | PTQ INT8 | train clean | 1041 | **6.9238 / 2.3303** | native-grid (1024×436)，标准方法 |
| 2026-05-11 R3 | M1 mainline | FP32 ckpt (test_sintel.py) | train clean | 1041 | 5.4649 / — | 旁路对照, ResizeNearestCrop @ 416×1024, 方法学不同, 不参与 Δ |
| 2026-05-11 R4 | M1 mainline | **FP32 ckpt (native grid)** | train clean | 1041 | **6.7915 / 2.1971** | apples-to-apples vs R2; ΔEPE = +0.1323 → **QAT 不立项** |
| 2026-05-11 R5 | M1 mainline | FP32 ckpt (test_sintel.py 默认) | train **final** | 1041 | **6.3117 / —** | 复现用户记忆 baseline；wrapper 默认指向 Final |
| 2026-05-11 R6 | M1 mainline | INT8 (test_sintel mode) | train final | 1041 | **7.7911 / 2.3706** | ResizeNearestCrop@416×1024 + clip 50；vs R5 Δ=+1.48 |
| 2026-05-11 R7 | M1 mainline | FP32 (test_sintel mode, 157×203 in) | train final | 1041 | **7.7059 / 2.3083** | 隔离纯量化：vs R6 Δ_pure_quant=+0.085；vs R5 Δ_downsample=+1.39 |
| 2026-05-11 R8 | M2 v3_acc | INT8 (test_sintel, 157×203 in, **no flow_scale, BUG**) | train final | 1041 | 10.6637 / 4.1021 | invalid — 忘乘 12.5 |
| 2026-05-11 R9 | M2 v3_efn_fps | INT8 (test_sintel, 157×203 in, **no flow_scale, BUG**) | train final | 1041 | 10.6724 / 4.0929 | invalid |
| 2026-05-11 R10 | M2 v3_light | INT8 (test_sintel, 157×203 in, **no flow_scale, BUG**) | train final | 1041 | 10.9277 / 4.4540 | invalid |
| 2026-05-11 R11 | M2 v3_acc | HPC FP32 (own evaluator, 416×1024 in) | train final | 1041 | **5.0898** / — | 复现 meta.json metric，validate HPC pipeline |
| 2026-05-11 R12 | M2 v3_acc | INT8 (test_sintel, 416×1024 in, flow_scale=12.5) | train final | 1041 | **5.2504** / 1.6180 | 同 HPC 方法学下 PTQ Δ=+0.16，validate 量化无损 |
| 2026-05-11 R13 | M2 v3_acc | INT8 (test_sintel, 157×203 in, flow_scale=12.5) | train final | 1041 | **8.3963** / 5.3634 | Δ vs mainline-INT8 +0.60 |
| 2026-05-11 R14 | **M2 v3_efn_fps** | INT8 (test_sintel, 157×203 in, flow_scale=12.5) | train final | 1041 | **7.3354 / 3.7575** | **Δ vs mainline-INT8 −0.45 → WIN**，165 ms inf |
| 2026-05-11 R15 | M2 v3_light | INT8 (test_sintel, 157×203 in, flow_scale=12.5) | train final | 1041 | **12.7268** / 11.0980 | Δ vs mainline-INT8 +4.94，最轻架构降分辨率敏感 |
| 2026-05-11 R16 | M2 v3_acc | **FP32** (test_sintel, 416×1024 in, flow_scale=12.5) | train final | 1041 | **5.0898** / 1.3641 | sanity vs HPC — match perfectly |
| 2026-05-11 R17 | M2 v3_acc | **FP32** (test_sintel, 157×203 in) | train final | 1041 | **8.2704** / 4.8348 | Δ_downsample +3.18, Δ_pure_quant +0.13 |
| 2026-05-11 R18 | M2 v3_efn_fps | **FP32** (test_sintel, 157×203 in) | train final | 1041 | **7.7501** / 3.4975 | Δ_downsample +2.86, Δ_pure_quant −0.41 (noise) |
| 2026-05-11 R19 | M2 v3_light | **FP32** (test_sintel, 157×203 in) | train final | 1041 | **10.5953** / 7.7511 | Δ_downsample +5.02, **Δ_pure_quant +2.13 ⚠** v3_light PTQ 异常 |

---

## Files Created / Modified

| File | Type | Phase | Notes |
|------|------|-------|-------|
| `plan/MCUFlowNet_Deployment/task_plan.md` | created | meta | 五阶段计划 |
| `plan/MCUFlowNet_Deployment/findings.md` | created | meta | 硬事实索引 |
| `plan/MCUFlowNet_Deployment/progress.md` | created | meta | 本文件 |
| `model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite` | (exist) | 1 | Vela 编译产物，已烧 |
| `tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite` | (exist) | 1 | 未走 Vela，待评 |
| `tools/eval/int8_sintel_eval.py` | created | 2 | INT8 evaluator, `--eval-grid {native,pred}` |
| `tools/eval/run_m1_int8_eval.sh` | created | 2 | INT8 wrapper |
| `tools/eval/fp32_sintel_eval.py` | created | 2 | FP32 evaluator (TF1 graph, same pipeline as INT8) |
| `tools/eval/run_m1_fp32_eval.sh` | created | 2 | FP32 wrapper via original test_sintel.py (416×1024) |
| `tools/eval/run_m1_fp32_native_eval.sh` | created | 2 | FP32 wrapper via new evaluator (native grid) |
| `plan/MCUFlowNet_Deployment/m1_int8_sintel_clean.json` | created | 2 | EPE report (pred-grid, BUG ref) |
| `plan/MCUFlowNet_Deployment/m1_int8_sintel_clean_native.json` | created | 2 | EPE report (INT8 native, 6.92) |
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean.log` | created | 2 | test_sintel.py log (5.46, Clean side-channel) |
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean_native.json` | created | 2 | EPE report (FP32 native, Clean, 6.79) |
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_final.log` | created | 2 | test_sintel.py log (Final, 6.31) |
| `plan/MCUFlowNet_Deployment/m1_int8_sintel_final_test_sintel.json` | created | 2 | EPE report (INT8 test_sintel Final, 7.79) |
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_final_test_sintel.json` | created | 2 | EPE report (FP32 test_sintel mode Final 157×203 in, 7.71) |
| `tools/model_export/edgeflownas_v3/run_export.py` | created | 3 | V3 subnet PTQ INT8 + Vela exporter |
| `tools/model_export/edgeflownas_v3/run_one.sh` | created | 3 | V3 export wrapper (sets PATH for vela) |
| `model_zoo/optical_flow/edgeflownas_v3/v3_acc/157x203/*_vela.tflite` | created | 3 | v3_acc Vela tflite |
| `model_zoo/optical_flow/edgeflownas_v3/v3_efn_fps/157x203/*_vela.tflite` | created | 3 | v3_efn_fps Vela tflite |
| `model_zoo/optical_flow/edgeflownas_v3/v3_light/157x203/*_vela.tflite` | created | 3 | v3_light Vela tflite |
| `plan/MCUFlowNet_Deployment/m2_v3_acc_sintel_final_test_sintel.json` | created | 3 | EPE report v3_acc (10.66) |
| `plan/MCUFlowNet_Deployment/m2_v3_efn_fps_sintel_final_test_sintel.json` | created | 3 | EPE report v3_efn_fps (10.67) |
| `plan/MCUFlowNet_Deployment/m2_v3_light_sintel_final_test_sintel.json` | created | 3 | EPE report v3_light (10.93, no-scale BUG) |
| `tools/eval/eval_int8_one.sh` | created | 3 | one-shot INT8 eval wrapper (uses test_sintel mode + Final list) |
| `plan/MCUFlowNet_Deployment/m2_v3_acc_sintel_final_test_sintel_s125.json` | created | 3 | EPE 8.40 (corrected, flow_scale=12.5) |
| `plan/MCUFlowNet_Deployment/m2_v3_efn_fps_sintel_final_test_sintel_s125.json` | created | 3 | EPE 7.34 (corrected) — winner |
| `plan/MCUFlowNet_Deployment/m2_v3_light_sintel_final_test_sintel_s125.json` | created | 3 | EPE 12.73 (corrected) |
| `tools/eval/fp32_v3_sintel_eval.py` | created | 3 | FP32 v3 evaluator (same pipeline as INT8 evaluator) |
| `plan/MCUFlowNet_Deployment/m2_v3_acc_fp32_157x203_test_sintel.json` | created | 3 | FP32 v3_acc @ 157x203, EPE 8.27 |
| `plan/MCUFlowNet_Deployment/m2_v3_efn_fps_fp32_157x203_test_sintel.json` | created | 3 | FP32 v3_efn_fps @ 157x203, EPE 7.75 |
| `plan/MCUFlowNet_Deployment/m2_v3_light_fp32_157x203_test_sintel.json` | created | 3 | FP32 v3_light @ 157x203, EPE 10.60 |

---

## Session Reboot Checklist

如果中断恢复，按这个顺序刷新上下文：
1. 读 `task_plan.md` → 看当前 Phase
2. 读 `findings.md` → 拿回硬事实
3. 读 `progress.md` 顶部 → 看上次干到哪
4. 验证关键路径仍存在：
   - `ls /mnt/g/AI_thesis/datasets/MPI-Sintel-complete/training/clean/alley_1/frame_0001.png`
   - `ls /home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite`
   - `/home/enmin/miniconda3/envs/vela/bin/python --version`
5. 看 `Next Action`，继续干
