# Progress Log: MCUFlowNet Deployment

按时间线追加，最新条目放最上面。

---

## Next Action

**放大 v3 input 充分利用 SRAM 余量**

V3 子网在 157×203 EPE 普遍比 mainline 差 +2.87 ~ +3.14。但 Vela SRAM peak 只有 1143 KiB（mainline 1430 KiB），还有 287 KiB 余量。要让 V3 体现 NAS 优势，应该放大 input 用满 arena。

具体步骤：
1. 找到 V3 Vela peak ≈ 1432 KiB 的最大 input。试以下尺寸，每个跑 export 看 Vela report：
   - 172×224 (NSGA-II 训练目标尺寸；预测 peak ~1386 KiB)
   - 200×256
   - 220×288
   - 选 Vela peak < 1432 KiB 且最大的
2. 在选定尺寸上重跑 3 个子网 (v3_acc / v3_efn_fps / v3_light) 全 Sintel Final EPE
3. 看是否反超 mainline 的 7.79
4. 同时检查板端 prev buffer 余量是否足够（当前 `remaining_after_arena=32`，arena 不变；如果 input 放大需要更大 prev buffer，要进 firmware 调整）
5. 选出 EPE 最低的子网烧到板上，flow_viewer 验证

如果放大 input 后 v3 仍打不过 mainline，则该轮 NAS 视为没有给出可部署 win，M2 退化为 mainline 平替；可以重启 NAS 搜索 v4。

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
| 2026-05-11 R8 | M2 v3_acc | INT8 (test_sintel, 157×203 in) | train final | 1041 | **10.6637 / 4.1021** | Vela peak 1143 KiB, 189.67 ms; Δ vs mainline +2.87 |
| 2026-05-11 R9 | M2 v3_efn_fps | INT8 (test_sintel, 157×203 in) | train final | 1041 | **10.6724 / 4.0929** | Vela peak 1143 KiB, 165.22 ms; Δ vs mainline +2.88 |
| 2026-05-11 R10 | M2 v3_light | INT8 (test_sintel, 157×203 in) | train final | 1041 | **10.9277 / 4.4540** | Vela peak 1143 KiB, 95.94 ms; Δ vs mainline +3.14 |

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
| `plan/MCUFlowNet_Deployment/m2_v3_light_sintel_final_test_sintel.json` | created | 3 | EPE report v3_light (10.93) |
| `tools/eval/eval_int8_one.sh` | created | 3 | one-shot INT8 eval wrapper (uses test_sintel mode + Final list) |

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
