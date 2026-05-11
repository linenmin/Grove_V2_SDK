# Progress Log: MCUFlowNet Deployment

按时间线追加，最新条目放最上面。

---

## Next Action

**跑 FP32 baseline（原 `test_sintel.py` 在 best.ckpt 上）作为 QAT 决策依据**

具体步骤：
1. 在 vela env 里跑 `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\test_sintel.py`（或拷贝路径到 WSL），传 `--checkpoint /home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/assets/checkpoints/best.ckpt --data_list .../MPI_Sintel_train_clean.txt`
   - 注意 patch_dim 默认 416×1024，建议改成与 INT8 评估对齐的 160×208 来直接比较；或者保留默认对齐 paper 数字。两者都跑一遍。
2. 数字记入 `findings.md` 第 10 节作为 FP32 对照
3. 比较 `Δ EPE = INT8 - FP32`：
   - Δ < 0.2：PTQ 已经足够，不做 QAT
   - 0.2 ≤ Δ < 0.5：边界，看高速场景具体退化情况
   - Δ ≥ 0.5：触发 QAT 立项
4. Phase 2 在此处收口，进入 Phase 3 (M2 部署)

---

## 2026-05-11 — Session 2

**完成：**
- 修 evaluator EPE 方法学 bug：加 `--eval-grid {native,pred}`，default native
- M1 INT8 重跑（native grid 1024×436）：**avg EPE 6.9238**（median 2.33，1041 帧，287s）
- 量级对齐论文 6.31，ΔEPE ≈ +0.6（vs 论文）
- 评估 GPU 适配性：硬件 OK（RTX 4060 Laptop），vela env TF 2.15 是 CUDA build 但缺 CUDA 12.2/cuDNN 8 lib；tf_work_hpc 不是 CUDA build。当前任务 (TFLite int8 + 小模型 FP32) 不需要 GPU；QAT 阶段再补 CUDA stack。

**新发现（已落 findings.md §10）：**
- 标准 OpFlow EPE 必须在 GT 原分辨率算，否则数字被同比例压缩
- INT8 vs 论文 FP32 ΔEPE ≈ +0.6；但需要本机严格 FP32 跑一次才算可靠 Δ

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
| 2026-05-11 R2 | M1 mainline | PTQ INT8 | train clean | 1041 | **6.9238 / 2.3303** | native-grid (1024×436)，标准方法，可比论文 6.31；ΔEPE vs 论文 ≈ +0.6 |
|     | M1 mainline | FP32 ckpt | train clean | — | TBD | next action — 本机严格跑一次 |

---

## Files Created / Modified

| File | Type | Phase | Notes |
|------|------|-------|-------|
| `plan/MCUFlowNet_Deployment/task_plan.md` | created | meta | 五阶段计划 |
| `plan/MCUFlowNet_Deployment/findings.md` | created | meta | 硬事实索引 |
| `plan/MCUFlowNet_Deployment/progress.md` | created | meta | 本文件 |
| `model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite` | (exist) | 1 | Vela 编译产物，已烧 |
| `tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite` | (exist) | 1 | 未走 Vela，待评 |
| `tools/eval/int8_sintel_eval.py` | created | 2 | INT8 evaluator, ~150 LOC, works |
| `tools/eval/run_m1_int8_eval.sh` | created | 2 | one-shot wrapper for M1 |
| `plan/MCUFlowNet_Deployment/m1_int8_sintel_clean.json` | created | 2 | EPE report (avg 1.847, n=1041) |
| `plan/MCUFlowNet_Deployment/m1_int8_sintel_clean.log` | created | 2 | full run log + summary |

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
