# Progress Log: MCUFlowNet Deployment

按时间线追加，最新条目放最上面。

---

## Next Action

**Phase 3：M2 (EdgeFlowNAS retrain_v3 子网) 部署铺垫**

立即可做：
1. 读 retrain_v3 plan 文档：`D:\Dataset\MCUFlowNet\EdgeFlowNAS\plan\retrain_v3\` → 搞清子网架构和 mainline 差多少
2. 检查训练产物清单：`D:\Dataset\MCUFlowNet\EdgeFlowNAS\outputs\retrain_v3_ft3d\retrain_v3_ft3d_run1\`
   - 是 ckpt？keras h5？savedmodel？
   - 网络定义文件路径？
3. 决定 export 路径：
   - 若结构和 `run_export.py` 现有 variant 表里某项相同 → 直接复用 `OPTICAL_FLOW_EXPORT_VARIANT=... OPTICAL_FLOW_CHECKPOINT_PREFIX=... bash scripts/export_optical_flow_144x192.sh`
   - 若结构不同 → 新增 variant module 到 `network/`，再走同样脚本
4. PTQ 导出 → Vela → 看 SRAM peak 是否在 1432 KiB 内（不在的话 fallback 到 144×192 / 调 arena）
5. 板端烧录 + flow_viewer 可视化验证
6. 用 `int8_sintel_eval.py` + `fp32_sintel_eval.py`（同一套 evaluator）跑 M2 EPE

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
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean.log` | created | 2 | test_sintel.py log (5.46, side-channel) |
| `plan/MCUFlowNet_Deployment/m1_fp32_sintel_clean_native.json` | created | 2 | EPE report (FP32 native, 6.79) |

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
