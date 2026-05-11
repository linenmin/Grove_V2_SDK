# Findings: MCUFlowNet Deployment

本文件记录所有"硬事实"和"踩过的坑"，新事实第一时间落盘。

---

## 1. 硬件与运行期约束

- 芯片：Himax WE2 = CM55M + Ethos-U55 NPU
- SRAM 总量：2 MiB，实际应用可控约 1.9 MiB
- 当前 Tensor Arena：1432 KiB（`common_config.h`，与 157x203 模型绑死）
- Flash 模型槽位：宏 `OPTICAL_FLOW_MODEL_FLASH_ADDR = 0x3AB7B000`，xmodem 偏移 `0xB7B000`
- 运行期边界：`157x203` 已验证稳定；`158x202`/`155x206` Vela 编译过但板端 `alloc prev buffer fail`；更大输入会 fallback
- Vela 报告峰值：`157x203 -> 160x208` 全档约 1430 KiB

## 2. 当前主线模型 (M1: EdgeFlowNet 原版 transpose-conv)

- 网络定义：`tools/model_export/optical_flow_144x192/network/MultiScaleResNet.py`
- 变体名 `mainline`，是 `run_export.py` 的默认 `DEFAULT_VARIANT`
- Float ckpt 路径（仓库自带，唯一一份带权重的）：
  `tools/model_export/optical_flow_144x192/assets/checkpoints/best.ckpt{.index,.data-*,.meta}`
- 输入 tensor：`[1, 157, 203, 6]`，6 通道 NHWC interleaved（prev RGB + curr RGB 按像素交错）
- 输出 tensor：`[1, 160, 208, 2]`，NHWC planar=0
- 量化参数（板端实测）：output `scale ≈ 0.426478, zp = -4`（文档老版本写 0.407547，重新导出时会略浮动；不影响）
- INT8 [-128, 127] → 像素位移约 [-64, +64]
- 当前发布产物：
  - PC 评估可用（标准 TFLite）：`tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite`
  - 板端烧录用（Vela）：`model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`
  - 两者权重 + 量化参数完全相同，Vela 只重排算子调度

## 3. 量化流水线（PTQ，无 QAT）

- 加载 float ckpt → 重建 TF1 graph → `TFLiteConverter.from_session(...)`
- `optimizations=[Optimize.DEFAULT]`、`supported_ops=[TFLITE_BUILTINS_INT8, TFLITE_BUILTINS]`
- `inference_input_type=int8`、`inference_output_type=int8`
- `representative_dataset_gen`：
  - 校准目录默认 `tools/model_export/optical_flow_144x192/assets/calibration/`
  - 仓库自带 `PERTURBED_market_3` + `PERTURBED_shaman_1` 各 50 帧
  - 流水线：连续帧对 `(img1, img2)` → `cv2.imread(BGR)` → `cv2.resize(W=203,H=157)` → `np.concatenate(axis=2)` → float32 → `expand_dims(0)` → `yield [tensor]`
  - 取 `min(100, len(pairs))` 对
- 最终 Vela：`run_vela(..., optimise="Size")` → `_vela.tflite`
- **整条链路无 QAT，无量化后 fine-tune，无 fakequant 训练**

## 4. 部署流水线 (Phase 1 已跑通)

- 导出：`bash scripts/export_optical_flow_144x192.sh`（脚本名仍带 144x192，但默认导 157x203）
- 烧写 + 抓 log：`bash .agent/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh --mode with-model --app-type optical_cam_oflow --port /dev/ttyACM0 --skip-build --capture-seconds 10 --keyword 'initial done' --keyword 'INVOKE' --extract-frames --max-frames 3 --model-arg '<vela_tflite_path> 0xB7B000 0x00000'`
- **前置：**
  - `usbipd attach --wsl Ubuntu-22.04 --busid 1-2`（板子 busid `1-2`，VID:PID `1a86:55d3` CH343 USB Serial → COM3 on Windows / `/dev/ttyACM0` on WSL）
  - `python3` 要能 import xmodem + pyserial → 把 `/home/enmin/miniconda3/envs/vela/bin` 加到 PATH 头部
- 烧完后 `usbipd detach --busid 1-2` 让 Windows 拿回 COM3
- 板端成功判据：日志同时命中 `model io: in(h=157,w=203,c=6) out(h=160,w=208,c=2)`、`[NPU_MODE]`、`initial done`、`INVOKE ... "resolution": [208, 160]`、`[viz] out=208x160 ...`

## 5. 实时可视化协议（板 → PC）

- 板端 firmware (`optical_cam_oflow`)：NPU 推完 → flow → `mag * 0.05` 系数 → HSV 渲染 → JPEG 编码 → UART 发包
- 握手：PC 发 `0xFC`，板回 `\r{"type":0,"name":"RAW_MODE",...}\n`
- 帧格式：`[0xAA][0x55][size_lo][size_hi][w_lo][w_hi][h_lo][h_hi][JPEG bytes...]`
- 波特率：921600
- Windows 端 viewer：`D:\Dataset\visualization\demo_vi\flow_viewer.py`，env `gpu_env` = `D:\Anaconda3\envs\gpu_env\python.exe`
- DISPLAY_SCALE=3，窗口 624×480（208×160 ×3）

## 6. 现有评估代码资产

- `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\test_sintel.py` — **只支持 float TF1 ckpt**，不能直接评 INT8 tflite
- `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\misc\processor.py` — `FlowPostProcessor` 类，多尺度累加 + EPE 计算逻辑
- `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\misc\FlowVisUtilsNP.py` — flow .flo 读取工具
- `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\dataset_paths\MPI_Sintel_train_clean.txt` — Sintel train clean 文件列表，相对路径 `Datasets/Sintel/training/{clean,flow}/<scene>/frame_XXXX.{png,flo}`
- **Sintel 实际路径**：`/mnt/g/AI_thesis/datasets/MPI-Sintel-complete/training/`（U 盘）
- INT8 evaluator 计划写在：`tools/eval/int8_sintel_eval.py`（待建）

## 7. M2 (EdgeFlowNAS retrain_v3) 已知信息

- 训练计划文档：`D:\Dataset\MCUFlowNet\EdgeFlowNAS\plan\retrain_v3\`
- 已下载的 HPC 训练产物：`D:\Dataset\MCUFlowNet\EdgeFlowNAS\outputs\retrain_v3_ft3d\retrain_v3_ft3d_run1\`
- 网络结构 vs mainline 的差异 / variant 表是否覆盖：**尚未核查**
- 量化校准数据是否可复用 mainline 的 calibration set：**尚未核查**

## 8. M3：占位

- 模型类型、权重位置、是否需要重训：**用户后续指定**

## 9. 已知风险 / 待排查

- **flow_viewer.py 大量 `[sync] Skipped N bytes` + `Corrupt JPEG`** — UART 同步打滑严重，FPS 待测。怀疑方向：板端 JPEG block 边界 vs PC 端 read chunk 不对齐 / 921600 baud 边际 / PC 串口驱动缓冲。Phase 2 不修，先把数字测出来再回头。
- **U 盘挂载稳定性** — Sintel 在 `/mnt/g`，跑长任务前先 `ls /mnt/g/AI_thesis` 验证挂载未掉
- **TF 2.15 TF1 graph 弃用警告** — `tf.losses.sparse_softmax_cross_entropy is deprecated` 等告警可忽略；`disable_eager_execution()` + TF1 graph + `Saver.restore` 在 TF 2.15 仍可工作
- **量化 scale 复现性** — `representative_dataset_gen` 取前 100 对帧，顺序由 `os.listdir`（无显式 sort）决定 → 同一台机器跑两次结果会不会一致需验证；如果浮动会影响 EPE 复现性，需要固定种子或排序

## 10. M1 Sintel EPE 评估结果（PTQ INT8）

### Methodology 踩坑：必须在 native (GT) 分辨率算 EPE

第一次跑得到 avg EPE 1.85（pred grid）— 看似很好但**不正确**。bug 在于把 GT flow downsample 到 160×208 并按比例缩小 flow magnitude，等于在一个被压缩 ~3-5× 的坐标系下算 EPE，数字被同比例压小。1.85 × √(1024/208 · 436/160) ≈ 6.77，和 EdgeFlowNet 论文 / 用户之前 FP32 baseline 6.31 量级一致。

**正确方法（EdgeFlowNet 论文 + 标准 optical-flow 评估）**：
- pred (160×208) → bilinear upsample 到 GT 分辨率 (1024×436)
- 同时把 flow vector 分量按 `x_scale = 1024/208`, `y_scale = 436/160` 放大
- 与原 GT (1024×436) 直接像素级 EPE

evaluator (`tools/eval/int8_sintel_eval.py`) 加了 `--eval-grid {native,pred}`，default `native`。`pred` 仅保留作 debug。

### 当前正确数字（native grid，2026-05-11）

- evaluator：`tools/eval/int8_sintel_eval.py --eval-grid native`
- tflite：`tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite`
- 数据：Sintel train clean，1041 帧对（23 scenes），全集
- 模型：157×203 in → 160×208 out → upsample 1024×436
- 硬件：WSL CPU vela env, XNNPACK 4 threads
- 耗时：286.7s (≈275 ms/frame)
- **avg EPE：6.9238**
- **median EPE：2.3303**（与 mean 差距大 → 长尾来自高速场景）
- 对照基准：EdgeFlowNet 论文 / 用户先前 FP32 baseline ≈ 6.31 → **PTQ INT8 ΔEPE ≈ +0.6**

### Per-scene（native grid）

| EPE 档 | scenes |
|---|---|
| 高速失败 (>10) | market_5 (24.4), ambush_4 (22.8), ambush_2 (19.6), ambush_6 (19.1), temple_3 (17.0), cave_2 (15.9), ambush_5 (11.8), market_6 (11.0) |
| 中段 (3–10) | cave_4 (8.98), temple_2 (6.12), ambush_7 (2.96) |
| 低速精度好 (<2) | sleeping_1 (1.02), sleeping_2 (1.09), shaman_2 (1.23), bandage_2 (1.41), bamboo_1 (1.48), alley_2 (1.80), shaman_3 (1.81), bamboo_2 (1.90), alley_1 (2.00) |

### 结论 / 下一步触发条件

- 真正干净的 ΔEPE 必须用**同一 pipeline**（157×203 input → 160×208 pred → upsample 到 1024×436 + flow 缩放 → 在 GT 原分辨率算 EPE）。已新写 `tools/eval/fp32_sintel_eval.py`，复用同一 graph 构造（MultiScaleResNet, NumOut=4, 取 [...,0:2]）+ 同一 resize_flow_to 工具。
- **苹果对苹果对比（2026-05-11，1041 帧 native grid）：**

| 配置 | avg EPE | median EPE | per-frame elapsed |
|---|---:|---:|---:|
| FP32 best.ckpt (TF1 graph) | **6.7915** | 2.1971 | 267 ms |
| PTQ INT8 (`optical_flow_157x203.tflite`) | **6.9238** | 2.3303 | 275 ms |
| **ΔEPE (INT8 − FP32)** | **+0.1323** | +0.1332 | — |

- **结论**：PTQ INT8 几乎无精度损失（相对 +1.9%）。**QAT 在 M1 上不需要**。
- 旁路对照：原 `test_sintel.py` 默认 ResizeNearestCrop @ 416×1024 grid，FP32 = 5.4649。这是另一套方法学，**不参与 Δ 计算**；用户记忆中的 6.31 大概率属于这一类（接近但配置/clip_val 等细节略不同）。
- Per-scene Δ：全部场景 INT8 vs FP32 偏差都在 ±0.5 以内；高 EPE 长尾来自 model 本身（输入 157×203 + 输出 INT8 ±64px 动态范围 + 多尺度累加结构），不是量化引入。

### 量化决策最终结论（M1）

- **PTQ INT8 已经足够，QAT 不立项**。Δ_纯量化 = +0.085 远低于 0.3 阈值。
- 想再压精度只能动 **input 分辨率 / 网络结构 / 训练数据**，不是量化方式。
- M2 retrain_v3 进入部署后，先复用同样 PTQ + 同样 evaluator 跑数；如果它的 Δ 突然变大，再考虑 QAT。

## 12. 决定切换：默认评估集 = Sintel **Final**（不再用 Clean）

- 用户先前的 baseline 6.31 来自 `wrappers/run_test.py` 默认配置，wrapper 把 `--dataset sintel` 隐式指向 **`MPI_Sintel_Final_train_list.txt`**（不是 Clean）。
- Final pass = 同样 1041 帧/23 scenes，但加了 motion blur / depth-of-field / 大气散射 / 合成阴影 → 更接近真实硬件场景，EPE 比 Clean 高 ~0.85。
- evaluator 默认数据集已切到 Final，wrapper `run_m1_int8_eval.sh` 默认 `LIST=...MPI_Sintel_Final_train_list.txt`。
- 想跑 Clean 仍可 `LIST=.../MPI_Sintel_train_clean.txt bash run_m1_int8_eval.sh`。

## 13. 完整 M1 Δ 矩阵（Final pass, 1041 帧, 2026-05-11）

evaluator 加了 `--ref-mode test_sintel`，用 `ResizeNearestCrop @ 416×1024 + clip_val=50 + flow vector 上采样到 patch grid` 复现 `test_sintel.py` 方法学。

| Config | input grid | eval grid | clip | EPE | 用途 |
|---|---|---|---|---:|---|
| FP32 (test_sintel.py 默认, 论文复刻) | 416×1024 | 416×1024 | ±50 | **6.3117** | 你之前的 baseline |
| FP32 (我的 evaluator, test_sintel mode) | 157×203 | 416×1024 | ±50 | 7.7059 | 隔离"降分辨率" |
| INT8 PTQ (test_sintel mode) | 157×203 | 416×1024 | ±50 | **7.7911** | 板端真实方法学 |
| FP32 (native, no clip) | 157×203 | 1024×436 | none | 6.7915 | 同 pipeline 苹果对苹果 (Clean) |
| INT8 PTQ (native, no clip) | 157×203 | 1024×436 | none | 6.9238 | 同 pipeline 苹果对苹果 (Clean) |

**Δ 拆解（Final pass, test_sintel methodology）：**

- **Δ_pure_quant = INT8 − FP32 (同 157×203 input) = +0.085 (+1.1%)** ← PTQ 损失，可忽略
- **Δ_downsample = FP32_157×203 − FP32_416×1024 = +1.39 (+22%)** ← 板上 input 分辨率限制损失
- **Δ_total vs 你的 6.31 = +1.48** ← 量化 + 降分辨率 合起来

**结论再次确认**：INT8 量化本身几乎不损失精度；板上看到的 EPE 上涨主要来自 157×203 输入分辨率上限（受 1432 KiB Tensor Arena 制约）。**QAT 在 M1 上无效，不立项**。要降 EPE 必须放更大的 input（需要 arena / 模型结构改造，参见 plan-018 系列对 158×202 失败的记录）或者换网络结构（M2 retrain_v3 的研究目标）。

## 11. 决策悬而未决

- QAT 触发阈值：FP32 → INT8 EPE 升高多少才值得做 QAT？
  - 待测：用 `EdgeFlowNet/code/test_sintel.py` 跑 `best.ckpt` 得 FP32 EPE，差值如果 >0.3 EPE 或者高速场景显著差，则启动 QAT
- M3 是什么模型？
- Phase 5 是否包括上 paper 的 latency 实测？板端 `algo_tick` 单位是 NPU clock，要换 ms

---

**更新规则**：每次发现新事实、改变假设、踩到新坑，立即追加到对应小节，不要等阶段结束。
