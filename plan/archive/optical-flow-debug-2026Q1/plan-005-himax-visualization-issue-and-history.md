> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 005：Himax 可视化问题调试经验结论（精简版）

## 1. 文档定位

- 本文件为 **经验结论清单**，用于后续快速复用。
- 已删除逐轮流水账，仅保留可验证、可执行、无歧义的结论。
- 历史原始证据仍在 `logs/pipeline/`，按日志路径回溯。

---

## 2. 固定前提（已验证）

1. 设备：Grove Vision AI(V2)（WE2）。
2. 浏览器链路：Windows Edge + Himax AI Web Toolkit（`index.html`）。
3. 串口波特率：`921600`。
4. 串口占用规则：同一时刻仅一端可占用（WSL2 或 Windows）。
5. 当前主应用：`optical_cam_oflow`。

---

## 3. 精炼调试结论（按优先级）

### C01 串口所有权是单占用，不可并发

- **结论**：WSL2 与 Windows 不能同时读写同一 CH343 串口。
- **判据**：WSL 能打开 `/dev/ttyACM0` 时，Windows COM 连接会失败或无数据。
- **标准动作**：Windows 验证前先 `usbipd detach --busid <BUSID>`；回到 WSL 前再 `usbipd attach --wsl <distro> --busid <BUSID>`。

### C02 仅 SPI 输出不足以驱动 Himax 页面预览

- **结论**：仅发送 SPI JPEG/META 时，页面可能 Connect 成功但 Preview/Log 仍无输出。
- **判据**：设备侧有 SPI 发送日志，但页面无 `INVOKE` 可见行为。
- **标准动作**：必须同时打通 UART JSON 输出链路。

### C03 页面联调最低协议是 UART JSON 握手 + INVOKE

- **结论**：至少要稳定输出 `NAME? / VER? / ID? / INFO? / MODEL? / INVOKE`。
- **判据**：WSL 抓串口能持续命中上述关键字，且 `INVOKE` 含 `image` 字段。
- **标准动作**：优先检查 `viz_uart_send_device_id_once()` 与 `viz_uart_send_invoke_jpeg()` 调用路径。

### C04 点击 `uart_send` 后设备需要能响应命令字节

- **结论**：页面按钮动作不是"被动等待"，设备需处理主机命令（0xFF/0xFE/0xFD）并回握手。
- **判据**：点击 `uart_send` 后若无回包，页面侧无新增状态。
- **标准动作**：保留并验证 `viz_uart_poll_host_cmd()` 非阻塞轮询逻辑。

### C05 UART 流中混入非 JSON 噪声会影响页面解析稳定性

- **结论**：高频 `dbg_printf`（特别是 JPEG size 对齐告警）会污染 JSON 流。
- **判据**：关键字偶发缺失或页面端表现不稳定，而设备端仍在运行。
- **标准动作**：压缩日志到低频、必要日志与 JSON 输出分离。

### C06 fallback INVOKE 图可作为"网页链路是否通"的金标准

- **结论**：固定测试图能显示，说明 Web Serial + JSON parser + 图像渲染链路可用。
- **判据**：页面出现红绿白条纹图且持续刷新。
- **标准动作**：先用 fallback 验证链路，再回头修真实图像源。

### C07 4X 子采样路径下真实 JPEG 产线不稳定，2X 可恢复

- **结论**：当前项目状态中，`4X` 路径出现 `real size=0` 与非 JPEG SOI 签名；`2X` 路径可恢复真实 JPEG。
- **判据**：
  - 4X：`cisdp=.../0`、`jpginfo real=0 fill=0`、`sig!=FFD8`、`jpeg_skip_count>0`。
  - 2X：`jpginfo real/fill` 稳定非零，`jpeg_skip_count=0`。
- **标准动作**：以 2X 作为可用基线推进功能联调；4X 单独开支线修复。

### C08 动态地址模式下需覆盖 4X WDMA 分配分支

- **结论**：4X 若未走动态 WDMA 分配，会与大块内存（含 arena）产生冲突风险。
- **判据**：启动日志中 `WD3_RAW[0]` 或 `wdma alloc fail: wd3=0x0`。
- **标准动作**：确保 `cisdp_wdma_addr_init()` 在 `DYNAMIC_ADDRESS` 下覆盖 4X 分支。

### C09 arena 过小会在模型初始化阶段失败

- **结论**：`tensor_arena_size` 低于模型门槛时，直接 `AllocateTensors fail`。
- **判据**：日志出现 `Failed to resize buffer` / `AllocateTensors fail`，且无 `initial done`。
- **标准动作**：先恢复能过 `AllocateTensors` 的尺寸，再谈 camera/JPEG 问题。

### C10 arena 过大又会压垮 camera WDMA 分配

- **结论**：arena 增大到临界以上会触发 `wdma alloc fail`，随后 `camera frame capture fail`。
- **判据**：有 `initial done`，但出现 `wd3=0x0` 与持续 frame fail 循环。
- **标准动作**：按 1KB 或 2KB 步进二分，不要跨大步反复试错。

### C11 当前环境下 2X 路径稳定上限是 1680KB

- **结论**：`1680KB` 可稳定；`1681KB` 已触发 WDMA 失败。
- **判据**：
  - 1680KB：`all_keywords_hit` 且 `camera_frame_capture_fail_count=0`。
  - 1681KB：`wdma alloc fail` + `missing_keywords=INVOKE/MODEL?/NAME?`。
- **标准动作**：后续 with-model 默认以 `1680KB` 为基线，不再上探无收益区间。

### C12 with-model 烧录必须严格按 app 的 FLASH_ADDR 宏选槽位

- **结论**：模型槽位必须来自目标 app 的 `common_config.h`，不能硬编码迁移。
- **判据**：错误槽位时会出现"模型似乎烧录成功但运行行为异常"。
- **标准动作**：使用 `get_model_slot.sh` 自动解析；多宏时必须显式 `--macro`。

### C13 非 Vela 模型与 Vela 模型失败模式不同

- **结论**：非 Vela 模型常见为算子不匹配或长时间无 INVOKE；Vela 模型更可控，失败多为纯内存门槛。
- **判据**：
  - 非 Vela：`Didn't find op for builtin opcode ...` 或长采样无 `INVOKE`。
  - Vela：模型可识别 Ethos-U，但可能报 `missing bytes`（arena不足）。
- **标准动作**：先确认输入模型是 Vela 产物，再做 resolver 与内存调优。

### C14 当前已验证可运行的 with-model 组合

- **结论**：以下组合已通过：
  - 模型：`/mnt/d/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/deployment/model/sram_test_modified_vela.tflite`
  - 槽位：`0xB7B000`
  - arena：`1680KB`
- **判据**：日志 `pipeline_with-model_optical_cam_oflow_20260223_002248.log` 中 `all_keywords_hit`。
- **标准动作**：后续功能开发先基于该组合作为回归底座。

### C15 "模型已运行"不等于"页面显示光流"

- **结论**：即使 `INVOKE` 持续输出，若 `image` 仍来自 camera JPEG，页面就只会显示普通摄像头画面。
- **判据**：页面画面是自然场景图而非光流伪彩图，且串口无 flow 渲染路径标记。
- **标准动作**：单独实现 `flow tensor -> viz image -> JPEG -> INVOKE.image` 路径。

### C16 低帧率与暗画面应在 camera_jpeg 模式先做基线

- **结论**：先调 sensor 曝光/增益和发送节流，再切 flow 可视化，避免观测窗口丢失。
- **判据**：切到 flow 后无法直接判断 sensor 参数是否正常。
- **标准动作**：按 `plan-006` 顺序先做性能与亮度，再做 flow 渲染。

### C17 调试过程必须"一次一因"，并先写 plan 再做下一轮

- **结论**：多变量同时修改会导致结论不可归因。
- **判据**：单次修改超过一个核心变量（如 arena + subsample + resolver）。
- **标准动作**：严格执行"一个假设 -> 一个改动 -> 一次验证 -> 先写 plan"。

### C18 每轮证据最小集合

- **结论**：保留最小证据即可复现判断，不需要粘贴全量 UART/base64。
- **判据**：每轮仅需 4 项：
  - 验证命令
  - 关键词命中结果
  - 关键计数（`invoke_count`/`jpeg_skip_count`/`camera_frame_capture_fail_count`）
  - 原始日志路径
- **标准动作**：统一使用 `extract_himax_keylog.sh` + 计划增量记录脚本。

---

## 4. 关键证据日志（保留索引）

1. fallback INVOKE 打通网页链路：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_223357.log`
2. 4X 路径 real JPEG 异常样例：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_231019.log`
3. 2X 路径 real JPEG 稳定基线：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_231522.log`
4. arena=1681KB 失败样例：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_233338.log`
5. arena=1680KB 成功样例：`logs/pipeline/pipeline_nomodel_optical_cam_oflow_20260222_233433.log`
6. with-model（Vela）成功样例：`logs/pipeline/pipeline_with-model_optical_cam_oflow_20260223_002248.log`

---

## 5. 当前未解决项

1. 页面仍显示 camera JPEG，不是光流渲染图（功能缺口，不是链路中断）。
2. 低帧率与偏暗仍需在 `camera_jpeg` 模式完成前置优化。

