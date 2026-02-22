# Plan 001: optical_sd 迭代“构建-烧录-观察”自动化方案

## 1. 背景与目标

当前你的人工流程是：

1. 改代码（`cvapp_yolov8n_ob.cpp` 等）  
2. 用 `build_img.sh` 构建并复制镜像到 Windows  
3. TeraTerm 手动进 Startup options -> 按 `1` 进入 Receiving mode  
4. 运行 Windows 的 `flash_img_opticalSD.bat` 烧录  
5. 再开 TeraTerm 看串口输出

目标是把这个流程尽量自动化，并让 agent 在“改完代码后”可以执行到“烧录并拿到日志输出”。

---

## 2. 从 README 和现有脚本得到的关键结论

- 根目录 `README.md` 已提供了 **Python 自动烧录路径**：`xmodem/xmodem_send.py`。
- `xmodem_send.py` 内部会：
  - 等待串口输出提示；
  - 自动发送字符 `1` 进入接收模式；
  - 发送 `output.img`；
  - 最后发送 `y` 触发重启。
- 仓库已有 `xmodem/serReadLoop.py`，可直接用来抓串口日志（等价于“自动化 TeraTerm 观察输出”）。
- `build_img.sh` 已实现构建 + 生成镜像 + 拷贝到 Windows 路径，说明“构建侧”已经接近可复用。

结论：**核心缺口不是烧录能力本身，而是“跨 WSL2/Windows 的串口访问与流程编排”。**

---

## 2.1 前提验证结果（2026-02-22）

本轮调试结论：

- `pip3 install -r xmodem/requirements.txt` 成功，`pyserial` 与 `xmodem` 依赖已满足。
- 通过 `usbipd` 将 CH343 设备映射到 WSL 后，WSL 侧出现 `\`/dev/ttyACM0\``。
- `python3 -c "import serial; serial.Serial('/dev/ttyACM0',921600,timeout=1)"` 可成功打开端口（`open ok`）。
- 因此，**方案 A 的关键前提已成立**（WSL 可直接访问串口，不再依赖 TeraTerm 手动传输）。
- 另外，WSL 内调用 `powershell.exe` / `cmd.exe` 仍报 `Exec format error`，所以当前不建议依赖跨端 exe 调用链路。

当前推荐策略：

- 主路径采用 **方案 A（WSL 端端到端）**。
- 方案 B 保留为备选，仅在方案 A 出现硬件不稳定时再排查。

---

## 3. 三种可行自动化方案（按推荐顺序）

## 方案 A（推荐）：WSL 端一条命令打通（构建 + 烧录 + 抓日志）

前提：

- USB 串口设备已映射到 WSL（例如 `/dev/ttyACM0`）。
- WSL 具备 `python3` + `pyserial` + `xmodem` 依赖。

流程：

- `build_img.sh` 生成 `we2_image_gen_local/output_case1_sec_wlcsp/output.img`
- 调 `python3 xmodem/xmodem_send.py --port=/dev/ttyACM0 --baudrate=921600 --protocol=xmodem --file=.../output.img`
- 再启动 `python3 xmodem/serReadLoop.py --port=/dev/ttyACM0 --baudrate=921600` 读 N 秒日志并落盘

优点：

- 全程在 WSL，agent 易编排、可重复、可保存日志。

风险：

- 设备重插或重启后，`/dev/ttyACM0` 可能变化（例如变为 `/dev/ttyACM1`），需要在脚本中做端口探测或参数化。
- 烧录阶段仍可能需要人工按一次 reset（取决于板子当前状态和握手时机）。

---

## 方案 B：WSL 构建 + Windows 烧录/日志（混合编排）

前提：

- 串口仅 Windows 可见（COMx），WSL 不可见。

流程：

- WSL 执行构建并输出 img 到 Windows 共享目录（你现在已经在做）
- Windows 侧脚本（PowerShell 或 bat）执行 `python xmodem\\xmodem_send.py --port=COMx ...`
- Windows 侧再执行 `python xmodem\\serReadLoop.py --port=COMx ...` 收集启动日志

优点：

- 最贴近你现有环境，改动最小。

风险：

- agent 从 WSL 直接控制 Windows 进程的稳定性取决于你是否允许 `powershell.exe`/`cmd.exe` 跨端调用。

---

## 方案 C：保留人工复位，自动化其余步骤（半自动）

流程：

- 自动跑构建、自动等待“请按 reset”、自动烧录、自动抓日志；
- 人只需在提示时按一次板子 Reset。

优点：

- 对硬件时序最稳，工程落地最快。

风险：

- 仍有人机交互点，但已大幅减少手工操作。

---

## 4. 第一个落地实验：在“复制 app”上打通 pipeline

为避免直接影响当前 `optical_sd` 主线，先做一个最小可回滚试验分支：

1. 复制 app 目录：`optical_sd` -> `optical_sd_poc_auto`  
2. 在 `makefile` 中把 `APP_TYPE` 切到 `optical_sd_poc_auto`  
3. 在复制 app 中只加一个最小可识别改动（例如串口打印版本号/构建时间）  
4. 执行自动化流水线，验证你能在日志里看到该标记  
5. 打通后再把流程迁移回正式 `optical_sd`

验收标准（DoD）：

- 能自动产出最新 `output.img`
- 能自动完成 xmodem 发送
- 设备重启后 10~30 秒内抓到串口日志
- 日志中出现“本次改动标记串”

---

## 5. 建议的脚本分层（后续实现）

- `scripts/pipeline/build_image.sh`
  - 只负责构建 + 产物检查 + 产物复制
- `scripts/pipeline/flash_image.py`
  - 封装 `xmodem_send.py` 调用（参数化 port/baud/file）
- `scripts/pipeline/capture_uart.py`
  - 封装 `serReadLoop.py`（支持超时、关键字检测、日志文件输出）
- `scripts/pipeline/run_pipeline.sh`
  - 串起 build -> flash -> capture，统一 exit code

这样将来无论是人手动执行，还是 agent 调用，都走同一入口。

---

## 6. 作为 Cursor Skill 的设计草案

可新增一个 skill（例如 `we2-optical-iter-loop`）：

- 输入参数：
  - `app_type`
  - `serial_port`（`/dev/ttyACM0` 或 `COMx`）
  - `baudrate`（默认 921600）
  - `capture_seconds`
  - `expect_keyword`（可选）
- 执行动作：
  1. 检查工具链/依赖
  2. 构建镜像
  3. 烧录
  4. 抓日志并保存到 `logs/`
  5. 输出“是否包含 expect_keyword”
- 失败策略（遵循“先调试不绕过”）：
  - 先报告失败阶段（build/flash/uart）
  - 打印关键输入输出（端口、img 路径、xmodem 返回值、首屏日志）
  - 不自动切换到替代逻辑，先请求人工确认环境问题

---

## 7. 调试优先策略（执行期必须遵守）

任何失败先做定位，不直接换方案绕过：

- build 失败：先定位编译错误文件与符号，再决定是否修改代码
- flash 失败：先看端口占用、波特率、reset 时机、xmodem 握手输出
- 无日志：先确认串口号/权限/是否被 TeraTerm 占用，再检查固件是否成功启动

建议每一步都输出：

- 输入参数快照（port、baudrate、img 路径）
- 关键状态点（进入 receiving、发送完成、重启确认）
- 首 100 行启动日志

---

## 8. 下一步（Plan 002 候选）

在你确认后，下一步做一个最小实现：

1. 新增 `scripts/pipeline/run_pipeline.sh`（先支持 WSL 端端到端）  
2. 新增 `scripts/pipeline/capture_uart.py`（固定超时抓日志）  
3. 用 `optical_sd_poc_auto` 跑通一次，验证日志关键字  
4. 再决定是否封装为 Cursor Skill

---

## 9. 实测记录（已打通，2026-02-22）

本节记录一次真实执行，目标是验证“复制 app + 自动 pipeline”是否可跑通。

### 9.1 实验变更

- 复制 app：`optical_sd` -> `optical_sd_poc_auto`
- 构建目标切换：`EPII_CM55M_APP_S/makefile`
  - `APP_TYPE = optical_sd_poc_auto`
- 在复制 app 注入验证标记：
  - 文件：`app/scenario_app/optical_sd_poc_auto/cvapp_yolov8n_ob.cpp`
  - 关键日志：`AUTO_POC_PIPELINE_20260222`

### 9.2 自动执行步骤与结果

1) 构建：
- 命令：`make clean && make -s --no-print-directory -j4`
- 结果：成功（编译日志中确认编译路径为 `optical_sd_poc_auto`）

2) 生成镜像：
- 命令：`we2_local_image_gen project_case1_blp_wlcsp.json`
- 结果：成功，产物存在：
  - `we2_image_gen_local/output_case1_sec_wlcsp/output.img`

3) 自动烧录（WSL 串口）：
- 命令：`python3 xmodem/xmodem_send.py --port=/dev/ttyACM0 --baudrate=921600 --protocol=xmodem --file=.../output.img`
- 结果：成功，日志包含：
  - `xmodem_send bin file result =  True`
  - `Firmware upgrade completed, restart WE2 ...`

4) 串口输出抓取：
- 直接用 `serReadLoop.py` 时日志文件为空（原因是其按 `readline` 读，容易错过/吞掉分段输出）。
- 改用原始字节流抓取后成功，文件：
  - `logs/pipeline/raw_uart_after_pipeline_20260222.log`
- 关键命中：
  - `Ethos-U55 device initialised`
  - `AUTO_POC_PIPELINE_20260222`
  - `initial done`
  - 多行 `[loop=...][frame=... ]` 推理输出

### 9.3 结论

- **pipeline 已打通**（复制 app 构建 -> 自动烧录 -> 自动抓取运行输出）。
- 方案 A 在当前机器上可作为主路径。
- 后续实现建议：将“串口抓取”从 `readline` 改为“原始字节流 + utf-8 ignore + 关键字检测”，提升稳定性。

### 9.4 串口抓取稳定性改造与复测（已完成）

改造文件：

- `xmodem/serReadLoop.py`

改造内容：

- `readline` -> 原始字节流 `ser.read(chunk_size)`
- UTF-8 解码方式：`errors='ignore'`
- 新增关键字检测：`--keyword`（可重复传入）
- 新增捕获时长：`--duration`
- 新增日志输出：`--log-file`
- 新增异常处理：串口断开/多进程占用时输出 `[SERIAL_ERROR]`，并返回退出码 `3`
- 关键字未命中返回退出码 `2`；全部命中返回 `0`

复测结果：

- 成功路径：
  - 命令包含 `--keyword=AUTO_POC_PIPELINE_20260222 --keyword=initial done`
  - 命中输出 `[KEYWORD_HIT]`，最终 `[SUMMARY] all_keywords_hit`
  - 退出码 `0`
- 失败路径：
  - 使用不存在关键字 `THIS_KEYWORD_SHOULD_NOT_EXIST`
  - 输出 `[SUMMARY] missing_keywords=[...]`
  - 退出码 `2`

结论：

- “原始字节流 + utf-8 ignore + 关键字检测”方案已验证可用，可作为后续 pipeline 默认抓取方式。

### 9.5 optical_sd 主用模型代码复测（复制 app，已完成）

目标：

- 针对当前主用代码 `app/scenario_app/optical_sd/cvapp_yolov8n_ob.cpp`，继续使用“复制 app”方式完成一次端到端验证。

执行：

- 新复制 app：`optical_sd_poc_model`（来源：`optical_sd` 当前版本）
- `APP_TYPE` 切换为 `optical_sd_poc_model`
- 注入验证标记：`AUTO_POC_MODEL_PIPELINE_20260222`
- 执行构建 -> 出图 -> xmodem 烧录 -> 串口关键字检测

结果：

- 烧录成功：`xmodem_send bin file result = True`
- 串口验证成功：
  - 命中 `AUTO_POC_MODEL_PIPELINE_20260222`
  - 命中 `initial done`
  - 输出 `[SUMMARY] all_keywords_hit`
  - 退出码 `0`
- 同时看到持续推理日志 `[loop=...][frame=...]`，说明模型应用运行正常。

### 9.6 USB 断连恢复步骤（必须纳入流程）

现象：

- 实测中出现过 `vhci_hcd: disconnect device`，导致 `/dev/ttyACM0` 消失，烧录脚本报 `Uart port open fail`。

恢复步骤（手动）：

1) Windows 管理员 PowerShell：
- `usbipd list`
- `usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>`
- `usbipd list`

2) WSL 验证：
- `ls -l /dev/ttyACM* 2>/dev/null`
- `python3 -c "import serial; s=serial.Serial('/dev/ttyACM0',921600,timeout=1); print('open ok'); s.close()"`

建议：

- 将上述 attach/验证步骤放到每次 pipeline 前的检查清单中，避免因串口丢失导致误判为代码问题。

---

## 10. Skill 落地（双模式 + 关键词触发）

已创建项目级 skill：

- `.cursor/skills/we2-optical-sd-pipeline/SKILL.md`
- `.cursor/skills/we2-optical-sd-pipeline/reference.md`
- `.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh`

能力：

- 双模式：
  - `nomodel`（只烧固件，快速迭代）
  - `with-model`（固件+模型全量烧录）
- 端到端：
  - 串口前置检查
  - 构建 + 出图
  - xmodem 烧录
  - UART 关键字验证
  - 日志落盘到 `logs/pipeline/`
- 故障提示：
  - 串口异常时给出 `usbipd attach` 恢复步骤（Windows + WSL）

关键词触发策略：

- 在 `SKILL.md` 的 `description` 和 `Trigger Keywords` 中覆盖了以下词汇，用于提升自动匹配概率：
  - `optical_sd`
  - `cvapp_yolov8n_ob.cpp`
  - `flash_img_opticalSD`
  - `flash_img_opticalSD_noModel`
  - `xmodem`
  - `usbipd attach`
  - `COM3`
  - `不烧模型`
  - `烧录模型`
  - `pipeline`

