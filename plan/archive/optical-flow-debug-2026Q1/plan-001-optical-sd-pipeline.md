> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 001: optical_sd 迭代"构建-烧录-观察"自动化方案

## 1. 背景与目标

当前你的人工流程是：

1. 改代码（`cvapp_yolov8n_ob.cpp` 等）
2. 用 `build_img.sh` 构建并复制镜像到 Windows
3. TeraTerm 手动进 Startup options -> 按 `1` 进入 Receiving mode
4. 运行 Windows 的 `flash_img_opticalSD.bat` 烧录
5. 再开 TeraTerm 看串口输出

目标是把这个流程尽量自动化，并让 agent 在"改完代码后"可以执行到"烧录并拿到日志输出"。

---

## 2. 从 README 和现有脚本得到的关键结论

- 根目录 `README.md` 已提供了 **Python 自动烧录路径**：`xmodem/xmodem_send.py`。
- `xmodem_send.py` 内部会：
  - 等待串口输出提示；
  - 自动发送字符 `1` 进入接收模式；
  - 发送 `output.img`；
  - 最后发送 `y` 触发重启。
- 仓库已有 `xmodem/serReadLoop.py`，可直接用来抓串口日志（等价于"自动化 TeraTerm 观察输出"）。
- `build_img.sh` 已实现构建 + 生成镜像 + 拷贝到 Windows 路径，说明"构建侧"已经接近可复用。

结论：**核心缺口不是烧录能力本身，而是"跨 WSL2/Windows 的串口访问与流程编排"。**

---

## 3. 前提验证结果（2026-02-22）

- `pip3 install -r xmodem/requirements.txt` 成功，`pyserial` 与 `xmodem` 依赖已满足。
- 通过 `usbipd` 将 CH343 设备映射到 WSL 后，WSL 侧出现 `/dev/ttyACM0`。
- 验证命令：`python3 -c "import serial; serial.Serial('/dev/ttyACM0',921600,timeout=1)"` 可成功打开端口（`open ok`）。
- **方案 A 的关键前提已成立**（WSL 可直接访问串口，不再依赖 TeraTerm 手动传输）。
- 另外，WSL 内调用 `powershell.exe` / `cmd.exe` 仍报 `Exec format error`，所以当前不建议依赖跨端 exe 调用链路。

当前推荐策略：主路径采用 **方案 A（WSL 端端到端）**。

---

## 4. 三种可行自动化方案

### 方案 A（推荐）：WSL 端一条命令打通

前提：USB 串口设备已映射到 WSL（例如 `/dev/ttyACM0`），WSL 具备 `python3` + `pyserial` + `xmodem` 依赖。

流程：
- `build_img.sh` 生成 `we2_image_gen_local/output_case1_sec_wlcsp/output.img`
- `python3 xmodem/xmodem_send.py --port=/dev/ttyACM0 --baudrate=921600 --protocol=xmodem --file=.../output.img`
- `python3 xmodem/serReadLoop.py --port=/dev/ttyACM0 --baudrate=921600`

风险：设备重插或重启后，`/dev/ttyACM0` 可能变化（例如变为 `/dev/ttyACM1`），需参数化。

### 方案 B：WSL 构建 + Windows 烧录/日志（混合编排）

前提：串口仅 Windows 可见（COMx），WSL 不可见。

风险：agent 从 WSL 直接控制 Windows 进程的稳定性取决于是否允许 `powershell.exe`/`cmd.exe` 跨端调用。

### 方案 C：保留人工复位（半自动）

自动跑构建、自动等待"请按 reset"、自动烧录、自动抓日志；人只需在提示时按一次板子 Reset。

---

## 5. 实测记录（已打通，2026-02-22）

### 5.1 实验：复制 app 验证 pipeline

- 复制 app：`optical_sd` -> `optical_sd_poc_auto`（已删除，仅用于PoC验证）
- 构建目标切换：`EPII_CM55M_APP_S/makefile` -> `APP_TYPE = optical_sd_poc_auto`
- 验证标记文件：`app/scenario_app/optical_sd_poc_auto/cvapp_yolov8n_ob.cpp`，关键日志：`AUTO_POC_PIPELINE_20260222`

**构建与烧录命令**：
```bash
make clean && make -s --no-print-directory -j4
we2_local_image_gen project_case1_blp_wlcsp.json
python3 xmodem/xmodem_send.py --port=/dev/ttyACM0 --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img
```

**关键产物路径**：`we2_image_gen_local/output_case1_sec_wlcsp/output.img`

**烧录成功标志**：`xmodem_send bin file result = True`、`Firmware upgrade completed, restart WE2 ...`

### 5.2 串口抓取稳定性改造

**改造文件**：`xmodem/serReadLoop.py`

**改造内容**：
- `readline` -> 原始字节流 `ser.read(chunk_size)`
- UTF-8 解码方式：`errors='ignore'`
- 新增关键字检测：`--keyword`（可重复传入）
- 新增捕获时长：`--duration`
- 新增日志输出：`--log-file`
- 新增异常处理：串口断开/多进程占用时输出 `[SERIAL_ERROR]`，返回退出码 `3`
- 关键字未命中返回退出码 `2`；全部命中返回 `0`

**结论**："原始字节流 + utf-8 ignore + 关键字检测"方案已验证可用。

---

## 6. USB 断连恢复步骤（必须纳入流程）

**现象**：`vhci_hcd: disconnect device`，导致 `/dev/ttyACM0` 消失，烧录脚本报 `Uart port open fail`。

**恢复步骤**：

1) Windows 管理员 PowerShell：
```powershell
usbipd list
usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>
usbipd list
```

2) WSL 验证：
```bash
ls -l /dev/ttyACM* 2>/dev/null
python3 -c "import serial; s=serial.Serial('/dev/ttyACM0',921600,timeout=1); print('open ok'); s.close()"
```

---

## 7. 调试优先策略

任何失败先做定位，不直接换方案绕过：
- build 失败：先定位编译错误文件与符号，再决定是否修改代码
- flash 失败：先看端口占用、波特率、reset 时机、xmodem 握手输出
- 无日志：先确认串口号/权限/是否被 TeraTerm 占用，再检查固件是否成功启动

建议每一步都输出：
- 输入参数快照（port、baudrate、img 路径）
- 关键状态点（进入 receiving、发送完成、重启确认）
- 首 100 行启动日志

---

## 8. Skill 落地（已完成）

已创建项目级 skill：
- `.cursor/skills/we2-optical-sd-pipeline/SKILL.md`
- `.cursor/skills/we2-optical-sd-pipeline/reference.md`
- `.cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh`

**能力**：
- 双模式：`nomodel`（只烧固件）、`with-model`（固件+模型全量烧录）
- 端到端：串口前置检查 → 构建 + 出图 → xmodem 烧录 → UART 关键字验证 → 日志落盘到 `logs/pipeline/`
- 故障提示：串口异常时给出 `usbipd attach` 恢复步骤

**关键词触发**：`optical_sd`、`cvapp_yolov8n_ob.cpp`、`xmodem`、`usbipd attach`、`pipeline` 等

