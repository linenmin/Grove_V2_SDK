> 历史材料：保留当时记录，当前部署按 docs/START_HERE.md；旧工具路径仅作追溯。

# Grove V2：Windows 连接核查与协作方案

核查日期：2026-09-24。代码基点：`2297e6f5a9ad375d18e5efe764755a8bfadf73d3`。本页区分实际观察、旧记录和建议；没有重新编译或刷写设备。

## 已核查的连接

Windows COM5，USB bridge `1A86:55D3`，序列号 `5AB9069839`；921600、8N1 收到 Himax 启动日志。发送字节数为零；打开串口期间收到启动日志，不能保证 DTR/RTS 打开行为对复位无影响。

日志报告 `Optical camera oflow app start`，模型输入 `172×224×6`，输出 `176×224×2`。随后 OV5647 寄存器 `0x4800` 写入失败（I²C `-60`），出现 `camera init fail`、`cv_optical_flow_init fail`。用户确认接了 OV5647。这里只证明固件读取了模型元数据，不能证明 NPU 推理成功，也不能确定 Flash 中模型的精确版本。

首轮原始证据在 `C:/00Work/Runs/GroveV2/connection-audit-20260924/passive-uart.bin`。设备后续状态须重新查询。

## 可直接使用的连接检查

在仓库根目录 PowerShell 执行：

```powershell
./scripts/grove_probe.ps1 -Action list
./scripts/grove_probe.ps1 -Action capture -SerialNumber 5AB9069839 -Seconds 8
```

脚本只接收数据，不发送握手、复位命令或刷写命令。先关闭占用串口的浏览器/预览程序；每次输出独立目录的原始字节、可读日志、JSON 摘要。退出码：0 收到数据（不等于推理验收通过）、2 连接/读取错误、3 摄像头初始化失败、4 未收到数据。USB ID 是桥接芯片身份，必须结合序列号和板端日志确认设备。

## 仓库实际问题

| 问题 | 证据与影响 |
| --- | --- |
| 接管入口循环且信息冲突 | `plan/README.md` 和 `docs/START_HERE.md` 互相要求先读；部署页坚持 157×203，plan-018 后续已记录 172×224 bilinear。需按模型版本分 profile，不把一个尺寸当所有模型的上限 |
| Windows 路径不成立 | `.agent/skills` 在 Git 中为 symlink，本次 Windows checkout 成为内容 `../.cursor/skills` 的普通文件；文档引用 `.agent/skills/...` 无法执行。真实脚本在 `.cursor/skills/we2-optical-sd-pipeline/scripts/` |
| 操作入口被藏在编辑器目录 | `scripts/run_optical_pipeline.sh` 在文档中出现但实际不存在；稳定工具应放 `tools/` 或 `scripts/`，skill 只说明如何调用 |
| 环境不可直接迁移 | 文档与导出默认值包含 `/home/enmin/...`；当前 WSL 为 Ubuntu-24.04，默认 shell PATH 未发现 `make`、`arm-none-eabi-gcc`。不能直接照旧命令复现 |
| 新鲜克隆无法 skip-build | 未生成 `we2_image_gen_local/output_case1_sec_wlcsp/output.img`；最短路线依赖旧电脑的构建产物 |
| 构建与 Flash 绑定 | pipeline 把编译、镜像生成、刷写、捕获连成一次调用；`--skip-build` 仍会刷写，不是观察模式 |
| 成功判据偏弱 | pipeline 默认查 `initial done`，帧提取失败仅警告；缺少期望模型 I/O、真实光流帧、错误标志、模型哈希联合验收 |
| 原始光流与图像预览混淆 | `tools/flow_viewer.py` 的 RAW 是 JPEG 二进制传输（0xFC 握手），不是原始 u/v 数值。不能把彩色 JPEG 用作机器人控制数值 |
| 潜在无界等待 | `xmodem/xmodem_send.py` 多处 `while(True)` 等 Bootloader 响应，需要总超时与明确失败状态 |

源码已经从 TFLite 读取 I/O 并动态分配内存（`pipeline/cvapp_optical_flow.cpp`），不能单凭配置头的 `160×208` 宏判断板端模型。旧文档的 1432 KiB 是历史条件，不应当作所有固件的固定预算。

## 推荐的目标工作流（尚未实现）

保留 Himax SDK 与当前光流固件。Windows 独占 USB 串口；WSL/Docker 只负责编译和打包，通过共享产物目录交接。日常不需要把设备在 Windows 与 WSL 之间来回 attach/detach。上游本来就支持 Windows Python XMODEM；这次验证了 Windows 接收，未验证新环境烧录。

建议一个统一命令入口，拆为 `doctor / build / flash / monitor / snapshot / verify`。前三个部署动作独立；`monitor` 不隐含刷写。CLI 输出 JSON、明确退出码和本次产物路径，agent 可直接调用；GUI/浏览器只作为同一服务的界面。

一个本机串口服务读取设备，向用户预览窗口和 agent 提供同一份最新帧、状态、日志。刷写时服务暂停接收并释放串口，结束后重新连接。这样用户浏览器不会与 agent 抢 COM。先做 CLI 和帧文件，确有多客户端需求再加服务；无需先引入 MCP。

每个部署 profile 记录模型/固件 SHA256、代码提交、工具链/Vela 版本、相机型号、输入输出/量化、Flash 布局、预期日志。`flash` 必须明确目标 profile；刷写前显示目标设备与文件，刷写后核对 I/O、内存初始化、推理与帧内容。精确板端哈希需要新增固件身份接口，不能从主机文件哈希推断板上已一致。

用户负责接线、镜头与场景、必要时按 RESET；agent 负责构建、校验、刷写流程、日志和帧分析。先用现有 XMODEM 完成一次已知配置验收，再逐步改善，不直接换掉底层 bootloader。

未来 Blender 硬件闭环需要独立的帧输入与数值 u/v 输出协议：带消息类型、长度、帧号、校验、超时及背压。现有 JPEG 预览不够。在 921600 bps、8N1 下，理论有效载荷上限为 92160 B/s；仅一对 172×224 RGB 原图就有 231168 B，传输至少约 2.51 秒，尚未计协议/推理/返回开销。连续流复用上一帧仍约 1.25 秒/新帧。需要按控制需求选择缩图、低频步进、SD 测试集或另行核查高速接口。

## 整理顺序

1. 先修复入口链接，给旧部署结论标日期/模型 profile；保留历史日志和上游 README。
2. 将工具从 `.cursor` 提升到稳定路径；旧路径保留薄包装兼容，不复制两套实现。
3. 固定构建环境、导入部署 manifest、拆分命令、增加超时与校验。模型导出继续由训练仓提供，SDK 消费带身份的部署包，避免长期维护两套网络定义。
4. 用已知模型完成一次构建→刷写→日志→帧验收后，再清理根目录试验脚本、转移大产物到 Runs/发布资产；不提前删除未知用途文件。

本轮只新增连接检查工具和本审计页，补充导航；上述部署重构、共享预览服务及数值协议仍是建议。

## 官方路线核对

访问日期均为 2026-09-24：

- [Himax 上游 SDK](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2)：已有 Linux/Windows 固件烧录路径；不能仅因使用 XMODEM 判定过时。
- [Seeed Himax SDK 教程](https://wiki.seeedstudio.com/grove_vision_ai_v2_himax_sdk/)：WSL 编译与 Windows Python 烧录仍有官方说明。
- [SenseCraft 预训练模型部署](https://sensecraft.seeed.cc/wiki/en/sensecraft-ai/guides/models/using-a-model/sensecraft-ai-pretrained-models-for-grove-vision-ai-v2/)：适合受支持的现成视觉模型和快速预览。尚无本轮证据证明它能直接接管当前六通道光流与自定义 u/v 接口，因此不建议直接替换研究固件。
