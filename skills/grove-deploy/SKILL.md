---
name: grove-deploy
description: 连接、诊断 Grove Vision AI V2（Himax WE2），按模型配置构建、打包、烧录和验收部署；适用于光流及后续模型的板端测试，不替代训练、量化或离线数据集评测。
---

# Grove 部署与测试

定位目标SDK的tools/grovectl.py与deployments/profiles，先读[入口](../../docs/START_HERE.md)。主机操作共用CLI，skill不复制实现、不绑定编辑器或/home/enmin。

- 连接：doctor枚举，按VID/PID及序列号选择；COM号不是永久身份。
- 复位：本机板（CH343，序列号5AB9069839，常为COM5）由主机复位，不请用户按RESET，也不请用户去终端运行脚本。同时拉高RTS和DTR约0.2 s再放下，板子重启进第一级引导程序，效果同RESET键；单拉一根无效（2026-10-11探测，C:/00Work/Runs/GroveV2/autoreset-probe-20261011/probe-result.json）。flash与monitor加`--auto-reset`：刷写在烧录器打开串口1 s后、监听在出现listening后各发一次脉冲（实现见tools/grove/serial_io.py与xmodem_autoreset.py；自写串口脚本复用pulse_reset，不另抄一份）。不加该选项时，grovectl仍提示按RESET。运行目录里早先抄的autoreset_xmodem.py副本，有的用pyserial默认打开串口（两线先拉高），可能只产生下降沿、复位失败，新任务不再沿用。其他板未经同样探测，仍请用户按RESET。
- 观察：monitor只接收。先等listening并保持进程存活，再复位，留足时间；STOP文件提前停止。空日志不能判定设备坏了。
- 部署：读[部署协议](../../docs/DEPLOYMENT.md)，明确app/model/布局，build→package→flash计划。已授权具体部署直接--execute；否则交付具体包供审批。传输后继续运行验收。
- 测试：读[测试规范](../../docs/TESTING.md)，判定最后一次启动、I/O、错误与有效帧。结构通过不等于身份、精度或预览FPS。
- 新模型：独立profile；尚无适配app/协议时先实现并说明缺口，不只换模型文件。
- 光流研究工具：[项目页](../../projects/optical_flow/README.md)，按需读取，研究环境与主机环境隔离。

Windows默认管USB，WSL/Linux管构建。monitor、浏览器、viewer和刷写程序不得抢占串口，释放后再交接。打开串口可能影响复位线；自写串口客户端先把RTS、DTR置低再打开（同monitor）。

当前光流映射地址0x3AB7B000对应烧录偏移0xB7B000，其他app重新核验。RAW可能仍是JPEG，不是u/v；Vela内存不涵盖全部运行缓冲。

OV5647 I²C -60为地址无应答，先查接线/相机/供电与配置，不先刷模型。保留当前证据，停止同样失败的盲目重试。

Runs保留完整产物，对话只给摘要与必要帧。不自动创建多份plan/快照或发Discord；历史只按需读，不把旧尺寸当全库默认。沿用用户已有授权，不重复确认。
