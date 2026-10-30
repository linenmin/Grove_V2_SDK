# Grove V2 多模型部署工作区

默认中文交流。先读 [使用入口](docs/START_HERE.md)，操作按 [grove-deploy skill](skills/grove-deploy/SKILL.md)，不必全文读取历史。

- tools/grovectl.py：统一主机入口，不依赖训练框架。
- deployments/profiles/：每个模型/应用组合独立配置，无全库默认模型。
- projects/optical_flow/：原光流导出、评测、预览与校准资产。
- EPII_CM55M_APP_S、we2_image_gen_local、xmodem、swd_debugging：底层SDK，保持路径兼容。
- docs/history：历史证据，不是当前操作指令。
- runs/（忽略提交）或指定外部Runs：构建、串口日志、帧与摘要。

观察不隐含刷写。已获授权的具体部署直接执行，无需重复确认；整理任务本身不授权替换板端固件。刷写须明确profile/设备并校验bundle。Windows管串口、WSL/Linux管编译；一次只有一个串口拥有者。请用户RESET前先等monitor输出listening并保持进程存活；用STOP文件结束监听。

新模型不继承旧I/O、量化、arena或Flash槽位。光流映射地址0x3AB7B000与烧录偏移0xB7B000不可混用。JPEG预览不是原始u/v。验收分清产物、传输、初始化/推理、数值精度和性能；日志关键词不能证明精确模型身份。

更新已有说明与项目记录；不强制每轮创建plan、快照或副本。历史失败保留，新证据另记。README.md保持上游说明，日常从docs/START_HERE.md进入。
