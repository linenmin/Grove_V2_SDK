# 2026-09-24 整理记录

入口：[START_HERE](START_HERE.md)。本轮在 maintenance/windows-device-audit 分支整理主机部署与测试，不改底层 SDK、现有模型二进制或上游 README。

## 文件分工

| 位置 | 用途 |
| --- | --- |
| tools/grovectl.py、tools/grove/ | 通用连接、配置校验、构建、打包、烧录及日志验收 |
| deployments/profiles/ | 按模型和应用保存布局、哈希与验收条件 |
| deployments/model-catalog.json | 26 个既有 TFLite 的路径、大小与 SHA256；不代表新部署结果 |
| projects/optical_flow/ | 光流导出、离线评测、BN 重估、演示和诊断 |
| skills/grove-deploy/ | 唯一部署 skill；个人 skills 中只有指向这里的入口 |
| docs/history/ | 旧计划、旧入口和审计记录，供回查 |
| 仓库外 Runs/GroveV2/ | 每次运行的日志、摘要、帧及构建产物 |

删除或移出工作树的 68 个文件见 [清单](history/removed-files.json)：包含 7 个旧 skill 及辅助治理文件、重复诊断脚本、硬编码评测快捷脚本和 23 个旧日志。旧 plan 迁入历史，研究代码迁入项目目录。迁移的 326 个权重/校准等资产与备份逐一核对哈希一致。

## 恢复

整理前提交：2297e6f5a9ad375d18e5efe764755a8bfadf73d3。完整工作树备份位于 C:/00Work/Backups/Grove_V2_SDK-before-cleanup-20260924/worktree.tar（不含 .git）；旁边保存初始修改补丁、提交号和历史日志。

备份 SHA256：DF4AF81F6D2E8E40B29BAFC0DF198341BB69D33ACF9AD40FC1B853EE8A49D4CE。

恢复时先解压到新的目录核对，再选择取回文件；不要直接覆盖正在工作的仓库。历史已跟踪文件也可从整理前 Git 提交取回。

## 验证与边界

- 主机自动测试 15 项通过，覆盖配置/哈希/布局、串口只读和释放、最后一次启动、有效 JPEG、超时与带空格路径。
- Python 语法编译、Shell 语法及当前入口 Markdown 链接检查通过。
- 真实成功日志重放识别 51 帧；真实相机失败日志正确判为失败。
- 新接收工具实机读取 22 帧；因为未复位、缺启动标志与 I/O，完整验收正确标为未通过。
- 构建 dry-run 通过。当前默认 WSL 缺 make，未完成固件编译；新烧录路径未做实机写入验证，本轮没有刷写。
- 未实现共享浏览器预览服务、板端哈希回读或通用数值张量协议；不同任务仍须适配板端应用。出图不等于数值精度通过。

本轮证据目录：C:/00Work/Runs/GroveV2/restructure-20260924/。研究导出/训练依赖与主机环境分离，未重跑完整训练和数据集评测。
