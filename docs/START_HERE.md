# Grove V2 部署与测试

本库使用一个主机入口、一个skill、每个模型一份profile。没有唯一默认模型。

## 环境与常用操作

Python 3.11+；在仓库根目录创建独立环境并安装 requirements-host.txt。本机已配置 C:/00Work/Envs/grove-host/Scripts/python.exe，以下python可用它替代。

```text
python -m venv .venv-host
# 使用新环境的python执行：
python -m pip install -r requirements-host.txt
python tools/grovectl.py doctor
python tools/grovectl.py profile deployments/profiles/optical-flow-observed-172x224.json
python tools/grovectl.py monitor --profile deployments/profiles/optical-flow-observed-172x224.json --seconds 120
```

monitor出现listening后再按RESET；先按后监听可能漏日志。新建运行目录下的STOP空文件可提前结束。串口打开可能影响复位线；monitor不发送设备数据。输出原始uart.bin、summary.json和少量帧；先读摘要，不粘贴base64。退出码3表示验收条件不足，也可能只是缺启动日志。

## 按需阅读

- [部署协议](DEPLOYMENT.md)：profile、编译、打包、烧录与新模型。
- [测试规范](TESTING.md)：运行、数值正确性、性能与证据。
- [硬件/诊断](KNOWLEDGE_BASE.md)：相机、串口、Flash和预览边界。
- [光流项目](../projects/optical_flow/README.md)：特定项目的导出/评测/预览。
- [整理记录](MAINTENANCE.md)：迁移删除与验证范围。
- [历史](history/README.md)：按需回查旧实验。

现有板端172×224模型曾完成相机/NPU初始化与推理，精确哈希未确认；观察profile不能用于刷写。本轮仓库整理不替换板上固件。
