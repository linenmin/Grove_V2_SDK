# 部署测试

| 层级 | 依据 | 边界 |
| --- | --- | --- |
| 产物 | app、代码/工具版本、模型/IMG哈希、Flash布局 | 不代表已上板 |
| 运行 | 当前启动初始化、预期I/O、连续结果、无失败 | 不证明身份或精度 |
| 数值 | 同一输入的主机参考/板端输出、布局、量化、容差 | 彩色JPEG不是u/v |
| 性能 | 固定输入、预热/重复、计时范围、实测arena | Vela估算、固件total与预览FPS分开 |

```text
python tools/grovectl.py verify --profile deployments/profiles/optical-flow-observed-172x224.json --log /path/to/uart.bin --out /path/to/new-verification
python -m unittest discover -s tests -v
```

verify只判定最后一次启动，避免旧成功掩盖新失败。检查必要标志、错误、I/O、JPEG解码和实际/声明尺寸，抽取最后三张帧。半条串口消息不算完整帧。passed只表示结构/日志通过，identity_verified始终false；精度另测。

每次运行独立目录保存session/profile、原始日志及SHA256、摘要和帧。不得重用latest目录的旧帧当新结果。长时监测可使用外部Runs目录。

退出码：0该命令完成/通过；2参数/依赖/运行错误；3日志未通过；130中断。flash另保留底层错误/124超时。doctor不证明固件编译环境就绪。

新任务顺序：训练仓交付固定产物→适配app与profile→初始化/I/O→固定输入数值对齐→重复性能/内存。分类、检测、分割不能仅换profile就宣称完成部署测验。Blender送图和数值光流返回需要新增协议，现有串口JPEG不满足该接口。
