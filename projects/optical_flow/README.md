# 光流项目

| 入口 | 用途 |
| --- | --- |
| export/optical_flow_144x192 | 原网络、TF导出、校准与Vela，历史名称，默认157×203 |
| export/edgeflownas_v3 | 需指定EFNAS_ROOT的外部代码导出 |
| evaluation/ | Sintel离线评测，非板端精度验收 |
| bn_recal/ | 历史BN重估，外部数据/源码另配 |
| demo_video/ | 主机渲染，面板时延为历史数字 |
| diagnostics/ | 主机模型/图像诊断 |
| flow_viewer.py | 串口JPEG预览，发送0xFC切换RAW且独占串口 |

模型仍在model_zoo。TensorFlow/OpenCV等研究依赖使用独立环境，requirements-host.txt仅供通用主机工具。旧训练/校准资产保留，不盲目删除。

```text
PYTHON_BIN=/path/to/python bash scripts/export_optical_flow_144x192.sh --skip-publish
python projects/optical_flow/flow_viewer.py --port COM5 --baud 921600
python projects/optical_flow/diagnostics/inspect_image.py --image /path/to/frame.jpg
```

compare_vela_vs_nonvela_tflite.py是历史源码参考：普通CPU TFLite解释器不能直接执行ethos-u自定义算子，不作为有效板端参考验证。本轮不重跑训练/离线精度。

[通用入口](../../docs/START_HERE.md) · [历史](../../docs/history/README.md)
