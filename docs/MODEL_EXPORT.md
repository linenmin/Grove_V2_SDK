# Model Export

本文件描述当前光流模型的导出入口。

当前默认导出基线：

- 输入：`157x203`
- 输出：`160x208`
- 默认发布模型：
  [optical_flow_157x203_vela.tflite](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite)

## 1. 当前推荐入口

使用仓库内脚本：

```bash
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh
```

这会调用：

- [run_export.py](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/run_export.py)

并默认把最终模型发布到：

- [optical_flow_157x203_vela.tflite](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite)

## 2. 当前默认依赖

- Python: `/home/enmin/miniconda3/envs/vela/bin/python`
- Checkpoint prefix:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/assets/checkpoints/best.ckpt`
- Calibration dir:
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/assets/calibration`

## 3. 当前仓库边界

当前仓库已经内置：

- 导出脚本
- 网络定义
- Python 辅助依赖
- Vela 包装器
- 默认 checkpoint
- 默认 calibration 数据

这意味着：

- 不需要再进入外部仓库执行导出
- 默认导出不再依赖外部 checkpoint 或 calibration 路径
- 当前一条命令默认导出的就是 `157x203 -> 160x208` 主线模型

仍然保留的环境前提是本机 Python 依赖和 Vela 可执行文件。

## 4. 常见覆盖方式

如果你要换 checkpoint：

```bash
OPTICAL_FLOW_CHECKPOINT_PREFIX=/path/to/best.ckpt \
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh
```

如果你要换 calibration 数据：

```bash
OPTICAL_FLOW_CALIBRATION_DIR=/path/to/calibration \
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh
```

如果你只想导出，不想覆盖发布模型：

```bash
/home/enmin/miniconda3/envs/vela/bin/python \
  /home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/run_export.py \
  --skip-publish
```

如果你要手动指定分辨率和发布路径：

```bash
/home/enmin/miniconda3/envs/vela/bin/python \
  /home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/run_export.py \
  --height 157 \
  --width 203 \
  --published-model /home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite
```
