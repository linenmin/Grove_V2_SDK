# Optical Flow Export Tool

这个目录承载当前光流模型的**仓库内导出入口**。

当前默认导出基线：

- 输入：`157x203`
- 输出：`160x208`
- 默认发布模型：
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`
- 默认导出变体：`mainline`

## 目录内容

- `run_export.py`
  当前导出主脚本，支持 `mainline` 和 bilinear shortlist 变体。
- `run_sram_test_bilinear.py`
  Bilinear 上采样实验骨架。随机初始化权重，只用于量化导出和 Vela 报告，不作为当前部署主线。
- `network/`
  从外部 `EdgeFlowNet/sramTest/network` 复制进来的最小网络定义。
- `misc/`
  从外部 `EdgeFlowNet/sramTest/misc` 复制进来的最小辅助依赖。
- `vela/`
  从外部 `EdgeFlowNet/vela` 复制进来的 Vela 包装器。
  同时保留一份当前 Ethos-U55/U65 支持算子参考：
  [vela/SUPPORTED_OPS.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/vela/SUPPORTED_OPS.md)
- `assets/checkpoints/`
  当前导出默认使用的 TensorFlow checkpoint。
- `assets/calibration/`
  当前导出默认使用的 calibration 图像集。

## 当前边界

已经复制进仓库的部分：

- 导出脚本
- 网络定义
- Python 辅助依赖
- Vela 包装器

当前默认已经内置到仓库的部分：

- TensorFlow checkpoint：`assets/checkpoints/best.ckpt*`
- Calibration dataset：`assets/calibration/`
- Vela 支持算子参考：`vela/SUPPORTED_OPS.md`

## 推荐入口

优先使用仓库根目录脚本：

- [export_optical_flow_144x192.sh](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh)

不要再手工去外部仓库执行 `EdgeFlowNet/sramTest/run_sram_test.py`。

## 变体导出

`run_export.py` 当前支持：

- `mainline`
- `baseline`
- `eca`
- `globalgate4x`
- `globalgate4x_bneckeca`
- `globalgate4x_bneckeca_skip8x`
- `globalgate4x_bneckeca_skip8x4x`
- `globalgate4x_bneckeca_skip8x4x2x`

默认仍是 `mainline`，不会覆盖当前对外发布主线。

如果要导出 bilinear shortlist：

```bash
OPTICAL_FLOW_EXPORT_VARIANT=globalgate4x_bneckeca_skip8x4x \
OPTICAL_FLOW_CHECKPOINT_PREFIX=/path/to/that_variant/best.ckpt \
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh --skip-publish
```

说明：

- bilinear 变体必须使用与该结构完全匹配的 checkpoint
- 未显式指定时，bilinear 导出结果会落到对应的 `output_bilinear_*` 目录
- bilinear 发布路径默认落到 `model_zoo/optical_flow/bilinear/<variant>/<HxW>/`

## 仍然存在的环境前提

虽然模型资产已经复制进仓库，导出时仍需要本机具备：

- Python
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Arm Vela
