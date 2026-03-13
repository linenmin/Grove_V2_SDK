# Minimal Deployment Route

本文件只保留**最小可工作路线**。目标不是解释全部背景，而是让 0-context agent 或新接手的人最快完成一次成功部署。

## 1. 先确认你要做的事

你现在的目标应该是：

- 使用 `157x203 -> 160x208` 光流模型
- 烧录 `optical_cam_oflow`
- 串口看到 `model io: in(h=157,w=203,c=6) out(h=160,w=208,c=2)`
- 提取出的 `INVOKE` 帧是光流图，而不是相机图

## 2. 最小前提

- 板子已通过 USB 连接到 WSL，可见 `/dev/ttyACM0`
- 固件工程已能使用现有 `output.img`
- 你手里有一个**已经验证过的** `157x203` 模型文件
- 当前默认模型路径：
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`

如果没有模型文件：

- 先看
  [docs/MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)
- 当前推荐入口是仓库内脚本
  `scripts/export_optical_flow_144x192.sh`

## 3. 一条命令的最小验证

```bash
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/.agent/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode with-model \
  --app-type optical_cam_oflow \
  --port /dev/ttyACM0 \
  --skip-build \
  --capture-seconds 10 \
  --keyword 'initial done' \
  --keyword 'INVOKE' \
  --extract-frames \
  --max-frames 3 \
  --model-arg '/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite 0xB7B000 0x00000'
```

## 4. 成功判据

打开 pipeline log，必须看到：

- `model io: in(h=157,w=203,c=6) out(h=160,w=208,c=2)`
- `[out_tensor=0] ... dims=[1,160,208,2]`
- `INVOKE ... "resolution": [208, 160]`

提取出的帧必须满足：

- 位置：
  [logs/flow_frames/latest](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/flow_frames/latest)
- 结果不是普通相机彩照
- 而是暗色或伪彩的光流图

## 5. 失败时先查什么

### 情况 A：`INVOKE` 变成 `320x240`

优先怀疑：

- 你刷进去的是旧的 fallback 实验模型
- 或者编译期缓存尺寸与模型输出尺寸不一致

这通常意味着：

- 推理可能还在跑
- 但显示已经 fallback 到相机 JPEG

### 情况 B：没有 `model io`

优先怀疑：

- 模型没有成功烧进去
- Flash 地址不对
- 串口抓取窗口太短

### 情况 C：有 `192x144`，但帧像相机图

优先怀疑：

- 抓到的是旧日志
- 提取帧目录没刷新
- 实际输出并未命中光流渲染路径

## 6. 继续往下看什么

- 想看当前主线完整说明：
  [docs/DEPLOYMENT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/DEPLOYMENT.md)
- 想重新导出当前 `157x203` 模型：
  [docs/MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)
- 想看当前知识与坑：
  [docs/KNOWLEDGE_BASE.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/KNOWLEDGE_BASE.md)
- 想看项目整理计划：
  [plan/plan-018-optical-flow-project-reorganization.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-018-optical-flow-project-reorganization.md)
