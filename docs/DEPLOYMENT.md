# Optical Flow Deployment Baseline

本文件只描述 **当前确认有效** 的部署主线，不记录历史试错过程。

如果你是第一次接手仓库，先看：

- [docs/START_HERE.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/START_HERE.md)
- [docs/MINIMAL_DEPLOYMENT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MINIMAL_DEPLOYMENT.md)

## 1. 当前唯一有效基线

- 模型分辨率：`157x203`
- 输入张量：`[1,157,203,6]`
- 输出张量：`[1,160,208,2]`
- 输出量化：`scale ≈ 0.407547`, `zp = -4`
- 可视化结果：`INVOKE resolution = [208,160]`，输出为光流图

## 2. 当前不要作为默认主线的状态

以下状态目前仅保留为实验记录：

- `144x192 -> 144x192` 模型
- `150x200 -> 160x208` 模型
- 当前仓库中的 `model_zoo/optical_cam_oflow/sram_test_modified_vela.tflite`

原因：

- `144x192` 虽然可工作，但不是当前 `1432 KiB` arena 下的最大稳定输入。
- `150x200 -> 160x208` 以及更大的若干实验模型会触发运行期内存失败或 fallback。
- 最终要么 `cv_optical_flow_init fail`，要么 `INVOKE.image` 回退成 `320x240` 相机 JPEG，而不是光流渲染图。

## 3. 当前有效部署流程

### Step 1：准备 `157x203` 模型

当前推荐入口已经切换到仓库内导出脚本：

```bash
bash /home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh
```

说明：

- 导出逻辑已经复制并内聚到本仓库。
- 默认发布模型路径是：
  `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite`
- 当前脚本名仍保留 `144x192` 历史名字，但默认导出参数已经切到 `157x203`。
- 具体覆盖方式见
  [docs/MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)。

模型验证标准：

- 输入必须是 `157x203x6`
- 输出必须是 `160x208x2`
- 当前验证过的运行上限是 `157x203`

### Step 2：烧录固件和模型

使用现有 pipeline 脚本进行 with-model 烧录：

```bash
bash .agent/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
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

串口日志中必须同时看到：

- `model io: in(h=157,w=203,c=6) out(h=160,w=208,c=2)`
- `[out_tensor=0] ... dims=[1,160,208,2]`
- `INVOKE ... "resolution": [208, 160]`

提取帧中应看到：

- 不是相机原图
- 而是暗色 / 伪彩 / 随运动变化的光流可视化图

## 5. 当前关键文件

- 部署脚本：
  [run_optical_pipeline.sh](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.agent/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh)
- 导出脚本：
  [export_optical_flow_144x192.sh](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/scripts/export_optical_flow_144x192.sh)
- 导出说明：
  [docs/MODEL_EXPORT.md](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/MODEL_EXPORT.md)
- 当前默认模型：
  [optical_flow_157x203_vela.tflite](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/optical_flow_157x203_vela.tflite)
- 现有场景代码：
  [cvapp_optical_flow.cpp](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/pipeline/cvapp_optical_flow.cpp)
- 可视化发布：
  [viz_publish.cpp](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/viz/viz_publish.cpp)
- 配置头：
  [common_config.h](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/common_config.h)

## 6. 备注

- 根目录 `README.md` 保持原 Seeed 仓库语义，不在此文档覆盖。
- 历史调试过程请看 `plan/` 下旧计划；本文件只保留当前主线事实。
- 2026-03-13 已验证当前 `1432 KiB` arena 下的最大稳定输入是 `157x203`。
- 同日验证结果表明：Vela 的 arena 估计与模型本体匹配，但不能覆盖运行期额外 `prev buffer` 开销；`158x202` 和 `155x206` 会在板端初始化阶段失败。
