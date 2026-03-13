# Current Bilinear Baseline

## 1. 当前关注模型

- 当前分析对象是 bilinear 版本的 decoder skeleton。
- 当前板端已验证可运行的版本：
  `172x224 -> 176x224`
- 对应 Vela 产物目录：
  [output_bilinear/172x224](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x224)

## 2. Vela SRAM peak 在哪里

### 2.1 当前板上可运行版本：`172x224 -> 176x224`

- Vela summary 峰值：
  `1386.00 KiB`
  见
  [optical_flow_bilinear_172x224_summary_Grove_Sys_Config.csv](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x224/optical_flow_bilinear_172x224_summary_Grove_Sys_Config.csv)
- per-layer 里 `Peak%=100.00` 的 op 是：
  `ResizeBilinear_1`
  见
  [optical_flow_bilinear_172x224_per-layer.csv](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x224/optical_flow_bilinear_172x224_per-layer.csv)
- detailed allocation 显示峰值发生在 decoder 尾段多个特征图并存时：
  - `Conv53/conv2d/Conv2D1` 持有 `630784 B`
  - `ResizeBilinear` / `add` 持有 `157696 B`
  - `ResizeBilinear_1` / `add_1` 新分配 `630784 B`
- 这一时刻总峰值：
  `1419264 B = 1386.00 KiB`
  见
  [detailed_performance.txt](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192/output_bilinear/172x224/detailed_performance.txt)

### 2.2 失败版本：`172x228 -> 176x240`

- Vela summary 峰值：
  `1485.00 KiB`
- per-layer 里 `Peak%=100.00` 的 op 仍然是：
  `ResizeBilinear_1`
- detailed allocation 峰值：
  `1520640 B = 1485.00 KiB`
- 板端 `AllocateTensors()` 报：
  `Requested: 1520720, available 1421976, missing: 98744`
- 这说明板端失败请求值与 Vela 峰值几乎一一对应，只差 `80 B` 级别的解释器额外开销。

## 3. 不要混淆的两件事

- **Vela peak 点**
  说的是模型内部 arena 热点，当前热点是 decoder 尾段 `ResizeBilinear_1`。
- **板端上机边界**
  还会受 `prev buffer`、`viz_buffers`、sensor buffers 等运行期开销影响。

## 4. 当前板端观测

### 4.1 `172x224 -> 176x224`

- 成功上板日志：
  [pipeline_with-model_optical_cam_oflow_20260313_171310.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_171310.log)
- 关键现象：
  - `model io: in(h=172,w=224,c=6) out(h=176,w=224,c=2)`
  - `resolution = [224, 176]`
  - `infer ≈ 178.5 ms`
  - `total ≈ 206.3 ms`
  - 算法帧率约 `400 / 82.7 ≈ 4.84 fps`

### 4.2 `172x228 -> 176x240`

- 失败日志：
  [pipeline_with-model_optical_cam_oflow_20260313_170148.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_170148.log)
- 关键现象：
  - `model io: in(h=172,w=228,c=6) out(h=176,w=240,c=2)`
  - `AllocateTensors fail`
  - 板端并未进入稳定 `INVOKE`

### 4.3 `172x224` 但 fallback 的旧记录

- 带 `320x240` 输出的记录：
  [pipeline_with-model_optical_cam_oflow_20260313_165902.log](/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_165902.log)
- 这类记录不能拿来判断 bilinear 模型不可运行，只说明当时可视化路径没有命中光流输出。
