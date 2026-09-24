# v3_efn_fps 157x203 FT3D fine-tuned model

This directory contains the FT3D-fine-tuned `v3_efn_fps` candidate used by the
host demo and the corresponding Vela-compiled deployment artifact.

- Host evaluation/demo model: `edgeflownas_v3_efn_fps_157x203.tflite`
- Grove Vision AI V2 deployment model: `edgeflownas_v3_efn_fps_157x203_vela.tflite`
- Input shape: `157x203x6`
- Output shape: `160x208x2`
- Recorded Sintel Final INT8 EPE: `6.60`
- Recorded Vela inference time: `165.22 ms`

SHA-256:

```text
c27c2721cdc6e42146ed16ea8ea1ac28ab3b74d02d1281d8c5527fb2fc05b79c  edgeflownas_v3_efn_fps_157x203.tflite
0c728abf7c0a6a26bdfb7be72621d3e92b6f7fba98b10cb64e46d8d97c20cff9  edgeflownas_v3_efn_fps_157x203_vela.tflite
```

The evaluation context and comparison table are recorded in
`plan/MCUFlowNet_Deployment/findings.md`.
