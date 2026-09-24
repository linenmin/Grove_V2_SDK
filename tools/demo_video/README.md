# Optical-flow demo renderer

`render_3panel_demo.py` renders a 2x2 comparison containing the source frame,
the EdgeFlowNet baseline, Ours-FPS, and Ours-light. Despite the historical
filename, the current output has four panels.

The script resolves its model paths relative to the repository root. Required
host-side INT8 models live under `model_zoo/`:

- `optical_flow/157x203/optical_flow_157x203.tflite`
- `optical_flow/edgeflownas_v3/v3_efn_fps/157x203_ft3d_ft/edgeflownas_v3_efn_fps_157x203.tflite`
- `optical_flow/edgeflownas_v3/v3_light/172x224_sintel_clean_ft/edgeflownas_v3_light_172x224.tflite`

Example:

```bash
python tools/demo_video/render_3panel_demo.py \
  --video /path/to/input.mp4 \
  --out /tmp/optical_flow_comparison.mp4
```

The Vela latency numbers shown in the panels are fixed measurements recorded in
the script; the host renderer does not benchmark the target device.
