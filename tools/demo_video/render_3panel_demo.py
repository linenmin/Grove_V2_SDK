"""Host-side 3-panel demo video for presentation.

Plays a Sintel scene through 3 INT8 TFLite models (Mainline / v3_efn_fps /
v3_light) and renders an MP4 with 3 side-by-side HSV optical-flow panels.

Each panel updates at its model's Vela-estimated inference rate (lower FPS
models visibly lag, faster models update smoothly) so the latency
difference is visually obvious. FPS counter overlay on each panel.

No training-method disclosure shown on the video (per presentation scope).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Model configs (paths + Vela inference times from prior measurements)
# ---------------------------------------------------------------------------
SEEED = "/home/enmin/Seeed_Grove_Vision_AI_Module_V2"

MODEL_CONFIGS = {
    "mainline": {
        "tflite": f"{SEEED}/tools/model_export/optical_flow_144x192/output/optical_flow_157x203.tflite",
        "in_h": 157, "in_w": 203,
        "flow_scale": 1.0,
        "vela_ms": 188.04,
        "label": "Mainline (orig transpose-conv)",
        "color": (200, 200, 200),
    },
    "v3_efn_fps": {
        "tflite": "/tmp/ft_v3_efn_fps_sintel_best/edgeflownas_v3_efn_fps_157x203.tflite",
        "in_h": 157, "in_w": 203,
        "flow_scale": 12.5,
        "vela_ms": 165.22,
        "label": "v3_efn_fps (NAS)",
        "color": (100, 220, 100),
    },
    "v3_light": {
        "tflite": "/tmp/v3_light_sintelFT/edgeflownas_v3_light_172x224.tflite",
        "in_h": 172, "in_w": 224,
        "flow_scale": 12.5,
        "vela_ms": 107.45,
        "label": "v3_light (NAS, smallest)",
        "color": (100, 180, 240),
    },
}

PANEL_W = 512   # display width per panel
PANEL_H = 392   # display height per panel (preserves ~aspect for both res)
GAP = 4
HEADER_H = 60   # space above each panel for labels


# ---------------------------------------------------------------------------
# Sintel I/O
# ---------------------------------------------------------------------------
def load_scene_frames(sintel_root: Path, scene: str, pass_name: str = "final") -> List[np.ndarray]:
    pass_dir = sintel_root / "training" / pass_name / scene
    frames: List[np.ndarray] = []
    for png in sorted(pass_dir.glob("frame_*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    if not frames:
        raise FileNotFoundError(f"no frames in {pass_dir}")
    return frames


# ---------------------------------------------------------------------------
# Inference + flow viz
# ---------------------------------------------------------------------------
def load_interpreter(tflite_path: str) -> Tuple:
    interp = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def run_model(interp, inp_details, out_details, img1: np.ndarray, img2: np.ndarray,
              in_h: int, in_w: int, flow_scale: float) -> np.ndarray:
    """Return float flow at the model's native output grid (out_h, out_w, 2)."""
    r1 = cv2.resize(img1, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    r2 = cv2.resize(img2, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    stack = np.concatenate([r1, r2], axis=2).astype(np.float32)
    in_scale, in_zp = inp_details["quantization"]
    q = np.clip(np.round(stack / in_scale + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp_details["index"], q[None, ...])
    interp.invoke()
    y = interp.get_tensor(out_details["index"])[0]  # H,W,2 int8
    out_scale, out_zp = out_details["quantization"]
    pred = (y.astype(np.float32) - out_zp) * out_scale * float(flow_scale)
    return pred


def hsv_flow_viz(flow: np.ndarray, panel_w: int, panel_h: int,
                 mag_coef: float = 0.05) -> np.ndarray:
    """Match the board's `viz_publish.cpp` rendering: HSV with mag*0.05 gain."""
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)  # [-pi, pi]
    h = ((ang / (2.0 * math.pi)) + 0.5) * 180.0  # opencv HSV: H in [0,180]
    s = np.ones_like(mag) * 255.0
    v = np.clip(mag * mag_coef * 255.0, 0.0, 255.0)
    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Composite + overlay
# ---------------------------------------------------------------------------
def make_header(panel_w: int, header_h: int, label: str, fps: float,
                accent: Tuple[int, int, int]) -> np.ndarray:
    header = np.zeros((header_h, panel_w, 3), dtype=np.uint8)
    header[:] = (24, 24, 24)
    cv2.rectangle(header, (0, header_h - 3), (panel_w, header_h), accent, -1)
    cv2.putText(header, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"{fps:.1f} FPS", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, accent, 2, cv2.LINE_AA)
    return header


def composite(panels: List[np.ndarray], headers: List[np.ndarray],
              panel_w: int, panel_h: int, header_h: int, gap: int) -> np.ndarray:
    n = len(panels)
    total_w = panel_w * n + gap * (n - 1)
    total_h = header_h + panel_h
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 16)
    for i, (head, pan) in enumerate(zip(headers, panels)):
        x0 = i * (panel_w + gap)
        canvas[0:header_h, x0:x0 + panel_w] = head
        canvas[header_h:header_h + panel_h, x0:x0 + panel_w] = pan
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="/mnt/g/AI_thesis/datasets/MPI-Sintel-complete")
    ap.add_argument("--scene", default="bamboo_1")
    ap.add_argument("--pass", default="final", dest="pass_name", choices=("clean", "final"))
    ap.add_argument("--input-fps", type=float, default=24.0,
                    help="Source video framerate (Sintel is 24fps).")
    ap.add_argument("--out-fps", type=int, default=30,
                    help="Output video framerate.")
    ap.add_argument("--out", default="/tmp/demo_3panel.mp4")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = use entire scene.")
    ap.add_argument("--mag-coef", type=float, default=0.05)
    args = ap.parse_args()

    print(f"[load] Sintel {args.scene} / {args.pass_name}")
    frames = load_scene_frames(Path(args.sintel_root), args.scene, args.pass_name)
    print(f"[load] {len(frames)} frames")
    if args.max_frames > 0:
        frames = frames[args.start_frame : args.start_frame + args.max_frames]
    else:
        frames = frames[args.start_frame:]
    if len(frames) < 2:
        raise SystemExit("need >=2 frames")

    # Load interpreters
    models = {}
    for name, cfg in MODEL_CONFIGS.items():
        if not os.path.exists(cfg["tflite"]):
            raise SystemExit(f"missing tflite for {name}: {cfg['tflite']}")
        interp, inp_d, out_d = load_interpreter(cfg["tflite"])
        models[name] = {
            **cfg, "interp": interp, "inp_d": inp_d, "out_d": out_d,
            "last_flow_idx": -1, "cached_viz": None,
            "fps": 1000.0 / cfg["vela_ms"],
        }
        print(f"[model] {name:<12}  tflite={cfg['tflite']}  "
              f"inf={cfg['vela_ms']:.1f}ms  fps={models[name]['fps']:.2f}")

    # Pre-decide playback structure.
    total_dur_ms = (len(frames) - 1) / args.input_fps * 1000.0
    total_out_frames = int(total_dur_ms / 1000.0 * args.out_fps)
    out_frame_period_ms = 1000.0 / args.out_fps
    print(f"[play] {total_dur_ms:.1f} ms source => {total_out_frames} out frames @ {args.out_fps}fps")

    # OpenCV VideoWriter
    head_h = HEADER_H
    pan_w, pan_h = PANEL_W, PANEL_H
    canvas_w = pan_w * len(models) + GAP * (len(models) - 1)
    canvas_h = head_h + pan_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, float(args.out_fps), (canvas_w, canvas_h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter open failed")

    # Cache: per-model dict of input_pair_idx -> hsv viz panel.
    cache_viz: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in models}

    # For each output frame, decide which input_pair is being displayed by each model.
    # A model's panel refreshes at intervals of vela_ms wall-clock. The flow
    # shown at out_t_ms is for the input pair (img_k, img_{k+1}) where k = the
    # SOURCE frame index at the most recent inference start time before out_t_ms.
    for out_idx in range(total_out_frames):
        out_t_ms = out_idx * out_frame_period_ms

        panels = []
        headers = []
        for name, m in models.items():
            last_compute_t_ms = math.floor(out_t_ms / m["vela_ms"]) * m["vela_ms"]
            # Map that wall-clock moment to a source frame index
            src_frame_idx = int(last_compute_t_ms / 1000.0 * args.input_fps)
            src_frame_idx = max(0, min(src_frame_idx, len(frames) - 2))

            if src_frame_idx not in cache_viz[name]:
                flow = run_model(m["interp"], m["inp_d"], m["out_d"],
                                 frames[src_frame_idx], frames[src_frame_idx + 1],
                                 m["in_h"], m["in_w"], m["flow_scale"])
                viz = hsv_flow_viz(flow, pan_w, pan_h, args.mag_coef)
                cache_viz[name][src_frame_idx] = viz
            panels.append(cache_viz[name][src_frame_idx])
            headers.append(make_header(pan_w, head_h, m["label"], m["fps"], m["color"]))

        canvas = composite(panels, headers, pan_w, pan_h, head_h, GAP)
        writer.write(canvas)

        if (out_idx + 1) % args.out_fps == 0:
            print(f"  rendered {out_idx + 1}/{total_out_frames} out frames "
                  f"(cache sizes: " +
                  ", ".join(f"{name}={len(cache_viz[name])}" for name in models) + ")")

    writer.release()
    print(f"[done] wrote {args.out}")
    print(f"[stats] inference cache hits per model: "
          + ", ".join(f"{name}={len(cache_viz[name])}" for name in models))


if __name__ == "__main__":
    main()
