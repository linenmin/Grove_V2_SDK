"""Host-side four-panel optical-flow demo video for presentation.

Plays a Sintel scene through 3 INT8 TFLite models (Mainline / v3_efn_fps /
v3_light) and renders an MP4 with the source plus three HSV flow panels.

Each panel updates at its model's Vela-estimated inference rate (lower FPS
models visibly lag, faster models update smoothly) so the latency
difference is visually obvious. FPS counter overlay on each panel.

No training-method disclosure shown on the video (per presentation scope).
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Model configs (paths + Vela inference times from prior measurements)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_PANEL_KEY = "_source"  # special key for the source frame panel

# Per-model `mag_coef` calibrates HSV V-channel brightness against the model's
# actual prediction magnitude scale. Mainline at 157x203 systematically
# under-predicts motion (trained at 480x640, never adapted to small input)
# so values are small and need a big gain; v3 subnets (after the *12.5
# flow_scale that undoes ft3d_flow_divisor) put out 10x larger values and
# need a proportionally smaller gain. Default ratio is approx 1:12.5.
MODEL_CONFIGS = {
    INPUT_PANEL_KEY: {
        "label": "Source",
        "color": (180, 180, 180),
        "vela_ms": 1.0,   # source updates every output frame (no fake latency)
    },
    "mainline": {
        "tflite": str(REPO_ROOT / "model_zoo/optical_flow/157x203/optical_flow_157x203.tflite"),
        "in_h": 157, "in_w": 203,
        "flow_scale": 1.0,
        "vela_ms": 188.04,
        "label": "EdgeFlowNet Baseline",
        "color": (200, 200, 200),
        "mag_coef": 8.0,
        "mag_floor": 0.4,
    },
    "v3_efn_fps": {
        "tflite": str(REPO_ROOT / "model_zoo/optical_flow/edgeflownas_v3/v3_efn_fps/157x203_ft3d_ft/edgeflownas_v3_efn_fps_157x203.tflite"),
        "in_h": 157, "in_w": 203,
        "flow_scale": 12.5,
        "vela_ms": 165.22,
        "label": "Ours-FPS",
        "color": (100, 220, 100),
        "mag_coef": 0.65,
        # v3 predicts higher magnitudes including subtle background motion;
        # gate it harder to match mainline's "clean static" feel.
        "mag_floor": 1.5,
    },
    "v3_light": {
        "tflite": str(REPO_ROOT / "model_zoo/optical_flow/edgeflownas_v3/v3_light/172x224_sintel_clean_ft/edgeflownas_v3_light_172x224.tflite"),
        "in_h": 172, "in_w": 224,
        "flow_scale": 12.5,
        "vela_ms": 107.45,
        "label": "Ours-light",
        "color": (100, 180, 240),
        "mag_coef": 0.65,
        "mag_floor": 1.5,
    },
}

PANEL_W = 640   # display width per panel
PANEL_H = 420   # display height per panel
GAP = 6
HEADER_H = 64   # space above each panel for labels
GRID_COLS = 2   # 2x2 layout for 4 panels
DEFAULT_MAG_COEF = 8.0   # bumped from 0.05 (board) -- Sintel motion is small


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


def load_video_frames(path: str, max_long_side: int = 1280) -> Tuple[List[np.ndarray], float]:
    """Read a full video file into a list of BGR frames + report native fps.

    Optionally cap the long-side resolution to keep memory in check
    (1440x1080 × 393 ≈ 1.8 GB; resize to 1280-wide ≈ 1.1 GB).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_long_side > 0:
            h, w = f.shape[:2]
            long_side = max(h, w)
            if long_side > max_long_side:
                s = max_long_side / float(long_side)
                f = cv2.resize(f, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, float(fps)


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
                 mag_coef: float = 0.05,
                 mag_floor: float = 0.5,
                 floor_softness: float = 0.25,
                 blur_sigma: float = 1.0) -> np.ndarray:
    """HSV optical-flow viz with smooth (sigmoid) magnitude suppression.

    Upgrades over the board's literal `mag * 0.05` rendering:

    - `blur_sigma`: Gaussian on the flow field before computing angle &
      magnitude. Removes per-pixel angle noise from INT8 quantization
      (each pixel gets a different discrete flow → wildly different
      arctan2 → speckle of bright colors).

    - SOFT magnitude floor via sigmoid weight (instead of a hard cliff).
      `mag_floor` sets the 50%-suppression point; `floor_softness` sets
      the transition width. Pixels well below `mag_floor` are smoothly
      dimmed (not abruptly black); pixels well above pass through to
      the normal linear V curve. This matches what optical-flow demo
      papers (e.g. flow_vis / Middlebury color-wheel) do to hide low-
      magnitude noise while keeping smooth transitions.

        soft_weight = 0.5 * (1 + tanh((mag - mag_floor) / floor_softness))
        V = clip(mag * mag_coef * 255, 0, 255) * soft_weight
    """
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    if blur_sigma > 0:
        fx = cv2.GaussianBlur(fx, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        fy = cv2.GaussianBlur(fy, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)
    h = ((ang / (2.0 * math.pi)) + 0.5) * 180.0
    s = np.ones_like(mag) * 255.0
    base_v = np.clip(mag * mag_coef * 255.0, 0.0, 255.0)
    if mag_floor > 0:
        softness = max(1e-3, float(floor_softness))
        soft_w = 0.5 * (1.0 + np.tanh((mag - mag_floor) / softness))
        v = base_v * soft_w
    else:
        v = base_v
    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Composite + overlay
# ---------------------------------------------------------------------------
def make_header(panel_w: int, header_h: int, label: str, fps: float,
                accent: Tuple[int, int, int], show_fps: bool = True) -> np.ndarray:
    header = np.zeros((header_h, panel_w, 3), dtype=np.uint8)
    header[:] = (24, 24, 24)
    cv2.rectangle(header, (0, header_h - 3), (panel_w, header_h), accent, -1)
    cv2.putText(header, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if show_fps:
        cv2.putText(header, f"{fps:.1f} FPS", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, accent, 2, cv2.LINE_AA)
    return header


def composite(panels: List[np.ndarray], headers,
              panel_w: int, panel_h: int, header_h: int, gap: int,
              cols: int = GRID_COLS) -> np.ndarray:
    """If header_h == 0 or headers is None, only the panels are drawn."""
    n = len(panels)
    rows = math.ceil(n / cols)
    cell_h = header_h + panel_h
    total_w = panel_w * cols + gap * (cols - 1)
    total_h = cell_h * rows + gap * (rows - 1)
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 16)
    for i in range(n):
        r = i // cols
        c = i % cols
        x0 = c * (panel_w + gap)
        y0 = r * (cell_h + gap)
        if header_h > 0 and headers is not None:
            canvas[y0:y0 + header_h, x0:x0 + panel_w] = headers[i]
        canvas[y0 + header_h:y0 + cell_h, x0:x0 + panel_w] = panels[i]
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="",
                    help="Path to a single video file (mp4 etc). When set, "
                         "renders the whole video as one scene and ignores "
                         "--scene/--scenes; native fps is auto-detected.")
    ap.add_argument("--sintel-root", default="/mnt/g/AI_thesis/datasets/MPI-Sintel-complete")
    ap.add_argument("--scene", default="bamboo_1",
                    help="Single-scene mode. Ignored if --scenes is given.")
    ap.add_argument("--scenes", default="",
                    help="Comma-separated list of Sintel scene names to "
                         "concatenate back-to-back (e.g. "
                         "'market_2,alley_2,bandage_1,temple_2'). Each scene "
                         "starts with fresh model state.")
    ap.add_argument("--scene-gap-sec", type=float, default=0.4,
                    help="Black frame gap between concatenated scenes.")
    ap.add_argument("--playback-speed", type=float, default=1.0,
                    help="<1.0 slows the video (1/0.8=1.25x longer). The "
                         "model inference FPS labels stay accurate; the "
                         "simulated wall-clock just stretches by 1/speed.")
    ap.add_argument("--pass", default="final", dest="pass_name", choices=("clean", "final"))
    ap.add_argument("--input-fps", type=float, default=24.0,
                    help="Source video framerate (Sintel is 24fps).")
    ap.add_argument("--out-fps", type=int, default=30,
                    help="Output video framerate.")
    ap.add_argument("--out", default="/tmp/demo_3panel.mp4")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = use entire scene.")
    ap.add_argument("--mag-coef", type=float, default=None,
                    help="Optional override: a single mag_coef applied to all "
                         "models. By default each model uses its own "
                         "MODEL_CONFIGS[name]['mag_coef'] (calibrated to balance "
                         "v3's flow_scale=12.5 multiplier vs mainline's 1.0).")
    ap.add_argument("--mag-floor", type=float, default=None,
                    help="Optional override: a single sigmoid soft-floor point "
                         "applied to all models. By default each model uses "
                         "its own MODEL_CONFIGS[name]['mag_floor'] (mainline "
                         "0.4, v3 2.0 — v3 predicts higher magnitudes including "
                         "background micro-motion, so it needs a harder gate).")
    ap.add_argument("--floor-softness", type=float, default=0.25,
                    help="Width of the soft-floor transition. Smaller = "
                         "more abrupt; larger = gentler. ~mag_floor/2 is sane.")
    ap.add_argument("--blur-sigma", type=float, default=1.0,
                    help="Gaussian blur sigma on flow field before viz, "
                         "in model-output grid pixels. Smooths per-pixel "
                         "angle noise. 0.0 = no blur.")
    ap.add_argument("--no-headers", action="store_true",
                    help="Drop the per-panel header strip (model label + FPS). "
                         "Useful for clean 4-panel-only export.")
    args = ap.parse_args()

    # Input mode: explicit video file beats Sintel scenes.
    video_mode = bool(args.video.strip())
    if video_mode:
        print(f"[video] {args.video}")
        video_frames, video_fps = load_video_frames(args.video)
        print(f"[video] {len(video_frames)} frames @ {video_fps:.2f} fps "
              f"(input_fps override)")
        scene_list = ["video"]
        args.input_fps = video_fps
    elif args.scenes.strip():
        scene_list = [s.strip() for s in args.scenes.split(",") if s.strip()]
        print(f"[scenes] {len(scene_list)}: {scene_list}")
    else:
        scene_list = [args.scene]
        print(f"[scenes] {len(scene_list)}: {scene_list}")

    # Load interpreters (skip the source panel which has no tflite)
    models = {}
    for name, cfg in MODEL_CONFIGS.items():
        if name == INPUT_PANEL_KEY:
            models[name] = {**cfg, "interp": None}
            continue
        if not Path(cfg["tflite"]).is_file():
            raise SystemExit(f"missing tflite for {name}: {cfg['tflite']}")
        interp, inp_d, out_d = load_interpreter(cfg["tflite"])
        models[name] = {
            **cfg, "interp": interp, "inp_d": inp_d, "out_d": out_d,
            "last_flow_idx": -1, "cached_viz": None,
            "fps": 1000.0 / cfg["vela_ms"],
        }
        eff_coef = args.mag_coef if args.mag_coef is not None else cfg.get("mag_coef", 1.0)
        print(f"[model] {name:<12}  inf={cfg['vela_ms']:.1f}ms  "
              f"fps={models[name]['fps']:.2f}  flow_scale={cfg['flow_scale']}  "
              f"mag_coef={eff_coef}")

    # Canvas geometry + writer (single writer; all scenes concatenate into it)
    head_h = 0 if args.no_headers else HEADER_H
    pan_w, pan_h = PANEL_W, PANEL_H
    n_panels = len(models)
    rows = math.ceil(n_panels / GRID_COLS)
    cell_h = head_h + pan_h
    canvas_w = pan_w * GRID_COLS + GAP * (GRID_COLS - 1)
    canvas_h = cell_h * rows + GAP * (rows - 1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, float(args.out_fps), (canvas_w, canvas_h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter open failed")

    speed = max(0.05, float(args.playback_speed))
    out_frame_period_ms = 1000.0 / args.out_fps
    total_written = 0
    gap_frames = int(max(0.0, args.scene_gap_sec) * args.out_fps)

    # Pre-render the gap frame: headers stay (when enabled), panel content
    # is subdued dark gray. With --no-headers the gap is just dark panels.
    def make_gap_frame() -> np.ndarray:
        gap_panel = np.full((pan_h, pan_w, 3), 28, dtype=np.uint8)
        gap_panels = [gap_panel] * len(models)
        if head_h > 0:
            gap_headers = []
            for name, m in models.items():
                show_fps = (name != INPUT_PANEL_KEY)
                fps_v = m["fps"] if show_fps else 0.0
                gap_headers.append(make_header(pan_w, head_h, m["label"], fps_v,
                                               m["color"], show_fps=show_fps))
        else:
            gap_headers = None
        return composite(gap_panels, gap_headers, pan_w, pan_h, head_h, GAP)

    gap_frame = make_gap_frame()

    for scene_i, scene_name in enumerate(scene_list):
        if video_mode:
            frames = video_frames  # the single user-supplied video
        else:
            try:
                frames = load_scene_frames(Path(args.sintel_root), scene_name, args.pass_name)
            except FileNotFoundError as e:
                print(f"[skip] {scene_name}: {e}")
                continue
        if args.max_frames > 0:
            frames = frames[args.start_frame : args.start_frame + args.max_frames]
        else:
            frames = frames[args.start_frame:]
        if len(frames) < 2:
            print(f"[skip] {scene_name}: <2 frames after trim")
            continue

        # Each scene resets per-model time origin + flow cache so mainline
        # doesn't carry stale flow into the next scene.
        cache_viz: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in models}
        sim_dur_ms = (len(frames) - 1) / args.input_fps * 1000.0
        playback_dur_ms = sim_dur_ms / speed
        n_out_frames = int(playback_dur_ms / 1000.0 * args.out_fps)
        print(f"[scene {scene_i+1}/{len(scene_list)}] {scene_name}: "
              f"{len(frames)} src frames -> {n_out_frames} out frames "
              f"({playback_dur_ms/1000:.1f}s at {speed}x)")

        for out_idx in range(n_out_frames):
            # Simulated wall-clock = output_time * playback_speed (slower
            # playback ⇒ less simulated time per output frame, so model
            # FPS labels stay accurate).
            out_t_ms = out_idx * out_frame_period_ms * speed
            src_now = int(out_t_ms / 1000.0 * args.input_fps)
            src_now = max(0, min(src_now, len(frames) - 1))

            panels = []
            headers = [] if head_h > 0 else None
            for name, m in models.items():
                if name == INPUT_PANEL_KEY:
                    if src_now not in cache_viz[name]:
                        cache_viz[name][src_now] = cv2.resize(
                            frames[src_now], (pan_w, pan_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    panels.append(cache_viz[name][src_now])
                    if headers is not None:
                        headers.append(make_header(pan_w, head_h, m["label"], 0.0,
                                                   m["color"], show_fps=False))
                    continue

                last_compute_t_ms = math.floor(out_t_ms / m["vela_ms"]) * m["vela_ms"]
                src_frame_idx = int(last_compute_t_ms / 1000.0 * args.input_fps)
                src_frame_idx = max(0, min(src_frame_idx, len(frames) - 2))
                if src_frame_idx not in cache_viz[name]:
                    flow = run_model(m["interp"], m["inp_d"], m["out_d"],
                                     frames[src_frame_idx], frames[src_frame_idx + 1],
                                     m["in_h"], m["in_w"], m["flow_scale"])
                    coef = args.mag_coef if args.mag_coef is not None else m["mag_coef"]
                    # Per-model mag_floor unless caller forces a single value.
                    if args.mag_floor is not None:
                        floor = args.mag_floor
                    else:
                        floor = m.get("mag_floor", 0.4)
                    viz = hsv_flow_viz(flow, pan_w, pan_h, coef,
                                       mag_floor=floor,
                                       floor_softness=args.floor_softness,
                                       blur_sigma=args.blur_sigma)
                    cache_viz[name][src_frame_idx] = viz
                panels.append(cache_viz[name][src_frame_idx])
                if headers is not None:
                    headers.append(make_header(pan_w, head_h, m["label"],
                                               m["fps"], m["color"]))

            canvas = composite(panels, headers, pan_w, pan_h, head_h, GAP)
            writer.write(canvas)
            total_written += 1

        cache_sizes = ", ".join(f"{n}={len(cache_viz[n])}" for n in models)
        print(f"  scene done. cache hits: {cache_sizes}")

        # Inter-scene gap: keep the 4 headers / titles visible, panel
        # content goes to a subdued dark-gray so the demo doesn't fully
        # black out between clips. Skip the gap after the last scene.
        if scene_i < len(scene_list) - 1 and gap_frames > 0:
            for _ in range(gap_frames):
                writer.write(gap_frame)
                total_written += 1

    writer.release()
    print(f"[done] wrote {args.out}  total {total_written} frames "
          f"= {total_written/args.out_fps:.1f}s @ {args.out_fps}fps")


if __name__ == "__main__":
    main()
