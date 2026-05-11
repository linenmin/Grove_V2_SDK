#!/usr/bin/env python3
"""Evaluate a PTQ INT8 TFLite optical-flow model on MPI-Sintel training.

Mirrors the inference numerics of the deployed Grove Vision AI V2 model:
    - input is the PRE-Vela INT8 TFLite (same weights & quant params as the _vela one).
    - input quantization is treated as `int8 = uint8 - 128` when scale=1.0, zp=-128.

Two evaluation modes, selectable via --ref-mode:

* "direct" (controlled by --eval-grid):
    - "native": upsample prediction to GT native resolution (1024x436) and
      rescale flow vectors by the pixel ratio. EPE pixel-for-pixel vs GT.
    - "pred":   legacy debug — downsample GT to pred grid (160x208) with flow
      magnitudes scaled DOWN. EPE in this mode is ~3.66x smaller; not paper-comparable.

* "test_sintel" (--ref-mode test_sintel, default in this project):
    Reproduce the EdgeFlowNet `test_sintel.py` methodology:
      1. Clip GT to [-clip_val, +clip_val]  (default clip_val=50)
      2. Stack [img1, img2, GT] (8ch) and ResizeNearestCrop the stack to
         (patch_h, patch_w)  (default 416 x 1024).
      3. Split back; further bilinear-resize img1, img2 down to the INT8 model
         input (157 x 203). Predict 160 x 208 flow.
      4. Bilinear-upsample prediction to (patch_h, patch_w) and rescale flow
         vectors by the ratio (since pred is in model-grid pixel units).
      5. EPE = mean(||pred - gt||) at (patch_h, patch_w).
    Apples-to-apples vs `test_sintel.py --uncertainity` on the same data list.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def read_flo(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"PIEH":
            raise ValueError(f"bad .flo magic in {path}: {magic!r}")
        w = np.frombuffer(f.read(4), np.int32)[0]
        h = np.frombuffer(f.read(4), np.int32)[0]
        data = np.frombuffer(f.read(2 * w * h * 4), np.float32)
    return data.reshape(int(h), int(w), 2)


def load_pair_list(list_path, sintel_root):
    pairs = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, c = line.split()
            mapped = []
            for rel in (a, b, c):
                rel = rel.replace("Datasets/Sintel//", "")
                rel = rel.replace("Datasets/Sintel/", "")
                mapped.append(os.path.join(sintel_root, rel))
            pairs.append(tuple(mapped))
    return pairs


def build_interpreter(tflite_path, num_threads=4):
    interp = tf.lite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def prepare_input(img1, img2, target_h, target_w, in_scale, in_zp):
    """BGR uint8 imread → resize to (target_w, target_h) → 6-ch concat → int8."""
    r1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    r2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    stack = np.concatenate([r1, r2], axis=2).astype(np.float32)  # H,W,6
    q = np.round(stack / in_scale + in_zp).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q[None, ...]  # 1,H,W,6


def _center_crop(arr_hw, target_h, target_w):
    """Center-crop a HxWx... array to (target_h, target_w). Pads with mirrors if too small."""
    h, w = arr_hw.shape[:2]
    if h < target_h or w < target_w:
        # mirror-pad just enough then crop (rare with 1024x436 -> 416x1024 since W only shrinks)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        arr_hw = np.pad(
            arr_hw,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="reflect",
        )
        h, w = arr_hw.shape[:2]
    y0 = int(np.ceil(h / 2 - target_h / 2))
    x0 = int(np.ceil(w / 2 - target_w / 2))
    return arr_hw[y0 : y0 + target_h, x0 : x0 + target_w]


def _closest_resize_dims(src_w, src_h, dst_w, dst_h):
    """Match the EdgeFlowNet `closest_resizing` helper."""
    src_ar = src_w / src_h
    dst_ar = dst_w / dst_h
    if dst_ar < src_ar:
        new_h = dst_h
        new_w = int(new_h * src_ar)
    else:
        new_w = dst_w
        new_h = int(new_w / src_ar)
    return new_w, new_h


def resize_nearest_crop_stack(stack_hwc, target_h, target_w):
    """Equivalent of misc.ImageUtils.ResizeNearestCrop on an HxWxC numpy array.

    Resizes spatially without aspect distortion (cv2 bilinear), then center-crops
    to exactly (target_h, target_w). Flow values inside the stack are NOT
    magnitude-rescaled (matches test_sintel.py)."""
    src_h, src_w = stack_hwc.shape[:2]
    new_w, new_h = _closest_resize_dims(src_w, src_h, target_w, target_h)
    resized = cv2.resize(stack_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _center_crop(resized, target_h, target_w)


def resize_flow_to(flow_hw2, new_h, new_w):
    """Bilinear-resize a [H,W,2] flow to (new_h,new_w) AND rescale magnitudes."""
    src_h, src_w = flow_hw2.shape[:2]
    if (src_h, src_w) == (new_h, new_w):
        return flow_hw2
    fx = new_w / float(src_w)
    fy = new_h / float(src_h)
    resized = cv2.resize(flow_hw2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    resized[..., 0] *= fx
    resized[..., 1] *= fy
    return resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--list", required=True, dest="list_path")
    ap.add_argument("--sintel-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = full list")
    ap.add_argument(
        "--ref-mode",
        choices=("direct", "test_sintel"),
        default="test_sintel",
        help="direct: use --eval-grid (native/pred). "
             "test_sintel: emulate EdgeFlowNet test_sintel.py "
             "(ResizeNearestCrop @ patch + clip_val + flow rescale to patch grid).",
    )
    ap.add_argument(
        "--eval-grid",
        choices=("native", "pred"),
        default="native",
        help="direct mode only. native = upsample pred to GT res.",
    )
    ap.add_argument("--clip-val", type=float, default=50.0,
                    help="GT clip (test_sintel mode). 0 = no clip.")
    ap.add_argument("--patch-h", type=int, default=416)
    ap.add_argument("--patch-w", type=int, default=1024)
    ap.add_argument("--report", default="")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    pairs = load_pair_list(args.list_path, args.sintel_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[init] {len(pairs)} sintel pairs")

    interp, inp, out = build_interpreter(args.tflite, args.threads)
    in_h, in_w = int(inp["shape"][1]), int(inp["shape"][2])
    out_h, out_w = int(out["shape"][1]), int(out["shape"][2])
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    print(f"[model] in {in_h}x{in_w} (scale={in_scale}, zp={in_zp}) "
          f"out {out_h}x{out_w} (scale={out_scale}, zp={out_zp})")

    epe_per_frame = []
    per_scene = {}  # scene -> list of EPE-per-pixel arrays
    skipped = 0
    t0 = time.time()

    for img1_path, img2_path, flo_path in tqdm(pairs, ncols=80):
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            skipped += 1
            continue
        try:
            gt = read_flo(flo_path)
        except Exception as e:
            print(f"[skip] {flo_path}: {e}")
            skipped += 1
            continue

        if args.ref_mode == "test_sintel":
            gt_clipped = np.clip(gt, -args.clip_val, args.clip_val) if args.clip_val > 0 else gt
            stack = np.concatenate([img1.astype(np.float32),
                                     img2.astype(np.float32),
                                     gt_clipped.astype(np.float32)], axis=2)
            stack_patch = resize_nearest_crop_stack(stack, args.patch_h, args.patch_w)
            i1_patch = stack_patch[..., 0:3]
            i2_patch = stack_patch[..., 3:6]
            gt_eval = stack_patch[..., 6:8]  # GT at (patch_h, patch_w), flow values unchanged
            r1 = cv2.resize(i1_patch, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            r2 = cv2.resize(i2_patch, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            stk = np.concatenate([r1, r2], axis=2)
            q = np.round(stk / in_scale + in_zp).astype(np.int32)
            x = np.clip(q, -128, 127).astype(np.int8)[None, ...]
        else:
            x = prepare_input(img1, img2, in_h, in_w, in_scale, in_zp)

        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y_i8 = interp.get_tensor(out["index"])[0]  # H,W,2 int8
        pred = (y_i8.astype(np.float32) - out_zp) * out_scale  # dequant @ out_h,out_w

        if args.ref_mode == "test_sintel":
            pred_eval = resize_flow_to(pred, args.patch_h, args.patch_w)
        elif args.eval_grid == "native":
            gt_h, gt_w = gt.shape[:2]
            pred_eval = resize_flow_to(pred, gt_h, gt_w)
            gt_eval = gt
        else:
            pred_eval = pred
            gt_eval = resize_flow_to(gt, out_h, out_w)

        diff = pred_eval - gt_eval
        epe_map = np.sqrt(np.sum(diff * diff, axis=-1))  # H,W
        frame_epe = float(epe_map.mean())
        epe_per_frame.append(frame_epe)

        scene = os.path.basename(os.path.dirname(img1_path))
        per_scene.setdefault(scene, []).append(frame_epe)

    elapsed = time.time() - t0
    if not epe_per_frame:
        print("[error] no valid frames")
        sys.exit(2)

    avg_epe = float(np.mean(epe_per_frame))
    median_epe = float(np.median(epe_per_frame))
    scene_summary = {
        s: {"n": len(v), "epe_mean": float(np.mean(v))}
        for s, v in sorted(per_scene.items())
    }

    print("\n----- SUMMARY -----")
    if args.ref_mode == "test_sintel":
        print(f"ref mode         : test_sintel (patch={args.patch_h}x{args.patch_w}, clip_val={args.clip_val})")
    else:
        print(f"ref mode         : direct, eval-grid={args.eval_grid}")
    print(f"frames evaluated : {len(epe_per_frame)}")
    print(f"skipped          : {skipped}")
    print(f"elapsed          : {elapsed:.1f}s "
          f"({elapsed / max(1, len(epe_per_frame)) * 1000:.1f} ms/frame)")
    print(f"average EPE      : {avg_epe:.4f}")
    print(f"median  EPE      : {median_epe:.4f}")
    print("per-scene EPE:")
    for s, info in scene_summary.items():
        print(f"  {s:<20}  n={info['n']:>3}  EPE={info['epe_mean']:.4f}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(
                {
                    "tflite": args.tflite,
                    "list": args.list_path,
                    "sintel_root": args.sintel_root,
                    "ref_mode": args.ref_mode,
                    "eval_grid": args.eval_grid,
                    "clip_val": args.clip_val,
                    "patch_hw": [args.patch_h, args.patch_w],
                    "n_frames": len(epe_per_frame),
                    "n_skipped": skipped,
                    "elapsed_sec": elapsed,
                    "avg_epe": avg_epe,
                    "median_epe": median_epe,
                    "in_shape": [in_h, in_w],
                    "out_shape": [out_h, out_w],
                    "in_quant": {"scale": float(in_scale), "zp": int(in_zp)},
                    "out_quant": {"scale": float(out_scale), "zp": int(out_zp)},
                    "per_scene": scene_summary,
                },
                f,
                indent=2,
            )
        print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
