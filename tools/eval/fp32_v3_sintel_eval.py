"""FP32 sintel EPE for retrain_v3 subnets — uses the same evaluator pipeline
as `int8_sintel_eval.py --ref-mode test_sintel`, just running the float TF1
graph instead of the INT8 TFLite interpreter.

Lets us isolate whether the EPE gap we see at 157×203 vs HPC's 416×1024 eval
is due to (a) input-resolution generalization or (b) PTQ INT8 quantization.
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

tf.compat.v1.disable_eager_execution()

EFNAS = "/mnt/d/Dataset/MCUFlowNet/EdgeFlowNAS"
if EFNAS not in sys.path:
    sys.path.insert(0, EFNAS)

from efnas.network.fixed_arch_models_v3 import FixedArchModelV3  # noqa: E402

CANDIDATES = {
    "v3_acc":     {"arch_code": "0,1,2,2,2,2,0,0,0,0,1", "ckpt": "model_v3_acc/checkpoints/sintel_best.ckpt"},
    "v3_efn_fps": {"arch_code": "2,0,0,2,2,1,0,0,0,0,0", "ckpt": "model_v3_efn_fps/checkpoints/sintel_best.ckpt"},
    "v3_light":   {"arch_code": "0,0,0,0,0,0,0,0,0,0,0", "ckpt": "model_v3_light/checkpoints/sintel_best.ckpt"},
}

RETRAIN_DIR = f"{EFNAS}/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1"


def read_flo(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"PIEH"
        w = np.frombuffer(f.read(4), np.int32)[0]
        h = np.frombuffer(f.read(4), np.int32)[0]
        return np.frombuffer(f.read(2 * w * h * 4), np.float32).reshape(h, w, 2)


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


def _center_crop(arr_hw, target_h, target_w):
    h, w = arr_hw.shape[:2]
    if h < target_h or w < target_w:
        pad_h = max(0, target_h - h); pad_w = max(0, target_w - w)
        arr_hw = np.pad(arr_hw, ((pad_h // 2, pad_h - pad_h // 2),
                                  (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                        mode="reflect")
        h, w = arr_hw.shape[:2]
    y0 = int(np.ceil(h / 2 - target_h / 2))
    x0 = int(np.ceil(w / 2 - target_w / 2))
    return arr_hw[y0 : y0 + target_h, x0 : x0 + target_w]


def _closest_resize_dims(src_w, src_h, dst_w, dst_h):
    src_ar = src_w / src_h; dst_ar = dst_w / dst_h
    if dst_ar < src_ar:
        return int(dst_h * src_ar), dst_h
    return dst_w, int(dst_w / src_ar)


def resize_nearest_crop_stack(stack_hwc, target_h, target_w):
    src_h, src_w = stack_hwc.shape[:2]
    new_w, new_h = _closest_resize_dims(src_w, src_h, target_w, target_h)
    resized = cv2.resize(stack_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _center_crop(resized, target_h, target_w)


def resize_flow_to(flow_hw2, new_h, new_w):
    src_h, src_w = flow_hw2.shape[:2]
    if (src_h, src_w) == (new_h, new_w):
        return flow_hw2
    fx = new_w / float(src_w); fy = new_h / float(src_h)
    resized = cv2.resize(flow_hw2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    resized[..., 0] *= fx
    resized[..., 1] *= fy
    return resized


def build_session(model_name, arch_code, height, width):
    arch = [int(x) for x in arch_code.split(",")]
    g = tf.Graph()
    with g.as_default():
        sess = tf.compat.v1.Session(graph=g)
        input_ph = tf.compat.v1.placeholder(tf.float32, [1, height, width, 6], name="in")
        is_training_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(False, dtype=tf.bool), shape=[], name="is_training")
        with tf.compat.v1.variable_scope(model_name):
            m = FixedArchModelV3(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=arch,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = m.build()
        # accumulate multi-scale predictions (same as HPC eval_step.accumulate_predictions)
        accum = None
        for p in preds:
            if accum is None:
                accum = p; continue
            accum = tf.compat.v1.image.resize_bilinear(
                accum, tf.shape(p)[1:3], align_corners=False, half_pixel_centers=False)
            accum = accum + p
        out = accum[..., 0:2]
        fwd = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{model_name}/")]
        ckpt = os.path.join(RETRAIN_DIR, CANDIDATES[model_name]["ckpt"])
        tf.compat.v1.train.Saver(var_list=fwd).restore(sess, ckpt)
    return sess, input_ph, out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, choices=sorted(CANDIDATES.keys()))
    ap.add_argument("--list", required=True, dest="list_path")
    ap.add_argument("--sintel-root", required=True)
    ap.add_argument("--height", type=int, default=157)
    ap.add_argument("--width", type=int, default=203)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-val", type=float, default=50.0)
    ap.add_argument("--patch-h", type=int, default=416)
    ap.add_argument("--patch-w", type=int, default=1024)
    ap.add_argument("--flow-scale", type=float, default=12.5)
    ap.add_argument("--normalize-input", action="store_true",
                    help="apply (x/255)*2 - 1 before feeding the network (training preproc).")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    pairs = load_pair_list(args.list_path, args.sintel_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[init] {len(pairs)} pairs | model={args.model_name} input={args.height}x{args.width}")
    print(f"[init] flow_scale={args.flow_scale} normalize_input={args.normalize_input}")

    sess, in_ph, out_t, g = build_session(args.model_name,
                                          CANDIDATES[args.model_name]["arch_code"],
                                          args.height, args.width)
    print(f"[model] out shape: {out_t.shape}")

    epe_per_frame = []
    per_scene = {}
    skipped = 0
    t0 = time.time()

    for img1_path, img2_path, flo_path in tqdm(pairs, ncols=80):
        img1 = cv2.imread(img1_path); img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            skipped += 1; continue
        try:
            gt = read_flo(flo_path)
        except Exception as e:
            print(f"[skip] {flo_path}: {e}"); skipped += 1; continue

        gt_c = np.clip(gt, -args.clip_val, args.clip_val) if args.clip_val > 0 else gt
        stack = np.concatenate([img1.astype(np.float32), img2.astype(np.float32),
                                 gt_c.astype(np.float32)], axis=2)
        stack_p = resize_nearest_crop_stack(stack, args.patch_h, args.patch_w)
        i1p, i2p = stack_p[..., 0:3], stack_p[..., 3:6]
        gt_eval = stack_p[..., 6:8]
        r1 = cv2.resize(i1p, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        r2 = cv2.resize(i2p, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        x = np.concatenate([r1, r2], axis=2).astype(np.float32)
        if args.normalize_input:
            x = (x / 255.0) * 2.0 - 1.0
        x = x[None, ...]
        pred = sess.run(out_t, feed_dict={in_ph: x})[0]
        if args.flow_scale != 1.0:
            pred = pred * float(args.flow_scale)

        pred_eval = resize_flow_to(pred, args.patch_h, args.patch_w)
        diff = pred_eval - gt_eval
        epe_map = np.sqrt(np.sum(diff * diff, axis=-1))
        frame_epe = float(epe_map.mean())
        epe_per_frame.append(frame_epe)
        scene = os.path.basename(os.path.dirname(img1_path))
        per_scene.setdefault(scene, []).append(frame_epe)

    elapsed = time.time() - t0
    avg = float(np.mean(epe_per_frame)); med = float(np.median(epe_per_frame))
    print(f"\n----- SUMMARY -----\nmodel : {args.model_name}\ninput : {args.height}x{args.width}")
    print(f"frames: {len(epe_per_frame)} skipped: {skipped} elapsed: {elapsed:.1f}s")
    print(f"avg EPE : {avg:.4f}\nmedian EPE : {med:.4f}")
    for s, v in sorted(per_scene.items()):
        print(f"  {s:<20}  n={len(v):>3}  EPE={float(np.mean(v)):.4f}")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"model": args.model_name, "input": [args.height, args.width],
                        "flow_scale": args.flow_scale, "normalize_input": args.normalize_input,
                        "patch_hw": [args.patch_h, args.patch_w], "clip_val": args.clip_val,
                        "avg_epe": avg, "median_epe": med, "n_frames": len(epe_per_frame),
                        "per_scene": {s: float(np.mean(v)) for s, v in per_scene.items()}}, f, indent=2)
        print(f"[report] {args.report}")
    sess.close()


if __name__ == "__main__":
    main()
