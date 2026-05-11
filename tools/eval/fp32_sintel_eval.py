#!/usr/bin/env python3
"""FP32 best.ckpt Sintel EPE — methodology-matched to int8_sintel_eval.py.

Same pipeline as the INT8 evaluator, but the network runs as a TF1 float graph
restored from `best.ckpt` instead of the INT8 TFLite interpreter:

    BGR uint8 -> resize to 157x203 -> 6-ch concat -> float32 -> net forward
    -> 160x208 flow -> upsample to GT native (1024x436) + rescale vectors
    -> EPE pixel-for-pixel against original GT

That gives a Delta EPE = INT8 - FP32 that isolates the contribution of PTQ
quantization (no resolution / methodology gap).
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

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = "/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/optical_flow_144x192"
if NETWORK_DIR not in sys.path:
    sys.path.insert(0, NETWORK_DIR)

from network.MultiScaleResNet import MultiScaleResNet  # noqa: E402

MODEL_CONFIG = {
    "InitNeurons": 32,
    "ExpansionFactor": 2.0,
    "NumSubBlocks": 2,
    "NumOut": 4,
    "NumBlocks": 1,
    "Padding": "same",
}


def read_flo(path):
    with open(path, "rb") as f:
        if f.read(4) != b"PIEH":
            raise ValueError(f"bad .flo magic: {path}")
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


def resize_flow_to(flow_hw2, new_h, new_w):
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


def build_session(checkpoint_prefix, height, width):
    g = tf.Graph()
    with g.as_default():
        input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[1, height, width, 6], name="input_image"
        )
        net = MultiScaleResNet(
            InputPH=input_ph,
            InitNeurons=MODEL_CONFIG["InitNeurons"],
            ExpansionFactor=MODEL_CONFIG["ExpansionFactor"],
            NumSubBlocks=MODEL_CONFIG["NumSubBlocks"],
            NumOut=MODEL_CONFIG["NumOut"],
            NumBlocks=MODEL_CONFIG["NumBlocks"],
            Padding=MODEL_CONFIG["Padding"],
        )
        outs = net.Network()
        # mainline multi-scale: accumulate via bilinear upsample then take 0:2
        accum = None
        for o in outs:
            if accum is None:
                accum = o
                continue
            accum = tf.compat.v1.image.resize_bilinear(accum, [o.shape[1], o.shape[2]])
            accum = accum + o
        final = accum[..., 0:2]  # [1, 160, 208, 2] float32

        saver = tf.compat.v1.train.Saver()
        sess = tf.compat.v1.Session(graph=g)
        saver.restore(sess, checkpoint_prefix)
    return sess, input_ph, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="/home/enmin/Seeed_Grove_Vision_AI_Module_V2/tools/model_export/"
        "optical_flow_144x192/assets/checkpoints/best.ckpt",
    )
    ap.add_argument("--list", required=True, dest="list_path")
    ap.add_argument("--sintel-root", required=True)
    ap.add_argument("--height", type=int, default=157)
    ap.add_argument("--width", type=int, default=203)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    pairs = load_pair_list(args.list_path, args.sintel_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[init] {len(pairs)} sintel pairs")
    print(f"[init] checkpoint = {args.checkpoint}")

    sess, in_ph, out_t = build_session(args.checkpoint, args.height, args.width)
    print(f"[model] in {args.height}x{args.width} (FP32) -> out {out_t.shape}")

    epe_per_frame = []
    per_scene = {}
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

        r1 = cv2.resize(img1, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        r2 = cv2.resize(img2, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        x = np.concatenate([r1, r2], axis=2).astype(np.float32)[None, ...]

        pred = sess.run(out_t, feed_dict={in_ph: x})[0]  # H,W,2 float32

        gt_h, gt_w = gt.shape[:2]
        pred_native = resize_flow_to(pred, gt_h, gt_w)
        diff = pred_native - gt
        epe_map = np.sqrt(np.sum(diff * diff, axis=-1))
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
    print(f"checkpoint       : {args.checkpoint}")
    print(f"eval grid        : native (GT res, methodology-matched to INT8)")
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
                    "checkpoint": args.checkpoint,
                    "list": args.list_path,
                    "sintel_root": args.sintel_root,
                    "eval_grid": "native",
                    "n_frames": len(epe_per_frame),
                    "n_skipped": skipped,
                    "elapsed_sec": elapsed,
                    "avg_epe": avg_epe,
                    "median_epe": median_epe,
                    "in_shape": [args.height, args.width],
                    "out_shape": [int(out_t.shape[1]), int(out_t.shape[2])],
                    "per_scene": scene_summary,
                },
                f,
                indent=2,
            )
        print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
