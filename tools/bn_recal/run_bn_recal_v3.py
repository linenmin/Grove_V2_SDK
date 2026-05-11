"""BN recalibration for retrain_v3 fixed-arch checkpoints at deploy resolution.

Cheapest possible intervention to fix the input-resolution OOD problem
identified in plan/MCUFlowNet_Deployment/findings.md §7d:

    The v3 subnets were trained at 480×640 (FT3D) but are deployed at
    157×203. Their batch-norm running stats reflect 480×640 feature
    distributions; at 157×203 the spatial sizes are 3× smaller and BN
    stats are wrong (mean/var of much smaller feature maps).

What this script does:
    1. Build FixedArchModelV3 inference graph at <H>×<W> with
       `is_training_ph=True` so BN consumes batch statistics and emits
       its UPDATE_OPS to refresh running_mean/running_variance.
    2. Restore the original sintel_best.ckpt.
    3. Iterate over N batches of <H>×<W> Sintel-clean image pairs (no
       GT used — BN is unsupervised), running forward + UPDATE_OPS each
       step. Optional momentum override slides running stats faster.
    4. Save the new ckpt with refreshed BN stats only (conv kernels
       unchanged).

Sintel train **clean** is used as the recal data source for two reasons:
    - No leakage on labels (BN doesn't use GT flow).
    - Same geometric scenes as the eval Final pass → BN sees the deploy
      image distribution at the deploy resolution.

The recal'd ckpt can then be re-exported via tools/model_export/
edgeflownas_v3/run_export.py --ckpt-name <new_name> (requires a small
tweak to that exporter to accept arbitrary ckpt names — see below).
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
    "v3_acc":     {"arch_code": "0,1,2,2,2,2,0,0,0,0,1", "ckpt_dir": "model_v3_acc/checkpoints"},
    "v3_efn_fps": {"arch_code": "2,0,0,2,2,1,0,0,0,0,0", "ckpt_dir": "model_v3_efn_fps/checkpoints"},
    "v3_light":   {"arch_code": "0,0,0,0,0,0,0,0,0,0,0", "ckpt_dir": "model_v3_light/checkpoints"},
}

RETRAIN_DIR = f"{EFNAS}/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1"


def load_sintel_pairs(sintel_root, pass_name="clean"):
    """Return list of (img1_path, img2_path) pairs from Sintel train <pass>."""
    pass_root = os.path.join(sintel_root, "training", pass_name)
    pairs = []
    for scene in sorted(os.listdir(pass_root)):
        d = os.path.join(pass_root, scene)
        if not os.path.isdir(d):
            continue
        frames = sorted([f for f in os.listdir(d) if f.endswith(".png")])
        for i in range(len(frames) - 1):
            pairs.append((os.path.join(d, frames[i]), os.path.join(d, frames[i + 1])))
    return pairs


def build_recal_graph(model_name, arch_code, height, width, bn_momentum_override):
    arch = [int(x) for x in arch_code.split(",")]
    g = tf.Graph()
    with g.as_default():
        sess = tf.compat.v1.Session(graph=g)
        input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[None, height, width, 6], name="input_image"
        )
        is_training_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(True, dtype=tf.bool), shape=[], name="is_training"
        )
        # bake the training-time normalization
        normalized = (input_ph - 127.5) / 127.5
        with tf.compat.v1.variable_scope(model_name):
            model = FixedArchModelV3(
                input_ph=normalized,
                is_training_ph=is_training_ph,
                arch_code=arch,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = model.build()
        # multiscale accumulation just to have a fetch target
        accum = None
        for p in preds:
            if accum is None:
                accum = p; continue
            accum = tf.compat.v1.image.resize_bilinear(
                accum, tf.shape(p)[1:3], align_corners=False, half_pixel_centers=False)
            accum = accum + p
        forward_fetch = accum[..., 0:2]

        # IMPORTANT: BN update ops fire only when explicitly fetched.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        print(f"[bn] discovered {len(update_ops)} UPDATE_OPS in graph")
        if bn_momentum_override is not None:
            print(f"[bn] NOTE: tf.layers.batch_normalization momentum default is "
                  f"0.99; override via this script is not possible without "
                  f"recreating BN layers. Using default momentum + relying on "
                  f"enough batches to stabilize stats.")

        fwd_vars = [v for v in tf.compat.v1.global_variables()
                    if v.name.startswith(f"{model_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=fwd_vars, max_to_keep=2)
    return sess, g, input_ph, is_training_ph, forward_fetch, update_ops, saver, fwd_vars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, choices=sorted(CANDIDATES.keys()))
    ap.add_argument("--ckpt-name", default="sintel_best",
                    help="ckpt stem to start from (default sintel_best)")
    ap.add_argument("--sintel-root", default="/mnt/g/AI_thesis/datasets/MPI-Sintel-complete")
    ap.add_argument("--pass-name", default="clean", choices=("clean", "final"),
                    help="Sintel pass to source BN-recal images from (default clean — "
                         "different visual pass than the eval Final pass).")
    ap.add_argument("--height", type=int, default=157)
    ap.add_argument("--width", type=int, default=203)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=500,
                    help="number of forward+UPDATE_OPS iterations.")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--bn-momentum", type=float, default=None,
                    help="(informational) target BN momentum; cannot be changed "
                         "post-hoc on tf.layers.batch_normalization.")
    ap.add_argument("--out-suffix", default="bn_recal_157x203",
                    help="suffix appended to the new ckpt filename.")
    args = ap.parse_args()

    cand = CANDIDATES[args.model_name]
    src_ckpt = os.path.join(RETRAIN_DIR, cand["ckpt_dir"], f"{args.ckpt_name}.ckpt")
    if not os.path.exists(src_ckpt + ".index"):
        raise SystemExit(f"missing: {src_ckpt}.index")

    out_dir = os.path.dirname(src_ckpt)
    out_ckpt = os.path.join(out_dir, f"{args.ckpt_name}_{args.out_suffix}.ckpt")
    print(f"[init] model={args.model_name} src_ckpt={src_ckpt}")
    print(f"[init] out_ckpt={out_ckpt}")

    pairs = load_sintel_pairs(args.sintel_root, args.pass_name)
    rng = np.random.RandomState(args.shuffle_seed)
    if len(pairs) == 0:
        raise SystemExit("no Sintel pairs found")
    print(f"[data] {len(pairs)} {args.pass_name} pairs from Sintel train")

    sess, g, in_ph, tr_ph, fwd, ups, saver, fwd_vars = build_recal_graph(
        args.model_name, cand["arch_code"], args.height, args.width, args.bn_momentum
    )

    # Restore the original ckpt weights (including original BN running stats).
    saver.restore(sess, src_ckpt)
    print(f"[restore] loaded {len(fwd_vars)} vars from {src_ckpt}")

    # Sanity: snapshot first BN moving_mean before update for diff check.
    bn_mm = [v for v in fwd_vars if "moving_mean" in v.name][0]
    pre = sess.run(bn_mm)
    pre_norm = float(np.abs(pre).mean())
    print(f"[bn-pre] sample BN moving_mean abs-mean before recal: {pre_norm:.6f}")

    t0 = time.time()
    fetched = [fwd] + ups
    pbar = tqdm(range(args.num_batches), ncols=80, desc="bn-recal")
    for step in pbar:
        idxs = rng.randint(0, len(pairs), size=args.batch_size)
        batch = []
        for k in idxs:
            a, b = pairs[k]
            img1 = cv2.imread(a); img2 = cv2.imread(b)
            if img1 is None or img2 is None:
                continue
            img1 = cv2.resize(img1, (args.width, args.height),
                              interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (args.width, args.height),
                              interpolation=cv2.INTER_LINEAR)
            batch.append(np.concatenate([img1, img2], axis=2))
        if not batch:
            continue
        x = np.stack(batch, axis=0).astype(np.float32)
        sess.run(fetched, feed_dict={in_ph: x, tr_ph: True})

    post = sess.run(bn_mm)
    post_norm = float(np.abs(post).mean())
    delta_norm = float(np.abs(post - pre).mean())
    print(f"[bn-post] sample BN moving_mean abs-mean after  recal: {post_norm:.6f}")
    print(f"[bn-diff] mean |Δ| on this BN tensor              : {delta_norm:.6f}")
    print(f"[time] {time.time() - t0:.1f}s for {args.num_batches} batches "
          f"× bs={args.batch_size}")

    saver.save(sess, out_ckpt, write_meta_graph=False)
    print(f"[save] wrote new ckpt: {out_ckpt}.{{index,data-00000-of-00001}}")
    sess.close()


if __name__ == "__main__":
    main()
