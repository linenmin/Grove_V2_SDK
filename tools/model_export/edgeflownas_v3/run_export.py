"""Export a retrain_v3 sub-net (v3_acc / v3_efn_fps / v3_light) to INT8 + Vela.

Mirrors `tools/model_export/optical_flow_144x192/run_export.py` but builds the
graph via `efnas.network.fixed_arch_models_v3.FixedArchModelV3` and loads the
HPC-trained checkpoint at `EdgeFlowNAS/outputs/retrain_v3_ft3d/.../checkpoints/`.

Pipeline:
    float ckpt -> TF1 graph at <H>x<W>x6  ->  PTQ INT8 (calibration via
    mainline calibration frames) -> Vela compile -> publish Vela .tflite.

Default input H,W = 157,203 to match the board's current 1432 KiB arena
(bilinear V3 has lower SRAM peak than the mainline transpose-conv, so this
should fit comfortably; larger inputs only if Vela report stays below arena).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = "/home/enmin/Seeed_Grove_Vision_AI_Module_V2"
EFNAS_ROOT = "/mnt/d/Dataset/MCUFlowNet/EdgeFlowNAS"
RETRAIN_DIR = os.path.join(
    EFNAS_ROOT, "outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1"
)
CALIB_DIR = os.path.join(
    REPO_ROOT,
    "tools/model_export/optical_flow_144x192/assets/calibration",
)

CANDIDATES = {
    "v3_acc":     {"arch_code": "0,1,2,2,2,2,0,0,0,0,1", "ckpt_dir": "model_v3_acc/checkpoints"},
    "v3_efn_fps": {"arch_code": "2,0,0,2,2,1,0,0,0,0,0", "ckpt_dir": "model_v3_efn_fps/checkpoints"},
    "v3_light":   {"arch_code": "0,0,0,0,0,0,0,0,0,0,0", "ckpt_dir": "model_v3_light/checkpoints"},
}

if EFNAS_ROOT not in sys.path:
    sys.path.insert(0, EFNAS_ROOT)

from efnas.network.fixed_arch_models_v3 import FixedArchModelV3  # noqa: E402

# vela wrapper (already used by mainline export)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools/model_export/optical_flow_144x192"))
from vela.vela_compiler import run_vela  # noqa: E402


def representative_dataset_gen(calib_dir: str, height: int, width: int):
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(calib_dir)
    pairs = []
    for sub in sorted(os.listdir(calib_dir)):
        d = os.path.join(calib_dir, sub)
        if not os.path.isdir(d):
            continue
        frames = sorted(glob.glob(os.path.join(d, "*.png")))
        for i in range(len(frames) - 1):
            pairs.append((frames[i], frames[i + 1]))
    n = min(100, len(pairs))
    if n == 0:
        raise RuntimeError(f"no calibration pairs in {calib_dir}")
    for i in range(n):
        a, b = pairs[i]
        img1 = cv2.imread(a)
        img2 = cv2.imread(b)
        if img1 is None or img2 is None:
            continue
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        t = np.concatenate([img1, img2], axis=2).astype(np.float32)
        yield [t[None, ...]]


def _accumulate_preds(preds):
    """Bilinear upsample + add (matches misc.utils.AccumPreds in EdgeFlowNet)."""
    accum = None
    for p in preds:
        if accum is None:
            accum = p
            continue
        h = int(p.shape[1])
        w = int(p.shape[2])
        accum = tf.compat.v1.image.resize_bilinear(accum, [h, w])
        accum = accum + p
    return accum


def build_inference_graph(model_name: str, arch_code, height: int, width: int):
    """Build inference graph that takes raw uint8-like input and normalizes
    internally to match training preprocessing `(x/255)*2 - 1`.

    The board firmware feeds `int8 = uint8 - 128` (scale=1.0, zp=-128), same
    convention as mainline. Internal normalization is folded by the
    TFLite converter; from the firmware's POV nothing changes."""
    arch_list = [int(x) for x in arch_code.split(",")]
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        input_ph = tf.compat.v1.placeholder(
            tf.float32, shape=[1, height, width, 6], name="input_image"
        )
        # Bake training-time preprocessing into the graph:
        #   train: (uint8 / 255) * 2 - 1  ==  (uint8 - 127.5) / 127.5
        normalized = (input_ph - 127.5) / 127.5
        is_training_const = tf.constant(False, dtype=tf.bool, name="is_training")
        with tf.compat.v1.variable_scope(model_name):
            model = FixedArchModelV3(
                input_ph=normalized,
                is_training_ph=is_training_const,
                arch_code=arch_list,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = model.build()  # [out_1_4, out_1_2, out_1_1] each NumOut=4
        accum = _accumulate_preds(preds)
        final_output = accum[..., 0:2]  # take 2 flow channels
        return sess, input_ph, final_output, graph


def restore_forward_vars(sess, graph, model_name: str, ckpt_prefix: str):
    """Restore only the forward-pass vars in `model_name/...` scope from ckpt.

    The ckpt has Adam slots and _grad_accum vars too; we ignore those by
    constructing a Saver over just the inference-side global vars (which the
    forward graph has actually created)."""
    with graph.as_default():
        fwd_vars = [v for v in tf.compat.v1.global_variables()
                    if v.name.startswith(f"{model_name}/")]
        if not fwd_vars:
            raise RuntimeError(f"no vars under scope {model_name!r}")
        saver = tf.compat.v1.train.Saver(var_list=fwd_vars)
        saver.restore(sess, ckpt_prefix)
        print(f"[+] restored {len(fwd_vars)} vars from {ckpt_prefix}")


def export_tflite(sess, input_ph, final_output, calib_dir, height, width, out_path):
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(
        sess, [input_ph], [final_output]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = lambda: representative_dataset_gen(
        calib_dir, height, width
    )
    tflite_bytes = converter.convert()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[+] INT8 TFLite saved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, choices=sorted(CANDIDATES.keys()))
    ap.add_argument("--ckpt-name", default="sintel_best",
                    help="filename stem under checkpoints/ (default sintel_best).")
    ap.add_argument("--height", type=int, default=157)
    ap.add_argument("--width", type=int, default=203)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--skip-vela", action="store_true")
    ap.add_argument("--skip-publish", action="store_true")
    args = ap.parse_args()

    cand = CANDIDATES[args.model_name]
    ckpt_prefix = os.path.join(RETRAIN_DIR, cand["ckpt_dir"], f"{args.ckpt_name}.ckpt")
    if not os.path.exists(ckpt_prefix + ".index"):
        raise SystemExit(f"checkpoint index missing: {ckpt_prefix}.index")

    out_dir = args.output_dir or os.path.join(
        TOOL_DIR, "output", f"{args.model_name}_{args.height}x{args.width}"
    )
    os.makedirs(out_dir, exist_ok=True)
    base = f"edgeflownas_{args.model_name}_{args.height}x{args.width}"
    tflite_path = os.path.join(out_dir, f"{base}.tflite")

    print(f"\n[1/3] building inference graph for {args.model_name} "
          f"arch={cand['arch_code']} at {args.height}x{args.width}")
    sess, in_ph, out_t, g = build_inference_graph(
        args.model_name, cand["arch_code"], args.height, args.width
    )
    print(f"[*] output shape: {out_t.shape}")

    print(f"\n[2/3] restoring weights from {ckpt_prefix}")
    restore_forward_vars(sess, g, args.model_name, ckpt_prefix)

    print(f"\n[3/3] PTQ INT8 export -> {tflite_path}")
    export_tflite(sess, in_ph, out_t, CALIB_DIR, args.height, args.width, tflite_path)
    sess.close()

    metrics = {"model_name": args.model_name, "arch_code": cand["arch_code"],
               "ckpt": ckpt_prefix, "input": [args.height, args.width],
               "tflite": tflite_path}

    if not args.skip_vela:
        print("\n[vela] compiling...")
        sram_mb, time_ms = run_vela(
            tflite_path, mode="verbose", output_dir=out_dir, optimise="Size"
        )
        vela_path = os.path.join(out_dir, f"{base}_vela.tflite")
        metrics["vela_tflite"] = vela_path
        metrics["sram_mb"] = sram_mb
        metrics["est_inference_ms"] = time_ms
        print(f"[+] Vela SRAM: {sram_mb:.3f} MB, est. inference: {time_ms:.2f} ms")

        if not args.skip_publish:
            zoo_dir = os.path.join(
                REPO_ROOT, "model_zoo/optical_flow/edgeflownas_v3",
                args.model_name, f"{args.height}x{args.width}",
            )
            os.makedirs(zoo_dir, exist_ok=True)
            published = os.path.join(zoo_dir, f"{base}_vela.tflite")
            shutil.copy2(vela_path, published)
            metrics["published"] = published
            print(f"[+] published Vela tflite -> {published}")

    print("\n[done]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
