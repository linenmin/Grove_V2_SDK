import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PREFIX = os.path.join(TOOL_DIR, "assets", "checkpoints", "best.ckpt")
DEFAULT_CALIBRATION_DIR = os.path.join(TOOL_DIR, "assets", "calibration")
DEFAULT_OUTPUT_DIR = os.path.join(TOOL_DIR, "output")
DEFAULT_PUBLISHED_MODEL = (
    "/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/157x203/"
    "optical_flow_157x203_vela.tflite"
)

HEIGHT = 157
WIDTH = 203
MODEL_CONFIG = {
    "InitNeurons": 32,
    "ExpansionFactor": 2.0,
    "NumSubBlocks": 2,
    "NumOut": 4,
    "NumBlocks": 1,
    "Padding": "same",
}

if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

try:
    from network.MultiScaleResNet import MultiScaleResNet
    from misc.utils import AccumPreds
    from vela.vela_compiler import run_vela
except ImportError as exc:
    print(f"[!] 导入失败: {exc}")
    print(f"[DEBUG] sys.path: {sys.path}")
    sys.exit(1)


def model_basename(height: int, width: int) -> str:
    return f"optical_flow_{height}x{width}"


def published_model_default(height: int, width: int) -> str:
    return (
        f"/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/"
        f"{height}x{width}/{model_basename(height, width)}_vela.tflite"
    )


def representative_dataset_gen(calibration_dir: str, height: int, width: int):
    if not os.path.isdir(calibration_dir):
        raise FileNotFoundError(f"calibration dir not found: {calibration_dir}")

    subdirs = [os.path.join(calibration_dir, d) for d in os.listdir(calibration_dir)]
    frame_pairs = []
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        frames = sorted(glob.glob(os.path.join(subdir, "*.png")))
        for idx in range(len(frames) - 1):
            frame_pairs.append((frames[idx], frames[idx + 1]))

    num_samples = min(100, len(frame_pairs))
    if num_samples == 0:
        raise RuntimeError(f"no calibration frame pairs found in: {calibration_dir}")

    for idx in range(num_samples):
        img1_path, img2_path = frame_pairs[idx]
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            continue
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        tensor = np.concatenate((img1, img2), axis=2).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0)
        yield [tensor]


def export_tflite(checkpoint_prefix: str,
                  calibration_dir: str,
                  output_dir: str,
                  height: int,
                  width: int):
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_basename(height, width)
    print(f"\n[1/2] 导出 TFLite 模型 ({height}x{width})...")

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        try:
            input_ph = tf.compat.v1.placeholder(
                tf.float32, shape=[1, height, width, 6], name="input_image"
            )
            model_obj = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=MODEL_CONFIG["InitNeurons"],
                ExpansionFactor=MODEL_CONFIG["ExpansionFactor"],
                NumSubBlocks=MODEL_CONFIG["NumSubBlocks"],
                NumOut=MODEL_CONFIG["NumOut"],
                NumBlocks=MODEL_CONFIG["NumBlocks"],
                Padding=MODEL_CONFIG["Padding"],
            )
            network_outputs = model_obj.Network()

            if isinstance(network_outputs, list) and len(network_outputs) > 1:
                print(f"[*] 网络输出 {len(network_outputs)} 个尺度:")
                for idx, tensor in enumerate(network_outputs):
                    print(f"    尺度 {idx}: {tensor.shape}")
                accum_out, _ = AccumPreds(network_outputs)
                final_output = accum_out[..., 0:2]
                print(f"[*] 使用多尺度累加 (AccumPreds): {final_output.shape}")
            else:
                final_output = (
                    network_outputs[-1]
                    if isinstance(network_outputs, list)
                    else network_outputs
                )
                print("[*] 使用单一输出")

            saver = tf.compat.v1.train.Saver()
            print(f"[*] 从 checkpoint 加载权重: {checkpoint_prefix}")
            saver.restore(sess, checkpoint_prefix)
            print("[+] 权重加载成功")

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
                calibration_dir, height, width
            )

            tflite_model = converter.convert()
            tflite_path = os.path.join(output_dir, f"{base_name}.tflite")
            with open(tflite_path, "wb") as handle:
                handle.write(tflite_model)
            print(f"[+] TFLite 已保存至: {tflite_path}")
            return tflite_path
        finally:
            sess.close()


def compile_with_vela(tflite_path: str, output_dir: str, height: int, width: int):
    print("\n[2/2] 调用 Vela 编译...")
    sram_mb, time_ms = run_vela(
        tflite_path, mode="verbose", output_dir=output_dir, optimise="Size"
    )
    if sram_mb is None:
        raise RuntimeError("Vela 编译未产生结果")

    base_name = model_basename(height, width)
    vela_path = os.path.join(output_dir, f"{base_name}_vela.tflite")

    print("\n" + "=" * 40)
    print("测试完成")
    print(f"模型分辨率: {height}x{width}")
    print(f"SRAM 占用: {sram_mb:.3f} MB")
    print(f"预估推理时间: {time_ms:.2f} ms")
    print(f"编译输出目录: {output_dir}")
    print("=" * 40)
    return vela_path


def publish_model(src_path: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"[+] 已复制发布模型到: {dst_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the current validated optical-flow model inside this repo."
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default=os.environ.get("OPTICAL_FLOW_CHECKPOINT_PREFIX", DEFAULT_CHECKPOINT_PREFIX),
        help="TensorFlow checkpoint prefix (default: external validated checkpoint).",
    )
    parser.add_argument(
        "--calibration-dir",
        default=os.environ.get("OPTICAL_FLOW_CALIBRATION_DIR", DEFAULT_CALIBRATION_DIR),
        help="Calibration image directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OPTICAL_FLOW_EXPORT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Local export output directory.",
    )
    parser.add_argument(
        "--published-model",
        default=os.environ.get("OPTICAL_FLOW_PUBLISHED_MODEL", ""),
        help="Where to copy the final Vela model after export. Empty means derive from resolution.",
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Do not copy the final Vela model to the published model path.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("OPTICAL_FLOW_EXPORT_HEIGHT", HEIGHT)),
        help="Model input height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("OPTICAL_FLOW_EXPORT_WIDTH", WIDTH)),
        help="Model input width.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    published_model = args.published_model or published_model_default(args.height, args.width)
    tflite_path = export_tflite(
        args.checkpoint_prefix, args.calibration_dir, args.output_dir, args.height, args.width
    )
    vela_model_path = compile_with_vela(tflite_path, args.output_dir, args.height, args.width)
    if not args.skip_publish:
        publish_model(vela_model_path, published_model)


if __name__ == "__main__":
    main()
