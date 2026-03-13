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
DEFAULT_VARIANT = "mainline"
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

VARIANT_TO_MODULE = {
    "mainline": "network.MultiScaleResNet",
    "baseline": "network.MultiScaleResNet_bilinear",
    "eca": "network.MultiScaleResNet_bilinear_eca",
    "globalgate4x": "network.MultiScaleResNet_bilinear_globalgate4x",
    "globalgate4x_bneckeca": "network.MultiScaleResNet_bilinear_globalgate4x_bneckeca",
    "globalgate4x_bneckeca_skip8x": "network.MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x",
    "globalgate4x_bneckeca_skip8x4x": "network.MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x",
    "globalgate4x_bneckeca_skip8x4x2x": "network.MultiScaleResNet_bilinear_globalgate4x_bneckeca_skip8x4x2x",
}

if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

try:
    from misc.utils import AccumPreds
    from vela.vela_compiler import run_vela
except ImportError as exc:
    print(f"[!] 导入失败: {exc}")
    print(f"[DEBUG] sys.path: {sys.path}")
    sys.exit(1)


def is_bilinear_variant(variant: str) -> bool:
    return variant != "mainline"


def load_network_class(variant: str):
    module_name = VARIANT_TO_MODULE[variant]
    module = __import__(module_name, fromlist=["MultiScaleResNet"])
    return module.MultiScaleResNet


def model_basename(height: int, width: int, variant: str) -> str:
    if variant == "mainline":
        return f"optical_flow_{height}x{width}"
    if variant == "baseline":
        return f"optical_flow_bilinear_{height}x{width}"
    return f"optical_flow_bilinear_{variant}_{height}x{width}"


def default_output_dir(height: int, width: int, variant: str) -> str:
    if variant == "mainline":
        return DEFAULT_OUTPUT_DIR
    root_dir = "output_bilinear" if variant == "baseline" else f"output_bilinear_{variant}"
    return os.path.join(TOOL_DIR, root_dir, f"{height}x{width}")


def published_model_default(height: int, width: int, variant: str) -> str:
    if variant == "mainline":
        return DEFAULT_PUBLISHED_MODEL
    return (
        "/home/enmin/Seeed_Grove_Vision_AI_Module_V2/model_zoo/optical_flow/bilinear/"
        f"{variant}/{height}x{width}/{model_basename(height, width, variant)}_vela.tflite"
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


def export_tflite(
    checkpoint_prefix: str,
    calibration_dir: str,
    output_dir: str,
    height: int,
    width: int,
    variant: str,
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = model_basename(height, width, variant)
    network_class = load_network_class(variant)

    print(f"\n[1/2] 导出 TFLite 模型 ({height}x{width})...")
    print(f"[*] 变体: {variant}")
    if is_bilinear_variant(variant) and checkpoint_prefix == DEFAULT_CHECKPOINT_PREFIX:
        print(
            "[!] 当前使用的仍是主线默认 checkpoint。若导出 bilinear 变体，"
            "请确保它来自该结构本身的训练。"
        )

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        try:
            input_ph = tf.compat.v1.placeholder(
                tf.float32, shape=[1, height, width, 6], name="input_image"
            )
            model_obj = network_class(
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
            try:
                saver.restore(sess, checkpoint_prefix)
            except Exception as exc:
                raise RuntimeError(
                    f"checkpoint restore failed for variant '{variant}'. "
                    "Use a checkpoint trained for this exact architecture."
                ) from exc
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


def compile_with_vela(tflite_path: str, output_dir: str, height: int, width: int, variant: str):
    print("\n[2/2] 调用 Vela 编译...")
    sram_mb, time_ms = run_vela(
        tflite_path, mode="verbose", output_dir=output_dir, optimise="Size"
    )
    if sram_mb is None:
        raise RuntimeError("Vela 编译未产生结果")

    base_name = model_basename(height, width, variant)
    vela_path = os.path.join(output_dir, f"{base_name}_vela.tflite")

    print("\n" + "=" * 40)
    print("测试完成")
    print(f"模型变体: {variant}")
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
        default=os.environ.get("OPTICAL_FLOW_EXPORT_OUTPUT_DIR", ""),
        help="Local export output directory. Empty means derive from variant.",
    )
    parser.add_argument(
        "--published-model",
        default=os.environ.get("OPTICAL_FLOW_PUBLISHED_MODEL", ""),
        help="Where to copy the final Vela model after export. Empty means derive from variant.",
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
    parser.add_argument(
        "--variant",
        default=os.environ.get("OPTICAL_FLOW_EXPORT_VARIANT", DEFAULT_VARIANT),
        choices=sorted(VARIANT_TO_MODULE.keys()),
        help="Export target. 'mainline' keeps the current deployed skeleton; the others are bilinear experiment variants.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.height, args.width, args.variant)
    published_model = args.published_model or published_model_default(
        args.height, args.width, args.variant
    )
    tflite_path = export_tflite(
        args.checkpoint_prefix,
        args.calibration_dir,
        output_dir,
        args.height,
        args.width,
        args.variant,
    )
    vela_model_path = compile_with_vela(
        tflite_path, output_dir, args.height, args.width, args.variant
    )
    if not args.skip_publish:
        publish_model(vela_model_path, published_model)


if __name__ == "__main__":
    main()
