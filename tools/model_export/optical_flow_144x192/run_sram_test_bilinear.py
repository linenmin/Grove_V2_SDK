import argparse
import os
import sys

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from misc.utils import AccumPreds
    from vela.vela_compiler import run_vela
except ImportError as exc:
    print(f"[!] 导入失败: {exc}")
    sys.exit(1)


MODEL_CONFIG = {
    "InitNeurons": 32,
    "ExpansionFactor": 2.0,
    "NumSubBlocks": 2,
    "NumOut": 4,
    "NumBlocks": 1,
    "Padding": "same",
}

VARIANT_TO_MODULE = {
    "baseline": "network.MultiScaleResNet_bilinear",
    "addskip": "network.MultiScaleResNet_bilinear_addskip",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the bilinear-upsample optical-flow skeleton with random weights "
            "and compile it with Vela."
        )
    )
    parser.add_argument("--height", type=int, default=156, help="Model input height.")
    parser.add_argument("--width", type=int, default=208, help="Model input width.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to output_bilinear/<HxW>/ under this tool.",
    )
    parser.add_argument(
        "--optimise",
        default="Size",
        choices=["Performance", "Size"],
        help="Vela optimisation strategy.",
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=sorted(VARIANT_TO_MODULE.keys()),
        help="Network variant to export.",
    )
    return parser.parse_args()


def load_network_class(variant: str):
    module_name = VARIANT_TO_MODULE[variant]
    module = __import__(module_name, fromlist=["MultiScaleResNet"])
    return module.MultiScaleResNet


def model_basename(height: int, width: int, variant: str) -> str:
    suffix = "" if variant == "baseline" else f"_{variant}"
    return f"optical_flow_bilinear{suffix}_{height}x{width}"


def default_output_dir(height: int, width: int, variant: str) -> str:
    root_dir = "output_bilinear" if variant == "baseline" else f"output_bilinear_{variant}"
    return os.path.join(CURRENT_DIR, root_dir, f"{height}x{width}")


def representative_dataset_gen(height: int, width: int):
    for _ in range(5):
        yield [np.random.uniform(0.0, 1.0, size=[1, height, width, 6]).astype(np.float32)]


def export_tflite(height: int, width: int, output_dir: str, variant: str):
    base_name = model_basename(height, width, variant)
    tflite_path = os.path.join(output_dir, f"{base_name}.tflite")
    network_class = load_network_class(variant)

    print(
        f"\n[1/2] 正在导出 Bilinear TFLite 模型 ({height}x{width})..."
        f" 变体: {variant}"
    )

    new_graph = tf.Graph()
    with new_graph.as_default():
        sess = tf.compat.v1.Session(graph=new_graph)
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
                accum_out, _ = AccumPreds(network_outputs)
                final_output = accum_out[..., 0:2]
                print(f"[*] 使用多尺度累加输出，共 {len(network_outputs)} 个尺度")
            else:
                final_output = network_outputs[-1] if isinstance(network_outputs, list) else network_outputs
                print("[*] 使用单一输出")

            print(f"[*] 最终输出 shape: {final_output.shape}")
            print("[*] 正在执行变量随机初始化...")
            sess.run(tf.compat.v1.global_variables_initializer())

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
            converter.representative_dataset = lambda: representative_dataset_gen(height, width)

            tflite_model = converter.convert()
            with open(tflite_path, "wb") as file_obj:
                file_obj.write(tflite_model)
            print(f"[+] TFLite 已保存至: {tflite_path}")
            return tflite_path
        finally:
            sess.close()


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.height, args.width, args.variant)
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = export_tflite(args.height, args.width, output_dir, args.variant)

    print("\n[2/2] 正在调用 Vela 进行编译 (Resize+Conv 方案)...")
    sram_mb, time_ms = run_vela(
        tflite_path, mode="verbose", output_dir=output_dir, optimise=args.optimise
    )
    if sram_mb is None or time_ms is None:
        return

    fps = 1000.0 / time_ms if time_ms > 0 else 0.0
    print("\n" + "=" * 40)
    print("Bilinear 测试完成")
    print(f"模型变体: {args.variant}")
    print(f"模型分辨率: {args.height}x{args.width}")
    print(f"SRAM 占用: {sram_mb:.3f} MB")
    print(f"预估推理时间: {time_ms:.2f} ms")
    print(f"预估 FPS: {fps:.2f}")
    print(f"结果目录: {output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()
