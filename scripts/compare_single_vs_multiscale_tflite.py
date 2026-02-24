#!/usr/bin/env python3
"""
Compare single-scale vs multi-scale optical-flow TFLite outputs on the same input frame pairs.

Input frames are expected to be grayscale PNGs (e.g., extracted INVOKE images in input-preview mode).
Each grayscale frame is replicated to 3 channels and paired as [prev_rgb, curr_rgb] -> [H,W,6].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


def _quant_params(tensor_detail: dict) -> Tuple[float, int]:
    q = tensor_detail.get("quantization_parameters", {})
    scales = q.get("scales", None)
    zero_points = q.get("zero_points", None)
    if scales is not None and len(scales) > 0:
        return float(scales[0]), int(zero_points[0])
    scale, zp = tensor_detail.get("quantization", (1.0, 0))
    if scale == 0:
        scale = 1.0
    return float(scale), int(zp)


class TFLiteFlowRunner:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        in_detail = self.interpreter.get_input_details()[0]
        out_detail = self.interpreter.get_output_details()[0]

        self.in_index = in_detail["index"]
        self.out_index = out_detail["index"]
        self.in_dtype = in_detail["dtype"]
        self.out_dtype = out_detail["dtype"]
        self.in_shape = tuple(int(v) for v in in_detail["shape"])
        self.out_shape = tuple(int(v) for v in out_detail["shape"])
        self.in_scale, self.in_zp = _quant_params(in_detail)
        self.out_scale, self.out_zp = _quant_params(out_detail)

    def run_pair(self, prev_u8: np.ndarray, curr_u8: np.ndarray) -> np.ndarray:
        pair = np.concatenate((prev_u8, curr_u8), axis=2).astype(np.float32)  # [H,W,6]
        pair = np.expand_dims(pair, axis=0)  # [1,H,W,6]

        if self.in_dtype == np.int8:
            q = np.round(pair / self.in_scale + self.in_zp)
            q = np.clip(q, -128, 127).astype(np.int8)
            input_tensor = q
        elif self.in_dtype == np.uint8:
            q = np.round(pair / self.in_scale + self.in_zp)
            q = np.clip(q, 0, 255).astype(np.uint8)
            input_tensor = q
        else:
            input_tensor = pair.astype(self.in_dtype)

        self.interpreter.set_tensor(self.in_index, input_tensor)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_index)

        if self.out_dtype == np.int8 or self.out_dtype == np.uint8:
            out_f = (out.astype(np.float32) - float(self.out_zp)) * float(self.out_scale)
        else:
            out_f = out.astype(np.float32)

        return out_f[0]  # [H,W,C]


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def gray_to_rgb3(gray: np.ndarray) -> np.ndarray:
    return np.stack((gray, gray, gray), axis=2)


def flow_mag(flow_hw2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(flow_hw2[:, :, 0]) + np.square(flow_hw2[:, :, 1]))


def render_mag_u8(mag: np.ndarray, vmax: float) -> np.ndarray:
    if vmax <= 1e-9:
        return np.zeros_like(mag, dtype=np.uint8)
    scaled = np.clip(mag / vmax, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def annotate(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, max(120, len(text) * 8), 16), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return out


def make_grid(images: Sequence[Image.Image], cols: int, bg: Tuple[int, int, int] = (8, 12, 20)) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), bg)
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        canvas.paste(img, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-scale vs multi-scale TFLite flow outputs.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory with frame_XXX.png input frames.")
    parser.add_argument("--single-model", type=Path, required=True, help="Path to single-scale .tflite model.")
    parser.add_argument("--multi-model", type=Path, required=True, help="Path to multi-scale .tflite model.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for visual artifacts.")
    parser.add_argument("--pairs", type=int, default=12, help="How many consecutive frame pairs to compare.")
    args = parser.parse_args()

    frame_paths = sorted(args.frames_dir.glob("frame_*.png"))
    if len(frame_paths) < 2:
        raise RuntimeError(f"need at least 2 frames in {args.frames_dir}")

    runner_single = TFLiteFlowRunner(args.single_model)
    runner_multi = TFLiteFlowRunner(args.multi_model)

    pair_count = min(args.pairs, len(frame_paths) - 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mags_single: List[np.ndarray] = []
    mags_multi: List[np.ndarray] = []
    prev_curr_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    pair_labels: List[str] = []

    for i in range(pair_count):
        prev = load_gray(frame_paths[i])
        curr = load_gray(frame_paths[i + 1])
        prev_rgb = gray_to_rgb3(prev)
        curr_rgb = gray_to_rgb3(curr)

        out_single = runner_single.run_pair(prev_rgb, curr_rgb)
        out_multi = runner_multi.run_pair(prev_rgb, curr_rgb)

        mags_single.append(flow_mag(out_single[:, :, :2]))
        mags_multi.append(flow_mag(out_multi[:, :, :2]))
        prev_curr_pairs.append((prev, curr))
        pair_labels.append(f"{frame_paths[i].stem}->{frame_paths[i+1].stem}")

    all_mags = np.concatenate(
        [np.asarray(m).reshape(-1) for m in (mags_single + mags_multi)],
        axis=0,
    )
    vmax = float(np.percentile(all_mags, 99.5))

    # Save per-pair side-by-side comparisons.
    for i in range(pair_count):
        prev, curr = prev_curr_pairs[i]
        s_u8 = render_mag_u8(mags_single[i], vmax)
        m_u8 = render_mag_u8(mags_multi[i], vmax)
        d_u8 = np.abs(s_u8.astype(np.int16) - m_u8.astype(np.int16)).astype(np.uint8)

        prev_img = annotate(Image.fromarray(prev, mode="L").convert("RGB"), "input prev")
        curr_img = annotate(Image.fromarray(curr, mode="L").convert("RGB"), "input curr")
        s_img = annotate(Image.fromarray(s_u8, mode="L").convert("RGB"), "single mag")
        m_img = annotate(Image.fromarray(m_u8, mode="L").convert("RGB"), "multi mag")
        d_img = annotate(Image.fromarray(d_u8, mode="L").convert("RGB"), "abs diff")

        # Normalize display size to input frame size for visual consistency.
        s_img = s_img.resize(prev_img.size, Image.Resampling.BILINEAR)
        m_img = m_img.resize(prev_img.size, Image.Resampling.BILINEAR)
        d_img = d_img.resize(prev_img.size, Image.Resampling.BILINEAR)

        tiles = [prev_img, curr_img, s_img, m_img, d_img]
        canvas = Image.new("RGB", (prev_img.width * len(tiles), prev_img.height), (8, 12, 20))
        for t, tile in enumerate(tiles):
            canvas.paste(tile, (t * prev_img.width, 0))

        pair_name = f"pair_{i+1:03d}_{i+2:03d}_{pair_labels[i]}.png"
        canvas.save(args.out_dir / pair_name)

    # Save compact contact sheets.
    single_tiles: List[Image.Image] = []
    multi_tiles: List[Image.Image] = []
    diff_tiles: List[Image.Image] = []
    for i in range(pair_count):
        s_u8 = render_mag_u8(mags_single[i], vmax)
        m_u8 = render_mag_u8(mags_multi[i], vmax)
        d_u8 = np.abs(s_u8.astype(np.int16) - m_u8.astype(np.int16)).astype(np.uint8)
        label = pair_labels[i]
        single_tiles.append(annotate(Image.fromarray(s_u8, mode="L").convert("RGB"), label))
        multi_tiles.append(annotate(Image.fromarray(m_u8, mode="L").convert("RGB"), label))
        diff_tiles.append(annotate(Image.fromarray(d_u8, mode="L").convert("RGB"), label))

    cols = 4
    single_sheet = make_grid(single_tiles, cols=cols)
    multi_sheet = make_grid(multi_tiles, cols=cols)
    diff_sheet = make_grid(diff_tiles, cols=cols)
    single_sheet.save(args.out_dir / "contact_sheet_single_mag.png")
    multi_sheet.save(args.out_dir / "contact_sheet_multi_mag.png")
    diff_sheet.save(args.out_dir / "contact_sheet_absdiff_single_vs_multi.png")

    combined = Image.new(
        "RGB",
        (single_sheet.width * 2, single_sheet.height),
        (8, 12, 20),
    )
    combined.paste(single_sheet, (0, 0))
    combined.paste(multi_sheet, (single_sheet.width, 0))
    combined = annotate(combined, "left: single-scale | right: multi-scale")
    combined.save(args.out_dir / "compare_single_vs_multi_contact_sheet.png")

    # Save numeric summary.
    mean_abs_diff = float(
        np.mean(
            [
                np.mean(np.abs(render_mag_u8(ms, vmax).astype(np.int16) - render_mag_u8(mm, vmax).astype(np.int16)))
                for ms, mm in zip(mags_single, mags_multi)
            ]
        )
    )
    summary = [
        f"frames_dir: {args.frames_dir}",
        f"single_model: {args.single_model}",
        f"multi_model: {args.multi_model}",
        f"pairs: {pair_count}",
        f"single_input_shape: {runner_single.in_shape}, output_shape: {runner_single.out_shape}",
        f"single_q: in(scale={runner_single.in_scale}, zp={runner_single.in_zp}), "
        f"out(scale={runner_single.out_scale}, zp={runner_single.out_zp})",
        f"multi_input_shape: {runner_multi.in_shape}, output_shape: {runner_multi.out_shape}",
        f"multi_q: in(scale={runner_multi.in_scale}, zp={runner_multi.in_zp}), "
        f"out(scale={runner_multi.out_scale}, zp={runner_multi.out_zp})",
        f"mag_render_vmax_p99.5: {vmax:.6f}",
        f"mean_absdiff_u8(single_vs_multi): {mean_abs_diff:.3f}",
    ]
    (args.out_dir / "README.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("[done] wrote artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()

