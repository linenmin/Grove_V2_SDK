#!/usr/bin/env python3
"""
从 pipeline UART 日志中提取 INVOKE 的 base64 图像并保存为 PNG。
用于 plan-008：让 Agent 能读取设备可视化输出，建立调试闭环。

用法:
  python3 scripts/extract_invoke_frames_from_log.py \
    --log logs/pipeline/pipeline_xxx.log \
    --output-dir logs/flow_frames/latest \
    --max-frames 10
"""
import argparse
import base64
import re
import sys
from pathlib import Path


def extract_invoke_images(log_path: str, output_dir: str, max_frames: int = 10) -> int:
    """从 log 中提取 INVOKE 的 image base64，解码并保存为 PNG。返回提取的帧数。"""
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    if not log_path.exists():
        print(f"[!] 日志不存在: {log_path}", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取日志（可能含非 UTF-8，忽略错误）
    try:
        content = log_path.read_bytes().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[!] 读取失败: {e}", file=sys.stderr)
        return 0

    # 匹配 "image": "<base64>"，base64 为 A-Za-z0-9+/= 及可能的换行
    # 注意：JSON 中 base64 可能不含换行，但为保险起见允许
    pattern = re.compile(r'"image"\s*:\s*"([A-Za-z0-9+/=\s]+)"')
    matches = pattern.findall(content)

    saved = 0
    for i, b64 in enumerate(matches):
        if saved >= max_frames:
            break
        b64_clean = b64.replace("\n", "").replace("\r", "").replace(" ", "")
        if not b64_clean:
            continue
        try:
            raw = base64.b64decode(b64_clean)
        except Exception as e:
            continue
        if len(raw) < 100:
            continue
        # 检查 JPEG 魔数
        if raw[:2] != b"\xff\xd8":
            continue
        out_path = output_dir / f"frame_{saved + 1:03d}.png"
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            img.save(out_path, "PNG")
            saved += 1
            print(f"[+] 保存: {out_path}")
        except Exception as e:
            # 若 PIL 不可用，直接保存为 JPEG
            out_path = output_dir / f"frame_{saved + 1:03d}.jpg"
            try:
                out_path.write_bytes(raw)
                saved += 1
                print(f"[+] 保存: {out_path} (JPEG)")
            except Exception as e2:
                continue

    return saved


def main():
    ap = argparse.ArgumentParser(description="从 pipeline UART 日志提取 INVOKE 图像")
    ap.add_argument("--log", required=True, help="pipeline 日志路径")
    ap.add_argument("--output-dir", default="logs/flow_frames/latest", help="输出目录")
    ap.add_argument("--max-frames", type=int, default=10, help="最多提取帧数")
    args = ap.parse_args()

    n = extract_invoke_images(args.log, args.output_dir, args.max_frames)
    print(f"\n[*] 共提取 {n} 帧")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
