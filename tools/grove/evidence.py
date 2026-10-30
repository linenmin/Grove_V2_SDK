"""Read logs without confusing a camera fallback, old boot, or corrupt frame with success."""
import base64
import io
import json
import re
import statistics
from collections import deque
from pathlib import Path


def analyze(log, profile, frames_dir=None, max_frames=3):
    raw = Path(log).read_bytes()
    text = raw.decode('utf-8', errors='replace')
    # Logs may contain several resets. Only the most recent boot can pass verification.
    segment = text[text.rfind('1st BL Modem Build'):] if '1st BL Modem Build' in text else text
    check = profile['verify']
    missing = [s for s in check['required'] if s not in segment]
    errors = []
    for pattern in check.get('forbidden_regex', [r'(?i)\bfail(?:ed|ure)?\b']):
        errors.extend(m.group(0) for m in re.finditer(pattern, segment))
    io_match = re.search(r'model io: in\(h=(\d+),w=(\d+),c=(\d+)\) out\(h=(\d+),w=(\d+),c=(\d+)\)', segment)
    shapes = {}
    if io_match:
        values = list(map(int, io_match.groups()))
        shapes = {'input_shape': [1, *values[:3]], 'output_shape': [1, *values[3:]]}
    for key in ('input_shape', 'output_shape'):
        if key in check and shapes.get(key) != check[key]:
            missing.append(f'{key}: expected {check[key]}, observed {shapes.get(key)}')
    frames, invalid_frames, resolutions = deque(maxlen=max_frames), 0, []
    valid_frames = 0
    for line in segment.splitlines():
        if not line.lstrip().startswith('{'):
            continue
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # Incomplete final serial line is not a valid frame.
        if item.get('name') != 'INVOKE':
            continue
        if item.get('code') != 0:
            errors.append(f'INVOKE code={item.get("code")}')
            continue
        data = item.get('data', {})
        if not data.get('image'):
            continue  # Some applications return numerical results, with min_frames=0.
        try:
            from PIL import Image
            content = base64.b64decode(data['image'], validate=True)
            with Image.open(io.BytesIO(content)) as im:
                im.load()
                size = list(im.size)
                format_name = im.format
            if format_name != 'JPEG' or size != data.get('resolution'):
                raise ValueError('JPEG dimensions differ from message')
            resolutions.append(size)
            if 'frame_resolution' in check and size != check['frame_resolution']:
                errors.append(f'Unexpected frame resolution {size}')
                continue
            frames.append(content)
            valid_frames += 1
        except (ValueError, OSError, KeyError):
            invalid_frames += 1
    if invalid_frames:
        errors.append(f'{invalid_frames} invalid image messages')
    if valid_frames < check.get('min_frames', 0):
        missing.append(f'frames: {valid_frames} < {check.get("min_frames", 0)}')
    saved = []
    if frames_dir and frames:
        directory = Path(frames_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for index, frame in enumerate(frames):
            target = directory / f'frame-{index:03d}.jpg'
            target.write_bytes(frame)
            saved.append(str(target.resolve()))
    timings = [tuple(map(float, m)) for m in re.findall(r'infer=([\d.]+)ms total=([\d.]+)ms', segment)]
    summary = {'profile': profile['id'], 'bytes': len(raw), 'latest_boot_only': True,
               'passed': bool(raw) and not missing and not errors,
               'missing': missing, 'errors': sorted(set(errors)), **shapes,
               'valid_frames': valid_frames, 'invalid_frames': invalid_frames,
               'frame_resolutions': sorted(set(map(tuple, resolutions))), 'saved_frames': saved,
               'identity_verified': False,
               'limits': 'Structural/log validation only; does not prove model identity, flow accuracy, or end-to-end preview FPS.'}
    if timings:
        summary['timing'] = {'samples': len(timings), 'scope': 'firmware-reported',
                             'infer_ms_median': statistics.median(t[0] for t in timings),
                             'total_ms_median': statistics.median(t[1] for t in timings)}
    return summary
