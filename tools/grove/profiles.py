"""Versioned profiles and immutable deployment bundles (stdlib only)."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_profile(path):
    profile = read_json(path)
    if profile.get('schema_version') != 1:
        raise ValueError('Unsupported profile schema_version')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', profile.get('id', '')):
        raise ValueError('Invalid profile id')
    app = profile.get('app', '')
    if not re.fullmatch(r'[A-Za-z0-9_]+', app) or not (ROOT / 'EPII_CM55M_APP_S/app/scenario_app' / app).is_dir():
        raise ValueError('Profile app must name an existing SDK scenario')
    if profile.get('mode') not in ('observe', 'deploy'):
        raise ValueError('Profile mode must be observe or deploy')
    device = profile.get('device', {})
    if not all(0 <= int(device[k], 0) <= 65535 for k in ('vid', 'pid')):
        raise ValueError('Invalid USB identifier')
    if type(device.get('baud')) is not int or not 1200 <= device['baud'] <= 3000000:
        raise ValueError('Invalid baud rate')
    check = profile.get('verify', {})
    if not check.get('required') or not all(isinstance(x, str) and x for x in check['required']):
        raise ValueError('verify.required must contain explicit success markers')
    for field in ('input_shape', 'output_shape'):
        if field in check and (len(check[field]) != 4 or any(type(n) is not int or n <= 0 for n in check[field])):
            raise ValueError(f'Invalid {field}')
    if type(check.get('min_frames', 0)) is not int or check.get('min_frames', 0) < 0:
        raise ValueError('Invalid min_frames')
    if 'frame_resolution' in check and (len(check['frame_resolution']) != 2 or any(type(n) is not int or n <= 0 for n in check['frame_resolution'])):
        raise ValueError('Invalid frame_resolution')
    for pattern in check.get('forbidden_regex', []):
        re.compile(pattern)
    return profile


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def layout_check(profile, models, firmware_size):
    layout = profile['flash']
    size = int(layout['size'], 0)
    start, end = (int(n, 0) for n in layout['firmware_region'])
    if not (0 <= start < end <= size) or firmware_size <= 0 or firmware_size > end - start:
        raise ValueError('Firmware exceeds declared region or layout invalid')
    intervals = [(start, end, 'firmware')]
    for model in models:
        address = int(model['address'], 0)
        length = model['bytes']
        # v1 deliberately supports only whole-model writes; offsets need a separate layout contract.
        if int(model.get('offset', '0x0'), 0) != 0:
            raise ValueError('Only zero model offsets are supported')
        if length <= 0 or address < 0 or address + length > size:
            raise ValueError('Model exceeds flash capacity')
        for lo, hi, label in intervals:
            if address < hi and lo < address + length:
                raise ValueError(f'Model overlaps {label}')
        intervals.append((address, address + length, model['path']))


def package(profile_path, firmware, output):
    profile = load_profile(profile_path)
    if profile['mode'] != 'deploy':
        raise ValueError('Observation-only profile cannot be packaged or flashed')
    output = Path(output)
    if output.exists():
        raise ValueError('Bundle output must be a new directory')
    firmware = Path(firmware).resolve(strict=True)
    models = []
    for index, item in enumerate(profile.get('models', [])):
        path = source_path(item['path']).resolve(strict=True)
        actual = digest(path)
        if actual != item.get('sha256'):
            raise ValueError(f'Model hash mismatch: {path}')
        models.append(dict(item, path=f'model-{index}.tflite', bytes=path.stat().st_size, source=str(path)))
    layout_check(profile, models, firmware.stat().st_size)
    output.mkdir(parents=True)
    shutil.copyfile(firmware, output / 'firmware.img')
    for model in models:
        shutil.copyfile(model['source'], output / model['path'])
    bundle = {'schema_version': 1, 'profile': profile,
              'firmware': {'path': 'firmware.img', 'bytes': firmware.stat().st_size, 'sha256': digest(output / 'firmware.img')},
              'models': models}
    write_json(output / 'bundle.json', bundle)
    validate_bundle(output / 'bundle.json')
    return bundle


def validate_bundle(manifest):
    manifest = Path(manifest).resolve(strict=True)
    bundle = read_json(manifest)
    if bundle.get('schema_version') != 1 or bundle['profile']['mode'] != 'deploy':
        raise ValueError('Not a deployment bundle')
    for item in [bundle['firmware'], *bundle['models']]:
        path = (manifest.parent / item['path']).resolve(strict=True)
        if not path.is_relative_to(manifest.parent):
            raise ValueError('Artifact escapes bundle directory')
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Bundle artifact changed: {path.name}')
    layout_check(bundle['profile'], bundle['models'], bundle['firmware']['bytes'])
    return bundle
