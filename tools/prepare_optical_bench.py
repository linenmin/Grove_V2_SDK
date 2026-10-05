"""Pin camera preprocessing and flash CRC for a compiled optical-flow model.

Use an accepted export report for explicit conventions. Does not flash hardware.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import zlib
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('--name', choices=['edge', 'MCUFlowNet-S', 'MCUFlowNet-L'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--export-report', type=Path,
    help='Accepted export.json; bypass historical model-name convention defaults')
p.add_argument('--fixtures', type=Path, help='Fixed-input fixtures.bin/json directory')
diagnostics = p.add_mutually_exclusive_group()
diagnostics.add_argument('--diagnostic-four-channel', action='store_true',
    help='Only for accepted final-slice removal diagnostics; dump four channels and skip timing')
diagnostics.add_argument('--diagnostic-fixed-output', action='store_true',
    help='Dump all three fixed u/v outputs and skip timing; requires accepted export and fixtures')
diagnostics.add_argument('--diagnostic-activations', action='store_true',
    help='Only for accepted two-output activation prefixes; dump features and skip camera inference')
a = p.parse_args()
diagnostic = a.diagnostic_four_channel or a.diagnostic_fixed_output or a.diagnostic_activations
if a.diagnostic_activations:
    assert a.export_report and a.fixtures
root = Path(__file__).resolve().parents[1]
data = a.model.read_bytes()
# Keep the historical slot when possible; larger models need an earlier start.
flash_address = 0xB7B000 if len(data) <= 0x1000000 - 0xB7B000 else 0xA00000
assert flash_address + len(data) <= 0x1000000
common = root / 'EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/common_config.h'
common_text, count = re.subn(r'(#define OPTICAL_FLOW_MODEL_FLASH_ADDR )0x[0-9A-Fa-f]+',
    lambda m: m[1] + f'0x{0x3A000000 + flash_address:08X}', common.read_text())
assert count == 1
if a.export_report:
    from ethosu.vela.tflite.Model import Model
    exported = json.loads(a.export_report.read_text())
    assert exported['status'] == 'passed'
    assert exported['model'] == {'edge':'edge', 'MCUFlowNet-S':'S', 'MCUFlowNet-L':'L'}[a.name]
    assert exported['output_convention'] == (
        'Internal quantized activations; no optical-flow displacement units' if a.diagnostic_activations else
        'Input-image pixel displacement u,v; no 12.5 multiplier; no clipping')
    subgraph = Model.GetRootAsModel(data,0).Subgraphs(0)
    assert subgraph.InputsLength() == 1
    assert subgraph.OutputsLength() == (2 if a.diagnostic_activations else 1)
    def io(index):
        tensor = subgraph.Tensors(index); quant = tensor.Quantization()
        assert tensor.Type() == 9 and quant.ScaleLength() == quant.ZeroPointLength() == 1
        return dict(shape=tensor.ShapeAsNumpy().tolist(),
                    quantization=[float(quant.Scale(0)),int(quant.ZeroPoint(0))])
    ii = io(subgraph.Inputs(0))
    all_outputs = [io(subgraph.Outputs(i)) for i in range(subgraph.OutputsLength())]
    oo = all_outputs[0]
    expected_outputs = (exported['exports']['int8']['outputs'] if a.diagnostic_activations else
                        [exported['exports']['int8']['output']])
    for actual, expected in [(ii,exported['exports']['int8']['input'])]+list(zip(all_outputs,expected_outputs)):
        assert actual['shape'] == expected['shape']
        assert actual['quantization'] == [expected['scale'],expected['zero_point']]
    normalized = not exported.get('edge_public',False)
    multiplier = 1.0
else:
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(a.model))
    ii, oo = it.get_input_details()[0],it.get_output_details()[0]
    assert ii['dtype'] == oo['dtype'] == np.int8
    ii = dict(shape=ii['shape'].tolist(),quantization=list(ii['quantization']))
    oo = dict(shape=oo['shape'].tolist(),quantization=list(oo['quantization']))
    all_outputs = [oo]
    normalized = a.name != 'edge'
    multiplier = 1.0 if a.name == 'edge' else 12.5
shape, output = ii['shape'],oo['shape']
assert shape[0] == output[0] == 1 and shape[-1] == 6
if a.diagnostic_activations:
    assert exported['diagnostic_only'] and exported['constant_buffers_byte_exact']
    assert exported['reference_kernel'] == 'BUILTIN_REF'
    assert exported['scope'] == 'Internal activation prefix diagnosis only; no benchmark EPE, calibration or weights'
    assert all(len(t['shape']) == 4 and t['shape'][0] == 1 for t in all_outputs)
    assert all(re.fullmatch('[a-z0-9_-]+', t['label']) for t in expected_outputs)
elif a.diagnostic_four_channel:
    assert a.export_report and a.fixtures and output[-1] == 4
    assert exported['scope'] == 'Final slice removal diagnostic only; no new benchmark EPE, calibration or weights'
    assert exported['constant_buffers_byte_exact']
else:
    assert output[-1] == 2
    if a.diagnostic_fixed_output:
        assert a.export_report and a.fixtures
scale, zero = ii['quantization']
assert scale > 0
x = np.arange(256, dtype=np.float32)
if normalized:
    x = x / 255 * 2 - 1
lut = np.clip(np.rint(x / scale + zero), -128, 127).astype(np.int8)
# Exactly the preprocessing used by export_grove/evaluate_grove, all byte values.
assert np.all(np.diff(lut.astype(np.int16)) >= 0)
crc = zlib.crc32(data)
name = f'{a.name}-{shape[2]}x{shape[1]}'
if a.diagnostic_four_channel:
    name += '-full4-diagnostic'
elif a.diagnostic_fixed_output:
    name += '-fixed-output-diagnostic'
elif a.diagnostic_activations:
    name += '-activation-diagnostic'
fixture_header = '#define FLOW_BENCH_FIXTURE_COUNT 0U\n'
fixture_models = []; fixture_required = []
if a.fixtures:
    assert a.export_report, 'Fixtures require explicit export conventions'
    f = json.loads((a.fixtures/'fixtures.json').read_text())
    blob_path = a.fixtures/'fixtures.bin'; blob = blob_path.read_bytes()
    assert f['status'] == 'passed' and len(f['pairs']) == 3
    assert f['source_sha256'] == exported['exports']['int8']['sha256']
    assert f['blob_sha256'] == hashlib.sha256(blob).hexdigest()
    assert f['blob_crc32'] == f'{zlib.crc32(blob):08x}' and f['blob_bytes'] == len(blob)
    assert f['input_bytes'] == int(np.prod(shape))
    assert f['output_bytes'] == sum(int(np.prod(t['shape'])) for t in all_outputs)
    assert f['input_scale'] == scale and f['input_zero_point'] == zero
    if a.diagnostic_activations:
        assert f['reference_kernel'] == 'BUILTIN_REF' and f['diagnostic_only']
        assert f['performance']['enabled'] is False
        assert f['outputs'] == expected_outputs
    else:
        assert [f['output_scale'],f['output_zero_point']] == oo['quantization']
        assert f['output_multiplier'] == multiplier
    assert f['images'] == ('normalized' if normalized else 'raw')
    assert f['acceptance'] == dict(max_abs_quantized_code=2,mean_abs_quantized_code=0.05)
    address = 0x200000
    assert 0x171000 <= address and address+len(blob) <= flash_address
    assert len(blob) == 3*(f['input_bytes']+f['output_bytes'])
    fixture_header = (f'#define FLOW_BENCH_FIXTURE_COUNT 3U\n'
        f'#define FLOW_BENCH_FIXTURE_ADDR 0x{0x3A000000+address:08X}U\n'
        f'#define FLOW_BENCH_FIXTURE_BYTES {len(blob)}U\n'
        f'#define FLOW_BENCH_FIXTURE_CRC 0x{zlib.crc32(blob):08x}U\n'
        f'#define FLOW_BENCH_INPUT_BYTES {f["input_bytes"]}U\n'
        f'#define FLOW_BENCH_OUTPUT_BYTES {f["output_bytes"]}U\n')
    fixture_models = [dict(path=str(blob_path.resolve()),sha256=f['blob_sha256'],address=hex(address),offset='0x0')]
    fixture_required = ['FLOW_FIXED_PASS count=3',
        'FLOW_FIXED_DIAGNOSTIC_DONE count=3' if diagnostic else
        'FLOW_FIXED_PERF_DONE warmup=5 measured=20']
    if a.diagnostic_activations:
        sizes = [int(np.prod(t['shape'])) for t in all_outputs]
        fixture_header += ('#define FLOW_BENCH_OUTPUT_COUNT 2U\n'
            'static const uint32_t flow_bench_output_bytes[2] = {' + ','.join(map(str,sizes)) + '};\n'
            'static const int32_t flow_bench_output_shapes[2][4] = {' +
            ','.join('{'+','.join(map(str,t['shape']))+'}' for t in all_outputs) + '};\n'
            'static const char *const flow_bench_output_names[2] = {' +
            ','.join(json.dumps(t['label']) for t in expected_outputs) + '};\n')
        fixture_required += [f'FLOW_PROBE_IO tensor={i} label={t["label"]} shape='+
            ','.join(map(str,t['shape']))+f' bytes={sizes[i]}' for i,t in enumerate(expected_outputs)]
a.out.mkdir(parents=True, exist_ok=False)
header = root / 'EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/bench_model.h'
header.write_text(
    '// Generated by tools/prepare_optical_bench.py; also archived in the build.\n'
    f'#define FLOW_BENCH_NAME "{name}"\n#define FLOW_BENCH_BYTES {len(data)}U\n'
    f'#define FLOW_BENCH_CRC 0x{crc:08x}U\n'
    f'#define FLOW_BENCH_OUTPUT_MULTIPLIER {multiplier}f\n' + fixture_header +
    ('#define FLOW_BENCH_DUMP_OUTPUT 1\n#define FLOW_BENCH_DIAGNOSTIC_ONLY 1\n' if diagnostic else '') +
    'static const int8_t flow_bench_input_lut[256] = {' + ','.join(map(str, lut.tolist())) + '};\n',
    encoding='utf-8')
profile = dict(schema_version=1, id=name.lower(), mode='deploy', app='optical_cam_oflow', camera='OV5647',
    device=dict(vid='0x1A86', pid='0x55D3', baud=921600),
    models=[dict(path=str(a.model.resolve()), sha256=hashlib.sha256(data).hexdigest(), address=hex(flash_address), offset='0x0')]+fixture_models,
    flash=dict(size='0x1000000', firmware_region=['0x0', '0x171000']),
    verify=dict(required=[f'FLOW_MODEL {name} crc={crc:08x} PASS', 'camera input init done',
        'Ethos-U55 device initialised', 'initial done']+fixture_required, forbidden_regex=['(?i)\\bfail(?:ed|ure)?\\b', '(?i)hardfault'],
        input_shape=shape, output_shape=output, min_frames=0 if diagnostic else 3, frame_resolution=[output[2], output[1]]),
    notes=('Four-channel final-slice diagnosis only; CPU u/v unchanged; timing disabled; not a benchmark model.'
           if a.diagnostic_four_channel else
           'Fixed-output numerical diagnosis only; model pinned to accepted export and fixtures to supplied hashes; timing disabled; not a benchmark result.'
           if a.diagnostic_fixed_output else
           'Internal activation numerical diagnosis only; two original tensors; reference CPU kernel; no EPE, timing or camera inference.'
           if a.diagnostic_activations else
           'Camera runtime check; LUT matches host INT8 preprocessing; CRC checks flash content, not cryptographic attestation. JPEG is not EPE evidence.'))
common.write_text(common_text, encoding='utf-8')
(a.out / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8')
(a.out / 'bench_model.h').write_bytes(header.read_bytes())
(a.out / 'io.json').write_text(json.dumps(dict(input_quantization=ii['quantization'],
    output_quantization=oo['quantization'], input_lut=lut.tolist(), crc32=f'{crc:08x}',
    output_quantizations=[t['quantization'] for t in all_outputs],
    output_multiplier=multiplier,input_convention='normalized' if normalized else 'raw',diagnostic_only=diagnostic,
    export_report_sha256=hashlib.sha256(a.export_report.read_bytes()).hexdigest() if a.export_report else None), indent=2) + '\n', encoding='utf-8')
print(a.out / 'profile.json')
