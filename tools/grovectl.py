#!/usr/bin/env python3
"""Grove V2 host entry point. Run --help; no operation implicitly flashes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from grove.profiles import ROOT, digest, load_profile, package, validate_bundle, write_json
from grove.evidence import analyze


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def new_run(root, action):
    directory = Path(root).resolve() / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + action + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    return directory


def build_command(profile, output, jobs):
    script = ROOT / 'scripts/build_firmware.sh'
    if os.name == 'nt':
        def linux_path(p):
            result = subprocess.run(['wsl', 'wslpath', '-a', Path(p).as_posix()], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        return ['wsl', 'bash', linux_path(script), '--app', profile['app'], '--out', linux_path(output), '--jobs', str(jobs)]
    return ['bash', str(script), '--app', profile['app'], '--out', str(output), '--jobs', str(jobs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('doctor', help='List host dependencies and USB devices; does not open serial')
    p = sub.add_parser('profile', help='Validate a profile and check model files')
    p.add_argument('profile')
    p = sub.add_parser('build', help='Build firmware in Linux/WSL; never flash')
    p.add_argument('--profile', required=True)
    p.add_argument('--runs', default=str(ROOT / 'runs'))
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('package', help='Create a hashed deployment bundle')
    p.add_argument('--profile', required=True)
    p.add_argument('--firmware', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('flash', help='Validate and show a flash plan; --execute sends it')
    p.add_argument('--bundle', required=True)
    p.add_argument('--port')
    p.add_argument('--serial')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--runs', default=str(ROOT / 'runs'))
    p = sub.add_parser('monitor', help='Receive UART only; save log, frames and verification')
    p.add_argument('--profile', required=True)
    p.add_argument('--port')
    p.add_argument('--serial')
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--runs', default=str(ROOT / 'runs'))
    p = sub.add_parser('verify', help='Verify an existing log against explicit model expectations')
    p.add_argument('--profile', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--out', required=True, help='New output directory; existing output is never reused')
    args = parser.parse_args()
    if args.action == 'doctor':
        result = {'python': sys.version.split()[0], 'root': str(ROOT),
                  'dependencies': {m: importlib.util.find_spec(m) is not None for m in ('serial', 'PIL', 'xmodem')},
                  'build_tools_in_host_path': {c: shutil.which(c) for c in ('make', 'arm-none-eabi-gcc', 'wsl')}}
        if result['dependencies']['serial']:
            from grove.serial_io import ports
            result['ports'] = ports()
        emit(result)
        return 0 if all(result['dependencies'].values()) else 2
    if args.action == 'package':
        emit(package(args.profile, args.firmware, args.out))
        return 0
    if args.action == 'flash':
        if not 1 <= args.timeout <= 3600:
            raise ValueError('timeout must be 1..3600 seconds')
        bundle = validate_bundle(args.bundle)
        profile = bundle['profile']
        from grove.serial_io import select_port
        device = select_port(profile, args.port, args.serial)
        manifest = Path(args.bundle).resolve()
        command = [sys.executable, str(ROOT / 'xmodem/xmodem_send.py'), '--port', device['port'],
                   '--baudrate', str(profile['device']['baud']), '--protocol', 'xmodem',
                   '--file', str(manifest.parent / bundle['firmware']['path']), '--overall-timeout', str(args.timeout)]
        for model in bundle['models']:
            command += ['--model', f'{manifest.parent / model["path"]} {model["address"]} 0x0']
        emit({'state': 'flash_plan', 'execute': args.execute, 'device': device, 'profile': profile['id'],
              'firmware': bundle['firmware'], 'models': bundle['models'], 'command': command})
        if not args.execute:
            return 0
        run = new_run(args.runs, 'flash')
        write_json(run / 'bundle.json', bundle)
        emit({'state': 'waiting_for_bootloader', 'run': str(run), 'action': 'Press RESET while flasher waits; do not open another serial client.'})
        try:
            with (run / 'flash.log').open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout + 5)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        summary = {'transfer_exit_code': code, 'runtime_verified': False, 'run': str(run),
                   'next': 'Run monitor and verify; transfer completion is not deployment acceptance.'}
        write_json(run / 'summary.json', summary)
        emit(summary)
        return code
    profile = load_profile(args.profile)
    if args.action == 'profile':
        from grove.profiles import source_path
        models = []
        for item in profile.get('models', []):
            file = source_path(item['path'])
            matches = file.is_file() and digest(file) == item.get('sha256')
            models.append({'path': str(file), 'hash_matches': matches})
        emit({'profile': profile, 'models': models})
        return 0 if all(m['hash_matches'] for m in models) else 2
    if args.action == 'build':
        if not 1 <= args.jobs <= 64:
            raise ValueError('jobs must be 1..64')
        if profile['mode'] != 'deploy':
            raise ValueError('Observation profile does not define a reproducible build')
        run = new_run(args.runs, 'build')
        command = build_command(profile, run, args.jobs)
        emit({'command': command, 'run': str(run), 'dry_run': args.dry_run})
        if args.dry_run:
            return 0
        with (run / 'build.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        summary = {'exit_code': result.returncode, 'run': str(run), 'profile': profile['id']}
        if result.returncode == 0:
            summary['firmware_sha256'] = digest(run / 'firmware.img')
        write_json(run / 'summary.json', summary)
        emit(summary)
        return result.returncode
    if args.action == 'monitor':
        if not 0 < args.seconds <= 3600:
            raise ValueError('seconds must be >0 and <=3600')
        from grove.serial_io import select_port, capture
        device = select_port(profile, args.port, args.serial)
        run = new_run(args.runs, 'monitor')
        write_json(run / 'session.json', {'profile': profile, 'device': device, 'started_at': datetime.now(timezone.utc).isoformat()})
        capture(device, profile['device']['baud'], args.seconds, run, emit)
        log = run / 'uart.bin'
    else:
        log = Path(args.log).resolve(strict=True)
        run = Path(args.out).resolve()
        run.mkdir(parents=True, exist_ok=False)
    summary = analyze(log, profile, run / 'frames')
    summary.update(log=str(log), log_sha256=digest(log), run=str(run))
    write_json(run / 'summary.json', summary)
    emit(summary)
    return 0 if summary['passed'] else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit({'error': 'interrupted', 'raw_log_preserved': True})
        sys.exit(130)
    except Exception as exc:
        emit({'error': str(exc), 'type': type(exc).__name__})
        sys.exit(2)
