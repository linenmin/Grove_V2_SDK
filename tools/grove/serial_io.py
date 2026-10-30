"""One serial owner, explicit duration, live evidence and graceful stop file."""
import time
from pathlib import Path


def ports():
    from serial.tools.list_ports import comports
    return [{'port': p.device, 'vid': p.vid, 'pid': p.pid, 'serial': p.serial_number,
             'description': p.description} for p in comports()]


def select_port(profile, port=None, serial_number=None):
    device = profile['device']
    matches = [p for p in ports() if p['vid'] == int(device['vid'], 0) and p['pid'] == int(device['pid'], 0)
               and (not port or p['port'] == port) and (not serial_number or p['serial'] == serial_number)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one device; use doctor then --port and/or --serial. Bridge ID is not board identity.')
    return matches[0]


def capture(device, baud, seconds, output, ready=print, serial_factory=None):
    if serial_factory is None:
        import serial
        serial_factory = serial.Serial
    port = serial_factory()
    port.port, port.baudrate, port.timeout = device['port'], baud, 0.2
    port.dtr, port.rts = False, False
    port.xonxoff, port.rtscts, port.dsrdtr = False, False, False
    output = Path(output)
    count = 0
    try:
        port.open()
        # Do not flush input: a boot may already be buffered.
        with (output / 'uart.bin').open('wb', buffering=0) as stream:
            ready({'state': 'listening', 'device': device, 'seconds': seconds, 'run': str(output),
                   'stop': str(output / 'STOP'), 'note': 'Press RESET now if boot evidence is needed. Opening COM can affect reset lines.'})
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and not (output / 'STOP').exists():
                chunk = port.read(8192)
                if chunk:
                    stream.write(chunk)
                    count += len(chunk)
    finally:
        port.close()
    return count
