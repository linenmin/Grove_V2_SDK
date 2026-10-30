"""Run xmodem/xmodem_send.py unchanged, but reset the board from the host.

grovectl flash --auto-reset calls this instead of xmodem_send.py with the same arguments.
One RTS+DTR pulse is sent 1 s after the flasher opens the port, while it waits for the
'1st BL Modem Build' banner. The port is opened with both lines low first: a default
pyserial open raises them, which leaves only a falling edge and the board keeps running
the old firmware (seen 2026-10-15 in CONTROL-PERCEPTION-01).
"""
import runpy
import sys
import threading
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
XMODEM = TOOLS.parent / 'xmodem/xmodem_send.py'
PULSE_DELAY = 1.0


def install(serial_class, delay=PULSE_DELAY):
    """Patch serial_class.open so the first open of the process is followed by one pulse."""
    sys.path.insert(0, str(TOOLS))
    from grove.serial_io import pulse_reset
    original = serial_class.open
    done = []

    def pulse(port):
        time.sleep(delay)
        pulse_reset(port)
        print('[autoreset] RTS+DTR pulse sent', flush=True)

    def patched_open(self, *args, **kwargs):
        first = not done
        if first:
            done.append(True)
            self.dtr, self.rts = False, False
        original(self, *args, **kwargs)
        if first:
            threading.Thread(target=pulse, args=(self,), daemon=True).start()

    serial_class.open = patched_open
    return done


def main():
    import serial
    install(serial.Serial)
    sys.path.insert(0, str(XMODEM.parent))
    sys.argv = [str(XMODEM)] + sys.argv[1:]
    runpy.run_path(str(XMODEM), run_name='__main__')


if __name__ == '__main__':
    main()
