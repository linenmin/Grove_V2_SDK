# MIT License
#
# Copyright (c) 2023 Himax Technologies, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/env python
import serial
import time
import sys
import argparse

DEF_TIMEOUT = 60
DEF_BAUDRATE = 115200
DEF_CHUNK_SIZE = 512
DEF_WINDOW_SIZE = 8192

def uart_open(ser, com, baudrate, timeout):
    ser.port = com
    ser.timeout = timeout
    ser.baudrate = baudrate
    ser.bytesize = serial.EIGHTBITS
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = 0
    ser.rtscts = 0
    ser.parity = serial.PARITY_NONE
    ser.open()
    print("Open Serial Port", ser.port)

def dev_init():
    global ser
    ser = serial.Serial()

    try:
        uart_open(ser=ser, com=args.port, baudrate=args.baudrate, timeout=args.timeout)
        ser.flushInput()
        ser.flushOutput()
    except:
        print("Uart port open fail")
        sys.exit(-1)

def monitor_uart():
    start = time.time()
    bytes_read = 0
    text_window = ""
    keyword_hit = {kw: False for kw in args.keyword}
    log_fp = open(args.log_file, "w", encoding="utf-8") if args.log_file else None

    try:
        while True:
            if args.duration > 0 and (time.time() - start) >= args.duration:
                break

            try:
                raw = ser.read(args.chunk_size)
            except serial.SerialException as e:
                print(f"\n[SERIAL_ERROR] {e}")
                return 3
            if not raw:
                continue

            bytes_read += len(raw)
            chunk = raw.decode("utf-8", errors="ignore")
            if not chunk:
                continue

            print(chunk, end="", flush=True)
            if log_fp:
                log_fp.write(chunk)
                log_fp.flush()

            if args.keyword:
                text_window = (text_window + chunk)[-DEF_WINDOW_SIZE:]
                for kw in args.keyword:
                    if (not keyword_hit[kw]) and (kw in text_window):
                        keyword_hit[kw] = True
                        print(f"\n[KEYWORD_HIT] {kw}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if log_fp:
            log_fp.close()
        ser.close()

    elapsed = time.time() - start
    print(f"\n[SUMMARY] elapsed={elapsed:.2f}s bytes={bytes_read}")

    if args.keyword:
        missing = [kw for kw, hit in keyword_hit.items() if not hit]
        if missing:
            print(f"[SUMMARY] missing_keywords={missing}")
            return 2
        print("[SUMMARY] all_keywords_hit")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",
                        required=True, type=str,
                        help="Serial device port: COMn for windows; /dev/ttyUSBn,/dev/ttySn for unix; /dev/tty.usbserial-abcde for MacOS")
    parser.add_argument("--baudrate",
                        default=DEF_BAUDRATE, type=lambda x: int(x,0),
                        help="Serial device baudrate. Default is " + str(DEF_BAUDRATE))
    parser.add_argument("--timeout",
                        default=DEF_TIMEOUT, type=lambda x: int(x,0),
                        help="Serial device timeout. Default is " + str(DEF_TIMEOUT))
    parser.add_argument("--duration",
                        default=0, type=float,
                        help="Capture duration in seconds. 0 means run forever.")
    parser.add_argument("--chunk-size",
                        default=DEF_CHUNK_SIZE, type=int,
                        help="Raw serial read size per iteration. Default is " + str(DEF_CHUNK_SIZE))
    parser.add_argument("--keyword",
                        action='append', default=[],
                        help='Keyword to detect in decoded output. Repeat for multiple keywords.')
    parser.add_argument("--log-file",
                        default="", type=str,
                        help="Optional output log file path.")
    args = parser.parse_args()

    dev_init()
    print('Device init successfully')

    exit_code = monitor_uart()
    sys.exit(exit_code)

