#!/usr/bin/env python3
"""
Grove Vision AI V2 - Optical Flow RAW Binary Viewer
====================================================
Receives raw binary JPEG frames from the WE2 MCU via UART,
decodes them with OpenCV, and displays with horizontal mirror.

Protocol:
  Host sends 0xFC to switch device to RAW mode.
  Device sends 8-byte header + raw JPEG bytes per frame:
    [0xAA][0x55][size_lo][size_hi][w_lo][w_hi][h_lo][h_hi][JPEG...]

Usage:
  python flow_viewer.py              # Auto-detect COM port
  python flow_viewer.py COM5         # Specify COM port
  python flow_viewer.py COM5 921600  # Specify baud rate

Requirements:
  pip install pyserial opencv-python numpy
"""

import sys
import time
import struct
import serial
import serial.tools.list_ports
import numpy as np
import cv2

# --- Configuration ---
DEFAULT_BAUD = 921600
SYNC_BYTE_0 = 0xAA
SYNC_BYTE_1 = 0x55
HEADER_SIZE = 8
MAX_JPEG_SIZE = 65535
DISPLAY_SCALE = 3  # Scale up the tiny 192x144 image for viewing
WINDOW_NAME = "Optical Flow - Grove Vision AI V2"


def find_com_port():
    """Auto-detect the Grove Vision AI V2 COM port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if "grove" in desc or "himax" in desc or "usb" in desc or "acm" in desc:
            print(f"[auto] Found port: {p.device} ({p.description})")
            return p.device
    # Fallback: return the first available port
    if ports:
        print(f"[auto] Using first available: {ports[0].device} ({ports[0].description})")
        return ports[0].device
    return None


def sync_to_header(ser):
    """Scan serial bytes until we find the 0xAA 0x55 sync marker."""
    skipped = 0
    while True:
        b = ser.read(1)
        if len(b) == 0:
            return False  # Timeout
        if b[0] == SYNC_BYTE_0:
            b2 = ser.read(1)
            if len(b2) == 0:
                return False
            if b2[0] == SYNC_BYTE_1:
                if skipped > 0:
                    print(f"[sync] Skipped {skipped} bytes to find header")
                return True
            else:
                skipped += 2
        else:
            skipped += 1


def read_frame(ser):
    """Read one complete frame: sync + header + JPEG payload."""
    if not sync_to_header(ser):
        return None, 0, 0

    # Read remaining 6 bytes of header (size, width, height as uint16 LE)
    hdr = ser.read(6)
    if len(hdr) < 6:
        return None, 0, 0

    jpeg_size, width, height = struct.unpack("<HHH", hdr)

    if jpeg_size == 0 or jpeg_size > MAX_JPEG_SIZE:
        print(f"[warn] Invalid JPEG size: {jpeg_size}")
        return None, 0, 0

    # Read JPEG payload
    jpeg_data = bytearray()
    remaining = jpeg_size
    while remaining > 0:
        chunk = ser.read(min(remaining, 4096))
        if len(chunk) == 0:
            print(f"[warn] Timeout reading JPEG payload ({jpeg_size - remaining}/{jpeg_size})")
            return None, 0, 0
        jpeg_data.extend(chunk)
        remaining -= len(chunk)

    return bytes(jpeg_data), width, height


def main():
    # Parse command line arguments
    port = None
    baud = DEFAULT_BAUD

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    if len(sys.argv) >= 3:
        baud = int(sys.argv[2])

    if port is None:
        port = find_com_port()
        if port is None:
            print("[error] No COM port found. Specify one: python flow_viewer.py COM5")
            sys.exit(1)

    print(f"[init] Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=2.0)
    time.sleep(0.5)

    # Send 0xFC to switch device to RAW binary mode
    print("[init] Sending 0xFC to enter RAW binary mode...")
    ser.write(bytes([0xFC]))
    ser.flush()
    time.sleep(0.5)

    # Drain any JSON ACK response
    if ser.in_waiting > 0:
        ack = ser.read(ser.in_waiting)
        print(f"[init] Device ACK: {ack.decode('utf-8', errors='replace').strip()}")

    print(f"[ready] Listening for binary JPEG frames...")
    print(f"[info] Press 'q' to quit, 'm' to toggle mirror, 's' to save screenshot")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    mirror_enabled = True
    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    try:
        while True:
            jpeg_data, width, height = read_frame(ser)

            if jpeg_data is None:
                continue

            # Decode JPEG
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[warn] Failed to decode JPEG ({len(jpeg_data)} bytes)")
                continue

            # Apply horizontal mirror (selfie view)
            if mirror_enabled:
                frame = cv2.flip(frame, 1)

            # Scale up for visibility
            display = cv2.resize(frame,
                                 (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
                                 interpolation=cv2.INTER_NEAREST)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Overlay info
            info = f"FPS: {fps_display:.1f} | {width}x{height} | JPEG: {len(jpeg_data)}B | Mirror: {'ON' if mirror_enabled else 'OFF'}"
            cv2.putText(display, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mirror_enabled = not mirror_enabled
                print(f"[toggle] Mirror: {'ON' if mirror_enabled else 'OFF'}")
            elif key == ord('s'):
                filename = f"flow_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"[save] Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n[exit] Interrupted by user")
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("[done] Viewer closed.")


if __name__ == "__main__":
    main()
