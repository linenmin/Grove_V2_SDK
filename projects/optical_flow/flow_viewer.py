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
  python flow_viewer.py                     # Auto-detect COM port
  python flow_viewer.py --port COM3         # Specify COM port
  python flow_viewer.py --port COM3 --baud 921600
  python flow_viewer.py --port COM3 --no-enhance   # Disable HSV V boost
  python flow_viewer.py --port COM3 --mag-coef 0.65 --mag-floor 1.5 --softness 0.1 --blur 0.5

Visualization tuning (matches host-side 3panel demo for v3_light):
  Board firmware bakes value = clip(mag * 0.05, 1.0) into the JPEG. With
  small motion this is too dim. We recover an estimated mag_px from the V
  channel, re-apply mag_coef (default 0.65 -> ~13x brighter), and a
  tanh "soft floor" to suppress quantization speckle below mag_floor.

Hotkeys:
  q  quit
  m  toggle mirror
  e  toggle enhancement (raw board JPEG vs demo-style)
  s  save screenshot

Requirements:
  pip install pyserial opencv-python numpy
"""

import argparse
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

# Board-side magnitude gain (in flow_render.cpp: mag_norm = mag * 0.05f).
# Note: board uses the *raw network* mag (in flow_divisor units), NOT pixel-units,
# so for v3* subnets the real motion in px is ~12.5x larger than what the board
# sees. We undo this with --flow-scale.
BOARD_MAG_COEF = 0.05

# Demo-style v3_light defaults (matches render_3panel_demo.py MODEL_CONFIGS["v3_light"]).
DEFAULT_FLOW_SCALE = 12.5  # v3_* trained with ft3d_flow_divisor=12.5; mainline=1.0
DEFAULT_MAG_COEF = 0.65    # new V gain on recovered mag (px-units)
DEFAULT_MAG_FLOOR = 1.5    # tanh soft floor center (px of motion)
DEFAULT_SOFTNESS = 0.3     # tanh soft floor width (looser than demo b/c INT8+JPEG noise)
DEFAULT_BLUR_SIGMA = 0.5   # gaussian blur on V before re-render (px)
DEFAULT_MEDIAN_K = 5       # median filter kernel on mag_px (0/1 disables; must be odd)


def enhance_flow_frame(bgr,
                       flow_scale=DEFAULT_FLOW_SCALE,
                       mag_coef=DEFAULT_MAG_COEF,
                       mag_floor=DEFAULT_MAG_FLOOR,
                       softness=DEFAULT_SOFTNESS,
                       blur_sigma=DEFAULT_BLUR_SIGMA,
                       median_k=DEFAULT_MEDIAN_K):
    """Re-grade the board's HSV-baked JPEG to match the demo video look.

    Board renders V = clip(network_mag * 0.05, 1.0) * 255 where network_mag is in
    flow_divisor units (not px). So for v3 (flow_scale=12.5) a 5 px motion comes
    out as V=5 -- almost black -- and the entire scene saturates only at ~250 px
    of motion. We invert that here.

    Pipeline:
      1. BGR -> HSV. Hue carries direction (board preserved it correctly).
      2. mag_px = V / (0.05 * 255) * flow_scale
      3. Optional gaussian blur on mag_px to damp INT8 + JPEG speckle.
      4. New V = clip(mag_px * mag_coef, 1.0) * sigmoid_softfloor(mag_px).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)   # uint8 input here
    v_u8 = hsv[:, :, 2]                          # uint8, 0..255

    # Median filter on V (uint8) first: kills isolated 8x8 JPEG MCU corruption
    # blocks from UART bit-flips at 921600 baud without smearing real motion.
    # Operate on uint8 V (not float mag_px) because cv2.medianBlur with
    # ksize >= 7 requires CV_8U.
    if median_k and median_k >= 3:
        k = int(median_k) | 1                    # force odd
        v_u8 = cv2.medianBlur(v_u8, k)

    mag_px = v_u8.astype(np.float32) * (flow_scale / (BOARD_MAG_COEF * 255.0))

    if blur_sigma > 0.0:
        mag_px = cv2.GaussianBlur(mag_px, ksize=(0, 0), sigmaX=blur_sigma)

    # cvtColor back at the end needs a float HSV array - lift to float32 now.
    hsv = hsv.astype(np.float32)

    val_norm = np.clip(mag_px * mag_coef, 0.0, 1.0)
    softness_safe = max(softness, 1e-3)
    soft_floor = 0.5 * (1.0 + np.tanh((mag_px - mag_floor) / softness_safe))
    v_new = np.clip(val_norm * soft_floor * 255.0, 0.0, 255.0)

    hsv[:, :, 1] = 255.0
    hsv[:, :, 2] = v_new
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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
                # Only print when we skipped a *lot* -- a real desync, not a
                # rejected false-sync that already burned a few bytes.
                if skipped > 4096:
                    print(f"[sync] Skipped {skipped} bytes to find header")
                return True
            else:
                skipped += 2
        else:
            skipped += 1


# Sanity bounds for header values. JPEG payload may contain spurious 0xAA 0x55
# sequences -- if the header that follows doesn't pass these checks we treat
# the match as a false-positive and resync.
MIN_W, MAX_W = 32, 1024
MIN_H, MAX_H = 32, 1024
MIN_JPEG = 256
LAST_GOOD = {"w": 0, "h": 0}


def header_is_plausible(jpeg_size, width, height):
    if jpeg_size < MIN_JPEG or jpeg_size > MAX_JPEG_SIZE:
        return False
    if width < MIN_W or width > MAX_W:
        return False
    if height < MIN_H or height > MAX_H:
        return False
    # Once we have seen a valid frame, lock the dimensions. The board sends a
    # fixed-resolution stream, so any later header with different w/h is a
    # false sync inside a JPEG payload.
    if LAST_GOOD["w"] and (width != LAST_GOOD["w"] or height != LAST_GOOD["h"]):
        return False
    return True


def read_frame(ser):
    """Read one complete frame: sync + header + JPEG payload.

    Robust against false 0xAA 0x55 matches inside JPEG payload: validates the
    header (size/width/height bounds + dimension lock) before committing the
    read. On rejection the function returns and the outer loop re-syncs.
    """
    if not sync_to_header(ser):
        return None, 0, 0

    # Read remaining 6 bytes of header (size, width, height as uint16 LE)
    hdr = ser.read(6)
    if len(hdr) < 6:
        return None, 0, 0

    jpeg_size, width, height = struct.unpack("<HHH", hdr)

    if not header_is_plausible(jpeg_size, width, height):
        # Don't print on every false sync -- they're frequent inside JPEG data.
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

    # JPEG must start with SOI 0xFFD8 -- last line of defense against false sync.
    if len(jpeg_data) >= 2 and (jpeg_data[0] != 0xFF or jpeg_data[1] != 0xD8):
        return None, 0, 0

    LAST_GOOD["w"], LAST_GOOD["h"] = width, height
    return bytes(jpeg_data), width, height


def parse_args():
    p = argparse.ArgumentParser(description="Grove Vision AI V2 Optical Flow Viewer")
    # Keep positional compat: `flow_viewer.py COM3 921600` still works.
    p.add_argument("pos_port", nargs="?", default=None,
                   help="(positional) COM port, e.g. COM3 / /dev/ttyACM0")
    p.add_argument("pos_baud", nargs="?", type=int, default=None,
                   help="(positional) baud rate")
    p.add_argument("--port", default=None, help="COM port (overrides positional)")
    p.add_argument("--baud", type=int, default=None, help="baud rate (overrides positional)")
    p.add_argument("--no-enhance", action="store_true",
                   help="Disable host-side V-channel re-grading; show raw board JPEG.")
    p.add_argument("--flow-scale", type=float, default=DEFAULT_FLOW_SCALE,
                   help=f"Train-time flow_divisor (12.5 for v3*, 1.0 for mainline). "
                        f"Default {DEFAULT_FLOW_SCALE}.")
    p.add_argument("--mag-coef", type=float, default=DEFAULT_MAG_COEF,
                   help=f"Magnitude gain for re-graded V (default {DEFAULT_MAG_COEF}).")
    p.add_argument("--mag-floor", type=float, default=DEFAULT_MAG_FLOOR,
                   help=f"Tanh soft-floor center in px (default {DEFAULT_MAG_FLOOR}).")
    p.add_argument("--softness", type=float, default=DEFAULT_SOFTNESS,
                   help=f"Tanh soft-floor width (default {DEFAULT_SOFTNESS}).")
    p.add_argument("--blur", type=float, default=DEFAULT_BLUR_SIGMA,
                   help=f"Gaussian sigma on recovered magnitude (default {DEFAULT_BLUR_SIGMA}).")
    p.add_argument("--median", type=int, default=DEFAULT_MEDIAN_K,
                   help=f"Median filter kernel for despeckling JPEG MCU corruption "
                        f"(odd, 0 disables; default {DEFAULT_MEDIAN_K}).")
    return p.parse_args()


def main():
    args = parse_args()
    port = args.port or args.pos_port
    baud = args.baud or args.pos_baud or DEFAULT_BAUD
    enhance_enabled = not args.no_enhance

    if port is None:
        port = find_com_port()
        if port is None:
            print("[error] No COM port found. Specify one: python flow_viewer.py COM5")
            sys.exit(1)

    print(f"[init] Opening {port} at {baud} baud...")
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.5
    # Try to prevent board reset on Windows by clearing DTR/RTS before open
    ser.dtr = False
    ser.rts = False
    ser.open()

    # Windows kernel ring buffer defaults to ~4KB which overflows in ~36ms at
    # 921600 baud -> byte loss inside JPEG payloads. Crank it to 1MB so a few
    # frames worth of data can sit unread without loss. No-op on Linux.
    try:
        ser.set_buffer_size(rx_size=1 << 20, tx_size=1 << 16)
        print("[init] RX buffer set to 1 MB")
    except Exception as exc:
        print(f"[init] set_buffer_size unavailable ({exc!r}) -- continuing")
    
    # Handshake loop: Repeatedly send 0xFC until we get the RAW_MODE ACK.
    # If the board resets, this waits for the app to finish booting.
    print("[init] Sending 0xFC to enter RAW binary mode (waiting for ACK)...")
    ack_received = False
    for _ in range(15):  # Try for up to ~7.5 seconds
        ser.write(bytes([0xFC]))
        ser.flush()
        time.sleep(0.1)
        
        if ser.in_waiting > 0:
            resp_bytes = ser.read(ser.in_waiting)
            resp_str = resp_bytes.decode('utf-8', errors='ignore')
            if 'RAW_MODE' in resp_str:
                print("\n[init] Device ACK: RAW_MODE activated!")
                ack_received = True
                break
            else:
                # Print device boot logs to console
                lines = resp_str.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        print(f"[dev] {line}")
        time.sleep(0.4)
        
    if not ack_received:
        print("\n[warn] Did not receive RAW_MODE ACK. Will try reading anyway...")

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
                print(f"[warn] Failed to decode JPEG ({len(jpeg_data)} bytes) - draining buffer")
                # A decode failure means the payload was corrupted mid-flight.
                # Anything still queued is likely the partial tail of the same
                # bad frame -- drop it to resync on the next clean header.
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                continue

            # Host-side re-grading to match the demo video look.
            if enhance_enabled:
                frame = enhance_flow_frame(frame,
                                           flow_scale=args.flow_scale,
                                           mag_coef=args.mag_coef,
                                           mag_floor=args.mag_floor,
                                           softness=args.softness,
                                           blur_sigma=args.blur,
                                           median_k=args.median)

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
            enh_tag = (f"Enh(c={args.mag_coef:.2f}, f={args.mag_floor:.1f})"
                       if enhance_enabled else "Enh:OFF")
            info = (f"FPS: {fps_display:.1f} | {width}x{height} | "
                    f"JPEG: {len(jpeg_data)}B | Mirror: {'ON' if mirror_enabled else 'OFF'} | {enh_tag}")
            cv2.putText(display, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mirror_enabled = not mirror_enabled
                print(f"[toggle] Mirror: {'ON' if mirror_enabled else 'OFF'}")
            elif key == ord('e'):
                enhance_enabled = not enhance_enabled
                print(f"[toggle] Enhancement: {'ON' if enhance_enabled else 'OFF'}")
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
