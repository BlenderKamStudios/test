#!/usr/bin/env python3
"""Prototype presence detector: CSI amplitude variance over a sliding window."""
import re, sys, time
from collections import deque

import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
BAUD = 921600          # matches firmware/csi_rx/csi_rx.ino
WINDOW = 100           # packets per window (~1 s at 100 pkt/s)
CALIB_SECONDS = 15     # room must be EMPTY and still during this time
K = 3.0                # threshold = mean + K * std of the empty-room baseline

csi_re = re.compile(r'"?\[([-\d,\s]+)\]"?\s*$')

def parse_amplitude(line: str):
    m = csi_re.search(line)
    if not m:
        return None
    vals = np.array([int(v) for v in m.group(1).split(",") if v.strip()], dtype=np.int8)
    if len(vals) < 2 or len(vals) % 2:
        return None
    imag, real = vals[0::2].astype(float), vals[1::2].astype(float)
    return np.hypot(real, imag)          # one amplitude per subcarrier

def motion_score(win: deque) -> float:
    a = np.array(win)                    # shape: (packets, subcarriers)
    return float(np.mean(np.var(a, axis=0)))   # avg variance across subcarriers

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    win = deque(maxlen=WINDOW)
    baseline, threshold = [], None
    t0 = time.time()
    print(f"Calibrating for {CALIB_SECONDS}s: keep the room EMPTY and still...")

    while True:
        line = ser.readline().decode(errors="ignore")
        if not line.startswith("CSI_DATA"):
            continue
        amp = parse_amplitude(line)
        if amp is None:
            continue
        # ignore packets with a different subcarrier count than the window
        if win and len(amp) != len(win[0]):
            win.clear()
        win.append(amp)
        if len(win) < WINDOW:
            continue

        score = motion_score(win)
        if threshold is None:
            baseline.append(score)
            if time.time() - t0 >= CALIB_SECONDS:
                threshold = np.mean(baseline) + K * np.std(baseline)
                print(f"Calibrated. Threshold = {threshold:.2f}")
        else:
            state = "PRESENT" if score > threshold else "empty  "
            print(f"\r{state}  score={score:8.2f}  threshold={threshold:.2f}", end="", flush=True)

if __name__ == "__main__":
    main()
