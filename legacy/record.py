#!/usr/bin/env python3
"""Record the receiver's raw serial output to a file, with a PC timestamp per line.

Usage: python record.py SECONDS [PORT] [NAME]
Writes recordings/<NAME or timestamp>.csv; lines are "<unix_time> <CSI_DATA line>".
Recordings can be replayed into server.py with --replay for testing without hardware.
"""
import os, sys, time

import serial

from detect import BAUD

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
port = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyACM1"
name = sys.argv[3] if len(sys.argv) > 3 else time.strftime("%Y%m%d-%H%M%S")
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, f"{name}.csv")

n = 0
with serial.Serial(port, BAUD, timeout=1) as ser, open(path, "w") as out:
    end = time.time() + seconds
    while time.time() < end:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("CSI_DATA"):
            out.write(f"{time.time():.4f} {line}\n")
            n += 1
print(f"{n} packets -> {path}")
