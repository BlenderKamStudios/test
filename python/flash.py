"""Build and flash the firmware with the settings from config/config.yaml.

    python python/flash.py tx --port /dev/ttyUSB0          # the transmitter
    python python/flash.py rx --port /dev/ttyUSB1 --id 1   # RX1
    python python/flash.py rx --port COM7 --id 2           # RX2 (Windows)
    python python/flash.py rx --compile-only --id 3        # just check that it builds
    python python/flash.py flags rx --id 1                 # print the compiler flags only

Uses arduino-cli (the one bundled with Arduino IDE 2 is found automatically)
with the ESP32 core "esp32:esp32" (Arduino-ESP32 3.x = ESP-IDF 5.x).
Values from config.yaml are passed as -D compiler flags, overriding the
defaults in firmware/common/csi_common/csi_common.h and firmware/rx/rx.ino.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

from common.config import PROJECT_ROOT, load_config

FIRMWARE = PROJECT_ROOT / "firmware"
CLI_CANDIDATES = [
    "/opt/arduino-ide/resources/app/lib/backend/resources/arduino-cli",
    str(Path.home() / "AppData/Local/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"),
    "C:/Program Files/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe",
    "/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli",
]


def find_cli(setting: str) -> str:
    if setting and setting != "auto":
        return setting
    found = shutil.which("arduino-cli")
    if found:
        return found
    for c in CLI_CANDIDATES + glob.glob("/usr/share/arduino-ide*/resources/app/lib/backend/resources/arduino-cli"):
        if Path(c).exists():
            return c
    sys.exit("arduino-cli not found. Install it (https://arduino.github.io/arduino-cli/) or set "
             "firmware.arduino_cli in config.yaml.")


def flags(cfg: dict, role: str, rx_id: int) -> list[str]:
    fw = cfg["firmware"]
    w, tx, rx = fw["wifi"], fw["tx"], fw["rx"]
    mac = ",".join("0x" + b for b in w["tx_mac"].lower().split(":"))
    if len(mac.split(",")) != 6:
        sys.exit(f"bad firmware.wifi.tx_mac {w['tx_mac']!r}")
    if not 1 <= int(w["channel"]) <= 13:
        sys.exit("firmware.wifi.channel must be 1..13")
    f = [f"-DCSI_CHANNEL={int(w['channel'])}", f"-DCSI_TX_MAC={mac}",
         f"-DTX_PACKETS_PER_SECOND={int(tx['packets_per_second'])}",
         f"-DTX_PAYLOAD_BYTES={int(tx['payload_bytes'])}",
         f"-DTX_PHY_HT20={1 if tx['phy'] == 'ht20' else 0}",
         f"-DTX_POWER_QDBM={int(tx['tx_power_qdbm'])}",
         f"-DTX_BAUD={int(tx['baud'])}", f"-DRX_BAUD={int(rx['baud'])}"]
    if role == "rx":
        b = lambda k: 1 if rx[k] else 0  # noqa: E731
        f += [f"-DRX_ID={rx_id}", f"-DRX_FILTER_TX_MAC={b('filter_tx_mac')}",
              f"-DRX_CSI_LLTF={b('csi_lltf')}", f"-DRX_CSI_HTLTF={b('csi_htltf')}",
              f"-DRX_CSI_STBC_HTLTF2={b('csi_stbc_htltf2')}", f"-DRX_CSI_LTF_MERGE={b('csi_ltf_merge')}",
              f"-DRX_CSI_CHANNEL_FILTER={b('csi_channel_filter')}"]
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tx", "rx", "flags"])
    ap.add_argument("role", nargs="?", choices=["tx", "rx"], help="for 'flags': tx or rx")
    ap.add_argument("--port", help="serial port of the board to flash")
    ap.add_argument("--id", type=int, default=0, help="receiver ID (1, 2, 3...); 0 = set later over serial")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    role = args.role if args.what == "flags" else args.what
    if role is None:
        ap.error("flags needs a role: tx or rx")
    known = {r["id"] for r in cfg["receivers"]}
    if role == "rx" and args.id and args.id not in known:
        ap.error(f"--id {args.id} is not in config.yaml receivers {sorted(known)}")
    fl = flags(cfg, role, args.id)
    if args.what == "flags":
        print(" ".join(fl))
        return 0
    if not args.compile_only and not args.port:
        ap.error("--port is required (or use --compile-only)")

    cli = find_cli(cfg["firmware"]["arduino_cli"])
    sketch = FIRMWARE / role
    build = sketch / "build" / (f"rx{args.id}" if role == "rx" else "tx")
    cmd = [cli, "compile", "--fqbn", cfg["firmware"]["fqbn"], "--library", str(FIRMWARE / "common" / "csi_common"),
           "--build-path", str(build), "--build-property", "compiler.cpp.extra_flags=" + " ".join(fl)]
    if not args.compile_only:
        cmd += ["--upload", "--port", args.port]
    cmd.append(str(sketch))
    print(f"{role.upper()}{' id ' + str(args.id) if role == 'rx' else ''}: {' '.join(fl)}")
    print("$", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print("\nBuild/upload failed. If the upload cannot connect: hold the BOOT button while "
              "'Connecting...' is shown, check the port, and close any serial monitor using it.")
    elif not args.compile_only:
        baud = cfg["firmware"]["rx" if role == "rx" else "tx"]["baud"]
        print(f"\nDone. Watch the output with:  python -m serial.tools.miniterm {args.port} {baud}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
