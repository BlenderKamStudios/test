"""Stage 3: collect CSI from RX1/RX2/RX3 and save the raw data (no display).

    python python/collect.py                      # serial ports from config.yaml, saves to data/raw/
    python python/collect.py --port 1=/dev/ttyUSB0 --port 2=/dev/ttyUSB1 --port 3=/dev/ttyUSB2
    python python/collect.py --duration 60 --label empty-room
    python python/collect.py --no-save            # just watch packet rates
    python python/collect.py --sim --duration 20  # test without hardware

Prints a status table every second. Stop with Ctrl+C (the file is closed
cleanly; even after a crash, everything up to the last flush is on disk).
"""
from __future__ import annotations

import argparse
import sys
import time

from collector import Hub, add_source_args, apply_port_overrides, make_source, make_writer
from common.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap, default_save=True)
    ap.add_argument("--duration", type=float, help="stop after this many seconds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_port_overrides(cfg, args.port)
    writer = make_writer(cfg, args)
    hub = Hub(cfg, writer, consumer_queue=1)       # nobody consumes records here
    source = make_source(cfg, args, hub)
    if writer:
        print(f"Saving to {writer.csv_path}")
    print("Receivers:", ", ".join(f"{r['name']}={r['port']}" for r in cfg["receivers"]), "| Ctrl+C to stop\n")
    source.start()

    t_start = time.time()
    try:
        while not hub.finished.is_set():
            time.sleep(1.0)
            for e in hub.pop_events():
                print("  *", e)
            lines = hub.summary_lines()
            elapsed = time.time() - t_start
            print(f"--- {elapsed:6.0f} s" + (f"  rows saved: {writer.rows}" if writer else ""))
            print("\n".join(lines), flush=True)
            if args.duration and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        hub.stop_event.set()
        time.sleep(0.3)
        if writer:
            with hub.lock:
                writer.close(hub.stats)
            print(f"Saved {writer.rows} records to {writer.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
