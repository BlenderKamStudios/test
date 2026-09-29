"""Replaying a recorded session without any hardware.

Reads csi.csv / csi.csv.gz (or a session folder) and feeds the records to the
hub with their ORIGINAL host_time, paced by the recorded time differences
divided by `speed` (speed 0 = as fast as possible). Everything downstream
(processing, baseline, heatmap) uses the recorded timestamps, so a replay at
10x speed gives the same results as the live run.
"""
from __future__ import annotations

import csv
import gzip
import threading
import time
from pathlib import Path

from common.records import record_from_row
from .hub import Hub


def resolve_csv(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("csi.csv", "csi.csv.gz"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no csi.csv(.gz) in {p}")
    return p


def iter_records(path: str | Path):
    p = resolve_csv(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="") as f:
        for row in csv.DictReader(f):
            try:
                yield record_from_row(row)
            except (ValueError, KeyError, TypeError):
                continue                  # a half-written last line after a crash


class ReplaySource:
    def __init__(self, cfg: dict, hub: Hub, path: str, speed: float = 1.0):
        self.hub, self.path, self.speed = hub, path, speed
        self.thread = None

    def records(self):
        """All records, unpaced (used by --snapshot mode and tools)."""
        return iter_records(self.path)

    def start(self):
        for rx in self.hub.stats:
            self.hub.set_connection(rx, "replay", True)
        self.thread = threading.Thread(target=self._run, daemon=True, name="replay")
        self.thread.start()

    def _run(self):
        t_rec0 = t_wall0 = None
        for rec in iter_records(self.path):
            if self.hub.stop_event.is_set():
                return
            if self.speed > 0:
                if t_rec0 is None:
                    t_rec0, t_wall0 = rec.host_time, time.time()
                delay = (rec.host_time - t_rec0) / self.speed - (time.time() - t_wall0)
                if delay > 0:
                    time.sleep(delay)
            self.hub.on_record(rec)
        self.hub.event("replay finished")
        self.hub.finished.set()
