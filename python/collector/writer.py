"""Saving raw data. One folder per recording session:

data/raw/20260929-153000_walk-test/
    csi.csv          one row per CSI record, all receivers interleaved in
                     arrival order (or csi.csv.gz if recording.compress)
    status.log       every '#' line from the boards, with laptop time
    session.json     config snapshot, firmware BOOT lines, counts, times

Nothing is processed before saving: csi_raw holds the exact int8 values.
"""
from __future__ import annotations

import csv
import gzip
import json
import time

from common.config import project_path
from common.records import CSV_COLUMNS, CsiRecord


class SessionWriter:
    def __init__(self, cfg: dict, label: str = "", source: str = ""):
        rcfg = cfg["recording"]
        name = time.strftime("%Y%m%d-%H%M%S") + (f"_{label}" if label else "")
        self.dir = project_path(rcfg["directory"]) / name
        self.dir.mkdir(parents=True, exist_ok=False)
        csv_name = "csi.csv.gz" if rcfg.get("compress") else "csi.csv"
        self.csv_path = self.dir / csv_name
        self._f = gzip.open(self.csv_path, "wt", newline="") if rcfg.get("compress") \
            else open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._f)
        self._csv.writerow(CSV_COLUMNS)
        self._status = open(self.dir / "status.log", "w")
        self.flush_every = rcfg.get("flush_seconds", 1.0)
        self._last_flush = time.time()
        self.rows = 0
        self.meta = {
            "format_version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": source,
            "label": label,
            "columns": CSV_COLUMNS,
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "firmware_boot": {},
        }
        self._write_meta()

    def write(self, rec: CsiRecord):
        self._csv.writerow(rec.to_row())
        self.rows += 1
        now = time.time()
        if now - self._last_flush >= self.flush_every:
            self._f.flush()
            self._status.flush()
            self._last_flush = now

    def write_status(self, t: float, rx_id, port: str, line: str):
        self._status.write(f"{t:.3f}\t{rx_id if rx_id is not None else '?'}\t{port}\t{line}\n")

    def note_boot(self, rx_id, line: str):
        if self.meta["firmware_boot"].get(str(rx_id)) != line:
            self.meta["firmware_boot"][str(rx_id)] = line
            self._write_meta()

    def _write_meta(self):
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def close(self, stats: dict | None = None):
        self._f.close()
        self._status.close()
        self.meta["closed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.meta["rows"] = self.rows
        if stats:
            self.meta["stats"] = {s.name: {"packets": s.packets, "lost_packets": s.lost_packets,
                                           "corrupt_lines": s.corrupt_lines} for s in stats.values()}
        self._write_meta()
