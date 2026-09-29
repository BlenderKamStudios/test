"""The Hub: where every data source delivers its records.

Serial readers, the replay reader and the simulator all call
hub.on_record(record). The hub
  1. writes the record to the raw CSV (if a writer is attached) — first, so
     raw data is saved even if the display cannot keep up;
  2. updates per-receiver statistics (rate, lost packets, errors);
  3. queues the record for consumers (live display) — this queue is bounded,
     a slow display drops old records from ITS view only, never from disk.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from common.records import CsiRecord


@dataclass
class RxStats:
    name: str
    port: str = "-"
    connected: bool = False
    packets: int = 0
    corrupt_lines: int = 0      # lines that failed to parse (serial glitches)
    lost_packets: int = 0       # TX packets missed, from gaps in tx_seq
    rate_hz: float = 0.0
    last_rssi: int | None = None
    last_time: float = 0.0
    fw_status: dict = field(default_factory=dict)   # last "# STATUS" of the firmware
    fw_boot: dict = field(default_factory=dict)     # last "# BOOT" of the firmware
    _last_seq: int = -1
    _win_count: int = 0
    _win_start: float = 0.0


class Hub:
    def __init__(self, cfg: dict, writer=None, consumer_queue: int = 50000):
        self.cfg = cfg
        self.writer = writer
        self.lock = threading.Lock()
        self.queue: deque[CsiRecord] = deque(maxlen=consumer_queue)
        self.stats = {r["id"]: RxStats(r["name"]) for r in cfg["receivers"]}
        self.events: deque[str] = deque(maxlen=500)      # recent messages (for displays)
        self._new_events: deque[str] = deque(maxlen=500)  # not yet printed by pop_events()
        self.stop_event = threading.Event()
        self.finished = threading.Event()        # set by sources that end (replay)

    # ---------------------------------------------------------------- input
    def on_record(self, rec: CsiRecord):
        with self.lock:
            if self.writer:
                self.writer.write(rec)
            s = self.stats.get(rec.receiver_id)
            if s is None:
                return
            s.packets += 1
            s.last_rssi, s.last_time = rec.rssi, rec.host_time
            if rec.tx_seq >= 0:
                gap = rec.tx_seq - s._last_seq - 1
                if s._last_seq >= 0 and 0 < gap < 100000:
                    s.lost_packets += gap
                s._last_seq = rec.tx_seq          # a TX reboot resets seq -> no gap counted
            s._win_count += 1
            if rec.host_time - s._win_start >= 1.0:
                if s._win_start:
                    s.rate_hz = s._win_count / (rec.host_time - s._win_start)
                s._win_start, s._win_count = rec.host_time, 0
            self.queue.append(rec)

    def on_corrupt(self, rx_id: int | None):
        with self.lock:
            if rx_id in self.stats:
                self.stats[rx_id].corrupt_lines += 1

    def on_status(self, rx_id: int | None, port: str, line: str, info: dict):
        with self.lock:
            if self.writer:
                self.writer.write_status(time.time(), rx_id, port, line)
            s = self.stats.get(rx_id)
            if s and info.get("kind") == "STATUS":
                s.fw_status = info
            elif s and info.get("kind") == "BOOT":
                s.fw_boot = info
                if self.writer:
                    self.writer.note_boot(rx_id, line)
            if line.startswith(("# ERROR", "# WARNING")):
                self.event(f"{port}: {line[2:]}", locked=True)

    def set_connection(self, rx_id: int, port: str, connected: bool):
        with self.lock:
            if rx_id in self.stats:
                self.stats[rx_id].port, self.stats[rx_id].connected = port, connected

    def event(self, msg: str, locked: bool = False):
        entry = f"{time.strftime('%H:%M:%S')} {msg}"
        if locked:
            self.events.append(entry)
            self._new_events.append(entry)
        else:
            with self.lock:
                self.events.append(entry)
                self._new_events.append(entry)

    def pop_events(self) -> list[str]:
        with self.lock:
            out = list(self._new_events)
            self._new_events.clear()
        return out

    # --------------------------------------------------------------- output
    def drain(self) -> list[CsiRecord]:
        with self.lock:
            out = list(self.queue)
            self.queue.clear()
        return out

    def summary_lines(self) -> list[str]:
        now = time.time()
        rows = [f"{'RX':<5}{'port':<16}{'conn':<6}{'packets':>9}{'rate/s':>8}{'lost':>7}"
                f"{'corrupt':>8}{'rssi':>6}{'fw_drop':>8}"]
        with self.lock:
            for s in self.stats.values():
                stale = now - s.last_time > 2.0 if s.last_time else True
                rate = 0.0 if stale and s.port not in ("replay", "sim") else s.rate_hz
                rows.append(f"{s.name:<5}{s.port[-15:]:<16}{('yes' if s.connected else 'no'):<6}"
                            f"{s.packets:>9}{rate:>8.1f}{s.lost_packets:>7}{s.corrupt_lines:>8}"
                            f"{(s.last_rssi if s.last_rssi is not None else '-'):>6}"
                            f"{s.fw_status.get('queue_drops', '-'):>8}")
        return rows
