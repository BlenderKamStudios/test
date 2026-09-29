"""Reading the three receivers over USB serial, one thread per port.

* Ports given explicitly in config.yaml belong to that receiver and are
  re-opened forever after a disconnect (USB cable pulled, board reset).
* Receivers with port "auto": every few seconds all serial ports are
  scanned; an unknown port is opened and listened to until the board prints
  a "# STATUS"/"# BOOT" line with its rx_id (or its MAC), then it is assigned.
  Ports that never identify themselves (e.g. the TX, other devices) are left
  alone for a while.
* Lines are read in bulk (read(in_waiting)) — pyserial's readline() is too
  slow for 3 x 100 lines/s and led to garbled lines in the earlier prototype.
* Corrupted lines are counted and skipped, never "repaired".
"""
from __future__ import annotations

import threading
import time

import serial
from serial.tools import list_ports

from common.records import ParseError, parse_csi_line, parse_status_line
from .hub import Hub


class SerialSource:
    def __init__(self, cfg: dict, hub: Hub):
        self.cfg, self.hub = cfg, hub
        self.scfg = cfg["serial"]
        self.claims: dict[int, str] = {}                  # rx_id -> port in use
        self.busy: set[str] = set()                       # ports with a reader thread
        # Ports that are not receivers (the TX, other devices). Never probed
        # again while they stay plugged in: on Linux, merely OPENING an ESP32
        # DevKit port resets the board, so re-probing would reboot the TX.
        self.not_receivers: set[str] = set()
        self.lock = threading.Lock()
        self.fixed = {r["port"]: r["id"] for r in cfg["receivers"] if r["port"] != "auto"}
        self.auto_ids = {r["id"] for r in cfg["receivers"] if r["port"] == "auto"}
        self.macs = {r["mac"].lower(): r["id"] for r in cfg["receivers"] if r.get("mac")}

    def start(self):
        for port, rx_id in self.fixed.items():
            self._spawn(port, rx_id)
        if self.auto_ids:
            threading.Thread(target=self._scan_loop, daemon=True, name="port-scan").start()

    # --------------------------------------------------------------- auto
    def _scan_loop(self):
        while not self.hub.stop_event.is_set():
            with self.lock:
                missing = self.auto_ids - set(self.claims)
            present = list_ports.comports()
            with self.lock:
                self.not_receivers &= {p.device for p in present}    # unplugged -> may be probed again
            if missing:
                for p in present:
                    dev = p.device
                    with self.lock:
                        skip = dev in self.busy or dev in self.fixed or dev in self.not_receivers
                    if not skip and _looks_like_usb_serial(p):
                        self._spawn(dev, None)
            self.hub.stop_event.wait(2.0)

    def _identify(self, info: dict, port: str) -> int | None:
        """Which configured auto receiver is this board? rx_id first, MAC second."""
        with self.lock:
            candidates = []
            try:
                candidates.append(int(info.get("rx_id", 0)))
            except ValueError:
                pass
            candidates.append(self.macs.get(info.get("mac", "").lower()))
            for rx_id in candidates:
                if rx_id in self.auto_ids and rx_id not in self.claims:
                    self.claims[rx_id] = port
                    return rx_id
        return None

    # -------------------------------------------------------------- reader
    def _spawn(self, port: str, rx_id: int | None):
        with self.lock:
            self.busy.add(port)
        threading.Thread(target=self._reader, args=(port, rx_id), daemon=True, name=f"serial-{port}").start()

    def _reader(self, port: str, fixed_rx: int | None):
        rx_id = fixed_rx
        reported_failure = False
        try:
            while not self.hub.stop_event.is_set():
                try:
                    ser = serial.Serial()
                    ser.port, ser.baudrate, ser.timeout = port, self.scfg["baud"], 0.1
                    # Opening a port normally toggles DTR/RTS, which on ESP32
                    # DevKits is wired to EN/IO0 and resets the board (or holds
                    # it in the bootloader). Keep both released.
                    ser.dtr = ser.rts = False
                    ser.open()
                except (serial.SerialException, OSError) as e:
                    if fixed_rx is None:
                        return                           # the scanner will try again later
                    if not reported_failure:          # say it once per outage, not every retry
                        reported_failure = True
                        self.hub.event(f"{port}: cannot open ({e}); retrying every "
                                       f"{self.scfg['reconnect_seconds']} s")
                    self.hub.stop_event.wait(self.scfg["reconnect_seconds"])
                    continue
                reported_failure = False
                if rx_id is not None:
                    self.hub.set_connection(rx_id, port, True)
                    self.hub.event(f"{port}: connected ({self.hub.stats[rx_id].name})")
                rx_id = self._read_loop(ser, port, rx_id)
                ser.close()
                if rx_id is not None:
                    self.hub.set_connection(rx_id, port, False)
                if fixed_rx is None:
                    return                               # auto port: release and let the scanner re-find it
                self.hub.stop_event.wait(self.scfg["reconnect_seconds"])
        finally:
            with self.lock:
                self.busy.discard(port)
                if fixed_rx is None and rx_id is not None and self.claims.get(rx_id) == port:
                    del self.claims[rx_id]

    def _read_loop(self, ser, port: str, rx_id: int | None) -> int | None:
        buf = b""
        opened = time.time()
        warned_mismatch = False
        while not self.hub.stop_event.is_set():
            try:
                chunk = ser.read(max(1, ser.in_waiting))
            except (serial.SerialException, OSError) as e:
                self.hub.event(f"{port}: disconnected ({e})")
                return rx_id
            if rx_id is None and time.time() - opened > self.scfg["identify_seconds"]:
                with self.lock:
                    self.not_receivers.add(port)
                self.hub.event(f"{port}: not a receiver, ignoring it while it stays plugged in")
                return None
            if not chunk:
                continue
            t = time.time()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            if len(buf) > 10000:                         # no newline for a long time: garbage
                buf = b""
            for raw in lines:
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith("CSI,"):
                    if rx_id is None:
                        continue                         # not identified yet
                    try:
                        rec = parse_csi_line(line, t, rx_id, port)
                    except ParseError:
                        self.hub.on_corrupt(rx_id)
                        continue
                    if rec.fw_rx_id not in (0, rx_id) and not warned_mismatch:
                        warned_mismatch = True
                        self.hub.event(f"{port}: board reports rx_id={rec.fw_rx_id} but config maps "
                                       f"this port to {rx_id}; using {rx_id}")
                    self.hub.on_record(rec)
                elif line.startswith("#"):
                    info = parse_status_line(line)
                    if rx_id is None and info and info.get("role") == "RX":
                        rx_id = self._identify(info, port)
                        if rx_id is not None:
                            self.hub.set_connection(rx_id, port, True)
                            self.hub.event(f"{port}: identified as {self.hub.stats[rx_id].name}")
                    self.hub.on_status(rx_id, port, line, info)
        return rx_id


def _looks_like_usb_serial(p) -> bool:
    # Skip built-in legacy ports (/dev/ttyS*) which exist on many PCs
    return p.vid is not None or "USB" in p.device or "ACM" in p.device or p.device.upper().startswith("COM")
