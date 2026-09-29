"""Simulated CSI for testing the software WITHOUT hardware.

It produces records in exactly the firmware's text format and pushes them
through the same parser as real data, so the whole pipeline is exercised.

The physics is a toy model — good enough to test code, NOT evidence that any
algorithm works on real data:
  * each TX->RX link has a fixed multipath channel (line of sight + a few
    random wall reflections): H[k] = sum_l a_l * exp(-j 2 pi f_k d_l / c)
  * a walking person adds a reflection path TX->person->RX and attenuates
    the line of sight when close to it (the same ellipse geometry the
    heatmap uses, so the heatmap will look better here than in reality)
  * ESP32 impairments: random common phase and phase slope per packet
    (unsynchronised clocks), AGC gain jitter, quantisation to int8,
    noise, garbage "first word", random packet loss.
The room is empty for simulation.empty_seconds (to record a baseline), then
a person walks a loop through the room.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from common.records import CsiRecord, format_csi_line, parse_csi_line
from preprocessing.csi_math import SUBCARRIERS
from .hub import Hub

C = 299_792_458.0
DF = 312.5e3                                   # subcarrier spacing, Hz


def channel_freq(ch: int) -> float:
    return (2407 + 5 * ch) * 1e6


class SimSource:
    def __init__(self, cfg: dict, hub: Hub, t0: float | None = None):
        self.cfg, self.hub = cfg, hub
        s = cfg["simulation"]
        self.pps, self.loss = s["packets_per_second"], s["packet_loss"]
        self.empty_s, self.speed = s["empty_seconds"], s["walk_speed"]
        self.rng = np.random.default_rng(s.get("seed"))
        self.room = cfg["room"]
        self.tx = np.array([self.room["tx"]["x"], self.room["tx"]["y"]])
        self.rx = {r["id"]: np.array([r["x"], r["y"]]) for r in cfg["receivers"]}
        self.fk = channel_freq(cfg["firmware"]["wifi"]["channel"]) + SUBCARRIERS * DF
        self.used = (np.abs(SUBCARRIERS) <= 26) & (SUBCARRIERS != 0)
        self.t0 = t0
        self.boot_us = {i: int(self.rng.integers(0, 2**31)) for i in self.rx}
        W, H = self.room["width"], self.room["height"]
        # static multipath per link: (path lengths, complex amplitudes)
        self.paths = {}
        for i, rxp in self.rx.items():
            d_los = np.linalg.norm(rxp - self.tx)
            d = np.concatenate([[d_los], d_los + self.rng.uniform(0.5, 8.0, 5)])
            a = np.concatenate([[1.0], self.rng.uniform(0.1, 0.5, 5)]) * np.exp(1j * self.rng.uniform(0, 2 * np.pi, 6))
            self.paths[i] = (d, a / d)
        # walking loop (fractions of the room)
        self.waypoints = np.array([[0.3, 0.2], [0.75, 0.3], [0.6, 0.8], [0.3, 0.75], [0.5, 0.5]]) * [W, H]
        seg = np.linalg.norm(np.diff(np.vstack([self.waypoints, self.waypoints[:1]]), axis=0), axis=1)
        self.seg_len, self.loop_len = seg, seg.sum()

    # ------------------------------------------------------------- ground truth
    def truth(self, t: float):
        """True person position at data time t (None while the room is empty)."""
        if self.t0 is None or t - self.t0 < self.empty_s:
            return None
        s = ((t - self.t0 - self.empty_s) * self.speed) % self.loop_len
        for k, L in enumerate(self.seg_len):
            if s <= L:
                a, b = self.waypoints[k], self.waypoints[(k + 1) % len(self.waypoints)]
                return a + (b - a) * (s / L)
            s -= L
        return self.waypoints[0]

    # ------------------------------------------------------------------ model
    def _channel(self, rx_id: int, person) -> np.ndarray:
        d, a = self.paths[rx_id]
        a = a.copy()
        d = d.copy()
        if person is not None:
            rxp = self.rx[rx_id]
            dp = np.linalg.norm(person - self.tx) + np.linalg.norm(rxp - person)
            excess = dp - d[0]
            a[0] *= 1 - 0.7 * np.exp(-(excess / 0.25) ** 2)          # body shadows the direct path
            refl = 0.5 * np.exp(-excess / 1.5) / dp * np.exp(1j * self.rng.normal(0, 0.1))
            d, a = np.append(d, dp), np.append(a, refl)
        return (a[None, :] * np.exp(-2j * np.pi * self.fk[:, None] * d[None, :] / C)).sum(axis=1)

    def _record(self, rx_id: int, n: int, t: float, person) -> CsiRecord:
        h = self._channel(rx_id, person)
        # ESP32 impairments
        h = h * np.exp(1j * (self.rng.uniform(0, 2 * np.pi) + SUBCARRIERS * self.rng.normal(0, 0.1)))
        rel_db = 20 * np.log10(np.mean(np.abs(h[self.used])) + 1e-12)
        gain = 22.0 / np.mean(np.abs(h[self.used])) * 10 ** (self.rng.normal(0, 0.5) / 20)   # AGC
        h = h * gain + (self.rng.normal(0, 0.7, 64) + 1j * self.rng.normal(0, 0.7, 64))
        h[~self.used] = 0
        q_real = np.clip(np.round(h.real), -128, 127).astype(int)
        q_imag = np.clip(np.round(h.imag), -128, 127).astype(int)
        # natural order (-32..31) -> ESP32 buffer order (0..31, -32..-1), [imag, real] pairs
        buf_real = np.concatenate([q_real[32:], q_real[:32]])
        buf_imag = np.concatenate([q_imag[32:], q_imag[:32]])
        raw = np.empty(128, int)
        raw[0::2], raw[1::2] = buf_imag, buf_real
        fwi = 1
        raw[:4] = self.rng.integers(-128, 128, 4)          # "first word invalid" garbage
        rssi = int(round(-20 + rel_db + self.rng.normal(0, 0.7)))
        ts = (self.boot_us[rx_id] + int((t - self.t0) * 1e6)) % 2**32
        rec = CsiRecord(t, rx_id, "sim", rx_id, n, n, ts, self.cfg["firmware"]["wifi"]["tx_mac"], rssi,
                        -95, self.cfg["firmware"]["wifi"]["channel"], 0, 1, 0, 0, 0, 0, 51, 0, fwi, 128,
                        raw.tolist())
        # round-trip through the real parser
        return parse_csi_line(format_csi_line(rec), t, rx_id, "sim")

    def records(self, duration: float | None = None, t0: float | None = None):
        """Generator of records in time order (unpaced)."""
        if self.t0 is None:
            self.t0 = t0 if t0 is not None else time.time()
        n = 0
        while duration is None or n / self.pps < duration:
            t = self.t0 + n / self.pps
            person = self.truth(t)
            if person is not None:
                person = person + self.rng.normal(0, 0.02, 2)      # body sway
            for rx_id in self.rx:
                if self.rng.random() >= self.loss:
                    yield self._record(rx_id, n, t, person)
            n += 1

    # --------------------------------------------------------- live (paced)
    def start(self):
        for rx in self.hub.stats:
            self.hub.set_connection(rx, "sim", True)
        threading.Thread(target=self._run, daemon=True, name="sim").start()

    def _run(self):
        last_state = None
        for rec in self.records(t0=time.time()):
            if self.hub.stop_event.is_set():
                return
            delay = rec.host_time - time.time()
            if delay > 0:
                time.sleep(delay)
            self.hub.on_record(rec)
            state = "room empty" if self.truth(rec.host_time) is None else "person walking"
            if state != last_state:
                last_state = state
                self.hub.event(f"simulation: {state}")
