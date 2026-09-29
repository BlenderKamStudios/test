"""Empty-room baseline and "CSI change" detection.

Why not simply  difference = current_CSI - baseline_CSI  on complex CSI?
    The ESP32's raw phase contains a random offset in every packet (TX and RX
    clocks are not synchronised), so two complex CSI vectors of the SAME
    empty room differ hugely. Their difference would be mostly clock noise.
    The amplitude is also scaled by the automatic gain control (AGC).
What we do instead:
    1. use the PROCESSED AMPLITUDE (AGC-normalised, outlier-filtered,
       smoothed; see pipeline.py) — phase is not used for the baseline;
    2. baseline = per-subcarrier mean  mu[k]  and standard deviation
       sigma[k]  of that amplitude while the room is empty;
    3. a per-packet distance between the current amplitude a[k] and mu[k]
       (the "metric", selectable in config.yaml):
         zscore      mean_k |a[k] - mu[k]| / sigma[k]
                     "how many empty-room standard deviations away, on average"
         euclidean   ||a - mu|| / ||mu||      relative change of the profile
         correlation 1 - Pearson(a, mu)        change of SHAPE only
    4. smooth that value over time (EMA) -> "change";
    5. divide by the change level seen while the room was empty (the
       threshold_percentile of the metric during the baseline) -> "ratio".
       ratio ≈ 1 or below: like the empty room; ratio >> 1: something changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .filters import Ema
from .pipeline import Frame

EPS = 1e-6


def change_metric(a: np.ndarray, mu: np.ndarray, sigma: np.ndarray, method: str) -> float:
    v = ~np.isnan(a) & ~np.isnan(mu)
    if v.sum() < 4:
        return float("nan")
    a, mu, sigma = a[v], mu[v], sigma[v]
    if method == "zscore":
        return float(np.mean(np.abs(a - mu) / (sigma + EPS)))
    if method == "euclidean":
        return float(np.linalg.norm(a - mu) / (np.linalg.norm(mu) + EPS))
    if method == "correlation":
        if np.std(a) < EPS or np.std(mu) < EPS:
            return float("nan")
        return float(1.0 - np.corrcoef(a, mu)[0, 1])
    raise ValueError(f"unknown baseline metric {method!r}")


@dataclass
class Baseline:
    mu: np.ndarray            # (64,) mean processed amplitude, empty room
    sigma: np.ndarray         # (64,) its standard deviation
    noise_level: float        # threshold_percentile of the metric while empty
    n_packets: int
    t_start: float
    t_end: float


class ReceiverChange:
    """Baseline + change value for one receiver."""

    def __init__(self, bcfg: dict):
        self.cfg = bcfg
        self.baseline: Baseline | None = None
        self.ema = Ema(bcfg.get("change_smoothing_alpha", 1.0))
        self.change = float("nan")     # smoothed metric
        self.ratio = float("nan")      # change / noise_level
        self._rec: list[np.ndarray] = []

    def add_baseline_frame(self, f: Frame):
        self._rec.append(f.amp_proc)

    def finish_baseline(self, t_start: float, t_end: float) -> str | None:
        """Returns an error message, or None on success."""
        frames, self._rec = self._rec, []
        if len(frames) < self.cfg.get("min_packets", 1):
            return f"only {len(frames)} packets (need {self.cfg.get('min_packets')})"
        A = np.stack(frames)
        mu, sigma = np.nanmean(A, axis=0), np.nanstd(A, axis=0)
        m = np.array([change_metric(a, mu, sigma, self.cfg["metric"]) for a in A])
        m = m[~np.isnan(m)]
        noise = float(np.percentile(m, self.cfg.get("threshold_percentile", 95))) if len(m) else float("nan")
        self.baseline = Baseline(mu, sigma, max(noise, EPS), len(frames), t_start, t_end)
        self.ema = Ema(self.cfg.get("change_smoothing_alpha", 1.0))
        return None

    def update(self, f: Frame):
        if self.baseline is None:
            return
        m = change_metric(f.amp_proc, self.baseline.mu, self.baseline.sigma, self.cfg["metric"])
        if np.isnan(m):
            return
        self.change = float(self.ema(np.array(m)))
        self.ratio = self.change / self.baseline.noise_level


class BaselineManager:
    """Runs the baseline workflow for all receivers:
    idle -> (start) -> waiting for data -> recording (duration, in DATA time,
    so replay/simulation work at any speed) -> ready."""

    def __init__(self, bcfg: dict, receiver_ids):
        self.cfg = bcfg
        self.rx = {i: ReceiverChange(bcfg) for i in receiver_ids}
        self.state = "idle"
        self.t_start = None
        self.errors: dict[int, str] = {}

    def start(self):
        self.state, self.t_start, self.errors = "waiting", None, {}
        for r in self.rx.values():
            r._rec = []

    def feed(self, rx_id: int, f: Frame):
        rc = self.rx.get(rx_id)
        if rc is None:
            return
        if self.state == "waiting":
            self.state, self.t_start = "recording", f.t
        if self.state == "recording":
            if f.t - self.t_start < self.cfg["duration_seconds"]:
                rc.add_baseline_frame(f)
            else:
                self._finish(f.t)
        if self.state == "ready":
            rc.update(f)

    def _finish(self, t_end):
        for i, rc in self.rx.items():
            err = rc.finish_baseline(self.t_start, t_end)
            if err:
                self.errors[i] = err
        self.state = "ready"

    def progress(self, t_now: float) -> float:
        if self.state != "recording" or self.t_start is None:
            return 0.0
        return min(1.0, (t_now - self.t_start) / self.cfg["duration_seconds"])

    def save(self, path: str | Path):
        data = {}
        for i, rc in self.rx.items():
            b = rc.baseline
            if b:
                data.update({f"rx{i}_mu": b.mu, f"rx{i}_sigma": b.sigma,
                             f"rx{i}_meta": np.array([b.noise_level, b.n_packets, b.t_start, b.t_end])})
        data["metric"] = np.array(self.cfg["metric"])
        np.savez(path, **data)

    def load(self, path: str | Path):
        d = np.load(path)
        if str(d["metric"]) != self.cfg["metric"]:
            raise ValueError(f"baseline file uses metric {d['metric']}, config says {self.cfg['metric']}")
        for i, rc in self.rx.items():
            if f"rx{i}_mu" in d:
                noise, n, t0, t1 = d[f"rx{i}_meta"]
                rc.baseline = Baseline(d[f"rx{i}_mu"], d[f"rx{i}_sigma"], float(noise), int(n), t0, t1)
        self.state = "ready"
