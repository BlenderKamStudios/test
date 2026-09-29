"""Streaming (packet-by-packet) filters for per-subcarrier time series.

Each filter takes one vector (one value per subcarrier) per packet and
returns the filtered vector. They keep their own small state, so they work
the same for live data, replay and simulation.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Hampel:
    """Outlier removal: if a value is further than n_sigmas robust standard
    deviations from the median of the last `window` values, replace it by
    that median. Robust sigma = 1.4826 * MAD (median absolute deviation).
    Removes single-packet spikes (e.g. AGC glitches) but keeps real changes
    that last longer than about window/2 packets."""

    def __init__(self, window: int = 11, n_sigmas: float = 3.0):
        self.buf: deque[np.ndarray] = deque(maxlen=window)
        self.n_sigmas = n_sigmas

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.buf.append(x)
        if len(self.buf) < 3:
            return x
        win = np.stack(self.buf)
        # plain median (much faster than nanmedian): a subcarrier with a NaN
        # in the window gets a NaN median, the comparison below is then False
        # and the value is passed through unchanged.
        med = np.median(win, axis=0)
        mad = 1.4826 * np.median(np.abs(win - med), axis=0)
        with np.errstate(invalid="ignore"):
            outlier = np.abs(x - med) > self.n_sigmas * np.maximum(mad, 1e-9)
        return np.where(outlier, med, x)


class Ema:
    """Exponential moving average: y = alpha*x + (1-alpha)*y_prev."""

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("ema alpha must be in (0, 1]")
        self.alpha, self.y = alpha, None

    def __call__(self, x):
        if self.y is None or np.shape(self.y) != np.shape(x):
            self.y = np.array(x, dtype=float, copy=True)
        else:
            self.y = np.where(np.isnan(self.y), x, self.alpha * x + (1 - self.alpha) * self.y)
        return self.y.copy()


class MovingAverage:
    def __init__(self, window: int):
        self.buf: deque[np.ndarray] = deque(maxlen=max(1, window))

    def __call__(self, x):
        self.buf.append(x)
        return np.nanmean(np.stack(self.buf), axis=0)


class Identity:
    def __call__(self, x):
        return x


def make_smoother(cfg: dict):
    method = cfg.get("method", "none")
    if method == "ema":
        return Ema(cfg["ema_alpha"])
    if method == "moving_average":
        return MovingAverage(cfg["window"])
    if method == "none":
        return Identity()
    raise ValueError(f"unknown smoothing method {method!r}")
