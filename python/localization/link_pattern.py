"""EXPERIMENTAL approximate position from the PATTERN of link activity.

Idea: a person mainly disturbs the links whose line (TX->RX) they are close
to. For every grid point p we predict how strongly each link WOULD react if
the person stood there (a sensitivity s_i(p), same ellipse model as
ellipse_weighting.py, but wider), and compare that predicted pattern with
the measured one:

    s_i(p)     = exp(-(excess_i(p) / pattern_width)^2)
    coverage(p)= max_i s_i(p)           how much any link "sees" point p
    s^_i(p)    = s_i(p) / coverage(p)   predicted pattern, largest link = 1
    a^_i       = a_i / max_j a_j        measured pattern,  largest link = 1
    L(p)       = exp(-sum_i (a^_i - s^_i(p))^2 / (2 sigma^2)) * coverage(p)^coverage_power

The map shows L (scaled to 0..1); the reported position is the L-weighted
centre of the cells above 60 % of the maximum, and "spread" is their radius.

Example (TX in the middle, RXs around it): only RX1's link active -> the
cells along the TX-RX1 line far from the TX match (1, 0, 0) best -> the
estimate lies on that line. All links active -> the area around the TX.

What this CANNOT do (say so in any report):
  * With 3 links the pattern only tells "near which link(s)", not WHERE
    along a link: expect errors of a metre or more.
  * The sensitivity model ignores wall reflections; a person far from all
    links can still change them and will then be placed on a link.
  * Two people, or other moving things, produce meaningless positions.
  * coverage_power trades off "far from the TX" vs "near the TX" — a
    tuning choice, not a measured fact.
"""
from __future__ import annotations

import numpy as np

from .base import Heatmap, SpatialEstimator


class LinkPattern(SpatialEstimator):
    name = "link_pattern"

    def __init__(self, room, links, params):
        super().__init__(room, links, params)
        width = params.get("pattern_width", 0.5)
        self.ids = [L.rx_id for L in links]
        S = []
        for L in links:
            d_tx = np.hypot(self.X - L.tx[0], self.Y - L.tx[1])
            d_rx = np.hypot(self.X - L.rx[0], self.Y - L.rx[1])
            excess = d_tx + d_rx - np.hypot(L.tx[0] - L.rx[0], L.tx[1] - L.rx[1])
            S.append(np.exp(-(excess / width) ** 2))
        self.S = np.array(S) if S else np.zeros((0,) + self.X.shape)
        self.coverage = self.S.max(axis=0) if S else np.zeros_like(self.X)
        self.S_hat = self.S / np.maximum(self.coverage, 1e-6)

    def estimate(self, activity):
        empty = Heatmap(self.x, self.y, np.zeros_like(self.X), label=self._label(),
                        info={"method": self.name, "present": False, "estimate": None})
        idx = [k for k, i in enumerate(self.ids) if activity.get(i) is not None and np.isfinite(activity[i])]
        if not idx:
            return empty
        a = np.array([max(activity[self.ids[k]], 0.0) for k in idx])
        amax = a.max()
        empty.info["strength"] = float(amax)
        if amax < self.params.get("presence_threshold", 1.0):
            return empty                                  # nothing moved enough: no position
        a_hat = a / amax
        d2 = ((self.S_hat[idx] - a_hat[:, None, None]) ** 2).sum(axis=0)
        sigma = self.params.get("pattern_sigma", 0.35)
        L = np.exp(-d2 / (2 * sigma ** 2)) * self.coverage ** self.params.get("coverage_power", 0.5)
        if L.max() <= 0:
            return empty
        heat = L / L.max()
        m = heat >= 0.6
        w = heat[m]
        ex, ey = float((w * self.X[m]).sum() / w.sum()), float((w * self.Y[m]).sum() / w.sum())
        spread = float(np.sqrt((w * ((self.X[m] - ex) ** 2 + (self.Y[m] - ey) ** 2)).sum() / w.sum()))
        return Heatmap(self.x, self.y, heat, label=self._label(),
                       info={"method": self.name, "present": True, "estimate": (ex, ey),
                             "spread": spread, "strength": float(amax)})

    @staticmethod
    def _label():
        return "EXPERIMENTAL approximate position from link-activity pattern (3 links, error >= ~1 m)"
