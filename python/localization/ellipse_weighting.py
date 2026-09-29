"""EXPERIMENTAL link-ellipse weighting (in the spirit of radio tomographic
imaging, heavily simplified).

Idea: a person mostly affects a TX->RX link when standing close to the direct
line between them. For a point p, the "excess path length"

    excess(p) = |TX - p| + |p - RX| - |TX - RX|

is 0 on the direct line and grows away from it; points with the same excess
form an ellipse with TX and RX as foci. We give each point a weight

    w(p) = exp(-(excess(p) / ellipse_width)^2)

and let each link paint its activity onto the room through that weight:

    heat(p) = sum_i activity_i * w_i(p)                       normalize: false
    heat(p) = sum_i activity_i * w_i(p) / (sum_i w_i(p) + f)  normalize: true

Without normalisation the area around the TX always lights up (every link
starts there, so all ellipses overlap). Normalising turns the map into a
weighted AVERAGE of link activities; f (weight_floor) keeps areas that no
link covers dark instead of dividing by ~0.

Limitations you should state in any report:
  * 3 links cannot pin down a position; a person anywhere along a link's
    ellipse gives the same map. Different people/positions can give the
    same three numbers.
  * Real CSI also changes when a person is FAR from the line (multipath
    reflections off walls), which this model ignores.
  * With the TX on one wall and all RXs on the opposite wall the links are
    nearly parallel, so the map resolves "which link" much better than
    "how far from the TX".
"""
from __future__ import annotations

import numpy as np

from .base import Heatmap, SpatialEstimator


class EllipseWeighting(SpatialEstimator):
    name = "ellipse_weighting"

    def __init__(self, room, links, params):
        super().__init__(room, links, params)
        width = params.get("ellipse_width", 0.3)
        self.weights = {}
        for L in links:
            d_tx = np.hypot(self.X - L.tx[0], self.Y - L.tx[1])
            d_rx = np.hypot(self.X - L.rx[0], self.Y - L.rx[1])
            excess = d_tx + d_rx - np.hypot(L.tx[0] - L.rx[0], L.tx[1] - L.rx[1])
            self.weights[L.rx_id] = np.exp(-(excess / width) ** 2)

    def estimate(self, activity):
        heat = np.zeros_like(self.X)
        wsum = np.zeros_like(self.X)
        for rx_id, w in self.weights.items():
            a = activity.get(rx_id)
            if a is None or not np.isfinite(a):
                continue
            heat += max(a, 0.0) * w
            wsum += w
        if self.params.get("normalize"):
            heat = heat / (wsum + self.params.get("weight_floor", 0.5))
        return Heatmap(self.x, self.y, heat, info={"method": self.name})
