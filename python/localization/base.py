"""Interface for spatial (heatmap) algorithms.

EXPERIMENTAL. A heatmap algorithm turns one "activity" number per TX->RX link
into a 2D map of the room. With 3 links this is a very under-determined
problem: the map shows WHERE THE LINKS ARE SENSITIVE weighted by how much
each link changed, not where a person is. Treat it as a visual aid.

To try a new algorithm: subclass SpatialEstimator, implement estimate(),
and register it in localization/__init__.py under a new name; then set
localization.method in config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Link:
    rx_id: int
    name: str
    tx: tuple[float, float]
    rx: tuple[float, float]


@dataclass
class Heatmap:
    x: np.ndarray             # (nx,) cell centres, metres
    y: np.ndarray             # (ny,)
    values: np.ndarray        # (ny, nx), arbitrary units, >= 0
    label: str = "experimental link-weighting map (NOT a position estimate)"
    info: dict = field(default_factory=dict)


class SpatialEstimator:
    name = "base"

    def __init__(self, room: dict, links: list[Link], params: dict):
        self.room, self.links, self.params = room, links, params
        res = params.get("grid_resolution", 0.05)
        x0, y0 = room.get("x0", 0.0), room.get("y0", 0.0)     # lower-left corner (TX-centred maps use -R)
        self.x = x0 + np.arange(res / 2, room["width"], res)
        self.y = y0 + np.arange(res / 2, room["height"], res)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def estimate(self, activity: dict[int, float]) -> Heatmap:
        """activity: rx_id -> non-negative number (0 = link looks like the
        empty room). Must return a Heatmap over self.x, self.y."""
        raise NotImplementedError


def links_from_config(cfg: dict) -> list[Link]:
    tx = (cfg["room"]["tx"]["x"], cfg["room"]["tx"]["y"])
    return [Link(r["id"], r["name"], tx, (r["x"], r["y"])) for r in cfg["receivers"]]
