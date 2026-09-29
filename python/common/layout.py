"""TX-centred board layout used by the dashboard.

Coordinates: metres, TX at (0, 0), seen from above. The map covers
-map_radius..+map_radius in x and y.

Where do the receiver positions come from?
  1. config/positions.yaml, if it exists (written by the dashboard's
     "Save positions" button after you dragged the RXs into place);
  2. otherwise: default directions (dashboard.rx_default_angles_deg) at
     dashboard.default_rx_distance.
"Auto-place" can then re-estimate each RX's DISTANCE from its RSSI with the
log-distance path-loss model

      RSSI(d) = RSSI(1 m) - 10 * n * log10(d)   ->   d = 10^((RSSI(1 m) - RSSI) / (10 n))

which is rough (typically +-50 %, worse through walls, bodies or with the
antenna pointing away). The DIRECTION of a receiver cannot be measured with
one antenna per board, so the current directions are kept: drag each RX
marker to where the board really is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import yaml

from .config import project_path


@dataclass
class Layout:
    radius: float
    rx: dict[int, tuple[float, float]] = field(default_factory=dict)
    origin: str = "default"      # default | file | rssi | edited

    def room(self) -> dict:
        """As a 'room' dict for the localization estimators."""
        return {"x0": -self.radius, "y0": -self.radius, "width": 2 * self.radius, "height": 2 * self.radius,
                "tx": {"x": 0.0, "y": 0.0}}


def load_layout(cfg: dict) -> Layout:
    d = cfg["dashboard"]
    lay = Layout(float(d["map_radius"]))
    path = project_path(d["positions_file"])
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        for r in cfg["receivers"]:
            p = (data.get("receivers") or {}).get(r["id"])
            if p:
                lay.rx[r["id"]] = (float(p["x"]), float(p["y"]))
        if data.get("map_radius"):
            lay.radius = float(data["map_radius"])
        lay.origin = "file"
    angles = d.get("rx_default_angles_deg", [90, 210, 330])
    for k, r in enumerate(cfg["receivers"]):
        if r["id"] not in lay.rx:
            a = math.radians(angles[k % len(angles)])
            dist = float(d["default_rx_distance"])
            lay.rx[r["id"]] = (dist * math.cos(a), dist * math.sin(a))
    return lay


def save_layout(cfg: dict, lay: Layout) -> str:
    path = project_path(cfg["dashboard"]["positions_file"])
    data = {"_help": "Receiver positions in metres, TX at (0, 0). Written by python/dashboard.py.",
            "map_radius": lay.radius,
            "receivers": {i: {"x": round(x, 3), "y": round(y, 3)} for i, (x, y) in lay.rx.items()}}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def distance_from_rssi(rssi: float, rssi_1m: float, n: float) -> float:
    return 10 ** ((rssi_1m - rssi) / (10 * n))


def place_by_rssi(cfg: dict, lay: Layout, median_rssi: dict[int, float]) -> dict[int, float]:
    """Move each RX along its current direction to the RSSI-estimated distance.
    Returns the (unclamped) estimated distances."""
    d = cfg["dashboard"]
    est = {}
    for i, rssi in median_rssi.items():
        if i not in lay.rx or rssi is None:
            continue
        dist = distance_from_rssi(rssi, d["rssi_at_1m"], d["path_loss_exponent"])
        est[i] = dist
        dist = min(max(dist, 0.3), 0.9 * lay.radius)
        x, y = lay.rx[i]
        ang = math.atan2(y, x) if (x or y) else 0.0
        lay.rx[i] = (dist * math.cos(ang), dist * math.sin(ang))
    lay.origin = "rssi"
    return est
