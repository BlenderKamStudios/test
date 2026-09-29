"""Combines several TX->RX links (one Sensor each) into one picture of the room.

Each link runs the single-link pipeline from sensing.py (movement, breathing).
On top of that, with receivers placed around the room, a heatmap shows where
the movement probably is:
  - Each link is sensitive in a thin ellipse around its TX->RX line (points
    whose detour TX->point->RX is short). Its weight on a map cell falls off
    with that detour.
  - Each link's activity is its movement score above the quiet level.
  - A cell's heat is the average activity of the links covering it, so an area
    seen by a quiet link stays cool, and areas no link covers stay dark.
It says "movement is somewhere in this area", never an exact spot.
Breathing and presence are taken from whichever link sees them best.
"""
import json, threading, time

import numpy as np

from sensing import Sensor

TICK_HZ = 15
RATIO_SMOOTH = 0.5          # s, smoothing of each link's movement ratio
QUIET = 0.5                 # movement ratio (1 = detection line) counted as no activity
FULL = 2.0                  # ratio shown as full heat
HEAT_SMOOTH = 1.0           # s, fade of the heatmap
CELL = 0.1                  # m, heatmap resolution
MARGIN = 1.0                # m, map border around the boards
ZONE = 0.35                 # m, detour at which a link's weight falls to ~60%
COVER = 0.3                 # cells covered by less than this total weight fade out


def load_layout(path):
    with open(path) as f:
        lay = json.load(f)
    tx = lay["tx"]
    for r in lay["receivers"]:
        r["mac"] = r["mac"].lower()
    return dict(tx=dict(name=tx.get("name", "TX"), x=float(tx["x"]), y=float(tx["y"])),
                receivers=[dict(name=r["name"], mac=r["mac"], x=float(r["x"]), y=float(r["y"]))
                           for r in lay["receivers"]])


class Link:
    def __init__(self, rx, tx):
        self.name, self.mac = rx["name"], rx["mac"]
        self.pos = np.array([rx["x"], rx["y"]])
        self.sensor = Sensor()
        self.port = None            # serial port it was found on, set by server.py
        self.ratio = 0.0            # smoothed movement ratio


class Tracker:
    def __init__(self, layout):
        self.layout = layout
        self.tx = np.array([layout["tx"]["x"], layout["tx"]["y"]])
        self.links = {r["name"]: Link(r, layout["tx"]) for r in layout["receivers"]}
        self.by_mac = {l.mac: l for l in self.links.values()}
        self.errors = {}            # extra problems reported by server.py (unknown boards...)
        self.lock = threading.Lock()
        self._init_grid()
        self.present_since = None
        self.latest = None
        self._last = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _init_grid(self):
        pts = np.array([self.tx] + [l.pos for l in self.links.values()])
        lo, hi = pts.min(axis=0) - MARGIN, pts.max(axis=0) + MARGIN
        self.nx, self.ny = (np.ceil((hi - lo) / CELL).astype(int) + 1)
        self.x0, self.y0 = lo
        gx = lo[0] + CELL * np.arange(self.nx)
        gy = lo[1] + CELL * np.arange(self.ny)
        P = np.stack(np.meshgrid(gx, gy), axis=-1)          # (ny, nx, 2)
        dtx = np.hypot(*(P - self.tx).transpose(2, 0, 1))
        W = []
        for l in self.links.values():
            detour = dtx + np.hypot(*(P - l.pos).transpose(2, 0, 1)) - np.hypot(*(l.pos - self.tx))
            W.append(np.exp(-(detour / ZONE) ** 2).ravel())
        self.W = np.array(W)                                 # (links, cells)
        self.cover = self.W.sum(axis=0)
        self.heat = np.zeros(self.nx * self.ny)

    # ---- controls, applied to every link --------------------------------------
    def calibrate(self):
        for l in self.links.values():
            l.sensor.calibrate()

    def set_sensitivity(self, value):
        for l in self.links.values():
            l.sensor.set_sensitivity(value)

    def _loop(self):
        while True:
            time.sleep(1 / TICK_HZ)
            try:
                snap = self._tick(time.time())
            except Exception as e:                        # keep the loop alive
                snap = dict(self.latest or {}, error=f"tracking: {e}")
            with self.lock:
                self.latest = snap

    def snapshot(self):
        with self.lock:
            return dict(self.latest) if self.latest else dict(t=time.time(), state="warming", links=[])

    # ---- combine the links ----------------------------------------------------
    def _tick(self, now):
        dt = 1 / TICK_HZ if self._last is None else min(1.0, now - self._last)
        self._last = now
        snaps = {name: l.sensor.snapshot(now) for name, l in self.links.items()}
        live = {n: s for n, s in snaps.items() if s["link"]["rate"] > 0}

        # Per-link smoothed movement ratio
        k = 1 - np.exp(-dt / RATIO_SMOOTH)
        for n, l in self.links.items():
            r = snaps[n]["motion"]["ratio"] if n in live else None
            l.ratio += ((r or 0.0) - l.ratio) * k

        states = {n: s["state"] for n, s in live.items()}
        vals = set(states.values())
        if not live:
            state = "nodata"
        elif vals & {"calib_countdown", "calibrating"}:
            state = "calib_countdown" if "calib_countdown" in vals else "calibrating"
        elif "moving" in vals:
            state = "moving"
        elif "still_breathing" in vals:
            state = "still_breathing"
        elif vals == {"warming"}:
            state = "warming"
        elif "maybe" in vals:
            state = "maybe"
        else:
            state = "empty"

        present = state in ("moving", "still_breathing", "maybe")
        if present and self.present_since is None:
            self.present_since = now
        if not present:
            self.present_since = None

        # The link that sees the most movement drives the movement meter
        ranked = sorted(live, key=lambda n: live[n]["motion"]["ratio"] or 0, reverse=True)
        primary = live[ranked[0]] if ranked else next(iter(snaps.values()))
        motion = dict(primary["motion"], link=ranked[0] if ranked else None)
        motion["events"] = sum(s["motion"]["events"] for s in snaps.values())
        agos = [s["motion"]["last_ago"] for s in snaps.values() if s["motion"]["last_ago"] is not None]
        motion["last_ago"] = min(agos) if agos else None

        # Breathing from the link that sees it best
        def breath_rank(n):
            b = live[n]["breathing"]
            return (b.get("status") == "detected", b.get("strength") or 0)
        bname = max(live, key=breath_rank) if live else None
        breathing = dict(live[bname]["breathing"], link=bname) if bname else dict(status="searching")

        dists = [s["disturbance"] for s in live.values() if s["disturbance"]]
        dists = [d for d in dists if len(d) == len(dists[0])] if dists else []
        disturbance = [round(float(x), 2) for x in np.mean(dists, axis=0)] if dists else None

        calibs = [s for s in live.values() if s["calib"]]
        ago = [s["calibrated_ago"] for s in live.values() if s["calibrated_ago"] is not None]
        warm = [s["warmup_left"] for s in live.values() if s["warmup_left"] is not None]
        warnings = [f"{n}: {s['calib_warning']}" for n, s in live.items() if s["calib_warning"]]
        errors = [f"{n}: {s['error']}" for n, s in snaps.items() if s["error"]] + list(self.errors.values())
        rates = [snaps[n]["link"]["rate"] for n in self.links]

        links = []
        for n, l in self.links.items():
            s = snaps[n]
            thr = s["motion"]["threshold"]
            links.append(dict(
                name=n, mac=l.mac, port=l.port, x=float(l.pos[0]), y=float(l.pos[1]),
                rate=s["link"]["rate"], rssi=s["link"]["rssi"], state=s["state"],
                ratio=s["motion"]["ratio"], smooth_ratio=round(l.ratio, 3),
                breathing=s["breathing"].get("status"), breath_strength=s["breathing"].get("strength"),
                history=[(t, round(v / thr, 3)) for t, v in s["history"]] if thr else [],
            ))

        return dict(
            t=now, state=state, error="; ".join(errors) or None,
            warmup_left=max(warm) if warm and state == "warming" else None,
            calib=calibs[0]["calib"] if calibs else None,
            calibrated_ago=min(ago) if ago else None,
            calib_warning=" ".join(warnings) or None,
            sensitivity=primary["sensitivity"],
            link=dict(rate=min(rates) if rates else 0, rssi=primary["link"]["rssi"],
                      subcarriers=primary["link"]["subcarriers"]),
            links=links,
            motion=motion,
            presence=dict(present=present,
                          for_s=None if self.present_since is None else now - self.present_since,
                          still_for_s=None if state != "still_breathing" else
                          min((s["presence"]["still_for_s"] for s in live.values()
                               if s["presence"]["still_for_s"] is not None), default=None)),
            breathing=breathing,
            disturbance=disturbance,
            heatmap=self._heatmap(dt, live),
            layout=dict(tx=self.layout["tx"], receivers=[dict(name=r["name"], x=r["x"], y=r["y"])
                                                         for r in self.layout["receivers"]]),
        )

    def _heatmap(self, dt, live):
        act = np.array([np.clip((l.ratio - QUIET) / (FULL - QUIET), 0, 1) if n in live else 0.0
                        for n, l in self.links.items()])
        target = (act @ self.W) / (self.cover + COVER)
        self.heat += (target - self.heat) * (1 - np.exp(-dt / HEAT_SMOOTH))
        i = int(self.heat.argmax())
        return dict(x0=round(float(self.x0), 3), y0=round(float(self.y0), 3), cell=CELL,
                    nx=int(self.nx), ny=int(self.ny),
                    values=[round(float(v), 2) for v in self.heat],
                    peak=round(float(self.heat[i]), 2),
                    peak_x=round(float(self.x0 + CELL * (i % self.nx)), 2),
                    peak_y=round(float(self.y0 + CELL * (i // self.nx)), 2),
                    activity={n: round(float(a), 2) for n, a in zip(self.links, act)})
