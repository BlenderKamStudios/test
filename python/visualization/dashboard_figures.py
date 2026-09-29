"""Figures used inside the dashboard window (python/dashboard.py).

PositionMap     TX-centred 2D map: receivers (draggable), links, the
                EXPERIMENTAL approximate position with its uncertainty and trail
OverviewFigure  compact graphs for the small panel: change per receiver,
                change over time, signal variation over time
The detailed graphs reuse CsiFigure / ChangeFigure from figures.py.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from matplotlib.patches import Circle

from localization import Link, make_estimator
from .figures import rx_color

GRAB_RADIUS_M = 0.35


class PositionMap:
    def __init__(self, cfg: dict, fig, layout, on_moved=None):
        self.cfg, self.fig, self.layout, self.on_moved = cfg, fig, layout, on_moved
        self.dcfg = cfg["dashboard"]
        self.rx_cfg = cfg["receivers"]
        R = layout.radius
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.1, top=0.9)
        self.im = self.ax.imshow(np.zeros((2, 2)), origin="lower", extent=(-R, R, -R, R), cmap="magma",
                                 vmin=0, vmax=1, interpolation="bilinear", alpha=0.95)
        for r in range(1, int(R) + 1):                                   # distance rings, 1 m apart
            self.ax.add_patch(Circle((0, 0), r, fill=False, ls=":", lw=0.8, ec="#888"))
            self.ax.text(r * 0.707, r * 0.707, f"{r} m", color="#999", fontsize=7)
        self.links, self.markers, self.labels = {}, {}, {}
        for c, r in enumerate(self.rx_cfg):
            i, col = r["id"], rx_color(c)
            self.links[i], = self.ax.plot([0, 0], [0, 0], "--", color=col, lw=1.3)
            self.markers[i], = self.ax.plot([0], [0], "s", color=col, ms=13, mec="w", mew=1.5, zorder=5)
            self.labels[i] = self.ax.text(0, 0, r["name"], color="w", fontsize=9, zorder=6,
                                          bbox=dict(fc=col, ec="none", pad=1.5))
        self.ax.plot(0, 0, "*", color="#ffd700", ms=20, mec="k", zorder=5)
        self.ax.text(0.15, -0.35, "TX", fontsize=9, zorder=6, bbox=dict(fc="#ffd700", ec="none", pad=1.5))
        self.trail, = self.ax.plot([], [], "-", color="#00e5ff", lw=2, alpha=0.6, zorder=7)
        self.spread = Circle((0, 0), 0.5, fill=True, fc="#00e5ff", ec="#00e5ff", alpha=0.18, lw=1.5, zorder=7,
                             visible=False)
        self.ax.add_patch(self.spread)
        self.you, = self.ax.plot([], [], "o", color="#00e5ff", ms=16, mec="w", mew=2, zorder=8,
                                 label="approximate position")
        self.you_txt = self.ax.text(0, 0, "", color="w", fontsize=9, fontweight="bold", zorder=9)
        self.truth, = self.ax.plot([], [], "x", color="#7CFC00", ms=14, mew=3, zorder=8,
                                   label="true position (simulation only)")
        self.ax.set_xlim(-R, R)
        self.ax.set_ylim(-R, R)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_facecolor("black")
        self.ax.set_title("APPROXIMATE position (experimental)", color="#b00", fontsize=10, fontweight="bold")
        self.status = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes, va="top", fontsize=9, color="w",
                                   zorder=10, wrap=True, bbox=dict(fc="#000000b0", ec="none", pad=4))
        fig.text(0.01, 0.01, "Drag the RX squares to where the boards really are (directions cannot be measured), "
                 "then 'Save positions'.\nBright = where the measured link pattern fits the model best. "
                 "Not a measured location; typical error is a metre or more.", fontsize=7.5, color="#444", wrap=True)
        self._drag = None
        self._heat = None
        self._pos = None
        self._trail = deque()
        self.rebuild()
        fig.canvas.mpl_connect("button_press_event", self._press)
        fig.canvas.mpl_connect("motion_notify_event", self._motion)
        fig.canvas.mpl_connect("button_release_event", self._release)

    # ----------------------------------------------------------- layout
    def rebuild(self):
        links = [Link(r["id"], r["name"], (0.0, 0.0), self.layout.rx[r["id"]]) for r in self.rx_cfg]
        self.est = make_estimator(self.cfg, self.layout.room(), links)
        self._heat = None
        for i, (x, y) in self.layout.rx.items():
            self.links[i].set_data([0, x], [0, y])
            self.markers[i].set_data([x], [y])
            self.labels[i].set_position((x + 0.18, y + 0.18))

    def _press(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or ev.button != 1:
            return
        best = min(self.layout.rx.items(), key=lambda kv: math.hypot(kv[1][0] - ev.xdata, kv[1][1] - ev.ydata))
        if math.hypot(best[1][0] - ev.xdata, best[1][1] - ev.ydata) <= GRAB_RADIUS_M:
            self._drag = best[0]

    def _motion(self, ev):
        if self._drag is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        R = self.layout.radius
        x, y = float(np.clip(ev.xdata, -R, R)), float(np.clip(ev.ydata, -R, R))
        self.layout.rx[self._drag] = (x, y)
        self.links[self._drag].set_data([0, x], [0, y])
        self.markers[self._drag].set_data([x], [y])
        self.labels[self._drag].set_position((x + 0.18, y + 0.18))
        self.fig.canvas.draw_idle()

    def _release(self, ev):
        if self._drag is not None:
            self._drag = None
            self.layout.origin = "edited"
            self.rebuild()
            if self.on_moved:
                self.on_moved()

    # ----------------------------------------------------------- update
    def update(self, st, message: str | None = None):
        bm = st.baseline
        floor = self.cfg["localization"]["activity_floor"]
        act = {i: max(0.0, rc.ratio - floor) for i, rc in bm.rx.items() if np.isfinite(rc.ratio)} \
            if bm.state == "ready" else {}
        hm = self.est.estimate(act)
        a = self.dcfg["heat_smoothing_alpha"]
        target = hm.values * min(1.0, hm.info.get("strength", 0) / 3.0) if hm.info.get("present") \
            else np.zeros_like(hm.values)
        self._heat = target if self._heat is None else a * target + (1 - a) * self._heat
        self.im.set_data(self._heat)

        now = st.now
        est = hm.info.get("estimate")
        if est is not None:
            pa = self.dcfg["position_smoothing_alpha"]
            self._pos = est if self._pos is None else (pa * est[0] + (1 - pa) * self._pos[0],
                                                       pa * est[1] + (1 - pa) * self._pos[1])
            self._trail.append((now, *self._pos))
        else:
            self._pos = None
        while self._trail and now - self._trail[0][0] > self.dcfg["trail_seconds"]:
            self._trail.popleft()
        tr = np.array(self._trail) if self._trail else np.zeros((0, 3))
        self.trail.set_data(tr[:, 1], tr[:, 2])
        if self._pos is not None:
            self.you.set_data([self._pos[0]], [self._pos[1]])
            self.spread.set_center(self._pos)
            self.spread.set_radius(1.0)            # honest scale: the real error is about a metre or more
            self.spread.set_visible(True)
            self.you_txt.set_position((self._pos[0] + 0.25, self._pos[1] - 0.35))
            self.you_txt.set_text("you? (approx.)")
        else:
            self.you.set_data([], [])
            self.spread.set_visible(False)
            self.you_txt.set_text("")
        truth = st.truth
        self.truth.set_data(*(([truth[0]], [truth[1]]) if truth is not None else ([], [])))

        if message:
            text = message
        elif bm.state == "idle":
            text = "Step 1: click 'Record empty room', then leave the room"
        elif bm.state in ("waiting", "recording"):
            text = f"Recording the EMPTY room … {bm.progress(now) * 100:3.0f} %  (stay out)"
        elif est is None:
            text = "No movement detected (links look like the empty room)"
        else:
            names = {r["id"]: r["name"] for r in self.rx_cfg}
            strong = [names[i] for i, v in sorted(act.items(), key=lambda kv: -kv[1]) if v >= 1.0]
            text = (f"Movement near: {', '.join(strong) or '-'}\n"
                    f"rough position ≈ ({self._pos[0]:+.1f}, {self._pos[1]:+.1f}) m  (circle = 1 m)")
        self.status.set_text(text)


class OverviewFigure:
    """Small graphs for the narrow side panel."""

    def __init__(self, cfg: dict, fig):
        self.cfg, self.fig = cfg, fig
        self.rx = cfg["receivers"]
        self.hist = cfg["visualization"]["history_seconds"] * 3
        self.ax_bar, self.ax_chg, self.ax_var = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
        fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.07, hspace=0.6)
        names = [r["name"] for r in self.rx]
        cols = [rx_color(c) for c in range(len(self.rx))]
        self.bars = self.ax_bar.bar(names, [0] * len(names), color=cols)
        self.ax_bar.axhline(1, color="k", ls="--", lw=0.8)
        self.ax_bar.set_title("change vs empty room (1 = empty)", fontsize=8)
        self.chg = {r["id"]: self.ax_chg.plot([], [], color=cols[c], lw=1)[0] for c, r in enumerate(self.rx)}
        self.ax_chg.axhline(1, color="k", ls="--", lw=0.8)
        self.ax_chg.set_title("change over time", fontsize=8)
        self.ax_chg.set_xlim(-self.hist, 0)
        self.var = {r["id"]: self.ax_var.plot([], [], color=cols[c], lw=1, label=r["name"])[0]
                    for c, r in enumerate(self.rx)}
        self.ax_var.set_title("CSI amplitude, % of its 10 s median (no baseline needed)", fontsize=8)
        self.ax_var.set_xlim(-cfg["visualization"]["history_seconds"], 0)
        self.ax_var.legend(fontsize=6, loc="upper left", ncol=3)
        for ax in (self.ax_bar, self.ax_chg, self.ax_var):
            ax.tick_params(labelsize=7)

    def update(self, st):
        top = 2.0
        for bar, r in zip(self.bars, self.rx):
            v = st.baseline.rx[r["id"]].ratio if st.baseline.state == "ready" else 0.0
            v = v if np.isfinite(v) else 0.0
            bar.set_height(v)
            top = max(top, v * 1.15)
        self.ax_bar.set_ylim(0, top)
        top = 2.0
        for r in self.rx:
            t, v = st.change_history(r["id"])
            self.chg[r["id"]].set_data(t - st.now, v)
            if len(v):
                top = max(top, float(np.nanmax(v)) * 1.1)
        self.ax_chg.set_ylim(0, top)
        lo, hi = 95.0, 105.0
        for r in self.rx:
            hist = list(st.pipelines[r["id"]].history)
            if len(hist) < 5:
                continue
            t = np.array([h.t for h in hist]) - st.now
            m = np.array([np.nanmean(h.amp_proc) for h in hist])
            y = 100 * m / np.nanmedian(m)
            self.var[r["id"]].set_data(t, y)
            if np.isfinite(y).any():
                lo, hi = min(lo, np.nanpercentile(y, 1)), max(hi, np.nanpercentile(y, 99))
        self.ax_var.set_ylim(lo - 2, hi + 2)
