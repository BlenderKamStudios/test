"""Matplotlib figures for the live display. Each figure has update(state).

CsiFigure      stage 4: per receiver amplitude vs subcarrier, amplitude vs
               time, phase vs subcarrier; RSSI of all receivers; link stats
ChangeFigure   stage 5: CSI change vs the empty-room baseline
HeatmapFigure  stage 6: EXPERIMENTAL spatial map
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from preprocessing.csi_math import SUBCARRIERS

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
RAW = "#9a9a9a"


def rx_color(i: int) -> str:
    return COLORS[i % len(COLORS)]


class CsiFigure:
    key = "csi"

    def __init__(self, cfg: dict, fig=None):
        self.cfg = cfg
        self.rx = cfg["receivers"]
        vcfg = cfg["visualization"]
        self.hist_s = vcfg["history_seconds"]
        self.time_sc = vcfg["time_subcarriers"]
        n = len(self.rx)
        self.fig = fig if fig is not None else plt.figure("CSI — live", figsize=(5 * n, 10))
        gs = GridSpec(4, n * 2, figure=self.fig, height_ratios=[1, 1, 1, 0.9], hspace=0.6, wspace=0.9)
        self.lines = {}
        for c, r in enumerate(self.rx):
            i, col = r["id"], rx_color(c)
            ax_a = self.fig.add_subplot(gs[0, 2 * c:2 * c + 2])
            ax_a.set_title(f"{r['name']}: amplitude vs subcarrier", fontsize=10)
            raw_line, = ax_a.plot(SUBCARRIERS, np.full(64, np.nan), ".-", color=RAW, lw=0.8, ms=3,
                                  label="raw |H| (left axis, ESP32 units)")
            ax_a.set_xlabel("subcarrier")
            ax_a.set_xlim(-32, 31)
            ax_a.tick_params(axis="y", labelcolor="#666")
            ax_p = ax_a.twinx()                  # right axis: processed amplitude (other units)
            proc_line, = ax_p.plot(SUBCARRIERS, np.full(64, np.nan), "-", color=col, lw=1.8,
                                   label="processed amplitude (right axis)")
            ax_p.tick_params(axis="y", labelcolor=col)

            ax_t = self.fig.add_subplot(gs[1, 2 * c:2 * c + 2])
            ax_t.set_title(f"{r['name']}: processed amplitude vs time", fontsize=10)
            t_lines = [ax_t.plot([], [], lw=1, label=f"sc {k}")[0] for k in self.time_sc]
            ax_t.set_xlim(-self.hist_s, 0)
            ax_t.set_xlabel("seconds (0 = now)")
            if c == 0:
                ax_t.legend(fontsize=7, loc="lower left", ncol=len(self.time_sc), framealpha=0.6)

            ax_ph = self.fig.add_subplot(gs[2, 2 * c:2 * c + 2])
            ax_ph.set_title(f"{r['name']}: phase vs subcarrier", fontsize=10)
            ph_raw, = ax_ph.plot(SUBCARRIERS, np.full(64, np.nan), ".", color=RAW, ms=3, label="raw phase")
            ph_clean, = ax_ph.plot(SUBCARRIERS, np.full(64, np.nan), "-", color=col, lw=1.5,
                                   label="sanitized phase" if cfg["preprocessing"]["phase_sanitization"] != "none"
                                   else "phase (valid sc)")
            ax_ph.set_ylim(-np.pi * 1.05, np.pi * 1.05)
            ax_ph.set_xlim(-32, 31)
            ax_ph.set_xlabel("subcarrier")
            ax_ph.set_ylabel("rad")
            self.lines[i] = dict(ax_a=ax_a, ax_p=ax_p, raw=raw_line, proc=proc_line,
                                 ax_t=ax_t, t_lines=t_lines, ph_raw=ph_raw, ph_clean=ph_clean)

        self.ax_rssi = self.fig.add_subplot(gs[3, :max(2, n * 2 - 2)])
        self.ax_rssi.set_title("RSSI (whole-packet signal strength, dBm) — not CSI", fontsize=10)
        self.ax_rssi.set_xlim(-self.hist_s, 0)
        self.ax_rssi.set_xlabel("seconds (0 = now)")
        self.rssi_lines = {r["id"]: self.ax_rssi.plot([], [], color=rx_color(c), lw=1.2, label=r["name"])[0]
                           for c, r in enumerate(self.rx)}
        self.ax_rssi.legend(fontsize=7, loc="upper left")
        self.ax_txt = self.fig.add_subplot(gs[3, max(2, n * 2 - 2):])
        self.ax_txt.axis("off")
        self.txt = self.ax_txt.text(0, 1, "", va="top", family="monospace", fontsize=8)
        self.title = self.fig.suptitle("", fontsize=11)
        self.fig.text(0.5, 0.945, "grey = RAW (amplitude: left axis, ESP32 units; phase: angle of raw CSI)     "
                      "colour = PROCESSED (amplitude: right axis, "
                      f"{cfg['preprocessing']['amplitude_normalization']}-normalised + filtered; "
                      f"phase: {cfg['preprocessing']['phase_sanitization']} sanitised)",
                      ha="center", fontsize=8, color="#444")

    def update(self, st):
        now = st.now
        rssi_all = []
        for i, L in self.lines.items():
            p = st.pipelines[i]
            f = p.last
            if f is None:
                continue
            L["raw"].set_ydata(np.where(f.valid, f.amp_raw, np.nan))
            L["proc"].set_ydata(f.amp_proc)
            _autoscale(L["ax_a"], f.amp_raw[f.valid])
            _autoscale(L["ax_p"], f.amp_proc)
            L["ph_raw"].set_ydata(np.where(f.valid, f.phase_raw, np.nan))
            L["ph_clean"].set_ydata(np.clip(f.phase_clean, -np.pi, np.pi))
            hist = list(p.history)
            if hist:
                t = np.array([h.t for h in hist]) - now
                A = np.stack([h.amp_proc for h in hist])
                vals = []
                for line, k in zip(L["t_lines"], self.time_sc):
                    y = A[:, k + 32]
                    line.set_data(t, y)
                    vals.append(y)
                _autoscale(L["ax_t"], np.concatenate(vals))
                r = np.array([h.rssi for h in hist], float)
                self.rssi_lines[i].set_data(t, r)
                rssi_all.append(r)
        if rssi_all:
            _autoscale(self.ax_rssi, np.concatenate(rssi_all), pad=2)
        self.txt.set_text(st.stats_text)
        self.title.set_text(f"Live CSI   source: {st.source}   {st.clock}")


class ChangeFigure:
    key = "change"

    def __init__(self, cfg: dict, fig=None):
        self.cfg = cfg
        self.rx = cfg["receivers"]
        self.hist_s = cfg["visualization"]["history_seconds"] * 3
        self.fig = fig if fig is not None else plt.figure("CSI change vs empty-room baseline", figsize=(12, 4.5))
        self.ax_bar, self.ax_t = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
        names = [r["name"] for r in self.rx]
        self.bars = self.ax_bar.bar(names, [0] * len(names), color=[rx_color(c) for c in range(len(names))])
        self.ax_bar.axhline(1.0, color="k", ls="--", lw=1)
        self.ax_bar.set_ylabel("change / empty-room noise level")
        self.bar_txt = [self.ax_bar.text(b.get_x() + b.get_width() / 2, 0, "", ha="center", va="bottom", fontsize=9)
                        for b in self.bars]
        self.ax_t.axhline(1.0, color="k", ls="--", lw=1, label="empty-room level")
        self.t_lines = {r["id"]: self.ax_t.plot([], [], color=rx_color(c), label=r["name"])[0]
                        for c, r in enumerate(self.rx)}
        self.ax_t.set_xlim(-self.hist_s, 0)
        self.ax_t.set_xlabel("seconds (0 = now)")
        self.ax_t.set_ylabel("change ratio")
        self.ax_t.legend(fontsize=8, loc="upper left")
        self.title = self.fig.suptitle("")
        self.fig.text(0.01, 0.01, f"metric: {cfg['baseline']['metric']} on processed amplitude; "
                      "ratio 1 = typical empty-room fluctuation.  Press 'b' to re-record the baseline.",
                      fontsize=8)
        self.fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    def update(self, st):
        bm = st.baseline
        if bm.state in ("idle", "waiting", "recording"):
            prog = bm.progress(st.now)
            self.title.set_text(f"BASELINE: keep the room EMPTY — recording {prog * 100:3.0f}% "
                                f"of {bm.cfg['duration_seconds']:.0f} s" if bm.state != "idle"
                                else "No baseline. Press 'b' to record one (room must be empty).")
            self.title.set_color("#b00")
            return
        err = "; ".join(f"RX{i}: {e}" for i, e in bm.errors.items())
        self.title.set_text("CSI change vs baseline" + (f"   (baseline failed for {err})" if err else ""))
        self.title.set_color("k")
        ymax = 2.0
        for bar, txt, r in zip(self.bars, self.bar_txt, self.rx):
            v = bm.rx[r["id"]].ratio
            v = 0 if not np.isfinite(v) else v
            bar.set_height(v)
            txt.set_position((bar.get_x() + bar.get_width() / 2, v))
            txt.set_text(f"{v:.2f}")
            ymax = max(ymax, v * 1.15)
        self.ax_bar.set_ylim(0, ymax)
        for r in self.rx:
            t, v = st.change_history(r["id"])
            self.t_lines[r["id"]].set_data(t - st.now, v)
        allv = np.concatenate([st.change_history(r["id"])[1] for r in self.rx] + [np.array([2.0])])
        self.ax_t.set_ylim(0, max(2.0, np.nanmax(allv) * 1.1))


class HeatmapFigure:
    key = "heatmap"

    def __init__(self, cfg: dict, estimator):
        self.cfg, self.est = cfg, estimator
        room = cfg["room"]
        self.fig, self.ax = plt.subplots(num="EXPERIMENTAL spatial map", figsize=(7.5, 6.8))
        ext = (0, room["width"], 0, room["height"])
        self.im = self.ax.imshow(np.zeros((len(estimator.y), len(estimator.x))), origin="lower", extent=ext,
                                 cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.04, label="link-weighted change (arb. units)")
        tx = (room["tx"]["x"], room["tx"]["y"])
        for c, r in enumerate(cfg["receivers"]):
            self.ax.plot([tx[0], r["x"]], [tx[1], r["y"]], "--", color=rx_color(c), lw=1)
            self.ax.plot(r["x"], r["y"], "s", color=rx_color(c), ms=10, mec="w")
            self.ax.annotate(r["name"], (r["x"], r["y"]), xytext=(6, 6), textcoords="offset points", color="w",
                             fontsize=9, bbox=dict(fc=rx_color(c), ec="none", pad=1.5))
        self.ax.plot(*tx, "^", color="#ffd700", ms=12, mec="k")
        self.ax.annotate("TX", tx, xytext=(6, 6), textcoords="offset points", fontsize=9,
                         bbox=dict(fc="#ffd700", ec="none", pad=1.5))
        self.truth, = self.ax.plot([], [], "o", mfc="none", mec="#00e5ff", mew=2, ms=14,
                                   label="true position (simulation only)")
        self.ax.set_xlim(0, room["width"])
        self.ax.set_ylim(0, room["height"])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.title = self.ax.set_title("")
        self.fig.suptitle("EXPERIMENTAL — link-weighting map, NOT a measured position", color="#b00",
                          fontsize=11, fontweight="bold")
        self.fig.text(0.01, 0.01, f"method: {cfg['localization']['method']}. Each link paints its CSI change "
                      "onto an ellipse around the TX–RX line.\n3 links cannot determine a position; "
                      "bright = 'somewhere near these links something changed'.", fontsize=7.5)
        self._vmax = 1.0

    def update(self, st):
        bm = st.baseline
        if bm.state != "ready":
            self.title.set_text("waiting for the baseline…")
            return
        act = {r["id"]: max(0.0, bm.rx[r["id"]].ratio - self.cfg["localization"]["activity_floor"])
               for r in self.cfg["receivers"] if np.isfinite(bm.rx[r["id"]].ratio)}
        hm = self.est.estimate(act)
        self._vmax = max(1.0, 0.95 * self._vmax, float(np.nanmax(hm.values)))   # slowly decaying colour scale
        self.im.set_data(hm.values)
        self.im.set_clim(0, self._vmax)
        self.title.set_text("link activity: " + ", ".join(f"RX{k}={v:.1f}" for k, v in act.items()))
        pos = st.truth
        if pos is not None:
            self.truth.set_data([pos[0]], [pos[1]])
            if not self.ax.get_legend():
                self.ax.legend(loc="upper right", fontsize=8)
        else:
            self.truth.set_data([], [])


def _autoscale(ax, values, pad=None):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    p = pad if pad is not None else max(span * 0.1, 1e-3 + abs(hi) * 0.02)
    cur_lo, cur_hi = ax.get_ylim()
    if (hi - lo + 2 * p) < 0.2 * (cur_hi - cur_lo):    # data far smaller than the axis: snap
        ax.set_ylim(lo - p, hi + p)
        return
    # expand immediately, shrink slowly -> less jumpy plots
    new_lo = lo - p if lo - p < cur_lo else cur_lo + 0.1 * (lo - p - cur_lo)
    new_hi = hi + p if hi + p > cur_hi else cur_hi + 0.1 * (hi + p - cur_hi)
    if new_hi > new_lo:
        ax.set_ylim(new_lo, new_hi)
