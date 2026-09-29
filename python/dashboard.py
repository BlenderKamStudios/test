"""One-window dashboard: TX-centred 2D map + graphs, with a swap button.

    python python/dashboard.py                  # live from the receivers (raw data is saved)
    python python/dashboard.py --sim            # simulated person walking, no hardware
    python python/dashboard.py --replay data/raw/<session>
    python python/dashboard.py --baseline-file data/raw/<session>/baseline-....npz

Layout: the map takes most of the window, the graphs a narrow strip.
"⇄ Swap" reverses that (graphs big, map small); the divider can also be dragged.

Workflow:
  1. Drag the RX squares on the map to where the boards really stand
     (TX = centre), or click "Auto-place" for a rough RSSI distance guess;
     "Save positions" stores them in config/positions.yaml.
  2. "Record empty room": you get 10 s to leave, then 30 s are recorded.
  3. Walk around: the map shows an APPROXIMATE, EXPERIMENTAL position.

Requires PyQt6 (pip install PyQt6).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from collector import Hub, add_source_args, apply_port_overrides, make_source, make_writer  # noqa: E402
from common.config import load_config, project_path  # noqa: E402
from common.layout import load_layout, place_by_rssi, save_layout  # noqa: E402
from visualization.dashboard_figures import OverviewFigure, PositionMap  # noqa: E402
from visualization.figures import ChangeFigure, CsiFigure  # noqa: E402
from visualization.state import LiveState  # noqa: E402


def sim_config(cfg, layout):
    """The simulator works in room coordinates starting at (0, 0): shift the
    TX-centred layout by +R so the simulated room matches the map."""
    import copy
    R = layout.radius
    c = copy.deepcopy(cfg)
    c["room"] = {"width": 2 * R, "height": 2 * R, "tx": {"x": R, "y": R}}
    for r in c["receivers"]:
        r["x"], r["y"] = layout.rx[r["id"]][0] + R, layout.rx[r["id"]][1] + R
    return c


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        super().__init__(Figure())
        self.setMinimumSize(120, 120)


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, cfg, args, hub, source, writer, state, layout, sim_shift=0.0):
        super().__init__()
        self.cfg, self.args, self.hub, self.source, self.writer, self.st = cfg, args, hub, source, writer, state
        self.layout_ = layout
        self.sim_shift = sim_shift
        self.dcfg = cfg["dashboard"]
        self.map_big = True
        self.countdown_end = None
        self.message, self.message_until = None, 0.0
        self.baseline_saved = bool(args.baseline_file)
        self.setWindowTitle("Wi-Fi CSI sensing — dashboard (experimental)")
        self.resize(1500, 860)

        # ------------------------------------------------------------ toolbar
        bar = QtWidgets.QHBoxLayout()
        def button(text, slot, tip):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            bar.addWidget(b)
            return b
        self.swap_btn = button("⇄  Swap map / graphs", self.swap, "Give most of the window to the other side")
        self.base_btn = button("Record empty room", self.record_baseline,
                               f"Countdown {self.dcfg['baseline_start_delay_seconds']} s to leave the room, then "
                               f"{cfg['baseline']['duration_seconds']} s recording")
        button("Auto-place RXs (RSSI)", self.auto_place,
               "Estimate each receiver's DISTANCE from its signal strength (rough). Directions are kept.")
        button("Save positions", self.save_positions, "Store receiver positions in " + self.dcfg["positions_file"])
        bar.addStretch(1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("font-family: monospace;")
        bar.addWidget(self.status)

        # ------------------------------------------------------------ panels
        self.map_canvas = Canvas()
        self.map = PositionMap(cfg, self.map_canvas.figure, layout, on_moved=self.on_moved)
        self.tabs = QtWidgets.QTabWidget()
        self.over_canvas, self.csi_canvas, self.chg_canvas = Canvas(), Canvas(), Canvas()
        self.overview = OverviewFigure(cfg, self.over_canvas.figure)
        self.csi = CsiFigure(cfg, self.csi_canvas.figure)
        self.change = ChangeFigure(cfg, self.chg_canvas.figure)
        self.tabs.addTab(self.over_canvas, "Overview")
        self.tabs.addTab(self.csi_canvas, "CSI details")
        self.tabs.addTab(self.chg_canvas, "Change")
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.map_canvas)
        self.splitter.addWidget(self.tabs)
        self.splitter.setChildrenCollapsible(False)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(bar)
        v.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        QtCore.QTimer.singleShot(0, self.apply_sizes)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.dcfg["update_interval_ms"])

    # ------------------------------------------------------------ actions
    def apply_sizes(self):
        w = max(self.splitter.width(), 800)
        big, small = int(w * 0.74), int(w * 0.26)
        self.splitter.setSizes([big, small] if self.map_big else [small, big])
        self.tabs.setCurrentIndex(0 if self.map_big else 1)
        self.swap_btn.setText("⇄  Swap: big graphs" if self.map_big else "⇄  Swap: big map")

    def swap(self):
        self.map_big = not self.map_big
        self.apply_sizes()

    def say(self, text, seconds=6.0):
        self.message, self.message_until = text, time.time() + seconds
        print("  *", text)

    def record_baseline(self):
        delay = 0 if (self.args.sim or self.args.replay) else self.dcfg["baseline_start_delay_seconds"]
        self.countdown_end = time.time() + delay
        self.baseline_saved = False
        self.st.baseline.state = "idle"

    def auto_place(self):
        med = {}
        for i, p in self.st.pipelines.items():
            r = [h.rssi for h in list(p.history)]
            if len(r) >= 20:
                med[i] = float(np.median(r))
        if not med:
            self.say("Auto-place: no data yet")
            return
        est = place_by_rssi(self.cfg, self.layout_, med)
        self.map.rebuild()
        names = {r["id"]: r["name"] for r in self.cfg["receivers"]}
        self.say("Auto-place (rough!): " + ", ".join(
            f"{names[i]} {med[i]:.0f} dBm ≈ {d:.1f} m" for i, d in est.items()) +
            ".  Now drag each RX to its real DIRECTION, then Save.", 12)

    def save_positions(self):
        self.say(f"Positions saved to {save_layout(self.cfg, self.layout_)}")
        self.layout_.origin = "file"

    def on_moved(self):
        self.say("Receiver moved — click 'Save positions' to keep it", 4)

    # --------------------------------------------------------------- loop
    def tick(self):
        st = self.st
        for rec in self.hub.drain():
            st.feed(rec)
        if not (self.args.sim or self.args.replay):
            st.now = time.time()
        if self.countdown_end is not None:
            left = self.countdown_end - time.time()
            if left <= 0:
                self.countdown_end = None
                st.baseline.start()
                self.say("Recording the empty room — stay out")
            else:
                self.message, self.message_until = f"LEAVE THE ROOM — empty-room recording starts in {left:.0f} s", \
                    time.time() + 1
        if st.baseline.state == "ready" and not self.baseline_saved:
            self.baseline_saved = True
            out = (self.writer.dir if self.writer else project_path("data/processed")) / \
                f"baseline-{time.strftime('%Y%m%d-%H%M%S')}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            st.baseline.save(out)
            self.say(f"Empty room recorded (saved {out.name}). Come back in and move around.")
            for i, e in st.baseline.errors.items():
                self.say(f"baseline for RX{i} failed: {e}", 10)
        if self.source_sim is not None:
            t = self.source_sim.truth(st.now)
            st.truth = None if t is None else (t[0] - self.sim_shift, t[1] - self.sim_shift)
        for e in self.hub.pop_events():
            print("  *", e)
        st.stats_text = "\n".join(self.hub.summary_lines() + [""] + list(self.hub.events)[-4:])

        msg = self.message if time.time() < self.message_until else None
        self.map.update(st, msg)
        self.map_canvas.draw_idle()
        figs = {0: (self.overview, self.over_canvas), 1: (self.csi, self.csi_canvas),
                2: (self.change, self.chg_canvas)}
        fig, canvas = figs[self.tabs.currentIndex()]
        fig.update(st)
        canvas.draw_idle()
        with self.hub.lock:
            rates = "  ".join(f"{s.name}:{s.rate_hz:4.0f}/s" + ("" if s.connected else "(off)")
                              for s in self.hub.stats.values())
        self.status.setText(rates + (f"   saving: {self.writer.rows} rows" if self.writer else ""))

    source_sim = None

    def closeEvent(self, ev):
        self.timer.stop()
        self.hub.stop_event.set()
        if self.writer:
            with self.hub.lock:
                self.writer.close(self.hub.stats)
            print(f"Saved {self.writer.rows} records to {self.writer.dir}")
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap, default_save=None)
    ap.add_argument("--baseline-file", help="load a saved empty-room baseline")
    ap.add_argument("--screenshot", metavar="PNG", help="(testing) save the window after --duration s and quit")
    ap.add_argument("--duration", type=float, default=30)
    args = ap.parse_args()
    if args.save is None:
        args.save = not (args.replay or args.sim)

    cfg = load_config(args.config)
    apply_port_overrides(cfg, args.port)
    layout = load_layout(cfg)
    writer = make_writer(cfg, args)
    hub = Hub(cfg, writer)
    src_cfg = sim_config(cfg, layout) if args.sim else cfg
    source = make_source(src_cfg, args, hub)
    label = f"replay {args.replay}" if args.replay else "simulation" if args.sim else "serial"
    st = LiveState(cfg, label, sim=source if args.sim else None)
    if args.baseline_file:
        st.baseline.load(args.baseline_file)

    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard(cfg, args, hub, source, writer, st, layout, sim_shift=layout.radius if args.sim else 0.0)
    win.source_sim = source if args.sim else None
    if (args.sim or args.replay) and not args.baseline_file:
        win.record_baseline()                  # simulation/replay start with an empty room
    win.show()
    # Ctrl+C / kill in the terminal: close the window properly (saves the recording).
    # The idle timer lets Python notice signals while Qt's event loop runs.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: win.close())
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(300)
    source.start()
    if writer:
        print(f"Saving raw data to {writer.csv_path}")
    if args.screenshot:
        def shot():
            win.grab().save(args.screenshot)
            print("saved", args.screenshot)
            win.close()
        QtCore.QTimer.singleShot(int(args.duration * 1000), shot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
