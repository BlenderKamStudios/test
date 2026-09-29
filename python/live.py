"""Stages 4-6: live CSI display, empty-room baseline / change, experimental heatmap.

    python python/live.py                          # receivers from config.yaml, all views, saves raw data
    python python/live.py --views csi              # stage 4 only (no baseline)
    python python/live.py --views csi,change       # stage 5
    python python/live.py --views change,heatmap   # stage 6
    python python/live.py --sim                    # no hardware: simulated room
    python python/live.py --replay data/raw/20260929-153000_walk --speed 2
    python python/live.py --baseline-file data/processed/baseline-....npz   # reuse a saved baseline
    python python/live.py --sim --snapshot out.png --duration 70            # headless: render PNGs

Keys (in any window): b = record a new empty-room baseline, q = quit.
"""
from __future__ import annotations

import argparse
import sys
import time

from collector import Hub, add_source_args, apply_port_overrides, make_source, make_writer
from common.config import load_config, project_path
from localization import make_estimator
from visualization.state import LiveState


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap, default_save=None)
    ap.add_argument("--views", default="csi,change,heatmap", help="comma list of csi, change, heatmap")
    ap.add_argument("--baseline-file", help="load a saved baseline instead of recording one")
    ap.add_argument("--snapshot", metavar="PNG", help="no window: process --duration seconds, save figures")
    ap.add_argument("--duration", type=float, default=60, help="seconds of data for --snapshot")
    args = ap.parse_args()
    if args.save is None:                       # save real measurements by default, not replays/simulations
        args.save = not (args.replay or args.sim)
    views = {v.strip() for v in args.views.split(",") if v.strip()}
    if not views <= {"csi", "change", "heatmap"}:
        ap.error(f"unknown view in {args.views}")

    cfg = load_config(args.config)
    apply_port_overrides(cfg, args.port)

    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not args.snapshot and matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo"):
        sys.exit("No interactive matplotlib backend available.\n"
                 "  Linux (Arch):   sudo pacman -S tk        Debian/Ubuntu: sudo apt install python3-tk\n"
                 "  or:             pip install PyQt6\n"
                 "  Windows/macOS:  the python.org installer includes Tk.")

    writer = make_writer(cfg, args)
    hub = Hub(cfg, writer)
    source = make_source(cfg, args, hub)
    label = f"replay {args.replay}" if args.replay else "simulation" if args.sim else "serial"
    st = LiveState(cfg, label, sim=source if args.sim else None)

    need_baseline = bool(views & {"change", "heatmap"})
    if args.baseline_file:
        st.baseline.load(args.baseline_file)
    elif need_baseline:
        st.baseline.start()
        print(f"Recording the empty-room baseline for {cfg['baseline']['duration_seconds']} s "
              "(data time) as soon as data arrives — keep the room EMPTY.")

    from visualization.figures import ChangeFigure, CsiFigure, HeatmapFigure
    figs = []
    if "csi" in views:
        figs.append(CsiFigure(cfg))
    if "change" in views:
        figs.append(ChangeFigure(cfg))
    if "heatmap" in views:
        figs.append(HeatmapFigure(cfg, make_estimator(cfg)))

    baseline_saved = [bool(args.baseline_file)]

    def after_feed():
        if st.baseline.state == "ready" and not baseline_saved[0]:
            baseline_saved[0] = True
            out = (writer.dir if writer else project_path("data/processed")) / \
                f"baseline-{time.strftime('%Y%m%d-%H%M%S')}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            st.baseline.save(out)
            hub.event(f"baseline ready, saved to {out}")
            for i, e in st.baseline.errors.items():
                hub.event(f"baseline for RX{i} failed: {e}")
        if st.sim is not None:
            st.truth = st.sim.truth(st.now)
        st.stats_text = "\n".join(hub.summary_lines() + [""] + list(hub.events)[-4:])

    # ---------------------------------------------------------- snapshot mode
    if args.snapshot:
        if not (args.sim or args.replay):
            sys.exit("--snapshot needs --sim or --replay")
        for rx in hub.stats:
            hub.set_connection(rx, "sim" if args.sim else "replay", True)
        recs = source.records(duration=args.duration) if args.sim else source.records()
        t_first = None
        for rec in recs:
            t_first = t_first or rec.host_time
            if rec.host_time - t_first > args.duration:
                break
            hub.on_record(rec)
            hub.queue.clear()
            st.feed(rec)
            if st.baseline.state == "ready" and not baseline_saved[0]:
                after_feed()
        after_feed()
        base = args.snapshot.removesuffix(".png")
        for fig in figs:
            fig.update(st)
            out = f"{base}-{fig.key}.png"
            fig.fig.savefig(out, dpi=90)
            print("saved", out)
        if writer:
            writer.close(hub.stats)
        return 0

    # -------------------------------------------------------------- live mode
    def on_key(ev):
        if ev.key == "b":
            st.baseline.start()
            baseline_saved[0] = False
            hub.event("recording a new baseline — keep the room empty")
        elif ev.key == "q":
            plt.close("all")

    for f in figs:
        f.fig.canvas.mpl_connect("key_press_event", on_key)

    def tick():
        for rec in hub.drain():
            st.feed(rec)
        if st.source == "serial":
            st.now = time.time()
        after_feed()
        for e in hub.pop_events():
            print("  *", e)
        for f in figs:
            f.update(st)
            f.fig.canvas.draw_idle()

    timer = figs[0].fig.canvas.new_timer(interval=cfg["visualization"]["update_interval_ms"])
    timer.add_callback(tick)
    timer.start()
    source.start()
    if writer:
        print(f"Saving raw data to {writer.csv_path}")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop_event.set()
        time.sleep(0.2)
        if writer:
            with hub.lock:
                writer.close(hub.stats)
            print(f"Saved {writer.rows} records to {writer.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
