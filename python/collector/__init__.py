"""Data sources (serial / replay / simulation), the hub, and the raw writer."""
from __future__ import annotations

import argparse

from .hub import Hub
from .writer import SessionWriter


def add_source_args(ap: argparse.ArgumentParser, default_save: bool):
    ap.add_argument("--config", help="config file (default config/config.yaml)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--replay", metavar="PATH", help="replay a recorded session folder or csi.csv(.gz)")
    src.add_argument("--sim", action="store_true", help="use simulated data (no hardware)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed factor (default 1 = real time)")
    ap.add_argument("--port", action="append", default=[], metavar="ID=PORT",
                    help="override a receiver's port, e.g. --port 1=/dev/ttyUSB0 --port 2=COM6")
    ap.add_argument("--label", default="", help="text appended to the session folder name")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--save", dest="save", action="store_true", default=default_save,
                   help="save raw data to data/raw/<session>/" + (" (default)" if default_save else ""))
    g.add_argument("--no-save", dest="save", action="store_false", help="do not save")


def apply_port_overrides(cfg: dict, overrides: list[str]):
    for o in overrides:
        rx_id, _, port = o.partition("=")
        for r in cfg["receivers"]:
            if str(r["id"]) == rx_id.strip():
                r["port"] = port.strip()
                break
        else:
            raise SystemExit(f"--port {o}: no receiver with id {rx_id}")


def make_source(cfg: dict, args, hub: Hub):
    if args.replay:
        from .replay_source import ReplaySource
        return ReplaySource(cfg, hub, args.replay, args.speed)
    if args.sim:
        from .sim_source import SimSource
        return SimSource(cfg, hub)
    from .serial_source import SerialSource
    return SerialSource(cfg, hub)


def source_name(args) -> str:
    return f"replay:{args.replay}" if args.replay else "sim" if args.sim else "serial"


def make_writer(cfg: dict, args):
    if not args.save:
        return None
    label = args.label or ("sim" if args.sim else "replay" if args.replay else "")
    return SessionWriter(cfg, label, source_name(args))
