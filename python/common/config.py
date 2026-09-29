"""Loading config/config.yaml.

Every tool calls load_config(); nothing else in the code base hard-codes
channel numbers, ports, positions or processing parameters.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["_path"] = str(path)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    ids = [r["id"] for r in cfg["receivers"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"receiver ids must be unique, got {ids}")
    for r in cfg["receivers"]:
        for key in ("id", "name", "port", "x", "y"):
            if key not in r:
                raise ValueError(f"receiver {r} is missing '{key}'")
    if cfg["serial"]["baud"] != cfg["firmware"]["rx"]["baud"]:
        raise ValueError("serial.baud must equal firmware.rx.baud")
    if cfg["firmware"]["tx"]["phy"] not in ("ht20", "11g"):
        raise ValueError("firmware.tx.phy must be 'ht20' or '11g' (11b gives no CSI)")


def project_path(p: str | Path) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def receiver_names(cfg: dict) -> dict[int, str]:
    return {r["id"]: r["name"] for r in cfg["receivers"]}
