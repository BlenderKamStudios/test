"""Per-receiver processing chain: raw record -> Frame (amplitude, phase, ...).

    raw int8 CSI
      -> complex H, subcarriers -32..31            (csi_math.to_complex)
      -> validity mask (null subcarriers, first_word_invalid)
      -> amplitude |H|                              = "raw amplitude"
      -> AGC normalisation (none/mean/rssi)         optional
      -> Hampel outlier filter over time            optional
      -> temporal smoothing (EMA / moving average)  optional  = "processed amplitude"
    phase: raw angle(H) and sanitized phase (per packet, no time smoothing)

Every step is controlled by the `preprocessing` section of config.yaml.
The raw record is never modified.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from common.records import CsiRecord
from . import csi_math
from .filters import Hampel, Identity, make_smoother

warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")


@dataclass
class Frame:
    t: float                  # laptop time of the packet (s)
    tx_seq: int
    rssi: int
    valid: np.ndarray         # (64,) bool: subcarrier usable in this packet
    amp_raw: np.ndarray       # (64,) |H|, raw ESP32 units
    amp_proc: np.ndarray      # (64,) processed amplitude (NaN where invalid)
    phase_raw: np.ndarray     # (64,) radians, -pi..pi
    phase_clean: np.ndarray   # (64,) sanitized phase (NaN where invalid), or raw if disabled


class ReceiverPipeline:
    def __init__(self, pcfg: dict, history_seconds: float = 10.0, max_rate_hz: float = 500.0):
        self.cfg = pcfg
        self.part = pcfg.get("csi_part", "lltf")
        self.base_valid = csi_math.used_mask(self.part) if pcfg.get("drop_null_subcarriers", True) \
            else np.ones(csi_math.N_SUBCARRIERS, bool)
        h = pcfg.get("hampel", {})
        self.hampel = Hampel(h.get("window", 11), h.get("n_sigmas", 3.0)) if h.get("enabled") else Identity()
        self.smoother = make_smoother(pcfg.get("smoothing", {"method": "none"}))
        self.history = deque(maxlen=int(history_seconds * max_rate_hz))
        self.history_seconds = history_seconds
        self.last: Frame | None = None
        self.skipped = 0          # records without the requested CSI part

    def process(self, rec: CsiRecord) -> Frame | None:
        h = csi_math.to_complex(rec.csi_raw, self.part)
        if h is None:
            self.skipped += 1
            return None
        valid = self.base_valid.copy()
        if self.cfg.get("mask_first_word_invalid", True) and rec.first_word_invalid:
            valid &= ~csi_math.invalid_first_word_mask(True)
        amp_raw = csi_math.amplitude(h)
        valid &= amp_raw > 0                              # all-zero entries carry no information

        amp = csi_math.normalize_amplitude(amp_raw, valid, self.cfg.get("amplitude_normalization", "none"),
                                           rssi=rec.rssi)
        amp = np.where(valid, amp, np.nan)
        amp = self.smoother(self.hampel(amp))

        phase_raw = csi_math.raw_phase(h)
        if self.cfg.get("phase_sanitization", "none") == "linear" and valid.sum() >= 4:
            phase_clean = csi_math.sanitize_phase(h, valid)
        else:
            phase_clean = np.where(valid, phase_raw, np.nan)

        f = Frame(rec.host_time, rec.tx_seq, rec.rssi, valid, amp_raw, amp, phase_raw, phase_clean)
        self.history.append(f)
        while self.history and f.t - self.history[0].t > self.history_seconds:
            self.history.popleft()
        self.last = f
        return f
