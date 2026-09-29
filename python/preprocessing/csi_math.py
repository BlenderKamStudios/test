"""From raw ESP32 CSI bytes to amplitude and phase.

Terminology used throughout the project
---------------------------------------
raw CSI        the int8 list the ESP32 sends: [imag, real, imag, real, ...]
               in the ESP32's buffer order. Stored unchanged in the CSV.
complex CSI    H[k] = real + j*imag for each subcarrier k, re-ordered to
               k = -32..31. Still "raw" physically, just re-arranged.
amplitude      |H[k]|. Changes when a person blocks or adds reflections.
phase (raw)    angle(H[k]) in radians. On the ESP32 it is dominated by
               random per-packet offsets (the TX and RX clocks are not
               synchronised: carrier frequency offset, sampling time offset,
               PLL phase). Raw phase jumps randomly from packet to packet.
phase (sanitized)
               raw phase, unwrapped along the subcarriers, with the best-fit
               straight line (offset + slope) removed. The random offsets
               mostly add exactly such a line, so what remains is the SHAPE
               of the phase across frequency, which is more stable.
RSSI           one number per packet (dBm) from the radio, NOT computed from
               CSI. Coarse (1 dB steps) and affected by AGC.
processed CSI  anything after normalisation / filtering / smoothing.
"""
from __future__ import annotations

import numpy as np

N_SUBCARRIERS = 64
SUBCARRIERS = np.arange(-32, 32)          # natural order, index i <-> subcarrier i-32

# Subcarriers that actually carry energy in each training field (20 MHz):
#   L-LTF  (802.11a/g):  -26..-1, 1..26   (52 used; DC and edges are empty)
#   HT-LTF (802.11n):    -28..-1, 1..28   (56 used)
_USED = {
    "lltf": (np.abs(SUBCARRIERS) <= 26) & (SUBCARRIERS != 0),
    "htltf": (np.abs(SUBCARRIERS) <= 28) & (SUBCARRIERS != 0),
}
# Byte offset of each part inside the ESP32 CSI buffer (secondary channel none)
_PART_OFFSET = {"lltf": 0, "htltf": 128}


def used_mask(part: str = "lltf") -> np.ndarray:
    return _USED[part].copy()


def to_complex(csi_raw, part: str = "lltf") -> np.ndarray | None:
    """Raw int8 list/array -> complex H of shape (64,) for subcarriers -32..31.

    ESP32 buffer layout (20 MHz, no secondary channel), per part:
        bytes 2k, 2k+1 = imag, real of buffer entry k (k = 0..63)
        buffer entry 0..31  -> subcarrier 0..31
        buffer entry 32..63 -> subcarrier -32..-1
    Returns None if the requested part is not in this record (e.g. HT-LTF of
    a legacy 11g packet).
    """
    raw = np.asarray(csi_raw, dtype=np.float64)
    off = _PART_OFFSET[part]
    if raw.shape[-1] < off + 128:
        return None
    seg = raw[..., off:off + 128]
    imag, real = seg[..., 0::2], seg[..., 1::2]
    h_buf = real + 1j * imag                        # buffer order
    return np.concatenate([h_buf[..., 32:], h_buf[..., :32]], axis=-1)   # -> -32..31


def invalid_first_word_mask(first_word_invalid) -> np.ndarray:
    """Buffer entries 0 and 1 (= subcarriers 0 and 1) are garbage when the
    ESP32 sets first_word_invalid. Returns a (..., 64) bool mask of BAD values."""
    fwi = np.asarray(first_word_invalid, dtype=bool)
    bad = np.zeros(fwi.shape + (N_SUBCARRIERS,), dtype=bool)
    bad[..., 32] = fwi          # subcarrier 0
    bad[..., 33] = fwi          # subcarrier 1
    return bad


def amplitude(h: np.ndarray) -> np.ndarray:
    return np.abs(h)


def normalize_amplitude(amp: np.ndarray, valid: np.ndarray, method: str, rssi=None) -> np.ndarray:
    """Remove the per-packet gain changes of the receiver's AGC.

    none : unchanged
    mean : divide by the mean amplitude over valid subcarriers -> the result
           is a dimensionless "shape" around 1.0; overall level changes vanish.
    rssi : rescale so the average power matches the packet RSSI, giving a
           linear amplitude that follows the true received level. Unit:
           amplitude relative to -100 dBm, i.e. 10 = -80 dBm, 100 = -60 dBm.
    """
    if method == "none":
        return amp
    mean = np.nanmean(np.where(valid, amp, np.nan), axis=-1, keepdims=True)
    mean = np.where(mean > 0, mean, np.nan)
    if method == "mean":
        return amp / mean
    if method == "rssi":
        if rssi is None:
            raise ValueError("rssi normalisation needs the RSSI")
        power = np.nanmean(np.where(valid, amp, np.nan) ** 2, axis=-1, keepdims=True)
        target = 10 ** ((np.asarray(rssi, dtype=float)[..., None] + 100) / 20)
        return amp * target / np.sqrt(np.where(power > 0, power, np.nan))
    raise ValueError(f"unknown amplitude_normalization {method!r}")


def raw_phase(h: np.ndarray) -> np.ndarray:
    return np.angle(h)


def sanitize_phase(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unwrap along the valid subcarriers and subtract the least-squares line
    a*k + b. Invalid subcarriers become NaN. Works on (64,) or (N, 64)."""
    h2 = np.atleast_2d(h)
    out = np.full(h2.shape, np.nan)
    k = SUBCARRIERS[valid].astype(float)
    ph = np.unwrap(np.angle(h2[:, valid]), axis=-1)
    A = np.vstack([k, np.ones_like(k)]).T                  # design matrix for a*k + b
    coef, *_ = np.linalg.lstsq(A, ph.T, rcond=None)        # (2, N)
    out[:, valid] = ph - (A @ coef).T
    return out[0] if h.ndim == 1 else out
