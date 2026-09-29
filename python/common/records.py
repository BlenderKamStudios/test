"""The CSI record: parsing the firmware's serial lines, and the CSV row format.

A firmware line looks like (one line, split here for reading):

    CSI,1,5321,5400,123456789,1a:00:00:00:00:00,-45,-93,11,0,1,0,0,0,0,51,0,1,128,
    3 -12 0 0 5 -14 ...

Field order is defined in firmware/rx/rx.ino (printRecord) and documented in
README.md, section "Dataset format". The CSV written by the collector has the
same fields plus laptop-side metadata (host_time, receiver_id, port).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Firmware fields in line order (after the "CSI" tag, before the CSI values)
FIRMWARE_FIELDS = [
    "fw_rx_id", "rx_count", "tx_seq", "esp_timestamp_us", "src_mac", "rssi",
    "noise_floor", "channel", "secondary_channel", "sig_mode", "mcs", "cwb",
    "rate", "stbc", "sig_len", "rx_state", "first_word_invalid", "csi_len",
]
_INT_FIELDS = [f for f in FIRMWARE_FIELDS if f != "src_mac"]

# CSV columns: laptop metadata first, then firmware fields, raw CSI last.
CSV_COLUMNS = ["host_time", "receiver_id", "port"] + FIRMWARE_FIELDS + ["csi_raw"]

_MAC_RE = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")


@dataclass
class CsiRecord:
    """One CSI measurement = one received TX packet at one receiver.

    Nothing here is processed: `csi_raw` holds exactly the int8 values the
    ESP32 produced, [imag0, real0, imag1, real1, ...] in the ESP32's buffer
    order. Use preprocessing.csi_math.to_complex() to get subcarriers.
    """
    host_time: float          # laptop clock (Unix seconds) when the line arrived
    receiver_id: int          # the laptop's receiver ID (from config.yaml mapping)
    port: str                 # serial port / "replay" / "sim"
    fw_rx_id: int             # receiver ID the firmware reported (0 = not set)
    rx_count: int             # firmware counter of accepted CSI records
    tx_seq: int               # TX packet number (-1 = unknown); same packet => same number on every RX
    esp_timestamp_us: int     # RX-local time in µs (32-bit, wraps every ~71.6 min)
    src_mac: str              # transmitter MAC the CSI was measured from
    rssi: int                 # packet signal strength, dBm (one per packet, not per subcarrier)
    noise_floor: int          # RF noise floor, dBm
    channel: int
    secondary_channel: int    # 0 = none (20 MHz)
    sig_mode: int             # 0 = legacy OFDM (11g), 1 = HT (11n)
    mcs: int
    cwb: int                  # 0 = 20 MHz, 1 = 40 MHz
    rate: int                 # legacy rate code (only meaningful for sig_mode 0)
    stbc: int
    sig_len: int              # packet length in bytes
    rx_state: int             # 0 = no error
    first_word_invalid: int   # 1 = first 4 CSI bytes are garbage (ESP32 HW limitation)
    csi_len: int              # number of int8 values in csi_raw
    csi_raw: list[int] = field(repr=False)

    def to_row(self) -> list:
        return [f"{self.host_time:.6f}", self.receiver_id, self.port] + \
            [getattr(self, f) for f in FIRMWARE_FIELDS] + [" ".join(map(str, self.csi_raw))]


class ParseError(ValueError):
    pass


def parse_csi_line(line: str, host_time: float, receiver_id: int = 0, port: str = "") -> CsiRecord:
    """Parse one 'CSI,...' line from the firmware. Raises ParseError on
    corrupted lines (serial glitches can cut or merge lines)."""
    parts = line.strip().split(",")
    n = len(FIRMWARE_FIELDS)
    if len(parts) != n + 2 or parts[0] != "CSI":
        raise ParseError(f"expected {n + 2} comma-separated fields, got {len(parts)}")
    values = dict(zip(FIRMWARE_FIELDS, parts[1:n + 1]))
    try:
        ints = {k: int(values[k]) for k in _INT_FIELDS}
        csi = [int(v) for v in parts[n + 1].split()]
    except ValueError as e:
        raise ParseError(str(e)) from None
    mac = values["src_mac"].lower()
    if not _MAC_RE.match(mac):
        raise ParseError(f"bad MAC {mac!r}")
    if len(csi) != ints["csi_len"] or len(csi) % 2:
        raise ParseError(f"csi_len={ints['csi_len']} but {len(csi)} values")
    if any(v < -128 or v > 127 for v in csi):
        raise ParseError("CSI value outside int8 range")
    return CsiRecord(host_time=host_time, receiver_id=receiver_id or ints["fw_rx_id"], port=port,
                     src_mac=mac, csi_raw=csi, **ints)


def format_csi_line(r: CsiRecord) -> str:
    """Inverse of parse_csi_line (used by the simulator, so simulated data
    goes through exactly the same parser as real data)."""
    return "CSI," + ",".join(str(getattr(r, f)) for f in FIRMWARE_FIELDS) + "," + " ".join(map(str, r.csi_raw))


def record_from_row(row: dict) -> CsiRecord:
    """CSV row (dict from csv.DictReader) -> CsiRecord."""
    kw = {k: int(row[k]) for k in _INT_FIELDS}
    return CsiRecord(host_time=float(row["host_time"]), receiver_id=int(row["receiver_id"]),
                     port=row["port"], src_mac=row["src_mac"],
                     csi_raw=[int(v) for v in row["csi_raw"].split()], **kw)


def parse_status_line(line: str) -> dict:
    """'# STATUS role=RX rx_id=1 mac=.. rate=99.8 ...' -> {'kind': 'STATUS', 'rx_id': '1', ...}.
    Returns {} for other '#' lines."""
    tokens = line.lstrip("#").split()
    if not tokens or tokens[0] not in ("STATUS", "BOOT"):
        return {}
    out = {"kind": tokens[0]}
    for t in tokens[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out
