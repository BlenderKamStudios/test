"""Convert a raw session into processed, analysis-friendly files.

    python python/export.py data/raw/20260929-153000_walk
    python python/export.py data/raw/20260929-153000_walk --wide-csv --align

Writes to data/processed/<session name>/ :
  csi.npz        NumPy arrays per receiver (binary, compact, fast to load):
                   rx{i}_host_time      (N,)   laptop time, s
                   rx{i}_esp_time_s     (N,)   receiver clock, s (wrap-around removed)
                   rx{i}_tx_seq         (N,)   TX packet number (-1 unknown)
                   rx{i}_rssi           (N,)   dBm
                   rx{i}_csi            (N,64) complex CSI, subcarriers -32..31 (raw values)
                   rx{i}_valid          (N,64) usable subcarrier mask
                   rx{i}_amplitude      (N,64) |CSI| raw units
                   rx{i}_amplitude_norm (N,64) normalised as preprocessing.amplitude_normalization
                   rx{i}_phase_sanitized(N,64) radians, NaN where invalid
                   subcarriers          (64,)
  csi_wide.csv   (--wide-csv) one row per record: metadata + amp/phase per
                 used subcarrier  (timestamp, receiver_id, rssi, channel,
                 sc_-26_amp, sc_-26_phase, ...)
  aligned.npz    (--align) only packets received by ALL receivers, matched by
                 tx_seq: tx_seq (M,), host_time (M,), csi (R, M, 64), rssi (R, M)

Only per-packet processing is applied here (normalisation, phase
sanitisation). Temporal filters (Hampel, smoothing) are live-display steps;
apply them in your analysis if you need them. The raw CSV is not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from collector.replay_source import iter_records, resolve_csv
from common.config import load_config, project_path
from preprocessing import csi_math


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder or csi.csv(.gz)")
    ap.add_argument("--config", help="default: the config saved in the session, else config/config.yaml")
    ap.add_argument("--wide-csv", action="store_true")
    ap.add_argument("--align", action="store_true")
    args = ap.parse_args()

    csv_path = resolve_csv(args.session)
    session_json = csv_path.parent / "session.json"
    if args.config:
        cfg = load_config(args.config)
    elif session_json.exists():
        cfg = json.loads(session_json.read_text())["config"]
    else:
        cfg = load_config()
    pcfg = cfg["preprocessing"]
    part = pcfg.get("csi_part", "lltf")

    by_rx: dict[int, list] = {}
    for rec in iter_records(csv_path):
        by_rx.setdefault(rec.receiver_id, []).append(rec)
    if not by_rx:
        sys.exit("no records")

    out_dir = project_path("data/processed") / csv_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"subcarriers": csi_math.SUBCARRIERS}
    per_rx = {}
    for rx, recs in sorted(by_rx.items()):
        recs = [r for r in recs if r.csi_len >= (256 if part == "htltf" else 128)]
        raw = np.array([r.csi_raw[:256 if part == "htltf" else 128] for r in recs], dtype=np.int8)
        h = csi_math.to_complex(raw, part)
        valid = np.broadcast_to(csi_math.used_mask(part), h.shape).copy()
        if pcfg.get("mask_first_word_invalid", True):
            valid &= ~csi_math.invalid_first_word_mask([r.first_word_invalid for r in recs])
        amp = np.abs(h)
        valid &= amp > 0
        rssi = np.array([r.rssi for r in recs])
        d = {
            "host_time": np.array([r.host_time for r in recs]),
            "esp_time_s": np.unwrap(np.array([r.esp_timestamp_us for r in recs], float), period=2**32) / 1e6,
            "tx_seq": np.array([r.tx_seq for r in recs]),
            "rssi": rssi,
            "csi": h.astype(np.complex64),
            "valid": valid,
            "amplitude": amp.astype(np.float32),
            "amplitude_norm": csi_math.normalize_amplitude(amp, valid, pcfg.get("amplitude_normalization", "none"),
                                                           rssi=rssi).astype(np.float32),
            "phase_sanitized": np.full(h.shape, np.nan, np.float32),
        }
        # phase sanitisation per packet (the valid mask can differ between packets)
        for key in {tuple(v) for v in valid}:
            m = np.all(valid == np.array(key), axis=1)
            d["phase_sanitized"][m] = csi_math.sanitize_phase(h[m], np.array(key))
        per_rx[rx] = (recs, d)
        arrays.update({f"rx{rx}_{k}": v for k, v in d.items()})
        print(f"RX{rx}: {len(recs)} records")
    np.savez_compressed(out_dir / "csi.npz", **arrays)
    print("wrote", out_dir / "csi.npz")

    if args.wide_csv:
        used = csi_math.SUBCARRIERS[csi_math.used_mask(part)]
        idx = used + 32
        with open(out_dir / "csi_wide.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["host_time", "receiver_id", "tx_seq", "rssi", "channel"] +
                       [f"sc_{k}_{q}" for k in used for q in ("amp", "phase")])
            for rx, (recs, d) in per_rx.items():
                amp = np.where(d["valid"], d["amplitude"], np.nan)[:, idx]
                ph = d["phase_sanitized"][:, idx]
                for n, r in enumerate(recs):
                    inter = np.empty(2 * len(idx))
                    inter[0::2], inter[1::2] = amp[n], ph[n]
                    w.writerow([f"{r.host_time:.6f}", rx, r.tx_seq, r.rssi, r.channel] +
                               ["" if np.isnan(v) else f"{v:.4f}" for v in inter])
        print("wrote", out_dir / "csi_wide.csv")

    if args.align:
        rxs = sorted(per_rx)
        common = None
        for rx in rxs:
            s = set(per_rx[rx][1]["tx_seq"][per_rx[rx][1]["tx_seq"] >= 0].tolist())
            common = s if common is None else common & s
        seqs = np.array(sorted(common or []))
        csi_a, rssi_a, t_a = [], [], []
        for rx in rxs:
            d = per_rx[rx][1]
            _, first = np.unique(d["tx_seq"], return_index=True)     # a TX reboot can repeat seq numbers
            pos = first[np.searchsorted(d["tx_seq"][first], seqs)]
            csi_a.append(d["csi"][pos])
            rssi_a.append(d["rssi"][pos])
            t_a.append(d["host_time"][pos])
        np.savez_compressed(out_dir / "aligned.npz", tx_seq=seqs, receivers=np.array(rxs),
                            host_time=np.min(t_a, axis=0) if len(seqs) else np.array([]),
                            csi=np.array(csi_a), rssi=np.array(rssi_a))
        print(f"wrote {out_dir / 'aligned.npz'}: {len(seqs)} packets seen by all {len(rxs)} receivers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
