#!/usr/bin/env python3
"""Live Wi-Fi sensing heatmap: reads the receivers, serves web/ on localhost.

Usage:
  python server.py                      # finds the receivers on any USB serial port
  python server.py --replay recordings/live-....csv.gz   # test without hardware
Then open http://localhost:8765 (opened automatically unless --no-browser).

Receivers and their positions come from layout.json. Each receiver reports its
own MAC in its status lines, so it is recognised whatever port it is on; ports
that never report one (the transmitter) are left alone.

All live data is also logged to recordings/live-*.csv.gz (about 50 MB per hour
per receiver), so any session can be replayed and analysed later. Log lines are
"<time> <receiver name> <CSI_DATA line>"; old single-receiver recordings
("<time> <CSI_DATA line>") replay as the first receiver.

Breathing tests started from the page are saved to recordings/breath-test-*.csv
(raw data, same format as record.py) with a .json next to it holding the
person's own breath count and what the system measured.
"""
import argparse, glob, gzip, json, os, re, signal, threading, time, webbrowser
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import serial

from detect import BAUD
from tracking import Tracker, load_layout

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
RECORDINGS = os.path.join(ROOT, "recordings")
LAYOUT = os.path.join(ROOT, "layout.json")
PORT_GLOBS = ("/dev/ttyACM*", "/dev/ttyUSB*")
IDENTIFY_SECONDS = 12       # receivers print their MAC every 5 s; silent ports are not receivers
STREAM_HZ = 15
RAW_SECONDS = 120           # raw lines kept in memory so a test can be saved afterwards
TEST_COUNTDOWN = 10
TEST_SECONDS = 60


class BreathTest:
    """Guided test: countdown, 60 s of the person counting their breaths, then their count."""

    def __init__(self, station):
        self.station = station
        self.lock = threading.Lock()
        self.state = None
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self):
        with self.lock:
            self.state = dict(phase="countdown", until=time.time() + TEST_COUNTDOWN)

    def cancel(self):
        with self.lock:
            self.state = None

    def _loop(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                st = self.state
                if not st:
                    continue
                now = time.time()
                if st["phase"] == "countdown" and now >= st["until"]:
                    self.state = dict(phase="recording", start=now, until=now + TEST_SECONDS, samples=[])
                elif st["phase"] == "recording":
                    b = self.station.tracker.snapshot()["breathing"]
                    st["samples"].append(dict(t=now, status=b.get("status"), bpm=b.get("bpm"),
                                              bpm_raw=b.get("bpm_raw"), strength=b.get("strength")))
                    if now >= st["until"]:
                        self._save(st)

    def _save(self, st):
        os.makedirs(RECORDINGS, exist_ok=True)
        name = time.strftime("breath-test-%Y%m%d-%H%M%S")
        rows = [r for r in list(self.station.raw) if st["start"] - 30 <= r[0] <= st["until"]]
        with open(os.path.join(RECORDINGS, name + ".csv"), "w") as f:
            f.writelines(f"{t:.4f} {rx} {line}\n" for t, rx, line in rows)
        # System's answer: median of the detected rates over the second half of the test
        late = [s for s in st["samples"] if s["t"] >= st["start"] + TEST_SECONDS / 2]
        detected = [s["bpm"] for s in late if s["status"] == "detected" and s["bpm"]]
        self.state = dict(phase="ask", name=name, samples=st["samples"], start=st["start"],
                          measured=float(np.median(detected)) if detected else None,
                          detected_share=len(detected) / max(1, len(late)))

    def label(self, count):
        with self.lock:
            st = self.state
            if not st or st["phase"] != "ask":
                return False
            meta = dict(counted_breaths=count, seconds=TEST_SECONDS, true_bpm=count * 60 / TEST_SECONDS,
                        measured_bpm=st["measured"], detected_share=st["detected_share"],
                        test_start=st["start"], samples=st["samples"])
            with open(os.path.join(RECORDINGS, st["name"] + ".json"), "w") as f:
                json.dump(meta, f, indent=1)
            self.state = dict(phase="done", true_bpm=meta["true_bpm"], measured=st["measured"],
                              detected_share=st["detected_share"], name=st["name"])
            return True

    def snapshot(self):
        with self.lock:
            st = self.state
            if not st:
                return None
            out = {k: v for k, v in st.items() if k not in ("samples",)}
            if "until" in st:
                out["left"] = max(0.0, st["until"] - time.time())
                out["total"] = TEST_COUNTDOWN if st["phase"] == "countdown" else TEST_SECONDS
            return out


MOTION_PHASES = [("still", 20), ("move", 15), ("still", 20), ("move", 15), ("still", 20)]
MOTION_COUNTDOWN = 5


class MotionTest:
    """Guided test: the page says when to stay still and when to move; saves what the system saw.

    Saved as recordings/motion-test-*.csv (raw data, same format as the live log)
    and .json (phase times plus the system's state 10 times per second), for tuning.
    """

    def __init__(self, station):
        self.station = station
        self.lock = threading.Lock()
        self.state = None
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self):
        with self.lock:
            self.state = dict(phase="countdown", until=time.time() + MOTION_COUNTDOWN)

    def cancel(self):
        with self.lock:
            self.state = None

    def _loop(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                st = self.state
                if not st:
                    continue
                now = time.time()
                if st["phase"] == "countdown" and now >= st["until"]:
                    t, phases = now, []
                    for kind, dur in MOTION_PHASES:
                        phases.append(dict(kind=kind, start=t, end=t + dur))
                        t += dur
                    self.state = dict(phase="running", start=now, until=t, phases=phases, samples=[])
                elif st["phase"] == "running":
                    snap = self.station.tracker.snapshot()
                    st["samples"].append(dict(t=now, state=snap.get("state"),
                                              links={l["name"]: [l["state"], l["ratio"]] for l in snap.get("links", [])}))
                    if now >= st["until"]:
                        self._save(st)

    def _save(self, st):
        os.makedirs(RECORDINGS, exist_ok=True)
        name = time.strftime("motion-test-%Y%m%d-%H%M%S")
        rows = [r for r in list(self.station.raw) if st["start"] - 5 <= r[0] <= st["until"]]
        with open(os.path.join(RECORDINGS, name + ".csv"), "w") as f:
            f.writelines(f"{t:.4f} {rx} {line}\n" for t, rx, line in rows)
        with open(os.path.join(RECORDINGS, name + ".json"), "w") as f:
            json.dump(dict(phases=st["phases"], samples=st["samples"]), f)
        self.state = dict(phase="done", name=name, **score_motion_test(st["phases"], st["samples"]))

    def snapshot(self):
        with self.lock:
            st = self.state
            if not st:
                return None
            now = time.time()
            out = dict(kind="motion", phase=st["phase"])
            if st["phase"] == "countdown":
                out.update(left=max(0.0, st["until"] - now), total=MOTION_COUNTDOWN)
            elif st["phase"] == "running":
                cur = next((p for p in st["phases"] if p["start"] <= now < p["end"]), st["phases"][-1])
                i = st["phases"].index(cur)
                out.update(step=cur["kind"], step_left=max(0.0, cur["end"] - now),
                           next=st["phases"][i + 1]["kind"] if i + 1 < len(st["phases"]) else None,
                           left=max(0.0, st["until"] - now), total=st["until"] - st["start"])
            else:
                out.update({k: v for k, v in st.items() if k != "phase"})
            return out


def score_motion_test(phases, samples, state_of=lambda s: s["state"]):
    """False alarms while still, detection while moving, and how long it takes to react."""
    def moving(s):
        return state_of(s) == "moving"
    still = [moving(s) for p in phases if p["kind"] == "still"
             for s in samples if p["start"] + 2 <= s["t"] < p["end"]]       # 2 s grace after moving
    move = [moving(s) for p in phases if p["kind"] == "move"
            for s in samples if p["start"] + 2 <= s["t"] < p["end"]]
    onsets, releases = [], []
    for p in phases:
        seg = [s for s in samples if p["start"] <= s["t"] < p["end"]]
        if p["kind"] == "move":
            hit = next((s["t"] for s in seg if moving(s)), None)
            onsets.append(None if hit is None else hit - p["start"])
        elif p is not phases[0]:
            rel = next((s["t"] for s in seg if not moving(s)), None)
            releases.append(None if rel is None else rel - p["start"])
    ok = lambda xs: [x for x in xs if x is not None]
    return dict(false_alarm=float(np.mean(still)) if still else None,
                detected=float(np.mean(move)) if move else None,
                onset_s=float(np.mean(ok(onsets))) if ok(onsets) else None,
                release_s=float(np.mean(ok(releases))) if ok(releases) else None,
                missed_moves=sum(o is None for o in onsets))


class Station:
    """The receivers' data flow: raw lines -> memory buffer + one sensing pipeline per link."""

    def __init__(self, layout, log=True):
        self.tracker = Tracker(layout)
        self.write_lock = threading.Lock()
        self.raw = deque(maxlen=int(RAW_SECONDS * 110 * len(layout["receivers"])))
        self.test = BreathTest(self)
        self.mtest = MotionTest(self)
        self.log = None
        if log:
            os.makedirs(RECORDINGS, exist_ok=True)
            self.log_path = os.path.join(RECORDINGS, time.strftime("live-%Y%m%d-%H%M%S.csv.gz"))
            self.log = gzip.open(self.log_path, "at")
            self._flushed = time.time()

    def feed(self, rx, line):
        line = line.strip()
        link = self.tracker.links.get(rx)
        if link and line.startswith("CSI_DATA") and csi_line_ok(line):
            now = time.time()
            self.raw.append((now, rx, line))
            try:
                link.sensor.feed(line, now)
            except (ValueError, OverflowError) as e:   # never let one odd line stop a receiver
                link.sensor.error = f"bad data line skipped ({e})"
            if self.log:
                with self.write_lock:
                    self.log.write(f"{now:.4f} {rx} {line}\n")
                    if now - self._flushed > 5:
                        self.log.flush()
                        self._flushed = now


CSI_RE = re.compile(r'^CSI_DATA,\d+,[0-9a-f:]{17},-?\d+,\d+,-?\d+,\d+,\d+,\d+,(\d+),"\[(-?\d+(?:,-?\d+)*)\]"$')


def csi_line_ok(line):
    """A complete line whose value count matches its length field (serial glitches garble lines)."""
    m = CSI_RE.match(line)
    return bool(m) and m.group(2).count(",") + 1 == int(m.group(1))


MAC_RE = re.compile(r"rx ((?:[0-9a-f]{2}:){5}[0-9a-f]{2})\s*$")


def read_port(station, port, busy, not_rx):
    """Read one serial port. Its lines are used once the board has said which receiver it is."""
    tracker, link = station.tracker, None
    try:
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = port, BAUD, 1
        ser.dtr = ser.rts = False              # don't hold the board in reset
        ser.open()
        with ser:
            start, buf = time.time(), b""
            while True:
                # Read in bulk: pyserial's readline() goes byte by byte, too slow for
                # several receivers at 921600 baud (the OS buffer overflows, lines get garbled)
                buf += ser.read(max(1, ser.in_waiting))
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    line = raw.decode(errors="ignore").strip()
                    if link:
                        station.feed(link.name, line)
                        continue
                    m = MAC_RE.search(line) if line.startswith("#") else None
                    if m:
                        found = tracker.by_mac.get(m.group(1))
                        if found is None:
                            tracker.errors[port] = f"board {m.group(1)} on {port} is not in layout.json"
                        elif found.port and found.port != port and found.port in busy:
                            tracker.errors[port] = f"{found.name} appears on both {found.port} and {port}"
                        else:
                            link, link.port, link.sensor.error = found, port, None
                            print(f"{link.name} ({link.mac}) found on {port}")
                            continue
                        not_rx.add(port)
                        return
                if link is None and time.time() - start > IDENTIFY_SECONDS:
                    not_rx.add(port)                   # the transmitter, or something else
                    return
    except (serial.SerialException, OSError) as e:
        if link:
            link.sensor.error = f"lost connection on {port} ({e})"
    finally:
        busy.discard(port)


def scan_ports(station, only=None):
    """Start a reader for every serial port that appears; forget ports that disappear."""
    busy, not_rx, started = set(), set(), time.time()
    while True:
        ports = set(only) if only else {p for g in PORT_GLOBS for p in glob.glob(g)}
        not_rx &= ports
        for p in list(station.tracker.errors):
            if p not in ports:
                del station.tracker.errors[p]
        for p in sorted(ports - busy - not_rx):
            if os.path.exists(p):
                busy.add(p)
                threading.Thread(target=read_port, args=(station, p, busy, not_rx), daemon=True).start()
        for l in station.tracker.links.values():
            if l.port is None and not l.sensor.error and time.time() - started > IDENTIFY_SECONDS:
                l.sensor.error = "not found on any USB port yet"
        time.sleep(3)


def replay(station, path):
    """Feed a recording in real time, looping, as if it were live."""
    first = next(iter(station.tracker.links))
    rows = []
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        try:
            for l in f:
                t, rest = l.rstrip("\n").split(" ", 1)
                rx, line = (first, rest) if rest.startswith("CSI_DATA") else rest.split(" ", 1)
                rows.append((float(t), rx, line))
        except (EOFError, ValueError):
            pass                        # log of a session that was killed: use what is there
    while True:
        start, t0 = time.time(), rows[0][0]
        for t, rx, line in rows:
            delay = (t - t0) - (time.time() - start)
            if delay > 0:
                time.sleep(delay)
            station.feed(rx, line)


class Handler(SimpleHTTPRequestHandler):
    station = None

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path != "/stream":
            return super().do_GET()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            while True:
                snap = self.station.tracker.snapshot()
                snap["test"] = self.station.test.snapshot() or self.station.mtest.snapshot()
                self.wfile.write(f"data: {json.dumps(snap, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()
                time.sleep(1 / STREAM_HZ)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/recalibrate":
                self.station.tracker.calibrate()
            elif url.path == "/sensitivity":
                self.station.tracker.set_sensitivity(float(q["value"][0]))
            elif url.path == "/test/start":
                self.station.mtest.cancel()
                self.station.test.start()
            elif url.path == "/mtest/start":
                self.station.test.cancel()
                self.station.mtest.start()
            elif url.path == "/mtest/cancel":
                self.station.mtest.cancel()
            elif url.path == "/test/cancel":
                self.station.test.cancel()
            elif url.path == "/test/label":
                if not self.station.test.label(int(q["count"][0])):
                    return self.send_error(409)
            else:
                return self.send_error(404)
        except (KeyError, ValueError):
            return self.send_error(400)
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="+", help="only use these serial ports (default: all USB serial ports)")
    ap.add_argument("--layout", default=LAYOUT, help="board positions and receiver MACs")
    ap.add_argument("--replay", help="replay a recordings/*.csv(.gz) file instead of the receivers")
    ap.add_argument("--http", type=int, default=8765, help="web server port")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    layout = load_layout(args.layout)
    station = Station(layout, log=not args.replay)
    source = partial(replay, station, args.replay) if args.replay else partial(scan_ports, station, args.ports)
    threading.Thread(target=source, daemon=True).start()

    Handler.station = station
    httpd = ThreadingHTTPServer(("127.0.0.1", args.http), partial(Handler, directory=WEB))
    httpd.daemon_threads = True
    url = f"http://localhost:{args.http}"
    names = ", ".join(r["name"] for r in layout["receivers"])
    print(f"Wi-Fi sensing display: {url}  (source: {args.replay or 'USB serial ports'}; receivers {names}; Ctrl+C to stop)")
    if station.log:
        print(f"Logging live data to {station.log_path}")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    # Stop cleanly on "kill" too, so the gzip log is closed properly
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if station.log:
            station.log.close()


if __name__ == "__main__":
    main()
