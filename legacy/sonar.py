#!/usr/bin/env python3
"""Sonar-style live display for the CSI presence detector.

Scope (left):
  - A sweep line rotates like a sonar/radar screen.
  - Each of the ~52 usable subcarriers has a fixed angle on the scope.
  - A blip's distance from the centre = how much that subcarrier is disturbed
    right now compared with the empty-room calibration (log scale).
  - The angle is the subcarrier (frequency), NOT the direction of the person:
    a single TX/RX link cannot tell where someone is, only that they are there.
Panel (right): state, motion score vs. threshold over the last 30 s, link stats.

Keys: R = recalibrate (room must be empty and still), Q / Esc = quit.
Usage: python sonar.py [/dev/ttyACM1]
"""
import math, sys, threading, time
from collections import deque

import numpy as np
import pygame
import serial

from detect import BAUD, CALIB_SECONDS, K, WINDOW, motion_score, parse_amplitude

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
SHORT = 30              # packets used for each subcarrier's live variability (~0.3 s)
HISTORY_SECONDS = 30
SWEEP_SECONDS = 2.5     # one full rotation of the sweep line

W, H = 1200, 720
BG = (8, 14, 11)
GRID = (28, 58, 42)
PHOSPHOR = (61, 220, 132)
TEXT = (226, 240, 231)
MUTED = (127, 154, 138)
PRESENT_COLOR = (255, 110, 84)
CALIB_COLOR = (240, 190, 70)


class Detector:
    """Reads the receiver in a background thread; same logic as detect.py."""

    def __init__(self, port):
        self.port = port
        self.lock = threading.Lock()
        self.error = None
        self.rssi = None
        self.packet_times = deque(maxlen=200)
        self.history = deque()          # (time, score)
        self.disturbance = None         # per-subcarrier live std / baseline std
        self.valid = None               # mask of usable (non-null) subcarriers
        self.reset()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        with self.lock:
            self.win = deque(maxlen=WINDOW)
            self.baseline_scores, self.baseline_std = [], []
            self.threshold = None
            self.score = None
            self.calib_start = None

    def state(self):
        with self.lock:
            now = time.time()
            rate = sum(1 for t in self.packet_times if now - t <= 1.0)
            return dict(
                error=self.error, rssi=self.rssi, rate=rate, score=self.score,
                threshold=self.threshold, history=list(self.history),
                disturbance=None if self.disturbance is None else self.disturbance.copy(),
                valid=None if self.valid is None else self.valid.copy(),
                calib_left=None if self.threshold is not None or self.calib_start is None
                else max(0.0, CALIB_SECONDS - (now - self.calib_start)),
            )

    def _run(self):
        while True:
            try:
                with serial.Serial(self.port, BAUD, timeout=1) as ser:
                    self.error = None
                    while True:
                        self._handle(ser.readline().decode(errors="ignore"))
            except (serial.SerialException, OSError) as e:
                self.error = f"{self.port}: {e}"
                time.sleep(2)

    def _handle(self, line):
        if not line.startswith("CSI_DATA"):
            return
        amp = parse_amplitude(line)
        if amp is None:
            return
        now = time.time()
        with self.lock:
            self.packet_times.append(now)
            try:
                self.rssi = int(line.split(",")[3])
            except (IndexError, ValueError):
                pass
            if self.win and len(amp) != len(self.win[0]):
                self.win.clear()
            self.win.append(amp)
            if len(self.win) < WINDOW:
                return

            a = np.array(self.win)
            self.score = motion_score(self.win)
            live_std = a[-SHORT:].std(axis=0)

            if self.threshold is None:
                if self.calib_start is None:
                    self.calib_start = now
                self.baseline_scores.append(self.score)
                self.baseline_std.append(live_std)
                if now - self.calib_start >= CALIB_SECONDS:
                    s = np.array(self.baseline_scores)
                    self.threshold = float(s.mean() + K * s.std())
                    self.ref_std = np.mean(self.baseline_std, axis=0) + 0.05
                    self.valid = a.mean(axis=0) > 1.0      # drop null/guard subcarriers
            else:
                self.disturbance = live_std / self.ref_std
                self.history.append((now, self.score))
                while self.history and now - self.history[0][0] > HISTORY_SECONDS:
                    self.history.popleft()


def ring_frac(ratio):
    """Scope radius (0..1) for a disturbance ratio: 1x at 0.2, 32x at the edge (log scale)."""
    return 0.2 + 0.8 * min(1.0, max(0.0, math.log10(max(ratio, 1.0)) / 1.5))


class Scope:
    def __init__(self, center, radius):
        self.c, self.r = center, radius
        self.lit = {}                   # subcarrier -> (time lit, radius fraction)
        self.last_angle = 0.0
        self.trail = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

    def point(self, angle, frac):
        return (self.c[0] + math.cos(angle) * frac * self.r,
                self.c[1] + math.sin(angle) * frac * self.r)

    def draw(self, screen, st, fonts, now):
        cx, cy = self.c
        pygame.draw.circle(screen, (5, 10, 8), self.c, self.r)
        for deg in range(0, 360, 30):
            pygame.draw.line(screen, GRID, self.c, self.point(math.radians(deg), 1.0), 1)
        for ratio, label in ((1, "1x"), (3, "3x"), (10, "10x"), (32, "32x")):
            f = ring_frac(ratio)
            pygame.draw.circle(screen, GRID, self.c, int(self.r * f), 1)
            screen.blit(fonts["tiny"].render(label, True, MUTED),
                        (cx + 4, cy - int(self.r * f) + 2))

        # Sweep with a fading trail
        angle = (now / SWEEP_SECONDS) % 1.0 * 2 * math.pi
        self.trail.fill((0, 0, 0, 0))
        for i in range(24):
            a = angle - math.radians(i * 1.5)
            alpha = int(90 * (1 - i / 24))
            pygame.draw.polygon(self.trail, (*PHOSPHOR, alpha), [
                (self.r, self.r),
                (self.r + math.cos(a) * self.r, self.r + math.sin(a) * self.r),
                (self.r + math.cos(a - 0.03) * self.r, self.r + math.sin(a - 0.03) * self.r)])
        screen.blit(self.trail, (cx - self.r, cy - self.r))
        pygame.draw.line(screen, PHOSPHOR, self.c, self.point(angle, 1.0), 2)

        # Light up subcarriers as the sweep passes them, then let them fade
        d, valid = st["disturbance"], st["valid"]
        if d is not None and valid is not None:
            idx = np.flatnonzero(valid)
            for k, i in enumerate(idx):
                a = 2 * math.pi * k / len(idx)
                crossed = (self.last_angle <= a < angle) if angle >= self.last_angle \
                    else (a >= self.last_angle or a < angle)
                if crossed:
                    frac = ring_frac(d[i])
                    self.lit[i] = (now, frac, a)
            for i, (t, frac, a) in self.lit.items():
                fade = max(0.0, 1 - (now - t) / SWEEP_SECONDS)
                if fade <= 0:
                    continue
                color = PRESENT_COLOR if frac > ring_frac(3) else PHOSPHOR
                glow = tuple(int(BG[j] + (color[j] - BG[j]) * fade) for j in range(3))
                size = 4 + int(4 * frac)
                pygame.draw.circle(screen, glow, self.point(a, frac), size)
        self.last_angle = angle

        pygame.draw.circle(screen, TEXT, self.c, 5)
        screen.blit(fonts["tiny"].render("RX", True, TEXT), (cx + 8, cy + 6))


def draw_panel(screen, st, fonts, rect, pulse):
    x, y, w, h = rect
    if st["error"]:
        state, color = "NO SIGNAL", CALIB_COLOR
    elif st["threshold"] is None:
        state = "CALIBRATING" if st["calib_left"] is not None else "WAITING"
        color = CALIB_COLOR
    elif st["score"] is not None and st["score"] > st["threshold"]:
        state, color = "PRESENT", PRESENT_COLOR
    else:
        state, color = "EMPTY", PHOSPHOR

    marker_r = 14 + (int(4 * pulse) if state == "PRESENT" else 0)
    pygame.draw.circle(screen, color, (x + 18, y + 36), marker_r)
    screen.blit(fonts["big"].render(state, True, TEXT), (x + 48, y + 12))
    if st["threshold"] is None and not st["error"]:
        hint = ("Waiting for data from the receiver..." if st["calib_left"] is None else
                f"Keep the area empty and still: {st['calib_left']:.0f} s left")
        screen.blit(fonts["small"].render(hint, True, MUTED), (x, y + 68))

    # Motion score over time, with the threshold
    gy, gh = y + 140, 280
    screen.blit(fonts["small"].render(f"Motion score, last {HISTORY_SECONDS} s", True, TEXT),
                (x, gy - 28))
    pygame.draw.rect(screen, (5, 10, 8), (x, gy, w, gh))
    hist, thr = st["history"], st["threshold"]
    if hist and thr:
        now = time.time()
        ymax = max(thr * 3, max(s for _, s in hist) * 1.1)
        to_xy = lambda t, s: (x + w - (now - t) / HISTORY_SECONDS * w,
                              gy + gh - min(s, ymax) / ymax * gh)
        for f in (0.25, 0.5, 0.75):
            pygame.draw.line(screen, GRID, (x, gy + gh * f), (x + w, gy + gh * f), 1)
        ty = gy + gh - thr / ymax * gh
        for sx in range(x, x + w, 12):
            pygame.draw.line(screen, CALIB_COLOR, (sx, ty), (min(sx + 6, x + w), ty), 1)
        screen.blit(fonts["tiny"].render(f"threshold {thr:.1f}", True, MUTED), (x + 4, ty - 16))
        pts = [to_xy(t, s) for t, s in hist]
        if len(pts) > 1:
            pygame.draw.lines(screen, PHOSPHOR, False, pts, 2)
        pygame.draw.circle(screen, color, pts[-1], 5)
        screen.blit(fonts["tiny"].render(f"{ymax:.0f}", True, MUTED), (x + 4, gy + 2))
        screen.blit(fonts["tiny"].render("-30 s", True, MUTED), (x, gy + gh + 4))
        screen.blit(fonts["tiny"].render("now", True, MUTED), (x + w - 26, gy + gh + 4))

    # Link stats
    sy = gy + gh + 40
    score = "-" if st["score"] is None else f"{st['score']:.2f}"
    thr_s = "-" if thr is None else f"{thr:.2f}"
    rssi = "-" if st["rssi"] is None else f"{st['rssi']} dBm"
    for i, (k, v) in enumerate((("Score", score), ("Threshold", thr_s),
                                ("Packets/s", str(st["rate"])), ("Signal (RSSI)", rssi))):
        cx = x + (i % 2) * (w // 2)
        cy = sy + (i // 2) * 56
        screen.blit(fonts["tiny"].render(k, True, MUTED), (cx, cy))
        screen.blit(fonts["mid"].render(v, True, TEXT), (cx, cy + 16))
    if st["error"]:
        screen.blit(fonts["tiny"].render(st["error"][:80], True, CALIB_COLOR), (x, sy + 116))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Wi-Fi Sonar: CSI presence detection")
    fonts = {"big": pygame.font.Font(None, 64), "mid": pygame.font.Font(None, 36),
             "small": pygame.font.Font(None, 26), "tiny": pygame.font.Font(None, 20)}
    det = Detector(PORT)
    scope = Scope((360, 350), 310)
    clock = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                det.reset()
                scope.lit.clear()

        now = time.time()
        st = det.state()
        screen.fill(BG)
        scope.draw(screen, st, fonts, now)
        draw_panel(screen, st, fonts, (740, 30, 420, 660), (math.sin(now * 8) + 1) / 2)
        footer = ("Distance from centre = how disturbed each subcarrier is vs. the empty room (1x = normal). "
                  "Angle = subcarrier, not direction.   R: recalibrate   Q: quit")
        screen.blit(fonts["tiny"].render(footer, True, MUTED), (40, H - 26))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
