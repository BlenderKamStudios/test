"""CSI sensing pipeline for one TX/RX link: movement, presence and breathing.

What one link can measure (and what it cannot):
  - Movement: how much the channel's shape (amplitude across subcarriers)
    changes over 0.3 s, compared with how much it would change from noise alone.
    Each packet is divided by its mean amplitude (cancels the ESP32's automatic
    gain jumps), packets are averaged into 0.1 s bins (averages the noise away),
    and each bin is compared with the bin 0.3 s earlier. The expected change of
    a still channel follows from the packet-to-packet noise, so the score is
    about 1 for a still room on any link, whatever its signal strength.
  - Breathing: the chest moves ~5-10 mm per breath, which shows up as a slow
    periodic change (0.13-0.7 Hz) in the subcarrier amplitudes when the person
    is otherwise still and near the link.
  - Not measured: position, identity, heart rate (heartbeat moves the chest
    ~0.1-0.5 mm, below what ESP32 CSI can resolve reliably).

Movement threshold (no calibration or learning needed):
  Movement = score above MOTION_ON (2.5x the noise-only change). Chosen on the
  labelled movement tests (recordings/motion-test-*): the old 1 s jitter score
  could not tell a still person from a moving one (it also rises when the signal
  gets weaker); this one separates them with AUC 0.97.

Breathing pipeline (every second, on the last VITAL_WINDOW seconds):
  1. Normalise each packet by its mean amplitude, resample to VITAL_FS Hz.
  2. Detrend, remove spikes (Hampel filter), keep the subcarriers whose energy
     is most concentrated in the breathing band, band-pass, then PCA across
     them: the first component is the breathing waveform.
  3. Rate from the spectral peak, cross-checked by counting breath peaks.
  4. Detected when the peak dominates the spectrum. After an optional
     empty-room calibration, the band energy must also exceed the empty room's,
     which allows a lower peak requirement.
"""
import threading, time
from collections import deque

import numpy as np

from detect import WINDOW, parse_amplitude

WARMUP_SECONDS = 3          # noise estimate needs 2 s of data first
MOTION_BIN = 0.1            # s, packets averaged per bin
MOTION_LAG = 3              # bins: each bin is compared with the one 0.3 s earlier
MOTION_SMOOTH = 2           # bins of change averaged into the score
NOISE_BINS = 20             # bins (2 s) of packet-to-packet noise for the expected change
MOTION_ON = 2.5             # movement = change above 2.5x what noise alone gives
FLOOR_WINDOW = 180.0        # seconds of scores behind the displayed quiet level
FLOOR_PERCENTILE = 10
EXIT_FRACTION = 0.7         # leave "moving" once the score stays below 0.7 x threshold...
EXIT_HOLD = 0.5             # ...for this many seconds (stops flickering)
LARGE_MOTION = 3.0          # score / threshold ratio counted as large movement
SENSITIVITY_STEPS = (0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
SHORT = 30                  # packets for each subcarrier's live variability
HISTORY_SECONDS = 60        # movement score history kept for the display
PRESENCE_HOLD = 60.0        # after movement stops, someone "may still be here" this long

CALIB_COUNTDOWN = 10        # seconds to leave the area after pressing Calibrate
CALIB_SECONDS = 20          # empty-room measurement (breathing baseline)

VITAL_FS = 20.0             # breathing analysis sample rate (Hz)
VITAL_WINDOW = 30.0         # seconds analysed for breathing
VITAL_MIN = 15.0            # seconds needed before a first estimate
BREATH_SUBCARRIER_SHARE = 0.3  # use the subcarriers where breathing stands out most
BREATH_BAND = (0.13, 0.7)   # Hz: 8-42 breaths/min
BROAD_BAND = (0.05, 2.0)    # Hz: reference band for "how dominant is the peak"
PEAK_FRACTION_MIN = 0.30    # share of broad-band energy in the breathing peak (calibrated)
PEAK_FRACTION_STRICT = 0.30 # the same without an empty-room calibration (live data: breathing gave 0.30-0.34)
POWER_RATIO_MIN = 2.0       # breathing-band energy vs. empty-room baseline
MOTION_SETTLE = 8.0         # seconds after large movement before breathing is trusted


def shape_motion_score(X):
    """X: (packets, subcarriers) normalised amplitudes -> decorrelation x 1000."""
    Z = X - X.mean(axis=1, keepdims=True)
    Z /= Z.std(axis=1, keepdims=True) + 1e-9
    m = Z.mean(axis=0)
    m = (m - m.mean()) / (m.std() + 1e-9)
    return float((1 - (Z @ m / Z.shape[1]).mean()) * 1000)


def _band_pass(X, fs, band):
    """Zero-phase FFT band-pass along axis 0."""
    n = len(X)
    F = np.fft.rfft(X, axis=0)
    f = np.fft.rfftfreq(n, 1 / fs)
    F[(f < band[0]) | (f > band[1])] = 0
    return np.fft.irfft(F, n, axis=0)


def _count_breaths(w, fs):
    """Breath peaks in the waveform: returns intervals between peaks (seconds)."""
    min_gap = int(fs / BREATH_BAND[1])
    floor = 0.3 * w.std()
    peaks, last = [], -min_gap
    for i in range(1, len(w) - 1):
        if w[i] > floor and w[i] >= w[i - 1] and w[i] > w[i + 1] and i - last >= min_gap:
            peaks.append(i)
            last = i
    return np.diff(peaks) / fs


def _hampel(X, half=10, k=3.0):
    """Replace spikes (more than k robust std devs from the local median) in each column."""
    pad = np.pad(X, ((half, half), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1, axis=0)
    med = np.median(win, axis=2)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=2) + 1e-9
    return np.where(np.abs(X - med) > k * mad, med, X)


def analyze_breathing(X, fs=VITAL_FS, prev_loading=None):
    """X: (samples, subcarriers) normalised amplitudes. Returns a dict of features."""
    n = len(X)
    X = X - X.mean(axis=0)
    ramp = np.arange(n) - (n - 1) / 2
    X = X - np.outer(ramp, (ramp @ X) / (ramp @ ramp))
    X = _hampel(X)

    Xb = _band_pass(X, fs, BREATH_BAND)
    # Subcarriers differ a lot in how well they see the chest: keep the ones
    # whose energy is most concentrated in the breathing band
    share = Xb.var(axis=0) / (_band_pass(X, fs, BROAD_BAND).var(axis=0) + 1e-12)
    keep = np.argsort(share)[-max(8, int(BREATH_SUBCARRIER_SHARE * X.shape[1])):]
    _, _, Vt = np.linalg.svd(Xb[:, keep], full_matrices=False)
    loading = np.zeros(X.shape[1])
    loading[keep] = Vt[0]
    if prev_loading is not None and len(prev_loading) == len(loading) and loading @ prev_loading < 0:
        loading = -loading       # keep the waveform's sign stable between updates
    wave = Xb @ loading

    # Spectrum of the same projection over a broader band
    broad = _band_pass(X, fs, BROAD_BAND) @ loading
    nfft = 8192
    spec = np.abs(np.fft.rfft(broad * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    in_breath = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
    in_broad = (f >= BROAD_BAND[0]) & (f <= BROAD_BAND[1])
    peak_f = f[in_breath][np.argmax(spec[in_breath])]
    near_peak = np.abs(f - peak_f) <= 0.04
    peak_fraction = spec[near_peak].sum() / max(spec[in_broad].sum(), 1e-12)

    intervals = _count_breaths(wave, fs)
    shown = (f >= 0.05) & (f <= 1.0)
    display_spec = spec[shown][:: max(1, shown.sum() // 80)]
    return dict(
        wave=wave, loading=loading, power=float(wave.var()),
        bpm=float(peak_f * 60), peak_fraction=float(peak_fraction),
        bpm_count=float(60 / intervals.mean()) if len(intervals) >= 2 else None,
        interval_cv=float(intervals.std() / intervals.mean()) if len(intervals) >= 3 else None,
        spectrum=display_spec / max(display_spec.max(), 1e-12),
    )



class Sensor:
    """Feed it CSI_DATA lines; read snapshot() for the current picture."""

    def __init__(self):
        self.lock = threading.Lock()
        self.error = None
        self.rssi = None
        self.sensitivity = 1.0
        self.packet_times = deque(maxlen=400)
        self.history = deque()                  # (time, score) for the display
        self.floor_samples = deque()            # (time, score), once per second, for the display
        self.floor = None
        self.vital = deque()                    # (time, normalised amplitude vector)
        self.valid = None
        self.ref_std = None
        self.disturbance = None
        self.win = deque(maxlen=WINDOW)
        self.score = None
        self._n = 0
        self._bin = None
        self._mbin = None                       # [bin index, sum, count, d1 sum, d1 count] being filled
        self._prev_a = None                     # previous normalised packet (for the noise estimate)
        self.bins = deque(maxlen=MOTION_LAG + 1)
        self.changes = deque(maxlen=MOTION_SMOOTH)
        self.noise = deque(maxlen=NOISE_BINS)   # (d1 sum, d1 count, packets) per bin
        self.first_packet = None

        self.calib = None                       # {"phase": "countdown"|"measuring", "until": t, ...}
        self.calibrated_at = None
        self.calib_warning = None
        self.baseline_power = None
        self._baseline_window = None            # (start, end) waiting for the breathing baseline

        self.moving = False
        self.calm_since = None
        self.last_motion = None
        self.last_large_motion = None
        self.motion_events = 0
        self.present_since = None
        self.still_since = None

        self.breath = dict(status="searching")
        self._breath_votes = deque(maxlen=4)
        self._bpm_recent = deque(maxlen=5)
        self._prev_loading = None
        threading.Thread(target=self._vitals_loop, daemon=True).start()

    # ---- controls ------------------------------------------------------------
    def calibrate(self, now=None):
        """Start an empty-room calibration: countdown to leave, then measure."""
        now = time.time() if now is None else now
        with self.lock:
            self.calib = dict(phase="countdown", until=now + CALIB_COUNTDOWN)
            self.calib_warning = None

    def set_sensitivity(self, value):
        with self.lock:
            self.sensitivity = min(SENSITIVITY_STEPS, key=lambda v: abs(v - value))

    def threshold(self):
        return None if self.score is None else MOTION_ON / self.sensitivity

    # ---- per packet ----------------------------------------------------------
    def feed(self, line, now=None):
        if not line.startswith("CSI_DATA"):
            return
        amp = parse_amplitude(line)
        if amp is None:
            return
        now = time.time() if now is None else now
        with self.lock:
            self.packet_times.append(now)
            self.first_packet = self.first_packet or now
            try:
                self.rssi = int(line.split(",")[3])
            except (IndexError, ValueError):
                pass
            if self.win and len(amp) != len(self.win[0]):
                self.win.clear()
                self.vital.clear()
                self.valid = None
                self.ref_std = None
                self._reset_motion()
            self.win.append(amp)
            self._resample(amp, now)
            if self.valid is None:
                if len(self.win) < WINDOW // 2:
                    return
                self.valid = np.array(self.win).mean(axis=0) > 1.0   # drop null/guard subcarriers
            a = amp[self.valid]
            a = a / (a.mean() + 1e-6)

            b = int(now / MOTION_BIN)
            m = self._mbin
            if m is not None and b != m[0]:
                self._close_bin(now, gap=b - m[0] > 1)
                m = None
            if m is None:
                m = self._mbin = [b, np.zeros_like(a), 0, 0.0, 0]
            m[1] += a
            m[2] += 1
            if self._prev_a is not None:
                m[3] += float(((a - self._prev_a) ** 2).mean())
                m[4] += 1
            self._prev_a = a

    def _reset_motion(self):
        self._mbin, self._prev_a = None, None
        self.bins.clear()
        self.changes.clear()
        self.noise.clear()

    def _close_bin(self, now, gap):
        """Every 0.1 s: movement score, then the movement state."""
        _, total, n, d1, nd = self._mbin
        if gap:                                      # packets missing: don't compare across the hole
            self.bins.clear()
            self.changes.clear()
        self.bins.append(total / n)
        self.noise.append((d1, nd, n))
        if len(self.bins) > MOTION_LAG:
            self.changes.append(float(((self.bins[-1] - self.bins[0]) ** 2).mean()))
        if len(self.changes) < MOTION_SMOOTH or len(self.noise) < NOISE_BINS:
            return
        d1s, nds, ns = (sum(x) for x in zip(*self.noise))
        expected = (d1s / max(nds, 1)) / (ns / len(self.noise))   # change a still channel would show
        self.score = float(np.mean(self.changes) / (expected + 1e-12))
        self._n += 1

        a = np.array(self.win)[-SHORT:, :][:, self.valid]
        live_std = (a / (a.mean(axis=1, keepdims=True) + 1e-6)).std(axis=0)
        self._update_calibration(now)
        self._update_floor(now)
        thr = self.threshold()
        if self.calib:
            return
        if self.ref_std is None:
            self.ref_std = live_std + 1e-3
        if not self.moving:                              # slowly learn the quiet spread per subcarrier
            self.ref_std += 0.02 * (live_std + 1e-3 - self.ref_std)
        self.disturbance = live_std / self.ref_std
        self.history.append((now, self.score))
        while self.history and now - self.history[0][0] > HISTORY_SECONDS:
            self.history.popleft()
        self._update_motion(now, self.score / thr)

    def _update_floor(self, now):
        """The quiet level shown on the page (about 1 on a still link); not used for detection."""
        if self._n % 10 == 0 and not self.calib:
            self.floor_samples.append((now, self.score))
            while self.floor_samples and now - self.floor_samples[0][0] > FLOOR_WINDOW:
                self.floor_samples.popleft()
            self.floor = float(np.percentile([v for _, v in self.floor_samples], FLOOR_PERCENTILE))

    def _update_calibration(self, now):
        c = self.calib
        if not c:
            return
        if c["phase"] == "countdown" and now >= c["until"]:
            self.calib = dict(phase="measuring", start=now, until=now + CALIB_SECONDS, scores=[])
        elif c["phase"] == "measuring":
            c["scores"].append(self.score)
            if now >= c["until"]:
                s = np.array(c["scores"])
                if np.mean(s > MOTION_ON) > 0.1:
                    self.calib_warning = "Something moved during calibration, so breathing detection may be less reliable. Calibrate again with the area empty."
                self._baseline_window = (c["start"], now)
                self.calibrated_at = now
                self.calib = None
                self.moving = False

    def _update_motion(self, now, ratio):
        if ratio >= 1:
            if not self.moving:
                self.motion_events += 1
            self.moving = True
            self.calm_since = None
            self.last_motion = now
            self.still_since = None
            if ratio >= LARGE_MOTION:
                self.last_large_motion = now
        elif self.moving:
            # Hysteresis: only call it still after a short calm period
            if ratio < EXIT_FRACTION:
                self.calm_since = self.calm_since or now
                if now - self.calm_since >= EXIT_HOLD:
                    self.moving = False
                    self.still_since = now
            else:
                self.calm_since = None
                self.last_motion = now

    def _resample(self, amp, now):
        """Average packets into 1/VITAL_FS bins for the breathing analysis."""
        norm = amp / (amp.mean() + 1e-6)
        b = int(now * VITAL_FS)
        if self._bin is None or b < self._bin[0] or b - self._bin[0] > 2 * VITAL_FS \
                or len(norm) != len(self._bin[1]):
            self._bin = [b, np.zeros_like(norm), 0]
            self.vital.clear()
        if b != self._bin[0]:
            if self._bin[2]:
                mean = self._bin[1] / self._bin[2]
                for k in range(self._bin[0], b):          # fill short gaps
                    self.vital.append((k / VITAL_FS, mean))
            self._bin = [b, np.zeros_like(norm), 0]
        self._bin[1] += norm
        self._bin[2] += 1
        while self.vital and now - self.vital[0][0] > max(VITAL_WINDOW, CALIB_SECONDS) + 5:
            self.vital.popleft()

    # ---- breathing, once per second ------------------------------------------
    def _vitals_loop(self):
        while True:
            time.sleep(1.0)
            try:
                self._vitals_tick(time.time())
            except Exception as e:                        # keep the loop alive
                self.error = f"breathing analysis: {e}"

    def _vitals_tick(self, now):
        with self.lock:
            if self.valid is None or not self.vital:
                return
            valid, prev = self.valid, self._prev_loading
            baseline_win = self._baseline_window
            if baseline_win:
                rows = [v for t, v in self.vital if baseline_win[0] <= t <= baseline_win[1]]
                self._baseline_window = None
            else:
                rows = [v for t, v in self.vital if t >= now - VITAL_WINDOW]
            if self.calib:
                self.breath = dict(status="calibrating")
                return
            t_end = self.vital[-1][0]
            large_recent = self.last_large_motion is not None and now - self.last_large_motion < MOTION_SETTLE

        if len(rows) < VITAL_MIN * VITAL_FS:
            return
        X = np.array(rows)[:, valid]
        res = analyze_breathing(X, VITAL_FS, prev)

        with self.lock:
            if baseline_win:
                self.baseline_power = max(res["power"], 1e-9)
                return
            self._prev_loading = res["loading"]
            calibrated = self.baseline_power is not None
            power_ratio = res["power"] / self.baseline_power if calibrated else None
            rates_agree = res["bpm_count"] is not None and abs(res["bpm_count"] - res["bpm"]) <= 0.25 * res["bpm"]
            # Slow drift piles up at the band edge; a real breathing peak sits inside the band
            off_edge = BREATH_BAND[0] + 0.02 < res["bpm"] / 60 < BREATH_BAND[1] - 0.02
            if calibrated:
                strong = res["peak_fraction"] >= PEAK_FRACTION_MIN and power_ratio >= POWER_RATIO_MIN
            else:
                strong = res["peak_fraction"] >= PEAK_FRACTION_STRICT
            candidate = strong and rates_agree and off_edge and not large_recent
            self._breath_votes.append(candidate)
            votes = list(self._breath_votes)
            detected = self.breath.get("status") == "detected"
            if not detected and votes[-2:] == [True, True]:
                detected = True
            elif detected and votes[-3:] == [False, False, False]:
                detected = False
            status = "motion" if large_recent else "detected" if detected else "searching"
            if candidate:
                self._bpm_recent.append(res["bpm"])

            # "Clarity" for the display: reaches 1.0 only when every check passes
            pf_min = PEAK_FRACTION_MIN if calibrated else PEAK_FRACTION_STRICT
            strength = min(1.0, max(0.0, res["peak_fraction"] / pf_min))
            if calibrated:
                strength *= min(1.0, max(0.0, np.log10(max(power_ratio, 1e-9)) / np.log10(POWER_RATIO_MIN)))
            if not (rates_agree and off_edge):
                strength = min(strength, 0.6)
            scale = np.sqrt(self.baseline_power) if calibrated else res["wave"].std() + 1e-9
            self.breath = dict(
                status=status,
                bpm=float(np.median(self._bpm_recent)) if detected and self._bpm_recent else None,
                bpm_raw=res["bpm"], bpm_count=res["bpm_count"],
                strength=float(strength) if status != "motion" else 0.0,
                peak_fraction=res["peak_fraction"],
                power_ratio=None if power_ratio is None else float(power_ratio),
                interval_cv=res["interval_cv"] if detected else None,
                wave=res["wave"][-int(20 * VITAL_FS)::2] / scale, wave_fs=VITAL_FS / 2, wave_end=t_end,
                spectrum=res["spectrum"], spectrum_max_hz=1.0,
            )

    # ---- read-out ------------------------------------------------------------
    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            rate = sum(1 for t in self.packet_times if now - t <= 1.0)
            thr = self.threshold()
            ratio = self.score / thr if thr and self.score is not None else None
            level = None if ratio is None else ("large" if ratio >= LARGE_MOTION else "small" if ratio >= 1 else "none")
            breathing = self.breath.get("status") == "detected"
            since_motion = None if self.last_motion is None else now - self.last_motion
            recent_motion = since_motion is not None and since_motion < PRESENCE_HOLD

            c = self.calib
            if c:
                state = "calib_countdown" if c["phase"] == "countdown" else "calibrating"
            elif thr is None:
                state = "warming"
            elif self.moving:
                state = "moving"
            elif breathing:
                state = "still_breathing"
            elif recent_motion:
                state = "maybe"
            else:
                state = "empty"

            present = state in ("moving", "still_breathing", "maybe")
            if present and self.present_since is None:
                self.present_since = now
            if not present:
                self.present_since = None

            b = dict(self.breath)
            for k in ("wave", "spectrum"):
                if k in b:
                    b[k] = [round(float(x), 3) for x in b[k]]
            valid = self.valid
            warm_left = None
            if state == "warming":
                warm_left = max(0.0, WARMUP_SECONDS - (now - (self.first_packet or now)))
            return dict(
                t=now, state=state, error=self.error, warmup_left=warm_left,
                calib=None if not c else dict(phase=c["phase"], left=max(0.0, c["until"] - now),
                                              total=CALIB_COUNTDOWN if c["phase"] == "countdown" else CALIB_SECONDS),
                calibrated_ago=None if self.calibrated_at is None else now - self.calibrated_at,
                calib_warning=self.calib_warning, sensitivity=self.sensitivity,
                link=dict(rate=rate, rssi=self.rssi,
                          subcarriers=int(valid.sum()) if valid is not None else None),
                motion=dict(score=self.score, threshold=thr, floor=self.floor, ratio=ratio, level=level,
                            events=self.motion_events, last_ago=since_motion),
                presence=dict(present=present,
                              for_s=None if self.present_since is None else now - self.present_since,
                              still_for_s=None if not present or self.moving or self.still_since is None
                              else now - self.still_since),
                breathing=b,
                disturbance=None if self.disturbance is None else [round(float(x), 2) for x in self.disturbance],
                history=[(round(t - now, 2), round(s, 3)) for t, s in list(self.history)[::5]],
            )
