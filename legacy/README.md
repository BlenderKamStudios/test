# ESP32 Wi-Fi Sensing: Human Presence Detection (Prototype)

Graduation project prototype: two ESP32 boards detect whether a person is present in a room using only Wi-Fi signals. No camera, no wearable, no extra sensor.

> **Status:** prototype / proof of concept. The goal of this stage is to get a working end-to-end pipeline (transmit → receive → detect). Accuracy, multi-person, and localisation come later.

---

## 1. How it works

The idea is "sonar-like", but the physics is slightly different from real sonar, so it helps to state it precisely:

- **Sonar** sends a pulse and times the echo.
- **Wi-Fi sensing** sends a steady stream of packets from one ESP32 (the **transmitter**) to another (the **receiver**). The signal reaches the receiver along many paths: directly, and bouncing off walls, furniture, and people.
- The receiver measures the **Channel State Information (CSI)** of every packet: the amplitude and phase of each OFDM subcarrier (about 52 usable subcarriers on 20 MHz 802.11n).
- In an **empty, still room** the CSI is almost constant. When a **person is present or moving**, they change the multipath pattern, so CSI **fluctuates**. Measuring that fluctuation (for example the variance of the amplitude over a short window) tells us whether someone is there.

```
   ┌──────────┐   Wi-Fi packets (~100/s)    ┌──────────┐   USB serial   ┌────────┐
   │ ESP32 TX │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶ │ ESP32 RX │ ─────────────▶ │  PC    │
   └──────────┘        ╲   person   ╱       └──────────┘   CSI lines    │ Python │
                        ╲  moves   ╱                                    └────────┘
                       (multipath changes)                         variance → present / empty
```

---

## 2. Hardware

| Item | Qty | Notes |
|---|---|---|
| ESP32 dev board (ESP32-DevKitC or similar) | 2 | Original ESP32, S3, C3 and C6 all expose CSI; this guide uses the original ESP32. |
| USB data cables (micro-USB or USB-C to match your board) | 2 | Must be **data** cables, not charge-only. |
| PC with a free USB port | 1 | Linux / macOS / Windows. |
| USB power bank or wall adapter | 1 | Optional, to power the transmitter once it is flashed. |

Placement for the first test: put the two boards **2–4 m apart**, at roughly table height, with the area between them clear. The person should walk through or stand in the line between them.

---

## 3. Software prerequisites

- **ESP-IDF v5.x** (Espressif's official framework). Install guide: <https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/>
- **Espressif `esp-csi`** repository, which provides ready-made CSI examples: <https://github.com/espressif/esp-csi>
- **Python 3.9+** with `pyserial` and `numpy`
- On Linux, your user must be in the serial group (`dialout` on Debian/Ubuntu, `uucp` on Arch):

```bash
sudo usermod -aG uucp $USER   # Arch;  use "dialout" on Debian/Ubuntu
# log out and back in afterwards
```

---

## 4. Step-by-step

### Step 1: Install ESP-IDF

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32
. ./export.sh          # run this in every new terminal
```

### Step 2: Get the CSI examples

```bash
cd ~
git clone https://github.com/espressif/esp-csi.git
cd esp-csi/examples/get-started
ls                     # csi_send  csi_recv
```

- `csi_send` : transmitter firmware (sends packets at a fixed rate).
- `csi_recv` : receiver firmware (collects CSI and prints it to the serial port).

### Step 3: Flash the transmitter

Plug in **board A** and find its port (`ls /dev/ttyUSB*` or `/dev/ttyACM*`; `COMx` on Windows).

```bash
cd ~/esp-csi/examples/get-started/csi_send
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Leave it running, or power it from a power bank afterwards. Press `Ctrl+]` to exit the monitor.

### Step 4: Flash the receiver

Plug in **board B** (it will probably appear as `/dev/ttyUSB1`).

```bash
cd ~/esp-csi/examples/get-started/csi_recv
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB1 flash monitor
```

You should see a stream of lines starting with `CSI_DATA,...`. If the stream is flowing, the RF link works. If not, see [Troubleshooting](#7-troubleshooting).

> The exact firmware, channel and packet-rate options can be adjusted with `idf.py menuconfig`. Keep both boards on the **same Wi-Fi channel**, and use the default packet rate (about 100 packets/s) for the prototype.

### Step 5: Read CSI on the PC

Close `idf.py monitor` first (only one program can hold the port). Then create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyserial numpy
```

Save the following as `detect.py`. It reads the receiver's serial output, computes the amplitude of each subcarrier, and looks at how much it fluctuates over a sliding window.

```python
#!/usr/bin/env python3
"""Prototype presence detector: CSI amplitude variance over a sliding window."""
import re, sys, time
from collections import deque

import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
BAUD = 921600          # matches the esp-csi csi_recv default; change if yours differs
WINDOW = 100           # packets per window (~1 s at 100 pkt/s)
CALIB_SECONDS = 15     # room must be EMPTY and still during this time
K = 3.0                # threshold = mean + K * std of the empty-room baseline

csi_re = re.compile(r'"?\[([-\d,\s]+)\]"?\s*$')

def parse_amplitude(line: str):
    m = csi_re.search(line)
    if not m:
        return None
    vals = np.array([int(v) for v in m.group(1).split(",") if v.strip()], dtype=np.int8)
    if len(vals) < 2 or len(vals) % 2:
        return None
    imag, real = vals[0::2].astype(float), vals[1::2].astype(float)
    return np.hypot(real, imag)          # one amplitude per subcarrier

def motion_score(win: deque) -> float:
    a = np.array(win)                    # shape: (packets, subcarriers)
    return float(np.mean(np.var(a, axis=0)))   # avg variance across subcarriers

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    win = deque(maxlen=WINDOW)
    baseline, threshold = [], None
    t0 = time.time()
    print(f"Calibrating for {CALIB_SECONDS}s: keep the room EMPTY and still...")

    while True:
        line = ser.readline().decode(errors="ignore")
        if not line.startswith("CSI_DATA"):
            continue
        amp = parse_amplitude(line)
        if amp is None:
            continue
        # ignore packets with a different subcarrier count than the window
        if win and len(amp) != len(win[0]):
            win.clear()
        win.append(amp)
        if len(win) < WINDOW:
            continue

        score = motion_score(win)
        if threshold is None:
            baseline.append(score)
            if time.time() - t0 >= CALIB_SECONDS:
                threshold = np.mean(baseline) + K * np.std(baseline)
                print(f"Calibrated. Threshold = {threshold:.2f}")
        else:
            state = "PRESENT" if score > threshold else "empty  "
            print(f"\r{state}  score={score:8.2f}  threshold={threshold:.2f}", end="", flush=True)

if __name__ == "__main__":
    main()
```

Run it:

```bash
python detect.py /dev/ttyUSB1
```

### Step 6: Test it

1. Start the script and **leave the room** (or stand completely still outside the link) for the 15-second calibration.
2. After it prints `Calibrated`, the state should read `empty`.
3. Walk between the two boards. The state should switch to `PRESENT` and the score should jump.
4. Leave the link area. The state should return to `empty` within a couple of seconds.

Record what you see in the table below so we have numbers for the report:

| Scenario | Expected | Observed score | Correct? |
|---|---|---|---|
| Empty room | empty | | |
| Person walking through the link | present | | |
| Person standing still in the link | present (harder) | | |
| Person in the room, off the link | present? (weaker) | | |
| Door opening / fan on (interference) | empty | | |

---

## 5. Known limitations of the prototype

- **Motion vs. presence.** Variance-based detection reacts mainly to *movement*. A perfectly still person is much harder to detect (breathing is a tiny signal). Handling this is a later-stage topic.
- **Coarse position.** With 1 TX + 3 RX (see §5a) the position is only the part of the room near the most disturbed link(s), not an exact spot, and a still person can be detected by breathing but not located.
- **Environment sensitive.** Moving furniture, other Wi-Fi traffic, and people outside the room change the baseline. Recalibrate when the setup changes.
- **Fixed threshold.** Threshold comes from a 15 s empty-room baseline, with no adaptation over time.
- **Serial format assumptions.** The parser assumes the `esp-csi` `csi_recv` output format (`CSI_DATA,...,"[i,r,i,r,...]"`). If a newer version of the example changes the column layout, adjust `parse_amplitude`.

---

## 5a. Three receivers: where and which way (`server.py`)

The live display now uses 1 transmitter and 3 receivers (Arduino firmware in `firmware/csi_tx`, `firmware/csi_rx`).

- **Board positions** go in `layout.json` (metres, seen from above; `x` = left/right, `y` = distance into the room from TX). Receivers are matched by MAC, so it does not matter which USB port each one is on. The default is a fan: TX at one wall, RX1/RX2/RX3 on the far side, left, middle and right.
- **Run:** `.venv/bin/python server.py`. It opens every USB serial port, waits for each receiver to report its MAC (status line every 5 s), and leaves the transmitter's port alone.
- **Where:** each link runs its own movement detector; the position is the midpoint of each link weighted by how far its movement score is above its own detection line.
- **Which way:** the trend of that position over the last 2 s, named after the board it heads for, plus the order in which the links started seeing movement (walking across the fan crosses RX1 → RX2 → RX3 or the reverse).
- **Logs:** `recordings/live-*.csv.gz` lines are `<time> <RXn> <CSI_DATA line>`; `--replay` plays them back (old single-receiver logs replay as RX1).

---

## 6. Roadmap (next stages)

1. Log raw CSI to file and build a small labelled dataset (empty / walking / sitting / standing).
2. Better features: subcarrier selection, PCA, Hampel filtering, phase sanitisation.
3. Breathing-rate detection for stationary presence.
4. Replace the threshold with a classifier (SVM / random forest, then a small neural network).
5. ~~Multiple receivers for coverage and coarse localisation.~~ Done: 3 receivers, see §5a.
6. Run detection on the ESP32 itself and report over MQTT / Wi-Fi to a dashboard.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Permission denied: /dev/ttyUSB0` | Add your user to the serial group (see §3) and log in again. |
| No port appears | Charge-only cable, or missing USB-serial driver (CP210x / CH340). Try another cable. |
| Flash fails to connect | Hold the **BOOT** button while flashing starts, then release. |
| Receiver prints nothing | The boards are on different channels, or the transmitter is not running. Reflash both with default settings. |
| `detect.py` shows no output | Wrong port or baud rate. Compare with `idf.py monitor` output and the baud rate in `menuconfig`. Also make sure nothing else has the port open. |
| Score is noisy in an empty room | Reduce nearby Wi-Fi traffic, move the boards away from metal objects, increase `CALIB_SECONDS`, or raise `K`. |
| Always says `PRESENT` | Calibration happened while someone was moving. Restart and keep the room empty. |

---

## 8. Project layout

```
Graduation project/
├── README.md        # this file
├── layout.json      # board positions (metres) and receiver MACs
├── server.py        # live 3D display, reads all receivers
├── tracking.py      # combines the links: position, direction
├── sensing.py       # single-link pipeline: movement, breathing
├── detect.py        # first prototype detector (one receiver)
├── record.py        # record one receiver's raw CSI
├── firmware/        # Arduino sketches: csi_tx, csi_rx
└── web/             # the 3D page served by server.py
```

## 9. References

- Espressif ESP-CSI: <https://github.com/espressif/esp-csi>
- ESP-IDF Wi-Fi driver, CSI API (`esp_wifi_set_csi_config`, `esp_wifi_set_csi_rx_cb`): <https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html>
