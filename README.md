# Wi-Fi CSI sensing prototype — 1 × ESP32 TX, 3 × ESP32 RX

A research prototype (Pre-Thesis 1) that measures Wi-Fi **Channel State Information (CSI)** on three
TX→RX links, records it losslessly, displays it live, compares it with an empty-room baseline, and
draws an **experimental** 2D heatmap. No machine learning is used anywhere.

```
                ESP32 TX  (ESP-NOW broadcast, 100 packets/s, channel 11, OFDM rate)
                   |
          Wi-Fi packets (one packet is heard by all three receivers)
                   |
      +------------+------------+
      |            |            |
    RX1          RX2          RX3      each: CSI callback -> text line on USB serial (921600 baud)
      |            |            |
      +------------+------------+
                   |
                Laptop  python/collect.py  -> data/raw/<session>/csi.csv   (raw, never modified)
                        python/live.py     -> live plots, baseline/change, experimental heatmap
                        python/export.py   -> data/processed/<session>/ (NumPy / wide CSV)
```

> **Honesty note.** The heatmap is *not* a position estimate. Three links cannot determine where a
> person is; the map only shows which links changed, spread over the region each link is most
> sensitive to. Every figure and file that shows it is labelled EXPERIMENTAL.

---

## Contents

1. [Technical verification (read first)](#1-technical-verification)
2. [Project layout](#2-project-layout)
3. [Installation](#3-installation)
4. [CSI in five minutes](#4-csi-in-five-minutes)
5. [Stage 1 — TX → RX1, CSI visible on serial](#stage-1--tx--rx1-csi-visible-on-serial)
6. [Stage 2 — TX → RX1/RX2/RX3](#stage-2--tx--rx1rx2rx3)
7. [Stage 3 — three receivers → laptop → CSV](#stage-3--three-receivers--laptop--csv)
8. [Stage 4 — live CSI visualisation](#stage-4--live-csi-visualisation)
9. [Stage 5 — empty-room baseline and change detection](#stage-5--empty-room-baseline-and-change-detection)
10. [Stage 6 — experimental 2D heatmap](#stage-6--experimental-2d-heatmap)
11. [Dataset format](#11-dataset-format)
12. [Replay and simulation](#12-replay-and-simulation)
13. [Known limitations](#13-known-limitations)

---

## 1. Technical verification

Everything below was checked against the ESP-IDF headers that are actually installed with the
toolchain (`~/.arduino15/packages/esp32/tools/esp32-libs/3.3.12/include/esp_wifi/...`), not from memory.

### 1.1 Which ESP-IDF version, and why

| | |
|---|---|
| Toolchain | **Arduino-ESP32 core 3.3.12**, driven by `arduino-cli` (bundled with Arduino IDE 2) |
| ESP-IDF inside it | **v5.5.5** (from the core's `versions.txt` and `IDF_VER` build flag; printed by the boards at boot as `idf=`) |
| Board | FQBN `esp32:esp32:esp32` = classic ESP32 (ESP32-D0WD in the WROOM-32 module) |
| CSI support | `CONFIG_SOC_WIFI_CSI_SUPPORT=y`, `CONFIG_ESP_WIFI_CSI_ENABLED=y` in the core's prebuilt sdkconfig |

Why this route instead of a bare `idf.py` project:

* The CSI functions are **pure ESP-IDF APIs** (`esp_wifi_set_csi_*`, `esp_now_*`). The Arduino core is
  ESP-IDF v5.5.5 plus a thin layer; the firmware only uses Arduino for `Serial`, `setup()/loop()`,
  `WiFi.mode()` (which starts the Wi-Fi driver) and `Preferences` (NVS storage).
* The Arduino core ships ESP-IDF precompiled with CSI enabled, so there is no `menuconfig` step and
  the same commands work on Windows and Linux.
* This toolchain was already used successfully with these exact boards (earlier prototype, now in
  `legacy/`): ~100 CSI packets/s per receiver, 64 subcarriers.

If you later move to plain ESP-IDF v5.5: the Wi-Fi/CSI code carries over unchanged; replace `Serial`
with `printf`/UART, `WiFi.mode()` with `esp_netif_init() + esp_event_loop_create_default() +
esp_wifi_init() + esp_wifi_set_mode(WIFI_MODE_STA) + esp_wifi_start()`, and `Preferences` with `nvs_*`.

### 1.2 CSI APIs available (ESP-IDF v5.5, classic ESP32)

| API | Purpose |
|---|---|
| `esp_wifi_set_csi_config(const wifi_csi_config_t*)` | which training fields to report, filtering, scaling |
| `esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx)` | register `void cb(void *ctx, wifi_csi_info_t *info)` |
| `esp_wifi_set_csi(bool en)` | switch CSI on/off |
| `wifi_csi_config_t` (classic ESP32) | `lltf_en, htltf_en, stbc_htltf2_en, ltf_merge_en, channel_filter_en, manu_scale, shift, dump_ack_en` |
| `wifi_csi_info_t` | `rx_ctrl` (metadata), `mac` (source), `dmac`, `first_word_invalid`, `buf` (int8 CSI), `len`, `hdr`, `payload`, `payload_len`, `rx_seq` |
| `wifi_pkt_rx_ctrl_t` (classic ESP32 layout) | `rssi, rate, sig_mode, mcs, cwb, smoothing, not_sounding, aggregation, stbc, fec_coding, sgi, noise_floor, ampdu_cnt, channel, secondary_channel, timestamp, ant, sig_len, rx_state` |

**Version differences you should know about** (not silently assumed):

* **ESP-NOW PHY rate.** ESP-IDF ≥ 5.4: `esp_now_set_peer_rate_config(peer, &esp_now_rate_config_t{phymode, rate, ...})`.
  Older versions: `esp_wifi_config_espnow_rate(ifx, rate)` (deprecated in 5.4+). `tx.ino` picks the
  right one with `#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)`.
* **Wi-Fi 6 chips (ESP32-C5/C6) use a different `wifi_csi_config_t`** (`wifi_csi_acquire_config_t`) and a
  different `rx_ctrl` layout. Irrelevant for the classic ESP32, but code copied from C6 examples
  will not compile here, and vice versa.
* **`noise_floor` sits at a different position in `wifi_pkt_rx_ctrl_t` on each chip** (ESP32 vs
  S2 vs S3/C3). We only access fields by name, so this is handled by the compiler.
* **ESP-NOW send callback signature changed in v5.5** (`esp_now_send_info_t*` instead of a MAC
  pointer). We do not use the send callback, so both versions are fine.
* `wifi_csi_info_t.payload/payload_len/rx_seq` are present in the v5.5 headers (verified). Very old
  versions (Arduino-ESP32 2.x = ESP-IDF 4.4) may not have them; the TX sequence number extraction
  would then have to be removed.

### 1.3 How ESP32-to-ESP32 CSI is configured here

* **No access point, no SSID/password.** The TX broadcasts **ESP-NOW** frames to `ff:ff:ff:ff:ff:ff`.
  All three receivers hear the *same* frame, so the same packet is measured on three links at once.
  (`config.yaml` still has `ssid/password` fields, set to `null`, to make that explicit.)
* **Fixed channel** (`esp_wifi_set_channel(11, WIFI_SECOND_CHAN_NONE)`, 20 MHz) on TX and RX.
* **Fixed TX MAC** (`esp_wifi_set_mac`, `1a:00:00:00:00:00`, locally administered). Receivers keep
  only CSI whose `info->mac` equals it, and ignore every other device in the building.
* **OFDM rate is mandatory.** ESP-NOW defaults to 1 Mbps 802.11b (DSSS), which has **no subcarriers
  and produces no CSI**. The TX sends HT20 MCS0 (6.5 Mbps) or 11g 6 Mbps.
* Power save off on both sides (`esp_wifi_set_ps(WIFI_PS_NONE)`), so timing and timestamps are stable.

### 1.4 How the TX generates traffic

`firmware/tx/tx.ino`: `esp_now_send()` in `loop()`, paced by `vTaskDelayUntil()` at
`packets_per_second` (default 100). The first 12 payload bytes are `magic "CSI1" + seq + tx_ms`.
Failed sends (TX queue full) are counted and printed every second.

### 1.5 How CSI is captured on three receivers

`firmware/rx/rx.ino` (same binary logic on all three):

1. `esp_now_init()` and **promiscuous mode** so the radio passes up broadcast frames;
2. `esp_wifi_set_csi_config()` with **L-LTF only** by default (64 subcarriers, 128 bytes),
   channel filter off (keeps subcarriers independent);
3. `esp_wifi_set_csi_rx_cb(onCsi)`: the callback runs in the Wi-Fi task, so it only filters by MAC,
   copies the data and the metadata into a FreeRTOS queue and returns;
4. `loop()` prints one text line per record; drops (queue full = serial too slow) are counted;
5. the receiver ID comes from the build (`--id N`) or the serial command `id N` (stored in NVS).

### 1.6 Limitations of one TX + three ESP32 receivers

See [section 13](#13-known-limitations) for the full list. The most important:

* The three receivers have **independent clocks**: their CSI phases cannot be compared with each other,
  so no angle-of-arrival or time-of-flight across boards. Each link gives amplitude + phase *shape* only.
* Each ESP32 has **one antenna** and an **automatic gain control (AGC)** that cannot be locked on the
  classic ESP32 via public APIs; absolute CSI amplitude jumps from packet to packet → normalisation.
* **Serial bandwidth**: 921600 baud ≈ 92 kB/s per receiver; a 64-subcarrier line is ~450 B, so
  100 packets/s uses about half. Enabling HT-LTF doubles the line length (keep ≤ ~60 packets/s then).
* **Three links cannot localise a person.** See the heatmap section.

### 1.7 Architecture in one paragraph

Hardware code (`firmware/`) only produces raw, lossless text records. On the laptop, *sources*
(serial / replay / simulator) push records into a *hub* which **first** writes them to disk and then
hands them to the display. Processing (`python/preprocessing/`) and the spatial algorithm
(`python/localization/`) are separate, swappable modules configured from `config/config.yaml`.

---

## 2. Project layout

```
firmware/
  common/csi_common/csi_common.h   settings shared by TX and RX (channel, TX MAC, packet header)
  tx/tx.ino                        transmitter
  rx/rx.ino                        receiver (RX1/RX2/RX3) — heavily commented CSI extraction
python/
  flash.py                         build + upload firmware with values from config.yaml
  collect.py                       stage 3: record raw CSI from all receivers (no GUI)
  live.py                          stages 4-6: live plots, baseline/change, heatmap
  dashboard.py                     one window: TX-centred approximate-position map + graphs, swap button
  export.py                        raw CSV -> NumPy (.npz) / wide CSV / packets aligned across RX
  common/        config loading, record parsing (the firmware line format lives here)
  collector/     serial reader (auto-detect, reconnect), replay, simulator, hub, CSV writer
  preprocessing/ csi_math (raw -> complex -> amplitude/phase), filters, pipeline, baseline
  localization/  EXPERIMENTAL spatial algorithms (base class + ellipse weighting)
  visualization/ matplotlib figures
config/config.yaml                 every tunable value, with comments
data/raw/        one folder per recording session (not in git)
data/processed/  exports, baselines (not in git)
legacy/          the earlier single-file prototype (web UI, sensing.py, ...) kept for reference
```

---

## 3. Installation

### 3.1 Firmware toolchain (once)

Arduino IDE 2 contains `arduino-cli`; `python/flash.py` finds it automatically (or install
`arduino-cli` separately and put it on `PATH`). Install the ESP32 core:

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32@3.3.12
```
(On this machine the CLI is `/opt/arduino-ide/resources/app/lib/backend/resources/arduino-cli` and the
core is already installed.)

USB drivers: the boards use CP2102 (`/dev/ttyUSB*`) or CH343 (`/dev/ttyACM*`) chips. Linux: your user
must be in the `uucp` (Arch) or `dialout` (Debian/Ubuntu) group. Windows: install the CP210x/CH343
driver if no COM port appears.

### 3.2 Python (once)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
All commands below assume the venv is active and the current directory is the project root.

### 3.3 Find the serial ports

```bash
python -m serial.tools.list_ports -v
```
Plug the boards in one at a time to see which port belongs to which board. Each board's factory MAC
(printed in its `# BOOT` line, or `esptool read-mac`) lets the collector recognise it on any port.

---

## 4. CSI in five minutes

Wi-Fi (802.11g/n, 20 MHz) splits the channel into **64 subcarriers** 312.5 kHz apart (numbered −32…31).
Every packet starts with a *training field* whose content the receiver knows. Comparing the received
training field with the known one gives, for every subcarrier *k*, a complex number

    H[k] = real + j·imag = |H[k]| · e^(j·phase[k])

— the **Channel State Information**. `|H[k]|` (amplitude) says how strongly frequency *k* arrived,
`phase[k]` how much it was delayed. Walls, furniture and people create many reflected copies of the
signal that add up differently at each frequency, so `|H[k]|` vs *k* has a characteristic
"fingerprint" shape. **A person moving changes the reflections, and therefore the fingerprint.**

What the ESP32 gives you, and what it does not:

| Quantity | What it is | Per | Quality on ESP32 |
|---|---|---|---|
| **RSSI** | received power of the whole packet, dBm | packet | coarse (1 dB), one number |
| **raw CSI** | int8 pairs `[imag, real]` per subcarrier | subcarrier | 8-bit, auto-scaled |
| **CSI amplitude** | `sqrt(real² + imag²)` | subcarrier | good, but scaled by AGC per packet |
| **CSI phase (raw)** | `atan2(imag, real)` | subcarrier | dominated by random clock offsets |
| **sanitized phase** | raw phase, unwrapped, linear trend removed | subcarrier | usable *shape*, no absolute value |
| **processed CSI** | anything normalised/filtered/smoothed | subcarrier | depends on `config.yaml` |

Buffer layout (L-LTF, 20 MHz): `buf = [imag(sc0), real(sc0), imag(sc1), real(sc1), …, imag(sc31),
real(sc31), imag(sc−32), real(sc−32), …, imag(sc−1), real(sc−1)]`. Only subcarriers −26…−1 and
1…26 carry energy; 0 (DC) and the edges are empty ("null") and are masked by default. When
`first_word_invalid = 1`, the first 4 bytes (subcarriers 0 and 1) are garbage and are masked too.
See `firmware/rx/rx.ino` (`onCsi`) and `python/preprocessing/csi_math.py` (`to_complex`).

---

## Stage 1 — TX → RX1, CSI visible on serial

**Files:** `firmware/common/csi_common/csi_common.h`, `firmware/tx/tx.ino`, `firmware/rx/rx.ino`,
`config/config.yaml` (`firmware:` section), `python/flash.py`.

**Commands** (replace ports; Windows: `COM5` etc.):

```bash
python python/flash.py tx --compile-only          # optional: check that it builds (~1 min first time)
python python/flash.py tx --port /dev/ttyUSB0      # flash the transmitter
python python/flash.py rx --port /dev/ttyUSB1 --id 1   # flash RX1
python -m serial.tools.miniterm /dev/ttyUSB0 115200    # watch the TX   (exit: Ctrl+])
python -m serial.tools.miniterm /dev/ttyUSB1 921600    # watch RX1
```

`flash.py` prints the `-D` flags it derived from `config.yaml` and the exact `arduino-cli` command it
runs, so you can also run that command by hand.

**Expected TX output** (every second, BOOT repeated every 30 s):
```
# BOOT role=TX fw=1.0 idf=v5.5.5 mac=1a:00:00:00:00:00 channel=11 pps=100 payload=32 phy=HT20-MCS0 tx_power_qdbm=0
# STATUS role=TX uptime_ms=5004 sent=500 send_errors=0 rate=100.0 channel=11
```

**Expected RX1 output** (≈100 `CSI` lines per second, one `# STATUS` per second):
```
# BOOT role=RX fw=1.0 idf=v5.5.5 rx_id=1 mac=8c:94:df:aa:53:84 channel=11 tx_mac=1a:00:00:00:00:00 filter_tx_mac=1 lltf=1 htltf=0 ...
CSI,1,5321,5400,123456789,1a:00:00:00:00:00,-45,-93,11,0,1,0,0,0,0,51,0,1,128,4 -7 0 0 -12 9 -13 8 ...
# STATUS role=RX rx_id=1 mac=8c:94:df:aa:53:84 uptime_ms=60001 csi_ok=5980 rate=99.0 csi_other=12 queue_drops=0 seq_gaps=3 no_seq=0 last_rssi=-45 last_mean_amp=17.3
```

**How to verify it works**

* TX: `rate=` ≈ `pps`, `send_errors=0`, on-board LED toggles every second.
* RX: `rate=` in STATUS ≈ TX rate (90–100 at 1–3 m); `csi_len` (19th field) = `128`;
  `sig_mode` (11th field) = `1` for HT20.
* The **TX sequence number** (4th field) must count up by 1 per line. `no_seq=0` in STATUS.
  *This is the one part of the firmware not yet exercised on hardware:* if `tx_seq` is always `-1`
  (and `no_seq` grows), CSI is still fine, only lost-packet counting and cross-receiver alignment
  are unavailable — report it and the payload offset search can be adjusted.
* Cover the RX antenna with your hand or walk between TX and RX: `last_rssi` and `last_mean_amp`
  change.

**Common failures**

| Symptom | Cause / fix |
|---|---|
| RX prints `WARNING RX: no CSI from the TX` | TX not powered / different channel / different `tx_mac` → reflash both from the same `config.yaml` |
| RX `rate` ≈ 0 but `csi_other` grows | TX MAC mismatch (check both BOOT lines) |
| No CSI at all, even with `filter_tx_mac: false` | TX sends 802.11b (e.g. `phy` wrong, or old firmware) |
| Upload fails "Failed to connect" | hold **BOOT** while "Connecting…" shows; close any serial monitor on that port |
| Garbage characters | wrong baud rate (TX 115200, RX 921600) |
| `WARNING RX: no receiver ID set` | flashed with `--id 0`: type `id 1` + Enter in the serial monitor |
| `queue_drops` grows | serial too slow: lower `packets_per_second` or disable HT-LTF |

**What the data means:** each `CSI` line = one TX packet as seen by RX1: its metadata (RSSI, channel,
PHY mode, receiver timestamp) and 128 numbers = 64 subcarriers × (imag, real). Nothing is processed yet.

---

## Stage 2 — TX → RX1/RX2/RX3

**Files:** none new. **Commands:**

```bash
python python/flash.py rx --port /dev/ttyUSB2 --id 2
python python/flash.py rx --port /dev/ttyUSB3 --id 3
# check each one:
python -m serial.tools.miniterm /dev/ttyUSB2 921600
python -m serial.tools.miniterm /dev/ttyUSB3 921600
```

**Expected output:** as in stage 1, with `rx_id=2`/`rx_id=3` and the first field of every `CSI` line
equal to the ID. **The same `tx_seq` values appear on all three receivers** (the same packets).

**Verify:** all three `rate≈100`. Note each board's `mac=` and put it into `config.yaml → receivers`
(already filled in for the current boards). Put the MAC-to-label mapping on the boards with tape.

**Common failures:** a board shows `rx_id=0` → flash again with `--id`; two boards with the same ID →
the collector assigns only the first; one receiver much lower rate → it is far away / behind metal /
USB hub power problem.

**What the data means:** three independent measurements of *the same* transmitted packet over three
different paths through the room.

---

## Stage 3 — three receivers → laptop → CSV

**Files:** `python/collect.py`, `python/collector/*`, `python/common/records.py`, `config.yaml →
receivers, serial, recording`.

**Commands:**

```bash
python python/collect.py                                   # ports "auto": recognised by rx_id / MAC
python python/collect.py --port 1=/dev/ttyUSB1 --port 2=/dev/ttyUSB2 --port 3=/dev/ttyUSB3
python python/collect.py --duration 60 --label empty-room  # 60 s, folder name suffix
python python/collect.py --no-save                         # only watch
python python/collect.py --sim --duration 10               # no hardware
```

**Expected output** (every second):
```
Saving to data/raw/20260929-153000_empty-room/csi.csv
  * 15:30:01 /dev/ttyUSB1: identified as RX1
  ...
---     12 s  rows saved: 3571
RX   port            conn    packets  rate/s   lost corrupt  rssi fw_drop
RX1  /dev/ttyUSB1    yes        1189    99.0      4       0   -41       0
RX2  /dev/ttyUSB2    yes        1192   100.0      1       0   -47       0
RX3  /dev/ttyUSB3    yes        1190    99.0      3       0   -44       0
```

**Verify:** `rate/s` ≈ 100 for all, `corrupt` stays ~0, `rows saved` grows by ~300/s. Pull a USB
cable: an event `disconnected` appears, `conn` = `no`; plug it back: it reconnects (auto ports
are re-detected within a few seconds). Afterwards:
`python python/export.py data/raw/<session> --align` reports how many packets all three received.

**Note:** on Linux, *opening* the serial port of these DevKits resets the board (the kernel toggles
DTR/RTS before pyserial can prevent it). Receivers therefore reboot once (~1 s) when the collector
starts. The TX port is probed once, recognised as "not a receiver" and then left alone; to be
safe, power the TX from a USB charger instead of the laptop.

**Common failures:** `conn no` for all with `auto` → boards not identified: check that they print
`# STATUS ... rx_id=` (flashed with this firmware) and that MACs in `config.yaml` match; "Permission
denied" → add user to `uucp`/`dialout` group and log in again; a port "busy" → close miniterm/Arduino IDE.

**What the data means:** see [Dataset format](#11-dataset-format). `lost` = TX packets that a receiver
never delivered (gaps in `tx_seq`: radio loss, corrupted lines, firmware queue drops). `fw_drop` =
records the receiver itself dropped because the serial link was busy.

---

## Stage 4 — live CSI visualisation

**Files:** `python/live.py`, `python/visualization/figures.py`, `python/preprocessing/{csi_math,filters,pipeline}.py`,
`config.yaml → preprocessing, visualization`.

**Commands:**

```bash
python python/live.py --views csi               # live from the receivers; raw data is saved too
python python/live.py --views csi --no-save
python python/live.py --views csi --sim         # try it without hardware first
```

**What you see** (one column per receiver):

1. *Amplitude vs subcarrier* — grey dots: **raw** `|H|` of the latest packet (ESP32 units, left axis);
   coloured line: **processed** amplitude (right axis).
2. *Processed amplitude vs time* — a few subcarriers (`visualization.time_subcarriers`) over the last 10 s.
3. *Phase vs subcarrier* — grey: **raw** phase (jumps every packet — that is expected); coloured:
   **sanitized** phase (slope and offset removed).
4. Bottom: **RSSI** of all receivers over time (not CSI), and the link statistics table.

**Verify:** empty room → the amplitude profile is stable, the time traces are flat-ish. Walk through
the room → the time traces of the receiver whose link you cross move strongly; the amplitude profile
changes shape. Standing still again → traces settle.

**Common failures:** "No interactive matplotlib backend" → `pip install PyQt6` or install Tk (see the
message); window very slow → raise `visualization.update_interval_ms`; everything flat even when
walking → check the processed amplitude is not over-smoothed (`smoothing.ema_alpha` too small).

**What the data means / processing chain** (each step switchable in `config.yaml → preprocessing`):

```
raw int8 -> complex H (-32..31) -> mask null / first-word-invalid subcarriers
         -> |H| (raw amplitude)
         -> AGC normalisation   none | mean | rssi
         -> Hampel outlier filter (per subcarrier over time)
         -> EMA or moving-average smoothing         = processed amplitude
phase:   angle(H) (raw)  ->  unwrap + remove least-squares line   = sanitized phase
```

`amplitude_normalization`:
* `mean` divides by the packet's mean amplitude: removes AGC jumps **and** any real overall
  attenuation — only the *shape* across subcarriers remains.
* `rssi` (default) rescales so the packet's average power equals its RSSI (unit: amplitude relative
  to −100 dBm, so 10 ≈ −80 dBm): removes the AGC's arbitrary scaling but **keeps** real attenuation, e.g. a body blocking the direct path. In the
  simulator this made the blocked link stand out; with `mean` it did not. Compare both on real data.

---

## Stage 5 — empty-room baseline and change detection

**Files:** `python/preprocessing/baseline.py`, `python/visualization/figures.py (ChangeFigure)`,
`config.yaml → baseline`.

**Workflow / commands:**

```bash
python python/live.py --views csi,change
```

1. Start the program, **leave the room** (or stand still outside the links). The change window shows
   "BASELINE: keep the room EMPTY — recording …%" for `baseline.duration_seconds` (30 s).
2. The baseline (per receiver: mean `mu[k]` and std `sigma[k]` of the processed amplitude, and the
   empty-room level of the change metric) is saved as `baseline-<time>.npz` in the session folder (or in `data/processed/` with `--no-save`).
3. From then on each receiver shows a live **change ratio**.
4. Press **b** in any window to record a new baseline (after moving furniture, opening doors...).
   Re-use a saved one: `--baseline-file data/raw/<session>/baseline-....npz`.

**Why not `current_CSI − baseline_CSI` on complex values?** The ESP32's raw phase has a random offset
in every packet, so two complex CSI vectors of the same empty room differ hugely — their difference
would measure clock noise, not the room. We compare **processed amplitudes** instead:

| `baseline.metric` | formula | meaning |
|---|---|---|
| `zscore` (default) | mean over k of `|a[k] − mu[k]| / sigma[k]` | how many empty-room standard deviations away, on average |
| `euclidean` | `‖a − mu‖ / ‖mu‖` | relative change of the whole profile |
| `correlation` | `1 − Pearson(a, mu)` | change of *shape* only, insensitive to level |

The metric is smoothed (EMA, `change_smoothing_alpha`) and divided by its own empty-room level
(`threshold_percentile`, default 95th percentile during the baseline) → **ratio**.
*ratio ≈ 1: like the empty room; ratio ≫ 1: something changed on this link.*

**Verify:** after the baseline, empty room → all bars ≲ 1–1.5. Walk across one link → that bar jumps
(typically 3–20×). Stand still in the room → values stay elevated but calmer (a still person
changes the static multipath).

**Common failures:** someone was in the room during the baseline → everything looks "unchanged" while
people are present and "changed" when empty: press **b** with the room empty; `baseline failed:
only N packets` → a receiver delivered too little data (see stage 3); slow drift upward over
hours → the environment changes (doors, temperature, people next door): re-record.

---

## Stage 6 — experimental 2D heatmap

**Files:** `python/localization/{base,ellipse_weighting,__init__}.py`, `python/visualization/figures.py
(HeatmapFigure)`, `config.yaml → room, receivers (x, y), localization`.

**Set up:** measure the room and the board positions (metres, origin in a corner, seen from above)
and enter them in `config.yaml` (`room.width/height`, `room.tx`, each receiver's `x, y`). Put the
boards at the same height (e.g. on 1 m stands).

```bash
python python/live.py                         # all three windows
python python/live.py --views change,heatmap
python python/live.py --sim                   # simulated walk; the true position is drawn as a cyan circle
```

**Algorithm (EXPERIMENTAL, `localization.method: ellipse_weighting`).** For each link TX→RXi and each
point p of a grid, the *excess path length* `e = |TX−p| + |p−RXi| − |TX−RXi|` is 0 on the direct
line and grows away from it (constant on ellipses with foci TX and RXi). Weight
`w_i(p) = exp(−(e/ellipse_width)²)`. Each link's activity `a_i = max(0, ratio_i − activity_floor)`
is spread over its ellipse: `heat(p) = Σ a_i·w_i(p) / (Σ w_i(p) + weight_floor)` (normalised, default)
or `Σ a_i·w_i(p)`.

**What it can and cannot show:** it shows *which link(s)* changed, drawn where those links are most
sensitive. It cannot say where along a link the person is; different positions can give the same
three numbers; reflections off walls also change links when the person is far from them.
With TX on one wall and all RXs on the opposite wall the links fan out from the TX, so the region
near the TX is always ambiguous. **Never report the brightest pixel as a measured location.**

**Replacing the algorithm:** subclass `SpatialEstimator` in `python/localization/`, implement
`estimate(activity) -> Heatmap`, register it in `METHODS` (`localization/__init__.py`), and set
`localization.method` in `config.yaml`. Nothing else changes.

**Verify:** in the simulator, the bright link follows the cyan circle roughly. In the real room: walk
slowly along each link in turn; the corresponding ellipse should light up.

---

## Dashboard — one window, TX-centred map + graphs

```bash
python python/dashboard.py            # live (raw data saved to data/raw/)
python python/dashboard.py --sim      # simulated person, no hardware
```

**Files:** `python/dashboard.py` (Qt window, needs PyQt6), `python/visualization/dashboard_figures.py`,
`python/common/layout.py`, `python/localization/link_pattern.py`, `config.yaml → dashboard, localization`.

* The **map** (left, large) puts the TX at the centre (0, 0) with distance rings every metre. The
  **graphs** (right, narrow) show an *Overview* tab (change per receiver, change over time, CSI
  amplitude variation); the *CSI details* and *Change* tabs are the stage 4/5 figures.
* **⇄ Swap** gives the big part of the window to the graphs (and opens *CSI details*); press again to
  go back. The divider can also be dragged.
* **Receiver positions.** The software *cannot* measure where the receivers are: one antenna per
  board gives no direction, and RSSI gives only a rough distance. So: drag each RX square to where
  the board really stands; **Auto-place RXs (RSSI)** moves each RX along its current direction to a
  distance guessed from its median RSSI (`RSSI(d) = RSSI(1 m) − 10·n·log10(d)`, typically ±50 %
  or worse); **Save positions** writes `config/positions.yaml`, which is loaded on the next start.
* **Record empty room** gives you `baseline_start_delay_seconds` (10 s) to leave, then records the
  baseline (30 s).
* **Approximate position (EXPERIMENTAL, `localization.method: link_pattern`).** For each map
  point the model predicts which links would react to a person there (same excess-path ellipses,
  wider), and compares that *pattern* with the measured link activity. The best-matching region is
  shaded, and its centre is the cyan "you? (approx.)" marker with a 1 m circle and a short trail.
  No position is shown while all links look like the empty room.
  In the simulator (a toy model built on the same geometry, so optimistic), the median error was
  **0.85 m** (90 %: 1.1 m), versus 1.75 m for always guessing the TX. A position was given only
  ~40 % of the time while walking, because many places change no link clearly. Real rooms will do
  worse: the pattern says *near which link(s)*, not *where along* them.

## 11. Dataset format

`data/raw/<YYYYMMDD-HHMMSS>[_label]/`

| File | Content |
|---|---|
| `csi.csv` (`.csv.gz` if `recording.compress`) | one row per CSI record, all receivers interleaved in arrival order |
| `status.log` | every `#` line from the boards: `laptop_time <tab> rx_id <tab> port <tab> line` |
| `session.json` | format version, config snapshot, firmware BOOT lines, row counts, lost/corrupt per RX |
| `baseline-*.npz` | baselines recorded during that session |

`csi.csv` columns (all fields after `port` come unchanged from the firmware line):

| Column | Unit | Meaning |
|---|---|---|
| `host_time` | s (Unix) | laptop clock when the line arrived (includes ~1–10 ms USB latency) |
| `receiver_id` | – | receiver as mapped by the laptop (config.yaml) |
| `port` | – | serial port (or `sim`/`replay`) |
| `fw_rx_id` | – | ID the firmware reported (0 = not set) |
| `rx_count` | – | firmware counter of accepted CSI records (jumps = firmware queue drops) |
| `tx_seq` | – | TX packet number; **identical on all receivers for the same packet**; −1 = unknown |
| `esp_timestamp_us` | µs | receiver's own clock at reception (32-bit, wraps every ~71.6 min); precise *within* one receiver, not comparable across receivers |
| `src_mac` | – | transmitter MAC |
| `rssi` | dBm | whole-packet received signal strength (not CSI) |
| `noise_floor` | dBm | RF noise floor estimate |
| `channel` / `secondary_channel` | – | primary channel; 0 = no secondary (20 MHz) |
| `sig_mode` | – | 0 = legacy 11g OFDM, 1 = 802.11n HT |
| `mcs` / `cwb` / `rate` / `stbc` | – | HT MCS index; 0 = 20 MHz; legacy rate code; STBC flag |
| `sig_len` | bytes | length of the received frame |
| `rx_state` | – | 0 = received without error |
| `first_word_invalid` | 0/1 | 1 = first 4 CSI bytes (subcarriers 0 and 1) are invalid |
| `csi_len` | – | number of values in `csi_raw` (128 = L-LTF only; 256 = L-LTF + HT-LTF) |
| `csi_raw` | int8 | space-separated `imag real imag real …` in ESP32 buffer order (see section 4) |

**Size:** ~500 bytes/row → 100 pkt/s × 3 RX ≈ 150 kB/s ≈ 0.5 GB/hour (`compress: true` ≈ 4× smaller).

**Processed formats** (`python python/export.py <session> [--wide-csv] [--align]`, written to
`data/processed/<session>/`): `csi.npz` (per receiver: complex CSI, amplitude, normalised amplitude,
sanitized phase, valid mask, times, RSSI), `csi_wide.csv` (one amp + phase column pair per used
subcarrier — the "spreadsheet" format), `aligned.npz` (only packets received by all receivers,
matched by `tx_seq`: `csi` shape `(receivers, packets, 64)`). Load with
`d = np.load("csi.npz"); d["rx1_amplitude"]`.

---

## 12. Replay and simulation

```bash
python python/live.py --replay data/raw/20260929-153000_walk             # real-time
python python/live.py --replay data/raw/20260929-153000_walk --speed 5   # 5x faster
python python/live.py --replay data/raw/... --snapshot figs/walk.png --duration 120   # PNGs, no window
python python/live.py --sim                                             # simulated room
python python/collect.py --sim --duration 30                            # simulated recording (saved)
```

Replay uses the recorded `host_time`, so baseline duration and plots behave the same at any speed.
Replays and simulations are **not** saved again unless `--save` is given.

The simulator (`python/collector/sim_source.py`) produces lines in the firmware's exact format and
feeds them through the same parser. Its room model (multipath + a walking person + ESP32 impairments:
random phase offsets, AGC jitter, int8 quantisation, invalid first word, packet loss) is a *toy*: it
uses the same ellipse geometry as the heatmap, so the heatmap looks better in simulation than it will
in reality. Use it to test code, not to validate methods. The first `simulation.empty_seconds` are an
empty room, so the baseline workflow can be tried.

---

## 13. Known limitations

* **No phase coherence between boards.** Independent oscillators → per-packet random phase offsets
  (CFO, sampling time offset, PLL). Only within-packet phase *shape* is usable; no AoA/ToF.
* **AGC / auto-scaling** changes the CSI scale per packet; not lockable on the classic ESP32 via
  public ESP-IDF APIs. Handled by normalisation, which also removes part of the real signal.
* **8-bit CSI** (int8), one antenna, 20 MHz → 52 usable subcarriers, limited dynamic range.
* **Serial bandwidth** caps rate × subcarriers (see 1.6). USB adds ms-level jitter to `host_time`;
  use `tx_seq` to align receivers and `esp_timestamp_us` for precise intervals within one receiver.
* **Shared channel.** Other Wi-Fi networks on the same channel cause collisions/lost packets and
  occupy airtime; pick a quiet channel (`nmcli dev wifi` / phone Wi-Fi analyser).
* **Environment drift.** Doors, furniture, people in adjacent rooms, even temperature change the
  static channel → re-record the baseline regularly.
* **Geometry.** One TX + three RX = three links, all meeting at the TX. Three numbers cannot
  determine a 2D position; the heatmap is a visualisation of link activity only.
* **Not yet validated on real people** for this code base (firmware compiles and the Python pipeline
  is tested with simulated and replayed data and with fake serial devices; stages 1–2 must be run
  on the hardware to confirm, in particular the `tx_seq` extraction).
