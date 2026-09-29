// CSI receiver (RX1 / RX2 / RX3) for the classic ESP32 (ESP32-WROOM-32).
//
// The same firmware runs on all three receivers. Which one it is (its
// "receiver ID") is set either at compile time (-DRX_ID=2, done by
// python/flash.py) or later over serial with the command "id 2".
//
// ============================================================================
// WHAT IS CSI? (short version — see README.md for the long one)
// ============================================================================
// A Wi-Fi OFDM signal is made of many narrow sub-channels ("subcarriers"),
// 64 of them in a 20 MHz channel, spaced 312.5 kHz apart. Every packet starts
// with a preamble whose content the receiver knows in advance (the "Long
// Training Field", LTF). By comparing what it received with what it expected,
// the receiver estimates, for each subcarrier k, a complex number
//
//      H[k] = |H[k]| * e^(j * phase[k])
//
// telling how much the channel attenuated (amplitude) and delayed (phase) that
// frequency. That list of complex numbers is the Channel State Information.
// A person in the room adds/blocks/moves reflection paths, which changes H[k].
//
// The ESP32 Wi-Fi driver exposes this estimate through three ESP-IDF APIs
// (all present in ESP-IDF v4.x and v5.x for the classic ESP32):
//   esp_wifi_set_csi_config(&cfg)   which LTFs to report, filtering, scaling
//   esp_wifi_set_csi_rx_cb(cb, ctx) function called for every CSI measurement
//   esp_wifi_set_csi(true)          switch CSI collection on
//
// ============================================================================
// OUTPUT FORMAT (one text line per packet, 921600 baud)
// ============================================================================
//   CSI,<rx_id>,<rx_count>,<tx_seq>,<esp_timestamp_us>,<src_mac>,<rssi>,
//       <noise_floor>,<channel>,<secondary_channel>,<sig_mode>,<mcs>,<cwb>,
//       <rate>,<stbc>,<sig_len>,<rx_state>,<first_word_invalid>,<csi_len>,
//       <csi values separated by spaces>
//
// Lines starting with '#' are status/diagnostic messages:
//   # BOOT   ...   printed at start-up and every 30 s (configuration)
//   # STATUS ...   printed every second (counters, packet rate)
// The field meanings are documented in README.md ("Dataset format").
//
// We deliberately send RAW I/Q values, not amplitude/phase: that is lossless,
// smaller on the serial line, and amplitude/phase are trivial to compute on
// the laptop (python/preprocessing/csi_math.py).

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <esp_idf_version.h>
#include <Preferences.h>
#include <csi_common.h>

// ---------------------------------------------------------------- settings
// 0 = "not set at compile time": use the ID stored with the serial command.
#ifndef RX_ID
#define RX_ID 0
#endif

// Which Long Training Fields to report (see wifi_csi_config_t below).
#ifndef RX_CSI_LLTF
#define RX_CSI_LLTF 1
#endif
#ifndef RX_CSI_HTLTF
#define RX_CSI_HTLTF 0
#endif
#ifndef RX_CSI_STBC_HTLTF2
#define RX_CSI_STBC_HTLTF2 0
#endif
#ifndef RX_CSI_LTF_MERGE
#define RX_CSI_LTF_MERGE 0
#endif
#ifndef RX_CSI_CHANNEL_FILTER
#define RX_CSI_CHANNEL_FILTER 0
#endif
// 1 = keep only CSI from the TX's MAC (recommended). 0 = keep everything
// (every access point / phone nearby) — only useful for exploration.
#ifndef RX_FILTER_TX_MAC
#define RX_FILTER_TX_MAC 1
#endif

static const uint8_t TX_MAC[6] = {CSI_TX_MAC};

// Largest CSI buffer the ESP32 can produce: L-LTF (128 B) + HT-LTF (128 B) +
// STBC-HT-LTF (128 B) for a 20 MHz channel.
static const size_t MAX_CSI_LEN = 384;

// One CSI measurement, copied out of the Wi-Fi driver's memory.
struct CsiRecord {
  uint32_t rx_count;        // our own counter of accepted CSI records
  int64_t tx_seq;           // TX sequence number from the payload, -1 if not found
  wifi_pkt_rx_ctrl_t ctrl;  // radio metadata (RSSI, channel, rate, timestamp...)
  uint8_t src_mac[6];
  bool first_word_invalid;
  uint16_t len;             // number of bytes in buf (2 bytes per subcarrier)
  int8_t buf[MAX_CSI_LEN];
};

static QueueHandle_t queue;
static uint8_t rxId = RX_ID;
static char ownMac[18];

// Counters (written only in the Wi-Fi task, read in loop(); aligned 32-bit
// reads/writes are atomic on the ESP32, so no lock is needed)
static uint32_t csiTotal = 0;    // all CSI callbacks
static uint32_t csiOther = 0;    // from other MACs (filtered out)
static uint32_t csiOk = 0;       // accepted (queued or dropped)
static uint32_t queueDrops = 0;  // queue full -> record lost (serial too slow)
static uint32_t noSeq = 0;       // accepted, but TX sequence number not found
static uint32_t seqGaps = 0;     // TX packets missed by this RX (from seq jumps)

// Finds our csi_tx_header_t inside the received frame and returns the TX
// sequence number, or -1. We search for the magic number instead of using a
// fixed offset, because the ESP-NOW frame wraps our payload in an 802.11
// action-frame header plus a vendor-specific element (~15 bytes).
static int64_t findTxSeq(const uint8_t *data, uint16_t len) {
  if (!data || len < sizeof(csi_tx_header_t)) return -1;
  for (uint16_t i = 0; i + sizeof(csi_tx_header_t) <= len; i++) {
    csi_tx_header_t h;
    memcpy(&h, data + i, sizeof(h));  // memcpy: data may be unaligned
    if (h.magic == CSI_PACKET_MAGIC) return h.seq;
  }
  return -1;
}

// ============================================================================
// THE CSI CALLBACK — called by the Wi-Fi driver for every received frame that
// has CSI. It runs inside the Wi-Fi task, so it must be FAST: no printing, no
// blocking. We copy the data and hand it to loop() through a FreeRTOS queue.
// ============================================================================
static void onCsi(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len == 0) return;
  csiTotal++;

  // info->mac is the SOURCE MAC of the frame this CSI was measured on.
  // Keep only our transmitter's frames: other devices have other positions,
  // other rates and random timing, which would just be noise for us.
  if (RX_FILTER_TX_MAC && memcmp(info->mac, TX_MAC, 6) != 0) {
    csiOther++;
    return;
  }

  static CsiRecord r;  // static: too big for the Wi-Fi task's stack
  r.rx_count = csiOk++;

  // info->rx_ctrl holds the radio metadata of the frame (wifi_pkt_rx_ctrl_t):
  //   rssi        received signal strength of the WHOLE packet, dBm. One number
  //               per packet — not per subcarrier. Coarse (1 dB steps).
  //   noise_floor RF noise floor estimate, dBm (classic ESP32 only field position)
  //   channel     primary channel the frame was received on
  //   sig_mode    0 = legacy 802.11b/g frame, 1 = 802.11n (HT) frame
  //   mcs, cwb    HT modulation index, bandwidth (0 = 20 MHz)
  //   timestamp   receiver's local time in microseconds (32-bit, wraps ~71 min)
  //   sig_len     length of the packet in bytes
  //   rx_state    0 = received without error
  r.ctrl = info->rx_ctrl;
  memcpy(r.src_mac, info->mac, 6);

  // info->buf is the CSI itself: a list of signed 8-bit integers, TWO per
  // subcarrier: [imaginary, real, imaginary, real, ...]  (imaginary FIRST).
  //   H[k] = buf[2k+1] + j*buf[2k]
  //
  // The order of subcarriers in buf for a 20 MHz channel, secondary channel
  // "none" (our setup), with only the L-LTF enabled (128 bytes = 64 values):
  //     buf index k = 0..31  -> subcarrier  0 .. 31
  //     buf index k = 32..63 -> subcarrier -32 .. -1
  // For an HT frame with HT-LTF also enabled, another 128 bytes follow with
  // the same order (HT-LTF). The laptop re-orders them to -32..31
  // (python/preprocessing/csi_math.py). Not every subcarrier carries signal:
  // L-LTF uses -26..-1 and 1..26 only; 0 (DC) and the edges are ~0 ("null").
  //
  // info->first_word_invalid: on the classic ESP32 the first 4 bytes of buf
  // (= the first 2 subcarriers in buf order: 0 and 1) can be garbage because
  // of a hardware limitation. We pass the flag on instead of zeroing, so the
  // raw data stays raw; the laptop masks those subcarriers.
  r.first_word_invalid = info->first_word_invalid;
  r.len = info->len > MAX_CSI_LEN ? MAX_CSI_LEN : info->len;
  memcpy(r.buf, info->buf, r.len);

  // Where does our TX sequence number sit? info->payload points to the frame
  // body (payload_len bytes) — our csi_tx_header_t is somewhere inside it.
  r.tx_seq = findTxSeq(info->payload, info->payload_len);
  if (r.tx_seq < 0) noSeq++;
  else {
    static int64_t lastSeq = -1;
    if (lastSeq >= 0 && r.tx_seq > lastSeq + 1 && r.tx_seq - lastSeq < 100000)
      seqGaps += (uint32_t)(r.tx_seq - lastSeq - 1);
    lastSeq = r.tx_seq;  // (a TX reboot makes seq go back to 0: no gap counted)
  }

  // Non-blocking send: if loop() cannot print fast enough the record is
  // dropped and counted, rather than stalling the Wi-Fi driver.
  if (xQueueSend(queue, &r, 0) != pdTRUE) queueDrops++;
}

static void fail(const char *what, esp_err_t err) {
  while (true) {
    Serial.printf("# ERROR RX: %s failed: %s\n", what, esp_err_to_name(err));
    delay(2000);
  }
}

static void printConfig() {
  Serial.printf("# BOOT role=RX fw=%s idf=%s rx_id=%u mac=%s channel=%d "
                "tx_mac=%02x:%02x:%02x:%02x:%02x:%02x filter_tx_mac=%d lltf=%d htltf=%d stbc_htltf2=%d "
                "ltf_merge=%d channel_filter=%d baud=%d\n",
                CSI_FIRMWARE_VERSION, esp_get_idf_version(), rxId, ownMac, CSI_CHANNEL,
                TX_MAC[0], TX_MAC[1], TX_MAC[2], TX_MAC[3], TX_MAC[4], TX_MAC[5],
                RX_FILTER_TX_MAC, RX_CSI_LLTF, RX_CSI_HTLTF, RX_CSI_STBC_HTLTF2,
                RX_CSI_LTF_MERGE, RX_CSI_CHANNEL_FILTER, RX_BAUD);
  if (rxId == 0) Serial.println("# WARNING RX: no receiver ID set. Type 'id 1' (or 2, 3) and press Enter.");
}

// Serial commands: "id <1..255>" stores the receiver ID in flash (NVS),
// "info" prints the configuration again.
static void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.startsWith("id ")) {
    int id = cmd.substring(3).toInt();
    if (RX_ID != 0) {
      Serial.printf("# ERROR RX: this build has a fixed ID (RX_ID=%d); reflash with RX_ID=0 to set it over serial\n", RX_ID);
    } else if (id >= 1 && id <= 255) {
      Preferences p;
      p.begin("csi", false);
      p.putUChar("rx_id", (uint8_t)id);
      p.end();
      rxId = id;
      Serial.printf("# OK RX: receiver ID set to %d (stored)\n", id);
    } else {
      Serial.println("# ERROR RX: usage: id <1..255>");
    }
  } else if (cmd == "info") {
    printConfig();
  } else if (cmd.length()) {
    Serial.println("# ERROR RX: unknown command (use 'id N' or 'info')");
  }
}

void setup() {
  Serial.setTxBufferSize(8192);  // absorb short bursts without blocking
  Serial.begin(RX_BAUD);
  delay(200);
  queue = xQueueCreate(64, sizeof(CsiRecord));

  // Receiver ID: compile-time value wins; otherwise the one stored in NVS.
  if (RX_ID == 0) {
    Preferences p;
    p.begin("csi", true);
    rxId = p.getUChar("rx_id", 0);
    p.end();
  }

  WiFi.mode(WIFI_STA);  // starts the Wi-Fi driver (no network connection)
  WiFi.disconnect();

  uint8_t m[6];
  esp_read_mac(m, ESP_MAC_WIFI_STA);  // factory MAC, identifies this board
  snprintf(ownMac, sizeof(ownMac), "%02x:%02x:%02x:%02x:%02x:%02x", m[0], m[1], m[2], m[3], m[4], m[5]);

  esp_err_t err;
  if ((err = esp_wifi_set_ps(WIFI_PS_NONE)) != ESP_OK) fail("esp_wifi_set_ps", err);
  if ((err = esp_wifi_set_channel(CSI_CHANNEL, WIFI_SECOND_CHAN_NONE)) != ESP_OK) fail("esp_wifi_set_channel", err);
  // ESP-NOW is initialised so the driver handles ESP-NOW action frames
  // normally; promiscuous mode makes the radio pass up every frame on the
  // channel (including broadcasts not addressed to us), so CSI is produced
  // for all of them. The MAC filter in onCsi() keeps only the TX.
  if ((err = esp_now_init()) != ESP_OK) fail("esp_now_init", err);
  if ((err = esp_wifi_set_promiscuous(true)) != ESP_OK) fail("esp_wifi_set_promiscuous", err);

  // --------------------------------------------------------------------------
  // CSI configuration (wifi_csi_config_t, classic-ESP32 layout; chips with
  // Wi-Fi 6 such as ESP32-C5/C6 use a different struct with the same name).
  // --------------------------------------------------------------------------
  wifi_csi_config_t cfg = {};
  // L-LTF: legacy long training field, present in EVERY OFDM frame (11g and
  // 11n). Gives 64 subcarriers = 128 bytes. This is our main measurement.
  cfg.lltf_en = RX_CSI_LLTF;
  // HT-LTF: extra training field only in 802.11n (HT) frames. Adds another
  // 64 subcarriers (128 bytes, -28..28 used). Doubles the serial traffic.
  cfg.htltf_en = RX_CSI_HTLTF;
  // STBC HT-LTF2: only for STBC frames (our TX does not send STBC).
  cfg.stbc_htltf2_en = RX_CSI_STBC_HTLTF2;
  // ltf_merge: report the average of L-LTF and HT-LTF instead of HT-LTF alone.
  // Off = keep the two measurements separate (more "raw").
  cfg.ltf_merge_en = RX_CSI_LTF_MERGE;
  // channel_filter: smooths neighbouring subcarriers in hardware. Off keeps
  // every subcarrier an independent measurement (better for research).
  cfg.channel_filter_en = RX_CSI_CHANNEL_FILTER;
  // manu_scale=false: the driver scales the CSI values automatically to fit
  // int8. NOTE: together with the receiver's automatic gain control (AGC)
  // this means the absolute amplitude can jump between packets — the laptop
  // compensates by normalising (see preprocessing). Not controllable via the
  // public API on the classic ESP32.
  cfg.manu_scale = false;
  cfg.shift = 0;
  cfg.dump_ack_en = false;  // do not report CSI of ACK frames

  if ((err = esp_wifi_set_csi_config(&cfg)) != ESP_OK) fail("esp_wifi_set_csi_config", err);
  if ((err = esp_wifi_set_csi_rx_cb(onCsi, nullptr)) != ESP_OK) fail("esp_wifi_set_csi_rx_cb", err);
  if ((err = esp_wifi_set_csi(true)) != ESP_OK) fail("esp_wifi_set_csi", err);

  printConfig();
}

// Prints one CSI record as a text line. Fields: see header comment / README.
static void printRecord(const CsiRecord &r) {
  static char line[2600];  // header ~130 chars + up to 384 values * 5 chars
  const wifi_pkt_rx_ctrl_t &c = r.ctrl;
  int n = snprintf(line, sizeof(line),
                   "CSI,%u,%lu,%lld,%lu,%02x:%02x:%02x:%02x:%02x:%02x,%d,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%u,",
                   rxId, (unsigned long)r.rx_count, (long long)r.tx_seq, (unsigned long)c.timestamp,
                   r.src_mac[0], r.src_mac[1], r.src_mac[2], r.src_mac[3], r.src_mac[4], r.src_mac[5],
                   (int)c.rssi, (int)c.noise_floor, (unsigned)c.channel, (unsigned)c.secondary_channel,
                   (unsigned)c.sig_mode, (unsigned)c.mcs, (unsigned)c.cwb, (unsigned)c.rate, (unsigned)c.stbc,
                   (unsigned)c.sig_len, (unsigned)c.rx_state, r.first_word_invalid ? 1 : 0, (unsigned)r.len);
  for (uint16_t i = 0; i < r.len && n < (int)sizeof(line) - 8; i++) {
    n += snprintf(line + n, sizeof(line) - n, i ? " %d" : "%d", r.buf[i]);
  }
  line[n++] = '\n';
  Serial.write((const uint8_t *)line, n);
}

void loop() {
  static CsiRecord r;
  static String cmd;
  static uint32_t lastStatusMs = 0, okAtLastStatus = 0;
  static float lastMeanAmp = 0;
  static int lastRssi = 0;

  // Print everything that is waiting (up to 50 ms of blocking wait).
  while (xQueueReceive(queue, &r, pdMS_TO_TICKS(50)) == pdTRUE) {
    printRecord(r);
    // Mean L-LTF amplitude of this packet, only for the human-readable status
    float s = 0;
    int cnt = 0;
    for (int k = 2; k + 1 < r.len && k < 128; k += 2, cnt++) s += sqrtf((float)r.buf[k] * r.buf[k] + (float)r.buf[k + 1] * r.buf[k + 1]);
    lastMeanAmp = cnt ? s / cnt : 0;
    lastRssi = r.ctrl.rssi;
    if (millis() - lastStatusMs >= 1000) break;
  }

  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') { handleCommand(cmd); cmd = ""; }
    else if (cmd.length() < 32) cmd += ch;
  }

  uint32_t now = millis();
  if (now - lastStatusMs >= 1000) {
    uint32_t ok = csiOk;
    float rate = (ok - okAtLastStatus) * 1000.0f / (now - lastStatusMs);
    Serial.printf("# STATUS role=RX rx_id=%u mac=%s uptime_ms=%lu csi_ok=%lu rate=%.1f csi_other=%lu "
                  "queue_drops=%lu seq_gaps=%lu no_seq=%lu last_rssi=%d last_mean_amp=%.1f\n",
                  rxId, ownMac, (unsigned long)now, (unsigned long)ok, rate, (unsigned long)csiOther,
                  (unsigned long)queueDrops, (unsigned long)seqGaps, (unsigned long)noSeq, lastRssi, lastMeanAmp);
    if (ok == okAtLastStatus && now > 3000)
      Serial.printf("# WARNING RX: no CSI from the TX in the last second (channel %d). Is the TX powered?\n", CSI_CHANNEL);
    if ((now / 1000) % 30 == 0) printConfig();
    lastStatusMs = now;
    okAtLastStatus = ok;
  }
}
