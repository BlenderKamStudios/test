// CSI transmitter (TX) for the classic ESP32 (ESP32-WROOM-32).
//
// What it does: broadcasts small ESP-NOW frames at a fixed rate on a fixed
// channel, from a fixed (spoofed) MAC address, at an OFDM PHY rate. Every
// frame is received by RX1, RX2 and RX3 at the same moment, and each of them
// measures the Channel State Information (CSI) of that frame.
//
// Why ESP-NOW broadcast and not a normal Wi-Fi network (SSID/password)?
//   * No access point or router needed, no association, no DHCP.
//   * One broadcast frame is heard by all three receivers -> the same packet
//     is measured on three links simultaneously (matched by sequence number).
//   * We fully control channel, packet rate and PHY rate.
//
// Serial output (115200 baud) is for humans only: status lines start with '#'.
// Once flashed, the TX only needs power (USB charger or power bank).

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_idf_version.h>
#include <csi_common.h>

static const uint8_t TX_MAC[6] = {CSI_TX_MAC};
static const uint8_t BROADCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static const uint8_t LED_PIN = 2;  // on-board LED on most ESP32 DevKit boards

#if TX_PAYLOAD_BYTES < 12 || TX_PAYLOAD_BYTES > 250
#error "TX_PAYLOAD_BYTES must be 12..250 (ESP-NOW v1 payload limit is 250)"
#endif

static uint8_t payload[TX_PAYLOAD_BYTES];
static uint32_t sent = 0, sendErrors = 0;

static void fail(const char *what, esp_err_t err) {
  // Never return: keep telling the user what went wrong.
  while (true) {
    Serial.printf("# ERROR TX: %s failed: %s\n", what, esp_err_to_name(err));
    delay(2000);
  }
}

static void printConfig() {
  Serial.printf("# BOOT role=TX fw=%s idf=%s mac=%02x:%02x:%02x:%02x:%02x:%02x channel=%d "
                "pps=%d payload=%d phy=%s tx_power_qdbm=%d\n",
                CSI_FIRMWARE_VERSION, esp_get_idf_version(),
                TX_MAC[0], TX_MAC[1], TX_MAC[2], TX_MAC[3], TX_MAC[4], TX_MAC[5],
                CSI_CHANNEL, TX_PACKETS_PER_SECOND, TX_PAYLOAD_BYTES,
                TX_PHY_HT20 ? "HT20-MCS0" : "11g-6M", TX_POWER_QDBM);
}

void setup() {
  Serial.begin(TX_BAUD);
  pinMode(LED_PIN, OUTPUT);
  delay(200);

  // Start the Wi-Fi driver in station mode but do NOT connect to any network.
  // (WiFi.mode() runs esp_netif_init + esp_wifi_init + esp_wifi_start for us.)
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  esp_err_t err;
  // Power save off: with modem sleep the radio would switch off between
  // beacons and our transmit timing would jitter.
  if ((err = esp_wifi_set_ps(WIFI_PS_NONE)) != ESP_OK) fail("esp_wifi_set_ps", err);
  // Fixed source MAC so the receivers can ignore every other Wi-Fi device.
  if ((err = esp_wifi_set_mac(WIFI_IF_STA, TX_MAC)) != ESP_OK) fail("esp_wifi_set_mac", err);
  // Fixed channel, 20 MHz (no secondary channel).
  if ((err = esp_wifi_set_channel(CSI_CHANNEL, WIFI_SECOND_CHAN_NONE)) != ESP_OK) fail("esp_wifi_set_channel", err);
#if TX_POWER_QDBM > 0
  if ((err = esp_wifi_set_max_tx_power(TX_POWER_QDBM)) != ESP_OK) fail("esp_wifi_set_max_tx_power", err);
#endif

  if ((err = esp_now_init()) != ESP_OK) fail("esp_now_init", err);
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BROADCAST, 6);
  peer.channel = CSI_CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;  // broadcast frames cannot be encrypted
  if ((err = esp_now_add_peer(&peer)) != ESP_OK) fail("esp_now_add_peer", err);

  // Select an OFDM rate. ESP-NOW otherwise sends at 1 Mbps 802.11b, which
  // has no subcarriers -> the receivers get no CSI at all.
  // API difference between ESP-IDF versions:
  //   >= v5.4: esp_now_set_peer_rate_config() (per peer, PHY mode + rate)
  //   <  v5.4: esp_wifi_config_espnow_rate() (per interface, rate only;
  //            deprecated in v5.4+)
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
  esp_now_rate_config_t rate = {};
  rate.phymode = TX_PHY_HT20 ? WIFI_PHY_MODE_HT20 : WIFI_PHY_MODE_11G;
  rate.rate = TX_PHY_HT20 ? WIFI_PHY_RATE_MCS0_LGI : WIFI_PHY_RATE_6M;
  if ((err = esp_now_set_peer_rate_config(BROADCAST, &rate)) != ESP_OK) fail("esp_now_set_peer_rate_config", err);
#else
  if ((err = esp_wifi_config_espnow_rate(WIFI_IF_STA, TX_PHY_HT20 ? WIFI_PHY_RATE_MCS0_LGI : WIFI_PHY_RATE_6M)) != ESP_OK)
    fail("esp_wifi_config_espnow_rate", err);
#endif

  // Fill the payload once; only the header changes per packet.
  for (int i = 0; i < TX_PAYLOAD_BYTES; i++) payload[i] = (uint8_t)i;
  printConfig();
}

void loop() {
  static TickType_t lastWake = xTaskGetTickCount();
  static uint32_t lastStatusMs = 0, sentAtLastStatus = 0;

  csi_tx_header_t hdr = {CSI_PACKET_MAGIC, sent, (uint32_t)millis()};
  memcpy(payload, &hdr, sizeof(hdr));
  // esp_now_send only queues the frame. It fails (ESP_ERR_ESPNOW_NO_MEM) if
  // the internal queue is full, i.e. if we send faster than the air allows.
  esp_err_t err = esp_now_send(BROADCAST, payload, sizeof(payload));
  if (err == ESP_OK) sent++;
  else sendErrors++;

  // LED toggles once per second: a quick "is the TX alive?" check.
  digitalWrite(LED_PIN, (sent / TX_PACKETS_PER_SECOND) % 2);

  uint32_t now = millis();
  if (now - lastStatusMs >= 1000) {
    float rate = (sent - sentAtLastStatus) * 1000.0f / (now - lastStatusMs);
    Serial.printf("# STATUS role=TX uptime_ms=%lu sent=%lu send_errors=%lu rate=%.1f channel=%d\n",
                  (unsigned long)now, (unsigned long)sent, (unsigned long)sendErrors, rate, CSI_CHANNEL);
    if (now < 1500 || (now / 1000) % 30 == 0) printConfig();  // repeat config now and then
    lastStatusMs = now;
    sentAtLastStatus = sent;
  }

  // Fixed-rate schedule: wake exactly every 1/PPS s (in FreeRTOS ticks).
  const TickType_t period = pdMS_TO_TICKS(1000 / TX_PACKETS_PER_SECOND);
  vTaskDelayUntil(&lastWake, period > 0 ? period : 1);
}
