// CSI receiver: listens for the transmitter's packets, reads the Channel State
// Information (CSI) of each one and prints it over USB serial for detect.py.
//
// Output, one line per packet:
//   CSI_DATA,seq,mac,rssi,rate,noise_floor,channel,timestamp,sig_len,len,"[i,r,i,r,...]"
// Lines starting with '#' are status messages and are ignored by detect.py.
// The status lines end with "rx <this board's MAC>" so server.py can tell the
// receivers apart whatever serial port they end up on.

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_mac.h>

// Must match csi_tx.ino
const uint8_t CHANNEL = 11;
const uint8_t TX_MAC[6] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

const uint32_t BAUD = 921600;
const size_t MAX_CSI_LEN = 384;

struct CsiPacket {
  uint32_t seq;
  int8_t rssi;
  uint8_t rate;
  int8_t noise_floor;
  uint8_t channel;
  uint32_t timestamp;
  uint16_t sig_len;
  uint16_t len;
  int8_t buf[MAX_CSI_LEN];
};

QueueHandle_t queue;
uint32_t received = 0;  // written by the Wi-Fi task, read in loop()
uint32_t dropped = 0;
String ownMac;

// Runs in the Wi-Fi task: copy the data out quickly and let loop() print it
void onCsi(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || memcmp(info->mac, TX_MAC, 6) != 0) return;

  static CsiPacket p;
  static uint32_t seq = 0;
  p.seq = seq++;
  p.rssi = info->rx_ctrl.rssi;
  p.rate = info->rx_ctrl.rate;
  p.noise_floor = info->rx_ctrl.noise_floor;
  p.channel = info->rx_ctrl.channel;
  p.timestamp = info->rx_ctrl.timestamp;
  p.sig_len = info->rx_ctrl.sig_len;
  p.len = min<size_t>(info->len, MAX_CSI_LEN);
  memcpy(p.buf, info->buf, p.len);
  if (info->first_word_invalid) memset(p.buf, 0, min<size_t>(4, p.len));

  received++;
  if (xQueueSend(queue, &p, 0) != pdTRUE) dropped++;
}

void fail(const char *what, esp_err_t err) {
  while (true) {
    Serial.printf("# ERROR: %s failed: %s\n", what, esp_err_to_name(err));
    delay(2000);
  }
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(BAUD);
  queue = xQueueCreate(32, sizeof(CsiPacket));

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  WiFi.setSleep(false);
  uint8_t m[6];
  esp_read_mac(m, ESP_MAC_WIFI_STA);   // from the chip; WiFi.macAddress() is 0 until Wi-Fi has started
  char buf[18];
  snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x", m[0], m[1], m[2], m[3], m[4], m[5]);
  ownMac = buf;

  esp_err_t err;
  if ((err = esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE)) != ESP_OK) fail("set channel", err);
  if ((err = esp_now_init()) != ESP_OK) fail("ESP-NOW init", err);
  // Promiscuous mode so broadcast frames reach the CSI hardware
  if ((err = esp_wifi_set_promiscuous(true)) != ESP_OK) fail("promiscuous", err);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;            // legacy LTF: 64 subcarriers, 128 bytes
  cfg.htltf_en = false;
  cfg.stbc_htltf2_en = false;
  cfg.ltf_merge_en = true;
  cfg.channel_filter_en = false; // keep subcarriers independent
  cfg.manu_scale = false;
  if ((err = esp_wifi_set_csi_config(&cfg)) != ESP_OK) fail("CSI config", err);
  if ((err = esp_wifi_set_csi_rx_cb(onCsi, nullptr)) != ESP_OK) fail("CSI callback", err);
  if ((err = esp_wifi_set_csi(true)) != ESP_OK) fail("enable CSI", err);

  Serial.printf("# CSI receiver: channel %u, waiting for %02x:%02x:%02x:%02x:%02x:%02x, rx %s\n",
                CHANNEL, TX_MAC[0], TX_MAC[1], TX_MAC[2], TX_MAC[3], TX_MAC[4], TX_MAC[5], ownMac.c_str());
}

void loop() {
  static CsiPacket p;
  static char line[2600];
  static uint32_t lastStatus = 0;

  if (xQueueReceive(queue, &p, pdMS_TO_TICKS(100)) == pdTRUE) {
    int n = snprintf(line, sizeof(line),
                     "CSI_DATA,%lu,%02x:%02x:%02x:%02x:%02x:%02x,%d,%u,%d,%u,%lu,%u,%u,\"[",
                     p.seq, TX_MAC[0], TX_MAC[1], TX_MAC[2], TX_MAC[3], TX_MAC[4], TX_MAC[5],
                     p.rssi, p.rate, p.noise_floor, p.channel, p.timestamp, p.sig_len, p.len);
    for (uint16_t i = 0; i < p.len; i++) {
      n += snprintf(line + n, sizeof(line) - n, i ? ",%d" : "%d", p.buf[i]);
    }
    snprintf(line + n, sizeof(line) - n, "]\"\n");
    Serial.print(line);
  }

  // Heartbeat every 5 s so an empty stream is easy to diagnose
  if (millis() - lastStatus >= 5000) {
    lastStatus = millis();
    Serial.printf("# status: %lu packets received, %lu dropped, rx %s\n", received, dropped, ownMac.c_str());
  }
}
