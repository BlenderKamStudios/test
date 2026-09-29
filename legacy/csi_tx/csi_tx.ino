// CSI transmitter: broadcasts small ESP-NOW packets at a fixed rate so the
// receiver can measure Channel State Information (CSI) on every packet.
// Only needs power once flashed (USB charger or power bank).

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

// Must match csi_rx.ino
const uint8_t CHANNEL = 11;
const uint8_t TX_MAC[6] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

const uint32_t PACKETS_PER_SECOND = 100;
const uint8_t BROADCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
const uint8_t LED_PIN = 2;  // on-board LED on most ESP32 DevKit boards

void fail(const char *what, esp_err_t err) {
  while (true) {
    Serial.printf("# ERROR: %s failed: %s\n", what, esp_err_to_name(err));
    delay(2000);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  WiFi.setSleep(false);

  esp_err_t err;
  // Fixed source MAC so the receiver can ignore every other Wi-Fi device
  if ((err = esp_wifi_set_mac(WIFI_IF_STA, TX_MAC)) != ESP_OK) fail("set MAC", err);
  if ((err = esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE)) != ESP_OK) fail("set channel", err);
  if ((err = esp_now_init()) != ESP_OK) fail("ESP-NOW init", err);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BROADCAST, 6);
  peer.channel = CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  if ((err = esp_now_add_peer(&peer)) != ESP_OK) fail("add peer", err);

  // ESP-NOW defaults to 1 Mbps 802.11b, which has no OFDM subcarriers and so
  // produces no CSI. Send at HT20 MCS0 (6.5 Mbps OFDM) instead.
  esp_now_rate_config_t rate = {};
  rate.phymode = WIFI_PHY_MODE_HT20;
  rate.rate = WIFI_PHY_RATE_MCS0_LGI;
  if ((err = esp_now_set_peer_rate_config(BROADCAST, &rate)) != ESP_OK) fail("set rate", err);

  Serial.printf("# CSI transmitter: MAC %s, channel %u, %lu packets/s\n",
                WiFi.macAddress().c_str(), CHANNEL, PACKETS_PER_SECOND);
}

void loop() {
  static uint32_t seq = 0;
  static TickType_t lastWake = xTaskGetTickCount();

  esp_now_send(BROADCAST, (const uint8_t *)&seq, sizeof(seq));
  seq++;

  // Blink slowly to show the transmitter is alive
  digitalWrite(LED_PIN, (seq / PACKETS_PER_SECOND) % 2);
  if (seq % (PACKETS_PER_SECOND * 5) == 0) {
    Serial.printf("# sent %lu packets\n", seq);
  }

  vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(1000 / PACKETS_PER_SECOND));
}
