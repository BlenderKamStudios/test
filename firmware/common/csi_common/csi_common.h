// csi_common.h — settings shared by the TX and RX firmware.
//
// Both sketches include this file, so the TX and the RXs can never disagree on
// the channel, the transmitter's MAC address or the packet layout.
//
// Every value below is only a DEFAULT. python/flash.py reads config/config.yaml
// and passes the real values to the compiler as -D flags, e.g.
//   -DCSI_CHANNEL=11 -DTX_PACKETS_PER_SECOND=100
// so normally you edit config/config.yaml, not this file.
//
// Target: classic ESP32 (ESP32-WROOM-32), Arduino-ESP32 core 3.3.x, which is
// built on ESP-IDF v5.5.x. All Wi-Fi/CSI calls in the sketches are plain
// ESP-IDF APIs (esp_wifi_*, esp_now_*); Arduino is only used for Serial,
// setup()/loop() and WiFi.mode() to start the Wi-Fi driver.

#pragma once
#include <stdint.h>

#define CSI_FIRMWARE_VERSION "1.0"

// ---------------------------------------------------------------- radio link
// 2.4 GHz Wi-Fi channel (1..13). TX and all RXs must use the same one.
// Pick a quiet channel (few nearby access points) for cleaner CSI.
#ifndef CSI_CHANNEL
#define CSI_CHANNEL 11
#endif

// The TX overrides its own MAC with this "locally administered" address
// (first byte 0x1a has the 0x02 bit set = locally administered, so it cannot
// clash with a real device). The RXs only keep CSI from frames with this
// source address and ignore every other Wi-Fi device in the building.
#ifndef CSI_TX_MAC
#define CSI_TX_MAC 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00
#endif

// ------------------------------------------------------------ TX parameters
// How many packets per second the TX broadcasts. Every packet produces one
// CSI record at EACH receiver. 100 Hz is plenty for human motion (walking
// changes the channel at a few Hz to ~30 Hz) and keeps the RX serial link at
// ~50 % load. Upper practical limit is ~500 (FreeRTOS tick = 1 ms).
#ifndef TX_PACKETS_PER_SECOND
#define TX_PACKETS_PER_SECOND 100
#endif

// ESP-NOW payload size in bytes (12..250). The payload content does not
// matter for CSI (CSI is measured on the preamble), only the first 12 bytes
// carry our magic number + sequence number. Longer = more airtime.
#ifndef TX_PAYLOAD_BYTES
#define TX_PAYLOAD_BYTES 32
#endif

// PHY mode / rate the TX sends at. THIS MATTERS: the ESP-NOW default is
// 1 Mbps 802.11b (DSSS), which has NO OFDM subcarriers and therefore produces
// NO CSI on the receiver. We must send an OFDM rate:
//   TX_PHY_HT20 = 1 -> 802.11n HT20, MCS0 (6.5 Mbps, long guard interval)
//   TX_PHY_HT20 = 0 -> 802.11g legacy OFDM, 6 Mbps
// Both give the 64-subcarrier L-LTF CSI; HT20 additionally carries an HT-LTF
// (see RX_CSI_HTLTF in rx/config.h).
#ifndef TX_PHY_HT20
#define TX_PHY_HT20 1
#endif

// Max TX power in units of 0.25 dBm (8..84, i.e. 2..21 dBm). 0 = leave default.
#ifndef TX_POWER_QDBM
#define TX_POWER_QDBM 0
#endif

// ------------------------------------------------------------ packet format
// First bytes of every TX payload. The RX searches for this magic number to
// find the TX sequence number inside the received frame, so the laptop can
// (a) count lost packets and (b) match the SAME packet across RX1/RX2/RX3.
#define CSI_PACKET_MAGIC 0x31495343u  // bytes 'C','S','I','1' (little endian)

typedef struct __attribute__((packed)) {
  uint32_t magic;   // CSI_PACKET_MAGIC
  uint32_t seq;     // increments by 1 per packet, starts at 0 on TX boot
  uint32_t tx_ms;   // TX uptime in ms when the packet was queued (debug only)
} csi_tx_header_t;

// ------------------------------------------------------------ serial links
#ifndef RX_BAUD
#define RX_BAUD 921600
#endif
#ifndef TX_BAUD
#define TX_BAUD 115200
#endif
