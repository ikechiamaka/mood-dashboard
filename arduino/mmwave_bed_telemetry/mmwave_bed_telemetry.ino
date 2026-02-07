/*
  NeuroSense Dummy Bed Telemetry Sender (ESP32)

  Purpose:
  - If the MR60 sensor is broken/unavailable, this sketch still sends realistic
    dummy bed telemetry so the NeuroSense dashboard shows live data + charts.

  What it sends:
  - presence, fall, rr, hr, confidence, firmware, capabilities, raw.simulated

  Requirements:
  - Fill in Wi-Fi + API URL + API key in `secrets.h`
  - Device + bed must already exist in NeuroSense (Admin > Devices / Beds)
*/

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include <time.h>

#include "secrets.h"

// Set to 1 later when you have a working MR60 sensor and want real presence/fall.
#define USE_MMWAVE 0

#if USE_MMWAVE
#include "Seeed_Arduino_mmWave.h"
#include <HardwareSerial.h>
static const int RADAR_RX = D7;
static const int RADAR_TX = D6;
HardwareSerial mmwaveSerial(1);
SEEED_MR60FDA2 mmWave;
#endif

const char* WIFI_SSID_VALUE = WIFI_SSID;
const char* WIFI_PASS_VALUE = WIFI_PASS;
const char* API_URL_VALUE = API_URL;
const char* API_KEY_VALUE = API_KEY;

// These must match what you created in the NeuroSense Admin UI.
const int FACILITY_ID = 1;
const char* DEVICE_ID = "DEV-001";
const char* BED_ID = "f2aeab6c-4093-4d71-b290-4e2203d892ac";

static uint32_t lastPostMs = 0;

static float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static uint32_t nowTs() {
  time_t t = time(nullptr);
  if (t > 1700000000) return (uint32_t)t;
  return (uint32_t)(millis() / 1000);
}

static void wifiConnect() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID_VALUE, WIFI_PASS_VALUE);

  Serial.print("WiFi connecting");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi OK, IP: ");
    Serial.println(WiFi.localIP());
    configTime(0, 0, "pool.ntp.org", "time.google.com");
  } else {
    Serial.println("WiFi FAILED (check SSID/PASS)");
  }
}

static bool simulatePresence() {
  // Occupied 90s, empty 20s, repeating.
  static bool presence = true;
  static uint32_t phaseStartMs = 0;
  const uint32_t occupiedMs = 90000;
  const uint32_t emptyMs = 20000;

  uint32_t now = millis();
  uint32_t phaseMs = presence ? occupiedMs : emptyMs;
  if (phaseStartMs == 0) phaseStartMs = now;
  if (now - phaseStartMs >= phaseMs) {
    presence = !presence;
    phaseStartMs = now;
  }
  return presence;
}

static bool simulateFall(bool presence) {
  if (!presence) return false;
  // Brief fall flag every ~4 minutes for demo.
  uint32_t s = millis() / 1000;
  return (s % (4 * 60) == 0);
}

static float simulateRR(float tSeconds) {
  float rr = 14.0f + 1.1f * sinf(tSeconds / 17.0f) + 0.4f * sinf(tSeconds / 4.7f);
  return clampf(rr, 9.0f, 24.0f);
}

static float simulateHR(float tSeconds) {
  float hr = 72.0f + 6.0f * sinf(tSeconds / 23.0f) + 1.5f * sinf(tSeconds / 6.3f);
  return clampf(hr, 55.0f, 115.0f);
}

static bool postTelemetry(bool presence, bool fall, float rr, float hr, float confidence) {
  if (WiFi.status() != WL_CONNECTED) return false;

  String json = "{";
  json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  json += "\"facility_id\":" + String(FACILITY_ID) + ",";
  json += "\"bed_id\":\"" + String(BED_ID) + "\",";
  json += "\"ts\":" + String(nowTs()) + ",";
  json += "\"presence\":" + String(presence ? "true" : "false") + ",";
  json += "\"fall\":" + String(fall ? "true" : "false") + ",";
  json += "\"rr\":" + String(rr, 2) + ",";
  json += "\"hr\":" + String(hr, 1) + ",";
  json += "\"confidence\":" + String(confidence, 2) + ",";
  json += "\"firmware\":\"dummy-vitals-1.0\",";
  json += "\"capabilities\":[\"presence\",\"fall\",\"rr\",\"hr\"],";
  json += "\"raw\":{\"simulated\":true}";
  json += "}";

  HTTPClient http;
  http.setTimeout(8000);
  http.begin(API_URL_VALUE);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer " + String(API_KEY_VALUE));

  int code = http.POST(json);
  String resp = http.getString();
  if (code < 0) {
    Serial.printf("HTTP error: %d (%s) url=%s\n", code, http.errorToString(code).c_str(), API_URL_VALUE);
  }
  http.end();

  Serial.printf("POST %d | %s\n", code, resp.c_str());
  return (code >= 200 && code < 300);
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("BOOT");
  Serial.printf("API_URL=%s\n", API_URL_VALUE);

  wifiConnect();

#if USE_MMWAVE
  mmwaveSerial.begin(115200, SERIAL_8N1, RADAR_RX, RADAR_TX);
  mmWave.begin(&mmwaveSerial);
  mmWave.setUserLog(0);
#endif

  Serial.println("Init done");
}

void loop() {
  bool presence = false;
  bool fall = false;

#if USE_MMWAVE
  if (mmWave.update(100)) {
    (void)mmWave.getHuman(presence);
    (void)mmWave.getFall(fall);
  }
#else
  presence = simulatePresence();
  fall = simulateFall(presence);
#endif

  if (millis() - lastPostMs >= 5000) {
    lastPostMs = millis();
    float tSeconds = millis() / 1000.0f;
    float rr = simulateRR(tSeconds);
    float hr = simulateHR(tSeconds);

    postTelemetry(presence, fall, rr, hr, 0.90f);
    Serial.printf("presence=%d fall=%d rr=%.2f hr=%.1f\n", presence, fall, rr, hr);
  }
}
