/*
  NeuroSense Dummy Bed Telemetry Sender (ESP32)

  Purpose:
  - If the MR60 sensor is broken/unavailable, this sketch still sends realistic
    dummy bed telemetry so the NeuroSense dashboard shows live data + charts.
  - If you now have the MR60BHA2 (breath/heart) device, set USE_MMWAVE to 1
    and the sketch will send real RR/HR from the sensor.

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

// Set to 1 when using a real MR60BHA2 (breath/heart) sensor.
#define USE_MMWAVE 1

#if USE_MMWAVE
#include "Seeed_Arduino_mmWave.h"
#include <HardwareSerial.h>
static const int RADAR_RX = D7;
static const int RADAR_TX = D6;
HardwareSerial mmwaveSerial(1);
SEEED_MR60BHA2 mmWave;
#endif

const char* WIFI_SSID_VALUE = WIFI_SSID;
const char* WIFI_PASS_VALUE = WIFI_PASS;
const char* API_URL_VALUE = API_URL;
const char* API_KEY_VALUE = API_KEY;

// These must match what you created in the NeuroSense Admin UI.
const int FACILITY_ID = 1;
const char* DEVICE_ID = "DEV-001";
const char* BED_ID = "83ffc988-c21c-440e-9670-ee05cfc25791";

static uint32_t lastPostMs = 0;
static float lastRealRR = 14.0f;
static float lastRealHR = 72.0f;

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

static bool postTelemetry(bool presence, bool fall, float rr, float hr, float confidence,
                          bool simulated, bool rrValid, bool hrValid, float distanceCm, bool distanceValid) {
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
  json += "\"firmware\":\"";
#if USE_MMWAVE
  json += "mr60bha2-live-1.0";
#else
  json += "dummy-vitals-1.0";
#endif
  json += "\",";
#if USE_MMWAVE
  json += "\"capabilities\":[\"presence\",\"rr\",\"hr\"],";
#else
  json += "\"capabilities\":[\"presence\",\"fall\",\"rr\",\"hr\"],";
#endif
  json += "\"raw\":{";
  json += "\"simulated\":" + String(simulated ? "true" : "false") + ",";
  json += "\"rr_valid\":" + String(rrValid ? "true" : "false") + ",";
  json += "\"hr_valid\":" + String(hrValid ? "true" : "false");
  if (distanceValid) {
    json += ",\"distance_cm\":" + String(distanceCm, 1);
  }
  json += "}";
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
  // Some library versions support user log suppression; safe to omit for compatibility.
#endif

  Serial.println("Init done");
}

void loop() {
  bool presence = false;
  bool fall = false;
  bool rrValid = true;
  bool hrValid = true;
  bool distanceValid = false;
  float distanceCm = 0.0f;
  float rr = 0.0f;
  float hr = 0.0f;
  float confidence = 0.90f;

#if USE_MMWAVE
  if (mmWave.update(100)) {
    // Requires recent BHA2 firmware/library for presence support.
    presence = mmWave.isHumanDetected();
    fall = false; // BHA2 does not provide fall detection.

    float rrRead = 0.0f;
    float hrRead = 0.0f;
    rrValid = mmWave.getBreathRate(rrRead);
    hrValid = mmWave.getHeartRate(hrRead);
    if (rrValid) {
      lastRealRR = clampf(rrRead, 4.0f, 40.0f);
    }
    if (hrValid) {
      lastRealHR = clampf(hrRead, 35.0f, 180.0f);
    }
    rr = lastRealRR;
    hr = lastRealHR;

    float distRead = 0.0f;
    distanceValid = mmWave.getDistance(distRead);
    if (distanceValid) {
      distanceCm = distRead;
    }

    if (rrValid && hrValid) confidence = 0.95f;
    else if (rrValid || hrValid) confidence = 0.70f;
    else confidence = 0.35f;
  } else {
    // Keep last known values if no fresh frame arrives.
    presence = false;
    fall = false;
    rr = lastRealRR;
    hr = lastRealHR;
    rrValid = false;
    hrValid = false;
    confidence = 0.25f;
  }
#else
  presence = simulatePresence();
  fall = simulateFall(presence);
  rr = 0.0f;
  hr = 0.0f;
#endif

  if (millis() - lastPostMs >= 5000) {
    lastPostMs = millis();
    bool simulated = false;
#if !USE_MMWAVE
    float tSeconds = millis() / 1000.0f;
    rr = simulateRR(tSeconds);
    hr = simulateHR(tSeconds);
    rrValid = true;
    hrValid = true;
    distanceValid = false;
    distanceCm = 0.0f;
    confidence = 0.90f;
    simulated = true;
#endif

    postTelemetry(presence, fall, rr, hr, confidence, simulated, rrValid, hrValid, distanceCm, distanceValid);
    Serial.printf("presence=%d fall=%d rr=%.2f (%d) hr=%.1f (%d) conf=%.2f\n",
                  presence, fall, rr, rrValid ? 1 : 0, hr, hrValid ? 1 : 0, confidence);
  }
}
