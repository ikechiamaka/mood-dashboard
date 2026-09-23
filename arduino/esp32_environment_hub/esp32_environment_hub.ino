/*
  MelX Health ESP32 Environment Hub

  Purpose:
  - Posts room/environment data into the existing MelX Health wall-event pipeline.
  - Designed for a normal ESP32 dev board.

  Default mode:
  - DUMMY_ENV_MODE=0
  - Sends measured values every 30 seconds; missing values are null
  - Requires only WiFi.h and HTTPClient.h

  Optional sensors (enable only after confirming wiring):
  - DHT22 for temperature/humidity
  - BH1750 for light
  - MQ135 analog air quality
  - PIR motion
  - analog noise sensor
*/

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "tls_ca.h"
#include <math.h>
#include <time.h>

#include "secrets.h"

#define DUMMY_ENV_MODE 0
#define POST_INTERVAL_MS 30000UL

#define USE_DHT22 0
#define USE_BH1750 0
#define USE_MQ135 0
#define USE_PIR 0
#define USE_NOISE_SENSOR 0

// Placeholder pin assignments for a typical ESP32 dev board.
// Adjust these to match your real wiring when moving beyond dummy mode.
#define DHT22_PIN 4
#define MQ135_PIN 34
#define PIR_PIN 27
#define NOISE_PIN 35
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

#if USE_DHT22
#include <DHT.h>
DHT dht(DHT22_PIN, DHT22);
#endif
#if USE_BH1750
#include <Wire.h>
#include <BH1750.h>
BH1750 lightMeter;
#endif

const char* WIFI_SSID_VALUE = WIFI_SSID;
const char* WIFI_PASSWORD_VALUE = WIFI_PASSWORD;
const char* SERVER_BASE_URL_VALUE = SERVER_BASE_URL;
const char* ENV_API_KEY_VALUE = ENV_API_KEY;
const char* ENV_DEVICE_ID_VALUE = ENV_DEVICE_ID;
const char* FACILITY_ID_VALUE = FACILITY_ID;
const char* BED_ID_VALUE = BED_ID;

static const char* SOURCE_NAME = "esp32_environment_hub";
static const char* FIRMWARE_NAME = "env_hub_v1";

struct EnvironmentSnapshot {
  float temperatureC;
  float humidity;
  const char* airQuality;
  float lightLevel;
  float noiseLevel;
  float confidence;
  int mq135Raw;
  bool motion;
};

static String jsonNumber(float value, int decimals) {
  return isfinite(value) ? String(value, decimals) : String("null");
}

static uint32_t lastPostMs = 0;
static uint32_t lastWifiAttemptMs = 0;

static float clampf(float value, float minValue, float maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

static uint32_t nowTs() {
  time_t t = time(nullptr);
  if (t > 1700000000) return (uint32_t)t;
  return (uint32_t)(millis() / 1000);
}

static String ipText() {
  if (WiFi.status() == WL_CONNECTED) {
    return WiFi.localIP().toString();
  }
  return "0.0.0.0";
}

static String jsonEscape(const String& input) {
  String out = "";
  for (size_t i = 0; i < input.length(); i++) {
    char c = input.charAt(i);
    if (c == '\"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

static const char* airQualityFromRaw(int raw) {
  if (raw < 700) return "GOOD";
  if (raw < 1200) return "OK";
  if (raw < 2000) return "POOR";
  return "ALERT";
}

static void wifiConnect() {
  Serial.print("WiFi connecting");
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID_VALUE, WIFI_PASSWORD_VALUE);
  lastWifiAttemptMs = millis();

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000UL) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi OK, IP: ");
    Serial.println(WiFi.localIP());
    configTzTime(TZ_INFO, "pool.ntp.org", "time.google.com");
  } else {
    Serial.println("WiFi FAILED");
  }
}

static void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  if (millis() - lastWifiAttemptMs < 15000UL) {
    return;
  }
  Serial.println("WiFi retrying");
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID_VALUE, WIFI_PASSWORD_VALUE);
  lastWifiAttemptMs = millis();
}

static bool readDht22(float& temperatureC, float& humidity) {
#if DUMMY_ENV_MODE
  (void)temperatureC;
  (void)humidity;
  return false;
#else
  #if USE_DHT22
    temperatureC = dht.readTemperature();
    humidity = dht.readHumidity();
    return isfinite(temperatureC) && isfinite(humidity);
  #else
    (void)temperatureC;
    (void)humidity;
    return false;
  #endif
#endif
}

static bool readBh1750(float& lightLevel) {
#if DUMMY_ENV_MODE
  (void)lightLevel;
  return false;
#else
  #if USE_BH1750
    lightLevel = lightMeter.readLightLevel();
    return isfinite(lightLevel) && lightLevel >= 0;
  #else
    (void)lightLevel;
    return false;
  #endif
#endif
}

static int readMq135Raw() {
#if DUMMY_ENV_MODE
  return -1;
#else
  #if USE_MQ135
    return analogRead(MQ135_PIN);
  #else
    return -1;
  #endif
#endif
}

static bool readPirMotion() {
#if DUMMY_ENV_MODE
  return false;
#else
  #if USE_PIR
    return digitalRead(PIR_PIN) == HIGH;
  #else
    return false;
  #endif
#endif
}

static float readNoiseLevel() {
#if DUMMY_ENV_MODE
  return -1.0f;
#else
  #if USE_NOISE_SENSOR
    int raw = analogRead(NOISE_PIN);
    return (float)raw;
  #else
    return -1.0f;
  #endif
#endif
}

static EnvironmentSnapshot simulateEnvironment() {
  float t = millis() / 1000.0f;
  int mq135Raw = 880 + (int)(120.0f * sinf(t / 29.0f)) + (int)(35.0f * sinf(t / 8.0f));
  mq135Raw = (int)clampf((float)mq135Raw, 650.0f, 1450.0f);

  EnvironmentSnapshot snapshot;
  snapshot.temperatureC = clampf(24.3f + 1.7f * sinf(t / 33.0f), 22.0f, 28.5f);
  snapshot.humidity = clampf(67.0f + 6.5f * sinf(t / 21.0f), 54.0f, 78.0f);
  snapshot.airQuality = airQualityFromRaw(mq135Raw);
  snapshot.lightLevel = clampf(120.0f + 55.0f * sinf(t / 12.0f), 30.0f, 260.0f);
  snapshot.noiseLevel = clampf(34.0f + 7.0f * sinf(t / 7.0f), 18.0f, 58.0f);
  snapshot.confidence = 0.95f;
  snapshot.mq135Raw = mq135Raw;
  snapshot.motion = ((millis() / 1000UL) % 90UL) < 6UL;
  return snapshot;
}

static EnvironmentSnapshot readEnvironmentSnapshot() {
#if DUMMY_ENV_MODE
  return simulateEnvironment();
#else
  EnvironmentSnapshot snapshot;
  snapshot.temperatureC = NAN;
  snapshot.humidity = NAN;
  snapshot.lightLevel = NAN;
  snapshot.noiseLevel = NAN;
  snapshot.mq135Raw = readMq135Raw();
  snapshot.motion = readPirMotion();
  snapshot.confidence = NAN;
  // Raw MQ135 ADC counts are not a calibrated air-quality category.
  snapshot.airQuality = "";

  float temp = 0.0f;
  float hum = 0.0f;
  if (readDht22(temp, hum)) {
    snapshot.temperatureC = temp;
    snapshot.humidity = hum;
  }

  float lux = 0.0f;
  if (readBh1750(lux)) {
    snapshot.lightLevel = lux;
  }

  float noise = readNoiseLevel();
  // Analog counts are retained only as raw data until calibrated to dB.
  (void)noise;

  return snapshot;
#endif
}

static String buildEnvironmentPayload(const EnvironmentSnapshot& snapshot) {
  String payload = "{";
  payload += "\"device_id\":\"" + jsonEscape(String(ENV_DEVICE_ID_VALUE)) + "\",";
  payload += "\"facility_id\":\"" + jsonEscape(String(FACILITY_ID_VALUE)) + "\",";
  payload += "\"bed_id\":\"" + jsonEscape(String(BED_ID_VALUE)) + "\",";
  payload += "\"event_type\":\"environment\",";
  payload += "\"source\":\"" + String(SOURCE_NAME) + "\",";
  payload += "\"temperature_c\":" + jsonNumber(snapshot.temperatureC, 2) + ",";
  payload += "\"humidity\":" + jsonNumber(snapshot.humidity, 1) + ",";
  payload += "\"air_quality\":\"" + jsonEscape(String(snapshot.airQuality)) + "\",";
  payload += "\"light_level\":" + jsonNumber(snapshot.lightLevel, 1) + ",";
  payload += "\"noise_level\":" + jsonNumber(snapshot.noiseLevel, 1) + ",";
  payload += "\"confidence\":" + jsonNumber(snapshot.confidence, 2) + ",";
  payload += "\"raw\":{";
  payload += "\"firmware\":\"" + String(FIRMWARE_NAME) + "\",";
  payload += "\"ts_epoch\":" + String(nowTs()) + ",";
  payload += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
  payload += "\"ip\":\"" + jsonEscape(ipText()) + "\",";
  payload += "\"mq135_raw\":" + String(snapshot.mq135Raw) + ",";
  payload += "\"noise_adc_raw\":" + jsonNumber(readNoiseLevel(), 0) + ",";
  payload += "\"motion\":" + String((DUMMY_ENV_MODE || USE_PIR) ? (snapshot.motion ? "true" : "false") : "null") + ",";
  payload += "\"dummy_env_mode\":" + String(DUMMY_ENV_MODE ? "true" : "false");
  payload += "}}";
  return payload;
}

static bool postEnvironment(const EnvironmentSnapshot& snapshot) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("POST skipped: WiFi offline");
    return false;
  }

  String url = String(SERVER_BASE_URL_VALUE) + "/api/v1/wall/event";
  String payload = buildEnvironmentPayload(snapshot);

  Serial.println("POST " + url);
  Serial.println(payload);

  if (time(nullptr) < 1700000000) return false;
  WiFiClientSecure secureClient;
  secureClient.setCACert(MELX_ROOT_CA);
  HTTPClient http;
  http.setTimeout(8000);

  if (!url.startsWith("https://") || !http.begin(secureClient, url)) {
    Serial.println("HTTP begin failed");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer " + String(ENV_API_KEY_VALUE));

  int code = http.POST(payload);
  String resp = http.getString();
  if (code < 0) {
    Serial.printf("HTTP error: %d (%s)\n", code, http.errorToString(code).c_str());
  }
  http.end();

  Serial.printf("HTTP code: %d\n", code);
  Serial.println(resp);
  return code >= 200 && code < 300;
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("BOOT: MelX Health ESP32 Environment Hub");
  Serial.printf("BASE_URL=%s\n", SERVER_BASE_URL_VALUE);

  pinMode(PIR_PIN, INPUT);
#if USE_DHT22
  dht.begin();
#endif
#if USE_BH1750
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
#endif
  wifiConnect();
}

void loop() {
  maintainWifi();

  if (millis() - lastPostMs >= POST_INTERVAL_MS) {
    lastPostMs = millis();
    EnvironmentSnapshot snapshot = readEnvironmentSnapshot();
    bool ok = postEnvironment(snapshot);
    Serial.printf(
      "environment sent=%d temp=%.2f humidity=%.1f aq=%s light=%.1f noise=%.1f conf=%.2f motion=%d\n",
      ok ? 1 : 0,
      snapshot.temperatureC,
      snapshot.humidity,
      snapshot.airQuality,
      snapshot.lightLevel,
      snapshot.noiseLevel,
      snapshot.confidence,
      snapshot.motion ? 1 : 0
    );
  }

  delay(50);
}
