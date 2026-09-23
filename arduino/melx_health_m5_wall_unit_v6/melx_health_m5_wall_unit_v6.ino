
/*
  MelX Health / La'i Tab5 Wall Unit v6
  Target: M5Stack Tab5 (ESP32-P4)

  Features:
  - Working embedded mood-scale UI
  - Correct grey selected-state for tapped emoji
  - Wi-Fi connection using Tab5 SDIO Wi-Fi pins
  - NTP time sync
  - POST mood_checkin to MelX Health backend:
      POST /api/v1/wall/event
  - Bearer device API key auth
  - On-screen status: Offline / Online / Sending / Saved / Failed

  Required libraries:
  - M5Unified
  - M5GFX

  Files in this sketch folder:
  - melx_health_m5_wall_unit_v6.ino
  - mood_scale_pngs.h
  - secrets.h
*/

#include <Arduino.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "tls_ca.h"
#include <time.h>

#include "mood_scale_pngs.h"
#include "secrets.h"

// Tab5 Wi-Fi SDIO pins from the M5Stack Tab5 Wi-Fi Arduino example.
#ifndef BOARD_SDIO_ESP_HOSTED_CLK
#define TAB5_SDIO_CLK GPIO_NUM_12
#define TAB5_SDIO_CMD GPIO_NUM_13
#define TAB5_SDIO_D0  GPIO_NUM_11
#define TAB5_SDIO_D1  GPIO_NUM_10
#define TAB5_SDIO_D2  GPIO_NUM_9
#define TAB5_SDIO_D3  GPIO_NUM_8
#define TAB5_SDIO_RST GPIO_NUM_15
#endif

struct MoodZone {
  int x;
  int y;
  int w;
  int h;
  int score;
  const char* label;
};

MoodZone zones[6];

int selectedMood = 0;

int screenW = 1280;
int screenH = 720;

int imgX = 50;
int imgY = 405;
int imgW = mood_scale_png_width;
int imgH = mood_scale_png_height;

uint32_t lastClockRedraw = 0;
uint32_t lastWifiAttempt = 0;
uint32_t lastStatusChange = 0;

String networkStatus = "Offline";
String postStatus = "Ready";
String lastServerResponse = "";

bool timeSynced = false;

struct CardPalette {
  uint16_t bg;
  uint16_t border;
  uint16_t title;
  uint16_t value;
  uint16_t subtitle;
};

// -------------------- UTILS --------------------

uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
  return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

CardPalette defaultCardPalette() {
  return {
    0xF7BE,
    0xC638,
    0x39E7,
    0x1082,
    0x4A49
  };
}

CardPalette wallUnitStatusPalette() {
  if (networkStatus == "Online") {
    return {
      rgb565(227, 252, 239),
      rgb565(34, 197, 94),
      rgb565(22, 101, 52),
      rgb565(21, 128, 61),
      rgb565(22, 101, 52)
    };
  }

  if (networkStatus == "Connecting") {
    return {
      rgb565(255, 247, 214),
      rgb565(245, 158, 11),
      rgb565(146, 64, 14),
      rgb565(180, 83, 9),
      rgb565(120, 53, 15)
    };
  }

  return {
    rgb565(254, 226, 226),
    rgb565(239, 68, 68),
    rgb565(127, 29, 29),
    rgb565(153, 27, 27),
    rgb565(127, 29, 29)
  };
}

String jsonEscape(const String& s) {
  String out = "";
  for (size_t i = 0; i < s.length(); i++) {
    char c = s.charAt(i);
    if (c == '\"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

String ipText() {
  if (WiFi.status() == WL_CONNECTED) {
    return WiFi.localIP().toString();
  }
  return "No IP";
}

String currentTimeText() {
  struct tm timeinfo;

  if (getLocalTime(&timeinfo, 50)) {
    timeSynced = true;
    char buf[8];
    strftime(buf, sizeof(buf), "%H:%M", &timeinfo);
    return String(buf);
  }

  // Fallback to uptime until NTP works.
  uint32_t secs = millis() / 1000;
  uint32_t mins = (secs / 60) % 60;
  uint32_t hrs  = (secs / 3600) % 24;

  char buf[8];
  snprintf(buf, sizeof(buf), "%02lu:%02lu", (unsigned long)hrs, (unsigned long)mins);
  return String(buf);
}

String currentEpochText() {
  time_t now;
  time(&now);

  if (now > 1700000000) {
    return String((unsigned long)now);
  }

  return String((unsigned long)(millis() / 1000));
}

void setPostStatus(const String& status) {
  postStatus = status;
  lastStatusChange = millis();
}

// -------------------- WIFI --------------------

void initWifiPins() {
#ifdef BOARD_SDIO_ESP_HOSTED_CLK
  WiFi.setPins(
    BOARD_SDIO_ESP_HOSTED_CLK,
    BOARD_SDIO_ESP_HOSTED_CMD,
    BOARD_SDIO_ESP_HOSTED_D0,
    BOARD_SDIO_ESP_HOSTED_D1,
    BOARD_SDIO_ESP_HOSTED_D2,
    BOARD_SDIO_ESP_HOSTED_D3,
    BOARD_SDIO_ESP_HOSTED_RESET
  );
#else
  WiFi.setPins(
    TAB5_SDIO_CLK,
    TAB5_SDIO_CMD,
    TAB5_SDIO_D0,
    TAB5_SDIO_D1,
    TAB5_SDIO_D2,
    TAB5_SDIO_D3,
    TAB5_SDIO_RST
  );
#endif
}

void startWifi() {
  Serial.println("WiFi: starting");

  networkStatus = "Connecting";
  initWifiPins();

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  lastWifiAttempt = millis();
}

void maintainWifi() {
  wl_status_t status = WiFi.status();

  if (status == WL_CONNECTED) {
    if (networkStatus != "Online") {
      networkStatus = "Online";
      Serial.print("WiFi connected. IP: ");
      Serial.println(WiFi.localIP());

      configTzTime(TZ_INFO, "pool.ntp.org", "time.nist.gov");
    }
    return;
  }

  networkStatus = "Offline";

  // Retry every 15 seconds without blocking the UI.
  if (millis() - lastWifiAttempt > 15000) {
    Serial.println("WiFi: retrying");
    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiAttempt = millis();
    networkStatus = "Connecting";
  }
}

// -------------------- UI --------------------

void drawCard(int x, int y, int w, int h, const String& title, const String& value, const String& subtitle = "", CardPalette palette = defaultCardPalette()) {
  auto& d = M5.Display;

  d.fillRoundRect(x, y, w, h, 18, palette.bg);
  d.drawRoundRect(x, y, w, h, 18, palette.border);

  d.setTextDatum(top_left);

  d.setTextColor(palette.title, palette.bg);
  d.setTextSize(2);
  d.drawString(title, x + 18, y + 18);

  d.setTextColor(palette.value, palette.bg);
  d.setTextSize(4);
  d.drawString(value, x + 18, y + 55);

  if (subtitle.length()) {
    d.setTextColor(palette.subtitle, palette.bg);
    d.setTextSize(2);
    d.drawString(subtitle, x + 18, y + h - 32);
  }
}

void drawHeader() {
  int margin = 40;
  int gap = 24;
  int cardCount = 4;

  int cardW = (screenW - (2 * margin) - ((cardCount - 1) * gap)) / cardCount;
  int cardH = 128;
  int y = 34;

  int x0 = margin;
  int x1 = margin + (cardW + gap);
  int x2 = margin + 2 * (cardW + gap);
  int x3 = margin + 3 * (cardW + gap);

  drawCard(x0, y, cardW, cardH, "Temperature", "--.- C", "sensor later");
  drawCard(x1, y, cardW, cardH, "Humidity", "--%", "sensor later");
  drawCard(x2, y, cardW, cardH, "Time", currentTimeText(), timeSynced ? "local" : "syncing");
  drawCard(x3, y, cardW, cardH, "Wall unit", networkStatus, ipText(), wallUnitStatusPalette());
}

void drawPrompt() {
  auto& d = M5.Display;

  d.setTextDatum(middle_center);

  d.setTextColor(0x1082, TFT_WHITE);
  d.setTextSize(4);
  d.drawString("Please rate your mood", screenW / 2, 240);

  d.setTextColor(0x4A49, TFT_WHITE);
  d.setTextSize(2);
  d.drawString("Tap the face that best matches how you feel right now.", screenW / 2, 290);
}

void computeMoodZones() {
  float centerRatios[6] = {
    0.057861f,
    0.234619f,
    0.411621f,
    0.588379f,
    0.765381f,
    0.941162f
  };

  const char* labels[6] = {"Very low", "Low", "Okay", "Good", "Great", "Excellent"};

  int zoneW = imgW / 7;
  int zoneH = imgH + 74;
  int zoneY = imgY - 12;

  for (int i = 0; i < 6; i++) {
    int cx = imgX + (int)(centerRatios[i] * imgW);

    zones[i] = {
      cx - zoneW / 2,
      zoneY,
      zoneW,
      zoneH,
      i + 1,
      labels[i]
    };
  }
}

void drawMoodImage() {
  auto& d = M5.Display;

  int imageIndex = selectedMood;
  if (imageIndex < 0 || imageIndex > 6) {
    imageIndex = 0;
  }

  bool ok = d.drawPng(mood_scale_pngs[imageIndex], mood_scale_png_lens[imageIndex], imgX, imgY);

  if (!ok) {
    d.setTextDatum(middle_center);
    d.setTextColor(TFT_RED, TFT_WHITE);
    d.setTextSize(3);
    d.drawString("Mood image failed to draw", screenW / 2, imgY + 50);
  }

  d.setTextDatum(top_center);
  d.setTextSize(2);
  d.setTextColor(0x1082, TFT_WHITE);

  for (int i = 0; i < 6; i++) {
    int cx = zones[i].x + zones[i].w / 2;
    d.drawString(String(i + 1) + " - " + zones[i].label, cx, imgY + imgH + 22);
  }
}

void drawFooter() {
  auto& d = M5.Display;

  d.fillRect(0, screenH - 54, screenW, 54, TFT_WHITE);
  d.setTextDatum(bottom_center);
  d.setTextSize(2);

  String msg;
  if (selectedMood >= 1 && selectedMood <= 6) {
    msg = "Mood: ";
    msg += String(selectedMood);
    msg += " - ";
    msg += zones[selectedMood - 1].label;
    msg += "   |   ";
    msg += postStatus;
  } else {
    msg = postStatus;
  }

  d.setTextColor(0x1082, TFT_WHITE);
  d.drawString(msg, screenW / 2, screenH - 26);

  if (lastServerResponse.length()) {
    d.setTextColor(0x5AEB, TFT_WHITE);
    String trimmed = lastServerResponse;
    if (trimmed.length() > 90) {
      trimmed = trimmed.substring(0, 90) + "...";
    }
    d.drawString(trimmed, screenW / 2, screenH - 4);
  }
}

void drawAll() {
  auto& d = M5.Display;

  d.fillScreen(TFT_WHITE);
  drawHeader();
  drawPrompt();
  drawMoodImage();
  drawFooter();
}

// -------------------- API --------------------

String buildMoodPayload(const MoodZone& z) {
  String payload = "{";
  payload += "\"device_id\":\"" + jsonEscape(String(WALL_DEVICE_ID)) + "\",";
  payload += "\"facility_id\":\"" + jsonEscape(String(FACILITY_ID)) + "\",";
  payload += "\"bed_id\":\"" + jsonEscape(String(BED_ID)) + "\",";
  payload += "\"event_type\":\"mood_checkin\",";
  payload += "\"mood_score\":" + String(z.score) + ",";
  payload += "\"mood_label\":\"" + jsonEscape(String(z.label)) + "\",";
  payload += "\"source\":\"tab5_wall_unit\",";
  payload += "\"ts_epoch\":" + currentEpochText() + ",";
  payload += "\"raw\":{";
  payload += "\"ui_version\":\"" + jsonEscape(String(WALL_UI_VERSION)) + "\",";
  payload += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
  payload += "\"ip\":\"" + jsonEscape(ipText()) + "\"";
  payload += "}";
  payload += "}";

  return payload;
}

bool postMoodCheckin(const MoodZone& z) {
  if (WiFi.status() != WL_CONNECTED) {
    lastServerResponse = "No Wi-Fi connection";
    setPostStatus("Failed: offline");
    return false;
  }

  String url = String(SERVER_BASE_URL) + "/api/v1/wall/event";
  String payload = buildMoodPayload(z);

  Serial.println("POST " + url);
  Serial.println(payload);

  if (time(nullptr) < 1700000000) {
    setPostStatus("Waiting for time sync");
    return false;
  }
  WiFiClientSecure secureClient;
  secureClient.setCACert(MELX_ROOT_CA);
  HTTPClient http;
  http.setTimeout(8000);

  if (!url.startsWith("https://") || !http.begin(secureClient, url)) {
    lastServerResponse = "HTTP begin failed";
    setPostStatus("Failed: HTTP setup");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  http.addHeader("Authorization", "Bearer " + String(WALL_API_KEY));

  int code = http.POST(payload);
  String response = http.getString();
  http.end();

  Serial.printf("HTTP code: %d\n", code);
  Serial.println(response);

  lastServerResponse = "HTTP " + String(code) + ": " + response;

  if (code >= 200 && code < 300) {
    setPostStatus("Saved");
    return true;
  }

  setPostStatus("Failed");
  return false;
}

// -------------------- TOUCH --------------------

bool inZone(int x, int y, MoodZone& z) {
  return x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h;
}

void handleTouch(int x, int y) {
  for (int i = 0; i < 6; i++) {
    if (inZone(x, y, zones[i])) {
      selectedMood = zones[i].score;

      setPostStatus("Sending...");
      lastServerResponse = "";
      drawAll();

      M5.Speaker.tone(880, 80);

      bool ok = postMoodCheckin(zones[i]);

      if (ok) {
        M5.Speaker.tone(1175, 80);
      } else {
        M5.Speaker.tone(220, 160);
      }

      drawAll();
      return;
    }
  }
}

// -------------------- SETUP / LOOP --------------------

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("BOOT: MelX Health Tab5 wall unit v6");

  auto cfg = M5.config();
  M5.begin(cfg);

  M5.Display.setRotation(1);
  M5.Display.setBrightness(200);

  screenW = M5.Display.width();
  screenH = M5.Display.height();

  Serial.printf("Display: %d x %d\n", screenW, screenH);

  imgW = mood_scale_png_width;
  imgH = mood_scale_png_height;
  imgX = (screenW - imgW) / 2;
  imgY = 405;

  if (imgY + imgH + 75 > screenH) {
    imgY = screenH - imgH - 95;
  }

  computeMoodZones();
  setPostStatus("Starting Wi-Fi...");
  drawAll();

  startWifi();

  uint32_t start = millis();
  while (millis() - start < 8000) {
    M5.update();
    maintainWifi();

    if (WiFi.status() == WL_CONNECTED) {
      break;
    }

    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    setPostStatus("Ready");
  } else {
    setPostStatus("Offline mode");
  }

  drawAll();
  Serial.println("Ready. Tap a face.");
}

void loop() {
  M5.update();
  maintainWifi();

  auto t = M5.Touch.getDetail();

  if (t.wasPressed()) {
    handleTouch(t.x, t.y);
  }

  if (millis() - lastClockRedraw > 30000) {
    lastClockRedraw = millis();

    if ((postStatus == "Saved" || postStatus == "Failed" || postStatus == "Failed: offline") &&
        millis() - lastStatusChange > 15000) {
      setPostStatus(WiFi.status() == WL_CONNECTED ? "Ready" : "Offline mode");
      lastServerResponse = "";
    }

    drawAll();
  }

  delay(10);
}
