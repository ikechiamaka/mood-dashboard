# ESP32 Environment Hub

This firmware track is for a normal ESP32 board acting as a room/environment hub for MelX Health.

It posts environment snapshots to:

- `POST /api/v1/wall/event`

using the `environment` event type and bearer API key authentication.

## Files

- `esp32_environment_hub.ino`
- `secrets.h.example`

Copy `secrets.h.example` to `secrets.h` and fill in real values before upload.

## Target Use

This device is intended to sit in the same room / care area as the patient bed and wall unit. It should be assigned to the same:

- `facility_id`
- `bed_id`

Suggested device ID:

- `ENV-001`

## Default Behavior

The sketch defaults to:

- `DUMMY_ENV_MODE=1`

That means it sends simulated temperature, humidity, air quality, light, and noise values every 30 seconds without requiring any external sensor libraries.

This is useful for:

- backend integration testing
- dashboard UI testing
- verifying device auth and room mapping before hardware wiring is complete

## Planned Sensor Placeholders

The sketch includes config switches and placeholders for:

- DHT22 for temperature and humidity
- BH1750 for light level
- MQ135 analog sensor for air-quality estimation
- PIR motion sensor
- analog noise sensor

While `DUMMY_ENV_MODE=1`, these sensor-specific integrations are not required.

## Endpoint Contract

The firmware posts to:

`SERVER_BASE_URL + "/api/v1/wall/event"`

Example payload:

```json
{
  "device_id": "ENV-001",
  "facility_id": "FAC123",
  "bed_id": "BED-01",
  "event_type": "environment",
  "source": "esp32_environment_hub",
  "temperature_c": 24.3,
  "humidity": 67,
  "air_quality": "OK",
  "light_level": 120,
  "noise_level": 34,
  "confidence": 0.95,
  "raw": {
    "firmware": "env_hub_v1",
    "ts_epoch": 1730000000,
    "wifi_rssi": -55,
    "ip": "192.168.1.100",
    "mq135_raw": 900,
    "motion": false
  }
}
```

## Setup

1. Create the device in MelX Health admin.
2. Assign it to the correct facility and bed.
3. Copy the revealed API key into `secrets.h`.
4. Set the Flask server LAN IP in `SERVER_BASE_URL`.
5. Upload to a normal ESP32 board.
6. Open Serial Monitor at `115200`.

The sketch prints:

- the full JSON payload
- HTTP status code
- response body

## Notes

- Do not use `localhost` for `SERVER_BASE_URL`.
- Use your Flask server's reachable LAN IP.
- If the board prints `HTTP code: -1`, check Wi-Fi, host IP, and firewall reachability first.
