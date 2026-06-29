# MelX Health Firmware Targets

This repository contains separate firmware tracks for separate pieces of MelX Health hardware. There is not one universal firmware image for all boards.

## Current Targets

### 1. M5Stack Tab5 wall unit
- Folder: `arduino/melx_health_m5_wall_unit_v6/`
- Purpose: patient-facing mood UI
- Endpoint: `POST /api/v1/wall/event`
- Suggested device ID: `WALL-001`
- Suggested capabilities: `["mood", "environment"]`

### 2. ESP32 environment hub
- Folder: `arduino/esp32_environment_hub/`
- Purpose: room/environment capture for temperature, humidity, air quality, light, noise, and motion later
- Endpoint: `POST /api/v1/wall/event`
- Suggested device ID: `ENV-001`
- Suggested capabilities: `["environment"]`

### 3. Seeed MR60BHA2 / XIAO bed sensor
- Folder: `arduino/mmwave_bed_telemetry/`
- Purpose: bed-side presence, respiratory rate, heart rate, confidence, and distance telemetry
- Endpoint: `POST /api/v1/telemetry`
- Suggested device ID: `BHA2-001`
- Suggested capabilities: `["presence", "rr", "hr", "distance"]`

## How Devices Tie Together

The backend treats these as separate physical devices. They are linked operationally by assigning them to the same:

- `facility_id`
- `bed_id`

That means one patient room/bed can have:

- `WALL-001` for mood check-ins
- `ENV-001` for room conditions
- `BHA2-001` for bed telemetry

All three should point to the same facility and bed in the MelX Health admin setup.

## Endpoint Mapping

- `WALL-001` -> `/api/v1/wall/event`
- `ENV-001` -> `/api/v1/wall/event`
- `BHA2-001` -> `/api/v1/telemetry`

## Backend / Admin Setup

Create the devices in the MelX Health admin UI before flashing firmware.

Suggested setup:

- `WALL-001`
  - `device_type=wall_unit`
  - assign to the target `facility_id`
  - assign to the target `bed_id`
  - capabilities: `["mood", "environment"]`

- `ENV-001`
  - `device_type=wall_unit` or `device_type=environment_hub`
  - assign to the same `facility_id`
  - assign to the same `bed_id`
  - capabilities: `["environment"]`

- `BHA2-001`
  - `device_type=bed_sensor`
  - assign to the same `facility_id`
  - assign to the same `bed_id`
  - capabilities: `["presence", "rr", "hr", "distance"]`

Each device receives its own bearer API key. The backend stores only the hash, so copy the key at creation or rotation time and place it into the matching `secrets.h`.

## Board Notes

- The Tab5 wall unit uses `WiFi.h` with `WiFi.setPins(...)` as required by the M5Stack Tab5 Wi-Fi examples/docs.
- The environment hub is intended for a normal ESP32 dev board.
- The bed telemetry firmware is intended for the Seeed MR60BHA2/XIAO track and posts to the telemetry endpoint.

## General Rules

- Do not use `localhost` in firmware config when testing against the Flask app on another machine.
- Use the Flask server's reachable LAN IP in `SERVER_BASE_URL` or `API_URL`.
- Keep each firmware target in its own folder with its own `secrets.h`.
