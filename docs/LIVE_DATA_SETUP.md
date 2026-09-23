# Live device pilot

The public splash page stays at `/`. Staff sign in at `/login` and use
`/dashboard`. Devices use HTTPS endpoints on the same deployed hostname.

Verified on 2026-09-23: use **https://www.melxhealth.com/login**. The apex
`https://melxhealth.com/login` currently returns 404 from GoDaddy infrastructure,
while `www` serves the DigitalOcean app. Configure devices with the `www` host.

## 1. Database and deployment

Create a DigitalOcean Managed PostgreSQL database in the app's region and add
the App Platform application as a trusted source. This is a paid resource.
Configure the app's encrypted `DATABASE_URL` with its connection details. Use
the provider's CA certificate and `sslmode=verify-full` for certificate and
hostname verification. Copy the public CA certificate to `certs/database-ca.crt`
in the deployment or supply a different readable `sslrootcert` path in the URI.
URL-encode special characters in the database password.

Use `.env.production.example` as the settings checklist. Keep all passwords,
device API keys and the actual DATABASE_URL out of Git and chat. With
`REQUIRE_POSTGRES=1`, the app refuses to silently fall back to SQLite.

Keep the existing run command:

```sh
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT wsgi:app
```

Deploy the tested code only after these settings are available. New production
databases contain no demo users, patients, readings or CSV imports. PostgreSQL
tables are initialized automatically. Database operations use separate
connections per thread, released at the end of each web request.

Reference: https://docs.digitalocean.com/products/app-platform/how-to/store-data/

## 2. Preserve existing records, or start clean

For a clean pilot, use the empty production database and create a real admin:

```sh
python manage.py create-admin --email YOUR_ADMIN_EMAIL --name "Administrator"
python manage.py create-facility --id 1 --name "YOUR FACILITY" --timezone Africa/Lagos
```

Run these in the deployed app console with its database environment. The admin
password is entered at a hidden prompt, must be at least 16 characters, and is
never included in shell history. Existing accounts are not overwritten. Use the
facility's actual IANA timezone, not necessarily Africa/Lagos.

If existing cloud data must be retained, stop incoming device traffic and pause
staff writes first. Export a consistent SQLite backup from the CURRENT running
app before redeploying. The committed `data/app.db` is not a backup of live data.
Using SQLite's backup API or its `.backup` command produces a consistent snapshot.
Store that backup securely outside the app's ephemeral filesystem.

On a trusted machine with the new DATABASE_URL and source snapshot:

```sh
python migrate_sqlite.py /secure/path/live-snapshot.db
```

The importer reads the source without modifying it, requires an empty target,
checks row counts, preserves IDs and hashes, and resets generated-ID sequences.
Data writes commit as one transaction; failures roll back imported rows.
Schema creation is separate and can remain after a failed import. The source
snapshot must have the current application schema. The importer preserves demo
records too: review imported accounts and reset or remove known demo accounts
before admitting production users. Retain the snapshot for rollback; do not
resume both old and new writers at the same time.

## 3. Register and configure the devices

Sign in with your real admin at `/login`. Create a bed, assign the patient when
appropriate, and register one device per physical board in Admin > Devices.
Use each device's own freshly generated API key. Facility IDs are numeric. Bed
IDs are the actual returned IDs, which may be UUIDs, not the visible bed label.

| Board | Device ID example | Firmware settings | Endpoint |
| --- | --- | --- | --- |
| MR60BHA2 / XIAO ESP32-C6 | BHA2-001 | API_URL, API_KEY, MELX_FACILITY_ID, MELX_DEVICE_ID, MELX_BED_ID | /api/v1/telemetry |
| ESP32 environment hub | ENV-001 | SERVER_BASE_URL, ENV_API_KEY, ENV_DEVICE_ID, FACILITY_ID, BED_ID | /api/v1/wall/event |
| M5Stack Tab5 | WALL-001 | SERVER_BASE_URL, WALL_API_KEY, WALL_DEVICE_ID, FACILITY_ID, BED_ID | /api/v1/wall/event |

Use the adjacent `secrets.h.example` in each firmware folder. Do not overwrite
working Wi-Fi values blindly. API_URL is the full HTTPS telemetry URL;
SERVER_BASE_URL is only `https://www.melxhealth.com` (no trailing slash). If that
hostname redirects, configure its final app hostname directly; devices should
not send API keys through a domain forwarding service.

All sketches now require HTTPS with CA verification and a synchronized clock.
`tls_ca.h` contains GTS Root R4 (verified for the current www server chain) and
ISRG Root X1. Verify the deployed server chain before flashing; update the CAs if the host uses a different
issuer. Never bypass verification with `setInsecure()`.

The bed sensor requires Seeed_Arduino_mmWave and the XIAO ESP32-C6 board package.
`USE_MMWAVE=1` selects real measurements. Missing RR/HR are sent as null; a missing
radar frame is not reported as an empty bed. The BHA2 does not detect falls.

The environment hub now defaults to `DUMMY_ENV_MODE=0`. All optional sensors are
disabled until wiring is confirmed. Enable only physically connected sensors.
For DHT22, install Adafruit DHT sensor library and Adafruit Unified Sensor, then
set USE_DHT22=1 and the actual pin. For BH1750, install the BH1750 library and
set USE_BH1750=1 plus actual SDA/SCL pins. These integrations now read the devices;
missing temperature, humidity and light values are null. MQ135 and microphone
ADC counts remain raw measurements until calibrated; they are not labeled as
air-quality categories or decibels. No placeholder 24 C / 60% readings are sent.
The Tab5 requires M5Unified, M5GFX and the correct Tab5 board package.

## 4. Acceptance checks

Start with one test bed and no identifiable patient information. At 115200 baud,
confirm Wi-Fi, time synchronization, and HTTP 200 with `ok: true`. Check that
the correct dashboard bed receives advancing timestamps and plausible physical
measurements. An HTTP success alone does not prove sensor accuracy.

Check wrong-key rejection (401), facility/bed mismatch rejection (403), loss and
restoration of Wi-Fi, missing sensor frames, and a device reboot. Verify null
measurements display as unavailable, not zero or an earlier valid reading.
Rotate a pilot device key and confirm the old key stops working. Reconnect the
app to PostgreSQL and confirm pilot records persist. Production rejects payloads
explicitly marked as simulated; firmware/hardware checks remain necessary.

## 5. Alerts

Keep one web worker and one app instance for the initial pilot. After verifying
staff contacts, shifts, thresholds and SMS delivery, set ENABLE_ALERT_MONITOR=1
and DISABLE_ALERT_MONITOR=0. Choose Termii or Twilio and configure its actual
credentials. Test with an agreed recipient before enabling operational alerts.
Production no longer treats mocked or failed SMS sends as delivered.

Do not run the background monitor in every replica. When scaling, disable the
web monitor and use exactly one separate `python alert_worker.py` worker with
the same DATABASE_URL. Backup/restore testing and physical sensor validation
remain part of the pilot before operational use.

## Verification

```sh
python -m pytest
# Optional real PostgreSQL integration tests use a fresh random schema:
TEST_POSTGRES_URL=postgresql://... python -m pytest tests/test_postgres.py
```

Use a disposable test database for TEST_POSTGRES_URL. The tests create and drop
only their own randomly named schemas. No tests connect to production by default.

Local verification on 2026-09-23 covered SQLite plus a PGlite PostgreSQL engine
accessed through the PostgreSQL wire protocol. It covered authenticated device
ingestion, dashboard retrieval, reconnect persistence, migration row counts,
generated IDs, rollback and refusal to import into a populated destination.
DigitalOcean connectivity and physical sensor behavior still need the pilot.

Compile checks use ESP32 Arduino core 3.3.12 and placeholder secrets. The
environment-hub check enables DHT22 and BH1750 to compile the actual sensor paths;
the checked-in defaults leave sensors disabled until wiring is confirmed.

All three compile checks passed:

- Bed sensor: `esp32:esp32:XIAO_ESP32C6` (85% program storage).
- Environment hub: `esp32:esp32:esp32` with DHT22/BH1750 enabled (80%).
- Wall unit: `esp32:esp32:m5stack_tab5` (25%).

The bundled CA certificates also verified the live `www` hostname from the
development machine. Compile and desktop TLS checks do not replace testing on
the boards. No physical devices were flashed and no cloud records were changed.
