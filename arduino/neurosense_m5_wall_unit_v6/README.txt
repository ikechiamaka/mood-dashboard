NeuroSense / La'i Tab5 Wall Unit v6
===================================

This version connects the working mood UI to:

POST /api/v1/wall/event

What it does:
- Shows the same embedded emoji mood UI.
- Keeps the corrected grey selected-state.
- Connects the Tab5 to Wi-Fi.
- Syncs time with NTP.
- Sends mood_checkin events to the Flask backend using Bearer API key auth.
- Shows status on screen: Starting Wi-Fi, Online, Sending, Saved, Failed.

Before upload:
1. Open secrets.h.
2. Set WIFI_SSID and WIFI_PASSWORD.
3. Set SERVER_BASE_URL.
   Example:
   http://192.168.1.50:5000
4. Set WALL_API_KEY to the wall_unit device API key from the NeuroSense backend.
5. Set WALL_DEVICE_ID, FACILITY_ID, and BED_ID to match the dashboard device assignment.

Important:
- Use a 2.4GHz Wi-Fi network.
- For local testing, do not use localhost in SERVER_BASE_URL. Use your computer/server IP address.
- Make sure Windows Firewall allows inbound traffic to the Flask port, usually 5000.
- The device must exist in the NeuroSense devices table and be assigned to the same facility_id and bed_id.

Expected backend response:
HTTP 200 or 201 with JSON like:
{"ok": true, "event_type": "mood_checkin", ...}

If it fails:
- HTTP 401/403: check WALL_API_KEY and Authorization Bearer setup.
- HTTP 400: check device_id, facility_id, bed_id, mood_score.
- HTTP -1 or timeout: check Wi-Fi, server IP, firewall, and Flask host binding.
