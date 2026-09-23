import time
from uuid import uuid4

from werkzeug.security import generate_password_hash, check_password_hash

def _seed_facility_bed_device(db_module):
    conn = db_module.get_conn()
    now_epoch = int(time.time())
    suffix = uuid4().hex[:8]
    facility_id = 1
    bed_id = f"BED-{suffix}"
    device_id = f"DEV-{suffix}"
    api_key = f"key-{suffix}"
    conn.execute(
        'INSERT OR IGNORE INTO facilities (id, name, timezone, created_at) VALUES (?, ?, ?, ?)',
        (facility_id, f'Facility {facility_id}', 'UTC', now_epoch),
    )
    conn.execute(
        '''
        INSERT OR REPLACE INTO beds (id, name, label, room, patient, facility_id, active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        ''',
        (bed_id, bed_id, bed_id, '101', None, facility_id, now_epoch),
    )
    conn.execute(
        '''
        INSERT OR REPLACE INTO devices (
            id, facility_id, bed_id, api_key_hash, firmware, capabilities, last_seen_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            device_id,
            facility_id,
            bed_id,
            generate_password_hash(api_key),
            'test-fw',
            '["presence","fall"]',
            None,
            now_epoch,
        ),
    )
    conn.commit()
    return facility_id, bed_id, device_id, api_key


def test_telemetry_ingest_with_bearer_key(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': int(time.time()),
                'presence': True,
                'fall': False,
                'confidence': 0.9,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body.get('ok') is True
        telemetry_id = body.get('telemetry_id')
        assert isinstance(telemetry_id, int)
        conn = db_module.get_conn()
        row = conn.execute('SELECT * FROM telemetry WHERE id = ?', (telemetry_id,)).fetchone()
        assert row is not None
        assert row['device_id'] == device_id
        assert row['bed_id'] == bed_id


def test_wall_event_rejects_missing_bearer_token(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, _api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 4,
            },
        )
        assert resp.status_code == 401


def test_wall_event_rejects_invalid_api_key(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, _api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 4,
            },
            headers={'Authorization': 'Bearer wrong-key'},
        )
        assert resp.status_code == 401


def test_wall_mood_checkin_stores_event(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    conn = db_module.get_conn()
    conn.execute(
        '''
        INSERT INTO patients (
            name, facility_id, bed_id, age, risk_level, primary_condition, allergies, care_focus, avatar_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        ('Wall Mood Patient', facility_id, bed_id, 41, 'Moderate', 'Test Condition', None, 'Observe mood', None),
    )
    conn.commit()
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 4,
                'source': 'tab5_wall_unit',
                'raw': {'ui_version': 'mood_embedded_v5'},
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['ok'] is True
        assert body['mood_score'] == 4
        assert body['mood_label'] == 'Good'
        row = db_module.get_conn().execute(
            'SELECT * FROM wall_events WHERE device_id = ? ORDER BY id DESC LIMIT 1',
            (device_id,),
        ).fetchone()
        assert row is not None
        assert row['event_type'] == 'mood_checkin'
        assert row['mood_score'] == 4
        assert row['mood_label'] == 'Good'
        assert row['patient_id'] is not None


def test_wall_mood_score_out_of_range_is_rejected(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 7,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert resp.status_code == 400
        assert 'mood_score' in (resp.get_json() or {}).get('error', '')


def test_wall_event_rejects_wrong_facility_or_bed(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        bad_facility = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id + 99,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 3,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert bad_facility.status_code == 403

        bad_bed = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': 'BED-WRONG',
                'event_type': 'mood_checkin',
                'mood_score': 3,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert bad_bed.status_code == 403


def test_wall_environment_event_stores_fields(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'environment',
                'temperature_c': 24.3,
                'humidity': 67,
                'air_quality': 'OK',
                'light_level': 120,
                'noise_level': 34,
                'source': 'tab5_wall_unit',
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['ok'] is True
        row = db_module.get_conn().execute(
            'SELECT * FROM wall_events WHERE device_id = ? AND event_type = ? ORDER BY id DESC LIMIT 1',
            (device_id, 'environment'),
        ).fetchone()
        assert row is not None
        assert float(row['temperature_c']) == 24.3
        assert float(row['humidity']) == 67.0
        assert row['air_quality'] == 'OK'


def test_bed_bundle_includes_wall_event_data(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        mood_resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'mood_checkin',
                'mood_score': 5,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert mood_resp.status_code == 200
        env_resp = client.post(
            '/api/v1/wall/event',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'event_type': 'environment',
                'temperature_c': 22.8,
                'humidity': 58,
                'air_quality': 'OK',
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert env_resp.status_code == 200

        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = facility_id

        bundle = client.get(f'/api/bed_bundle?bed_id={bed_id}')
        assert bundle.status_code == 200
        payload = bundle.get_json()
        assert payload['latest_wall_mood']['mood_label'] == 'Great'
        assert float(payload['latest_wall_environment']['temperature_c']) == 22.8
        assert len(payload['recent_wall_mood_events']) >= 1
        assert len(payload['recent_wall_environment_events']) >= 1

        beds = client.get('/api/beds')
        assert beds.status_code == 200
        rows = beds.get_json()
        target = next(item for item in rows if item['id'] == bed_id)
        assert target['latest_mood_label'] == 'Great'
        assert float(target['latest_room_temperature']) == 22.8
        assert float(target['latest_humidity']) == 58.0
        assert target['latest_air_quality'] == 'OK'
        assert target['last_wall_event_at']


def test_alert_monitor_creates_fall_alert(app_db, monkeypatch):
    flask_app, db_module = app_db
    monkeypatch.setenv('ALERT_ENABLE_FALL', '1')
    monkeypatch.setenv('TERMII_API_KEY', '')
    monkeypatch.setenv('TWILIO_ACCOUNT_SID', '')
    monkeypatch.setenv('TWILIO_AUTH_TOKEN', '')
    monkeypatch.setenv('TWILIO_FROM_NUMBER', '')
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        baseline = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': int(time.time()),
                'presence': True,
                'fall': False,
                'confidence': 0.95,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert baseline.status_code == 200
        baseline_id = int(baseline.get_json().get('telemetry_id'))

        ingest = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': int(time.time()),
                'presence': True,
                'fall': True,
                'confidence': 0.95,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert ingest.status_code == 200
    flask_app._alert_cursor_id = baseline_id
    processed = flask_app._alert_monitor_cycle()
    assert processed >= 1
    conn = db_module.get_conn()
    alert = conn.execute(
        'SELECT * FROM alerts WHERE bed_id = ? AND type = ? ORDER BY id DESC LIMIT 1',
        (bed_id, 'FALL'),
    ).fetchone()
    assert alert is not None
    assert alert['status'] == 'open'


def test_alert_monitor_creates_and_resolves_stale_telemetry_alert(app_db, monkeypatch):
    flask_app, db_module = app_db
    monkeypatch.setenv('ALERT_STALE_SEC', '60')
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    old_ts = int(time.time()) - 600
    with flask_app.app.test_client() as client:
        stale_ingest = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': old_ts,
                'presence': True,
                'fall': False,
                'confidence': 0.95,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert stale_ingest.status_code == 200

    flask_app._alert_cursor_id = db_module.db_get_latest_telemetry_id()
    flask_app._alert_monitor_cycle()
    conn = db_module.get_conn()
    stale_alert = conn.execute(
        'SELECT * FROM alerts WHERE bed_id = ? AND type = ? ORDER BY id DESC LIMIT 1',
        (bed_id, 'STALE_TELEMETRY'),
    ).fetchone()
    assert stale_alert is not None
    assert stale_alert['status'] == 'open'

    with flask_app.app.test_client() as client:
        fresh_ingest = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': int(time.time()),
                'presence': True,
                'fall': False,
                'confidence': 0.95,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert fresh_ingest.status_code == 200

    flask_app._alert_monitor_cycle()
    refreshed = db_module.get_conn().execute(
        'SELECT * FROM alerts WHERE id = ?',
        (int(stale_alert['id']),),
    ).fetchone()
    assert refreshed is not None
    assert refreshed['status'] == 'resolved'


def test_schedule_rejects_unknown_staff_contact(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
            sess['csrf_token'] = 'testtoken'
        resp = client.post(
            '/api/schedule',
            json={
                'name': 'Day Shift',
                'start': '07:00',
                'end': '19:00',
                'days': [0, 1, 2],
                'staff_ids': [999999],
            },
            headers={'X-CSRF-Token': 'testtoken'},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body
    assert 'staff_contact_id' in (body.get('error') or '')


def test_alert_types_endpoint_returns_catalog(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.get('/api/alert_types')
        assert resp.status_code == 200
        rows = resp.get_json()
        assert isinstance(rows, list)
        types = {item.get('type') for item in rows}
        assert 'FALL' not in types
        assert 'RR_LOW' in types
        assert 'STALE_TELEMETRY' in types
        assert 'DEVICE_OFFLINE' in types


def test_alert_types_endpoint_includes_fall_when_enabled(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('ALERT_ENABLE_FALL', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.get('/api/alert_types')
        assert resp.status_code == 200
        rows = resp.get_json()
        assert isinstance(rows, list)
        types = {item.get('type') for item in rows}
        assert 'FALL' in types


def test_admin_can_rotate_device_api_key(app_db):
    flask_app, db_module = app_db
    _, _, device_id, old_api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
            sess['csrf_token'] = 'testtoken'
        resp = client.post(
            f'/api/admin/devices/{device_id}/rotate_key',
            json={},
            headers={'X-CSRF-Token': 'testtoken'},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body
        new_key = body.get('api_key')
        assert isinstance(new_key, str) and len(new_key) > 10
        assert new_key != old_api_key
        row = db_module.get_conn().execute('SELECT api_key_hash FROM devices WHERE id = ?', (device_id,)).fetchone()
        assert row is not None
        assert check_password_hash(row['api_key_hash'], new_key)


def test_admin_can_delete_device_and_ingest_fails(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
            sess['csrf_token'] = 'testtoken'
        resp = client.delete(
            f'/api/admin/devices/{device_id}',
            headers={'X-CSRF-Token': 'testtoken'},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and body.get('status') == 'deleted'

        ingest = client.post(
            '/api/v1/telemetry',
            json={
                'device_id': device_id,
                'facility_id': facility_id,
                'bed_id': bed_id,
                'ts': int(time.time()),
                'presence': True,
                'fall': False,
                'confidence': 0.9,
            },
            headers={'Authorization': f'Bearer {api_key}'},
        )
        assert ingest.status_code in (401, 403)

def test_telemetry_ingest_accepts_utf16_json_body(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    payload = {
        "device_id": device_id,
        "facility_id": facility_id,
        "bed_id": bed_id,
        "ts": int(time.time()),
        "presence": True,
        "fall": False,
        "confidence": 0.9,
    }
    body_utf16 = db_module.json.dumps(payload).encode("utf-16")
    with flask_app.app.test_client() as client:
        resp = client.post(
            "/api/v1/telemetry",
            data=body_utf16,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200


def test_telemetry_ingest_accepts_wrapped_json_string_body(app_db):
    flask_app, db_module = app_db
    facility_id, bed_id, device_id, api_key = _seed_facility_bed_device(db_module)
    payload = {
        "device_id": device_id,
        "facility_id": facility_id,
        "bed_id": bed_id,
        "ts": int(time.time()),
        "presence": True,
        "fall": False,
        "confidence": 0.9,
    }
    wrapped = db_module.json.dumps(db_module.json.dumps(payload)).encode("utf-8")
    with flask_app.app.test_client() as client:
        resp = client.post(
            "/api/v1/telemetry",
            data=wrapped,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200
