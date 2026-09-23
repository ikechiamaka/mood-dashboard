from test_bed_monitoring import _seed_facility_bed_device


def test_invalid_sensor_values_are_not_saved_as_measurements(app_db):
    app_module, db = app_db
    facility, bed, device, key = _seed_facility_bed_device(db)
    response = app_module.app.test_client().post('/api/v1/telemetry', json={
        'device_id': device, 'facility_id': facility, 'bed_id': bed,
        'rr': 14, 'hr': 72, 'raw': {'rr_valid': False, 'hr_valid': False},
    }, headers={'Authorization': f'Bearer {key}'})
    assert response.status_code == 200
    row = db.db_get_latest_telemetry_for_bed(bed)
    assert row['rr'] is None and row['hr'] is None


def test_production_rejects_simulated_device_data(app_db, monkeypatch):
    app_module, db = app_db
    facility, bed, device, key = _seed_facility_bed_device(db)
    monkeypatch.setenv('FLASK_ENV', 'production')
    client = app_module.app.test_client()
    for route, flag in (('/api/v1/telemetry', 'simulated'), ('/api/v1/wall/event', 'dummy_env_mode')):
        response = client.post(route, json={
            'device_id': device, 'facility_id': facility, 'bed_id': bed,
            'raw': {flag: True}, 'event_type': 'environment',
        }, headers={'Authorization': f'Bearer {key}'})
        assert response.status_code == 400
    assert db.db_get_latest_telemetry_for_bed(bed) is None


def test_production_database_starts_without_demo_data(tmp_path, monkeypatch):
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('FLASK_ENV', 'production')
    monkeypatch.setenv('REQUIRE_POSTGRES', '0')
    monkeypatch.setenv('MELX_HEALTH_DB_PATH', str(tmp_path / 'clean.db'))
    import db
    db.close_conn()
    db.init_db()
    for table in ('users', 'patients', 'readings', 'checkins'):
        assert db.get_conn().execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
    db.close_conn()


def test_postgres_required_prevents_sqlite_fallback(monkeypatch):
    import pytest
    import db
    db.close_conn()
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('REQUIRE_POSTGRES', '1')
    with pytest.raises(RuntimeError, match='DATABASE_URL is required'):
        db.get_conn()


def test_production_sms_failure_is_not_reported_as_delivered(app_db, monkeypatch):
    app_module, _ = app_db
    monkeypatch.setenv('FLASK_ENV', 'production')
    monkeypatch.setenv('TERMII_API_KEY', '')
    monkeypatch.setattr(app_module, '_send_twilio_sms', lambda *_: (False, {'reason': 'not configured'}))
    sent, metadata = app_module._send_sms('+10000000000', 'Pilot test')
    assert sent is False
    assert metadata['provider'] == 'none'
