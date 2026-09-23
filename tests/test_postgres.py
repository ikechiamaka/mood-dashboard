"""Integration tests: set TEST_POSTGRES_URL to a disposable PostgreSQL server."""
import importlib
import os
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest
from werkzeug.security import generate_password_hash


@pytest.fixture
def postgres(monkeypatch):
    url = os.getenv('TEST_POSTGRES_URL')
    if not url:
        pytest.skip('TEST_POSTGRES_URL is not configured')
    import psycopg
    from psycopg import sql
    schema = 'test_' + uuid4().hex
    with psycopg.connect(url, autocommit=True) as control:
        control.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query['options'] = '-csearch_path=' + schema
    scoped = urlunsplit(parts._replace(query=urlencode(query)))
    monkeypatch.setenv('DATABASE_URL', scoped)
    monkeypatch.setenv('DATABASE_SCHEMA', schema)
    monkeypatch.setenv('FLASK_ENV', 'production')
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.setenv('FLASK_SECRET_KEY', 'integration-test-only')
    monkeypatch.setenv('SEED_DEMO_DATA', '0')
    monkeypatch.setenv('CHATBOT_PROVIDER', 'local')
    monkeypatch.setenv('DISABLE_ALERT_MONITOR', '1')
    for name in ('users', 'db', 'app'):
        previous = sys.modules.pop(name, None)
        if name == 'db' and previous:
            previous.close_conn()
    db = importlib.import_module('db')
    try:
        yield db
    finally:
        db.close_conn()
        with psycopg.connect(url, autocommit=True) as control:
            control.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_postgres_device_to_dashboard_and_restart(postgres):
    db = postgres
    app = importlib.import_module('app').app
    app.config['TESTING'] = True
    conn = db.get_conn()
    assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0
    conn.execute('INSERT INTO facilities (id, name, timezone, created_at) VALUES (?, ?, ?, ?)',
                 (1, 'Pilot', 'UTC', int(time.time())))
    conn.commit()
    db.db_insert_user('pilot@example.test', generate_password_hash('test-password-long'), 'super_admin', 'Pilot', None, [])
    client = app.test_client()
    with client.session_transaction() as session:
        session['user'] = 'pilot@example.test'
        session['role'] = 'super_admin'
    bed_response = client.post('/api/beds', json={'facility_id': 1, 'name': 'Pilot bed', 'label': 'Pilot bed'})
    assert bed_response.status_code == 201, bed_response.get_json()
    bed = bed_response.get_json()['id']
    device_response = client.post('/api/admin/devices', json={'facility_id': 1, 'bed_id': bed, 'device_id': 'PILOT-1'})
    assert device_response.status_code == 201, device_response.get_json()
    key = device_response.get_json()['api_key']
    payload = {'device_id': 'PILOT-1', 'facility_id': 1, 'bed_id': bed, 'rr': 17.5, 'hr': 78, 'presence': True}
    headers = {'Authorization': 'Bearer ' + key}
    assert client.post('/api/v1/telemetry', json=payload).status_code == 401
    result = client.post('/api/v1/telemetry', json=payload, headers=headers)
    assert result.status_code == 200, result.get_json()
    patient = db.db_insert_patient('Pilot', 1, bed)
    assert isinstance(patient['id'], int)
    db.close_conn()
    result = client.post('/api/v1/wall/event', json={**payload, 'event_type': 'mood_checkin', 'mood_score': 4}, headers=headers)
    assert result.status_code == 200, result.get_json()
    result = client.get('/api/bed_bundle', query_string={'bed_id': bed})
    assert result.status_code == 200, result.get_json()
    assert result.get_json()['latest_telemetry']['hr'] == 78
    conn = db.get_conn()
    conn.execute('INSERT INTO readings (timestamp, temperature, humidity, bed_id, patient_id) VALUES (?, ?, ?, ?, ?)',
                 ('2026-09-23T12:00:00+00:00', 24.5, 55.0, bed, patient['id']))
    conn.commit()
    frame = sys.modules['app']._load_sensor_dataframe(patient['id'])
    assert frame.iloc[0]['Temperature'] == 24.5
    db.close_conn()
    db.init_db()
    assert db.db_get_latest_telemetry_for_bed(bed)['hr'] == 78
    conn = db.get_conn()
    conn.execute('INSERT INTO facilities (id, name, timezone, created_at) VALUES (2, ?, ?, ?)', ('Rollback', 'UTC', 1))
    conn.rollback()
    assert conn.execute('SELECT id FROM facilities WHERE id = 2').fetchone() is None


def test_migration_is_verified_and_refuses_populated_target(postgres, tmp_path, monkeypatch):
    from migrate_sqlite import migrate
    db = postgres
    source = tmp_path / 'snapshot.db'
    with monkeypatch.context() as local:
        local.setenv('DATABASE_URL', '')
        local.setenv('MELX_HEALTH_DB_PATH', str(source))
        db.init_db()
        db.db_insert_user('migrated@example.test', generate_password_hash('test-password-long'), 'super_admin', 'Migrated', None, [])
        db.close_conn()
    migrate(source)
    assert db.db_get_user_by_email('migrated@example.test')['name'] == 'Migrated'
    new_user = db.db_insert_user('next@example.test', generate_password_hash('test-password-long'), 'super_admin', 'Next', None, [])
    assert new_user['id'] > 1
    db.close_conn()
    with pytest.raises(RuntimeError, match='Destination must be empty'):
        migrate(source)
