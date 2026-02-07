import json
from datetime import datetime
import pandas as pd
import os
import pytest

@pytest.fixture
def client(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as c:
        # Simulate login session
        with c.session_transaction() as sess:
            sess['user'] = 'demo@example.com'
        yield c

def test_predict_mood_basic(client):
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "Temperature": 70.0,
        "Humidity": 45.0,
        "MQ-2": "OK",
        "BH1750FVI": 250.0,
        "Radar": 1,
        "Ultrasonic": 60.0,
        "song": 1
    }
    resp = client.post('/api/predict_mood', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'predicted_mood' in data
    assert isinstance(data['predicted_mood'], int)


def test_predict_mood_invalid(client):
    resp = client.post('/api/predict_mood', json={"Temperature": 70})
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data


def test_dashboard_data(client, monkeypatch):
    # Monkeypatch loader to supply deterministic dataframe
    import app as a
    import pandas as pd
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    df = pd.DataFrame({
        'timestamp': [now.isoformat(), (now - timedelta(days=1)).isoformat()],
        'Temperature': [70, 72],
        'Humidity': [40, 42],
        'MQ-2': ['OK','OK'],
        'BH1750FVI': [200, 250],
        'Radar': [1,0],
        'Ultrasonic': [50,55],
        'mood':[3,4],
        'song':[1,2]
    })
    def fake_loader(): return df
    monkeypatch.setattr(a, '_load_sensor_dataframe', fake_loader)
    # Invalidate cache by calling cached aggregates with new marker
    resp = client.get('/api/dashboard_data')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'summary' in data
    assert 'donutSlices' in data


def test_staff_cannot_list_staff(client):
    with client.session_transaction() as sess:
        sess['user'] = 'staff@example.com'
        sess['role'] = 'staff'
    resp = client.get('/api/staff')
    assert resp.status_code == 403


def test_csrf_required_for_api_post(client):
    import app as flask_app
    import db as db_module
    prev_testing = flask_app.app.config.get('TESTING')
    flask_app.app.config['TESTING'] = False
    try:
        patient = db_module.db_insert_patient('Test Patient', None, None)
        patient_id = patient.get('id')
        with client.session_transaction() as sess:
            sess['user'] = 'staff@example.com'
            sess['role'] = 'staff'
            sess['assigned_patient_ids'] = [patient_id]
            sess['csrf_token'] = 'testtoken'
        resp = client.post('/api/goals', json={'patient_id': patient_id, 'title': 'Goal'})
        assert resp.status_code == 403
        resp = client.post(
            '/api/goals',
            json={'patient_id': patient_id, 'title': 'Goal'},
            headers={'X-CSRF-Token': 'testtoken'}
        )
        assert resp.status_code == 201
    finally:
        flask_app.app.config['TESTING'] = prev_testing
