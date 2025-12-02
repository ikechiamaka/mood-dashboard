from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import glob
import os
import joblib
import csv
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import openai
from uuid import uuid4
from werkzeug.utils import secure_filename

from data_store import load_json_list, save_json_list

from preprocessing import engineer_features, ensure_feature_order, FEATURE_COLUMNS
from users import (
    authenticate,
    get_user_by_email,
    db_insert_user,
    db_list_users,
    db_delete_user_by_email,
    db_update_user,
)
from db import (
    init_db,
    db_list_staff,
    db_insert_staff,
    db_delete_staff,
    db_list_beds,
    db_insert_bed,
    db_list_schedule,
    db_insert_schedule,
    db_list_patients,
    db_get_patient_by_id,
    db_insert_patient,
    get_conn,
)
from functools import lru_cache
from datetime import datetime as dt
from typing import Any, Dict



load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-me-dev-secret')
remember_days = os.getenv('SESSION_REMEMBER_DAYS', '14')
try:
    remember_days_int = int(remember_days)
except ValueError:
    remember_days_int = 14
app.permanent_session_lifetime = timedelta(days=max(1, remember_days_int))
init_db()

# NOTE: Demo credentials are stored hashed in users.py (environment overrideable)

# Load trained mood model
from sklearn.dummy import DummyClassifier
import warnings
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mood_model.pkl')
with warnings.catch_warnings(record=True) as wlist:
    try:
        mood_model = joblib.load(MODEL_PATH)
        # Filter noisy sklearn pickle version warnings
        filtered = [w for w in wlist if 'InconsistentVersionWarning' not in str(w.message)]
        for w in filtered:
            logging.warning(w.message)
    except Exception as e:
        logging.warning(f"Failed to load model at {MODEL_PATH}: {e}. Using DummyClassifier.")
        dummy = DummyClassifier(strategy='most_frequent')
        synth = pd.DataFrame([[0]*len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
        dummy.fit(synth, [3])
        mood_model = dummy


JOURNAL_PATH = os.path.join('data', 'journal.csv')
CHECKIN_PATH = os.path.join('data', 'checkins.csv')
GOALS_FILE = 'goals.json'
AVATAR_UPLOAD_SUBDIR = 'uploads'
AVATAR_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', AVATAR_UPLOAD_SUBDIR)
ALLOWED_AVATAR_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
SUPPORT_REQUESTS_FILE = 'support_requests.json'
PREDICTIONS_PATH = os.path.join('data', 'predictions.csv')


def _normalize_score(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    span = high - low
    if span <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / span)) * 100)


def _environment_score(temp_mean: float | None, humid_mean: float | None) -> float:
    if temp_mean is None and humid_mean is None:
        return 0.0
    temp_component = 100.0
    if temp_mean is not None:
        ideal_low, ideal_high = 68.0, 74.0
        tolerance = 12.0
        if temp_mean < ideal_low:
            diff = ideal_low - temp_mean
        elif temp_mean > ideal_high:
            diff = temp_mean - ideal_high
        else:
            diff = 0.0
        temp_component = max(0.0, 100.0 - (diff / tolerance) * 60.0)
    humid_component = 100.0
    if humid_mean is not None:
        ideal_low, ideal_high = 35.0, 55.0
        tolerance = 25.0
        if humid_mean < ideal_low:
            diff = ideal_low - humid_mean
        elif humid_mean > ideal_high:
            diff = humid_mean - ideal_high
        else:
            diff = 0.0
        humid_component = max(0.0, 100.0 - (diff / tolerance) * 60.0)
    return float(max(0.0, min(100.0, 0.6 * temp_component + 0.4 * humid_component)))


def _score_to_band(score: float | None) -> str:
    if score is None:
        return 'Unknown'
    if score >= 70:
        return 'High'
    if score >= 40:
        return 'Moderate'
    return 'Low'


def _score_to_quality(score: float | None) -> str:
    if score is None:
        return 'Unknown'
    if score >= 70:
        return 'Good'
    if score >= 40:
        return 'Fair'
    return 'Needs Attention'


def _compute_health_scores(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {'mood': 0.0, 'movement': 0.0, 'light': 0.0, 'overallScore': 0.0}
    numeric = df.copy()
    for col in ['Temperature', 'Humidity', 'Ultrasonic', 'BH1750FVI', 'mood']:
        if col in numeric:
            numeric[col] = pd.to_numeric(numeric[col], errors='coerce')
    mood_mean = numeric['mood'].mean(skipna=True) if 'mood' in numeric else None
    movement_mean = numeric['Ultrasonic'].mean(skipna=True) if 'Ultrasonic' in numeric else None
    temp_mean = numeric['Temperature'].mean(skipna=True) if 'Temperature' in numeric else None
    hum_mean = numeric['Humidity'].mean(skipna=True) if 'Humidity' in numeric else None
    light_mean = numeric['BH1750FVI'].mean(skipna=True) if 'BH1750FVI' in numeric else None

    mood_score = _normalize_score(mood_mean, 1.0, 6.0)
    movement_score = _normalize_score(movement_mean, 15.0, 120.0)
    light_score = _normalize_score(light_mean, 150.0, 800.0)
    environment_val = _environment_score(temp_mean, hum_mean)
    environment_score = max(0.0, min(100.0, 0.7 * environment_val + 0.3 * light_score))
    overall = float(max(0.0, min(100.0, (mood_score + movement_score + environment_score) / 3)))
    return {
        'mood': mood_score,
        'movement': movement_score,
        'light': environment_score,
        'overallScore': overall,
    }


def _ensure_csv_columns(path, header):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return
    existing_header = rows[0]
    if existing_header == header:
        return
    index_map = {col: idx for idx, col in enumerate(existing_header)}
    upgraded = []
    for row in rows[1:]:
        new_row = []
        for col in header:
            idx = index_map.get(col)
            new_row.append(row[idx] if idx is not None and idx < len(row) else '')
        upgraded.append(new_row)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(upgraded)


def _append_csv_rows(path, header, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _ensure_csv_columns(path, header)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)


def _seed_demo_logs_if_needed():
    try:
        patients = db_list_patients()
    except Exception:
        return
    if not patients:
        return
    patient_summaries = []
    for entry in patients:
        pid = entry.get('id')
        if pid is None:
            continue
        patient_summaries.append({'id': pid, 'name': entry.get('name', 'Patient')})
    if not patient_summaries:
        return
    journal_header = ['user', 'patient_id', 'timestamp', 'mood', 'text']
    checkins_header = ['user', 'patient_id', 'timestamp', 'mood', 'stress']
    _ensure_csv_columns(JOURNAL_PATH, journal_header)
    _ensure_csv_columns(CHECKIN_PATH, checkins_header)
    journal_existing = set()
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('patient_id')
                if pid and str(pid).isdigit():
                    journal_existing.add(int(pid))
    checkin_existing = set()
    if os.path.exists(CHECKIN_PATH):
        with open(CHECKIN_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('patient_id')
                if pid and str(pid).isdigit():
                    checkin_existing.add(int(pid))
    base_user = os.getenv('DEMO_SUPER_EMAIL', 'superadmin@demo.com')
    now = datetime.utcnow()
    phrases = [
        'had a calm afternoon walk.',
        'completed their breathing exercises.',
        'responded well to music therapy.',
        'enjoyed time with family.',
        'slept soundly through the night.',
        'was engaged during group session.'
    ]
    journal_rows = []
    checkin_rows = []
    for idx, info in enumerate(patient_summaries):
        pid = info['id']
        name = info.get('name', 'Patient')
        timestamp = (now - timedelta(hours=idx * 6)).replace(microsecond=0).isoformat()
        if pid not in journal_existing:
            mood_value = 3 + (idx % 3)
            note = f"{name} {phrases[idx % len(phrases)]}"
            journal_rows.append([base_user, pid, timestamp, mood_value, note])
        if pid not in checkin_existing:
            mood_value = 3 + (idx % 2)
            stress_value = 2 + ((idx + 1) % 3)
            checkin_rows.append([base_user, pid, timestamp, mood_value, stress_value])
    _append_csv_rows(JOURNAL_PATH, journal_header, journal_rows)
    _append_csv_rows(CHECKIN_PATH, checkins_header, checkin_rows)
_seed_demo_logs_if_needed()











# Configure logging so you can see the exception
logging.basicConfig(level=logging.INFO)

# Pull from .env or fall back to your literal (if you really must)
raw_key = os.getenv("OPENAI_API_KEY", "")  # Do not fallback to hardcoded key in production.

# Replace any non-ascii hyphens with ASCII hyphens:
key = raw_key.replace("\u2011", "-")

# Optional extra cleanup if you suspect any other invisible chars:
key = "".join(ch for ch in key if ord(ch) < 128)

openai.api_key = key if key else None


@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify(error="Unauthorized"), 401

    data = request.get_json(force=True)
    user_msg = (data or {}).get("message", "").strip()
    if not user_msg:
        return jsonify(error="No message provided"), 400

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a helpful assistant in a hospital dashboard."},
                {"role": "user",   "content": user_msg}
            ]
        )
        # Access the reply correctly:
        assistant_reply = resp.choices[0].message.content
        return jsonify(reply=assistant_reply)
    except Exception as e:
        app.logger.error("OpenAI chat error", exc_info=e)
        return jsonify(error="Chat API error"), 500







@app.route('/api/weekly_insights')
def weekly_insights():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    pid_raw = request.args.get('patient_id')
    pid = None
    try:
        pid = int(pid_raw) if pid_raw else None
    except Exception:
        pid = None

    df = _load_sensor_dataframe(pid)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        now = pd.Timestamp.now()
        week_ago = now - pd.Timedelta(days=7)
        df = df[df['timestamp'] >= week_ago]
        if 'date_only' not in df.columns:
            df['date_only'] = df['timestamp'].dt.date
    else:
        df = pd.DataFrame()

    if df.empty:
        return jsonify({"narrative": "No data available for the past week."})

    # Ensure numeric
    for col in ['mood','Ultrasonic','Temperature','BH1750FVI']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute stats
    avg_mood = df['mood'].mean() if 'mood' in df else None
    avg_move = df['Ultrasonic'].mean() if 'Ultrasonic' in df else None
    avg_temp = df['Temperature'].mean() if 'Temperature' in df else None
    avg_light = df['BH1750FVI'].mean() if 'BH1750FVI' in df else None
    mood_series = df['mood'].dropna() if 'mood' in df else pd.Series(dtype=float)
    max_row = df.loc[mood_series.idxmax()] if not mood_series.empty else None
    min_row = df.loc[mood_series.idxmin()] if not mood_series.empty else None

    # Build narrative
    parts = []
    if pd.notnull(avg_mood):
        parts.append(f"Over the last 7 days your average mood was {avg_mood:.1f}/6.")
    if max_row is not None:
        d = max_row['timestamp'].strftime("%b %d")
        parts.append(f"Your highest mood ({int(max_row['mood'])}) was on {d}.")
    if min_row is not None:
        d = min_row['timestamp'].strftime("%b %d")
        parts.append(f"Your lowest mood ({int(min_row['mood'])}) was on {d}.")
    if pd.notnull(avg_move):
        parts.append(f"Average movement was {avg_move:.0f} cm.")
    if pd.notnull(avg_temp):
        parts.append(f"Avg. temperature: {avg_temp:.1f} deg F; light: {avg_light:.0f} lux.")

    narrative = " ".join(parts) if parts else "No data available for the past week."

    return jsonify({"narrative": narrative})



@app.route('/api/checkins', methods=['GET', 'POST'])
def checkins():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    pid_raw = request.args.get('patient_id')
    pid_filter = None
    if pid_raw:
        try:
            pid_filter = int(pid_raw)
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid patient_id'}), 400
        if not _patient_access_ok(pid_filter):
            return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'POST':
        payload = request.get_json() or {}
        ts = payload.get('timestamp')
        mood = payload.get('mood', '')
        stress = payload.get('stress', '')
        notes = (payload.get('notes') or '').strip()
        pid_payload = payload.get('patient_id')
        if pid_payload is None:
            if pid_filter is None:
                return jsonify({'error': 'patient_id required'}), 400
            pid_payload = pid_filter
        try:
            pid_int = int(pid_payload)
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid patient_id'}), 400
        if not _patient_access_ok(pid_int):
            return jsonify({'error': 'Forbidden'}), 403
        user = session['user']

        header = ['user', 'patient_id', 'timestamp', 'mood', 'stress', 'notes']
        _append_csv_rows(CHECKIN_PATH, header, [[user, pid_int, ts, mood, stress, notes]])
        return jsonify({'status': 'ok'})

    entries = []
    if os.path.exists(CHECKIN_PATH):
        header = ['user', 'patient_id', 'timestamp', 'mood', 'stress', 'notes']
        _ensure_csv_columns(CHECKIN_PATH, header)
        with open(CHECKIN_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_pid = row.get('patient_id')
                if pid_filter is not None:
                    if not row_pid:
                        continue
                    try:
                        if int(row_pid) != pid_filter:
                            continue
                    except ValueError:
                        continue
                entries.append({
                    'user': row.get('user', ''),
                    'timestamp': row.get('timestamp', ''),
                    'mood': row.get('mood', ''),
                    'stress': row.get('stress', ''),
                    'notes': row.get('notes', '')
                })
    entries.sort(key=lambda e: e['timestamp'], reverse=True)
    return jsonify(entries)


def _load_goals() -> list[dict]:
    data = load_json_list(GOALS_FILE)
    return data if isinstance(data, list) else []


def _save_goals(items: list[dict]) -> None:
    save_json_list(GOALS_FILE, items)


@app.route('/api/goals', methods=['GET', 'POST'])
def goals_collection():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if request.method == 'POST':
        payload = request.get_json(force=True) or {}
        pid_raw = payload.get('patient_id') or request.args.get('patient_id')
    else:
        payload = {}
        pid_raw = request.args.get('patient_id')

    if pid_raw is None:
        return jsonify({'error': 'patient_id required'}), 400

    try:
        pid_filter = int(pid_raw)
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid patient_id'}), 400

    if not _patient_access_ok(pid_filter):
        return jsonify({'error': 'Forbidden'}), 403

    goals = _load_goals()

    if request.method == 'GET':
        results = [g for g in goals if int(g.get('patient_id', -1)) == pid_filter]
        results.sort(key=lambda g: g.get('created_at', ''), reverse=True)
        return jsonify(results)

    title = (payload.get('title') or '').strip()
    if not title:
        return jsonify({'error': 'title required'}), 400
    due_date_raw = (payload.get('due_date') or '').strip()
    due_date = due_date_raw or None
    notify = bool(payload.get('notify'))
    now_iso = datetime.utcnow().isoformat()
    goal = {
        'id': uuid4().hex,
        'patient_id': pid_filter,
        'title': title,
        'status': 'active',
        'due_date': due_date,
        'notify': notify,
        'created_at': now_iso,
        'updated_at': now_iso,
        'created_by': session.get('user'),
    }
    goals.append(goal)
    _save_goals(goals)
    return jsonify(goal), 201


@app.route('/api/goals/<goal_id>', methods=['PATCH', 'DELETE'])
def goal_detail(goal_id: str):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    goals = _load_goals()
    target = next((g for g in goals if g.get('id') == goal_id), None)
    if not target:
        return jsonify({'error': 'Not found'}), 404
    if not _patient_access_ok(target.get('patient_id')):
        return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'DELETE':
        goals = [g for g in goals if g.get('id') != goal_id]
        _save_goals(goals)
        return jsonify({'status': 'deleted'})

    payload = request.get_json(force=True) or {}
    allowed = {'status', 'title', 'due_date', 'notify'}
    updated = False
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key == 'status':
            if value not in ('active', 'completed'):
                continue
        if key == 'notify':
            value = bool(value)
        if key == 'due_date':
            value = (value or '').strip() or None
        if key == 'title':
            value = (value or '').strip()
            if not value:
                continue
        target[key] = value
        updated = True
    if updated:
        target['updated_at'] = datetime.utcnow().isoformat()
        _save_goals(goals)
    return jsonify(target)


# === New: Journal entries endpoint ===
@app.route('/api/journal_entries', methods=['GET', 'POST'])
def journal_entries():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    pid_raw = request.args.get('patient_id')
    pid_filter = None
    if pid_raw:
        try:
            pid_filter = int(pid_raw)
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid patient_id'}), 400
        if not _patient_access_ok(pid_filter):
            return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'POST':
        payload = request.get_json() or {}
        ts = payload.get('timestamp')
        text_value = (payload.get('text') or '').strip()
        mood = payload.get('mood', '')
        pid_payload = payload.get('patient_id')
        if pid_payload is None:
            if pid_filter is None:
                return jsonify({'error': 'patient_id required'}), 400
            pid_payload = pid_filter
        try:
            pid_int = int(pid_payload)
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid patient_id'}), 400
        if not _patient_access_ok(pid_int):
            return jsonify({'error': 'Forbidden'}), 403
        user = session['user']

        header = ['user', 'patient_id', 'timestamp', 'mood', 'text']
        _append_csv_rows(JOURNAL_PATH, header, [[user, pid_int, ts, mood, text_value]])
        return jsonify({'status': 'ok'})

    entries = []
    if os.path.exists(JOURNAL_PATH):
        header = ['user', 'patient_id', 'timestamp', 'mood', 'text']
        _ensure_csv_columns(JOURNAL_PATH, header)
        with open(JOURNAL_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_pid = row.get('patient_id')
                if pid_filter is not None:
                    if not row_pid:
                        continue
                    try:
                        if int(row_pid) != pid_filter:
                            continue
                    except ValueError:
                        continue
                entries.append({
                    'timestamp': row.get('timestamp', ''),
                    'mood': row.get('mood', ''),
                    'text': row.get('text', '')
                })
    # Sort newest first
    entries.sort(key=lambda e: e['timestamp'], reverse=True)
    return jsonify(entries)


def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to maintain backward compatibility; delegates to shared module."""
    df = engineer_features(df, drop_original_timestamp=False)
    return ensure_feature_order(df)


def _record_support_event(kind: str, details: Dict[str, Any]) -> None:
    payload = {
        'id': uuid4().hex,
        'kind': kind,
        'submitted_at': datetime.utcnow().isoformat() + 'Z',
        'details': {k: (v or '') for k, v in details.items()},
    }
    existing = load_json_list(SUPPORT_REQUESTS_FILE)
    existing.append(payload)
    # Keep file from growing unbounded
    if len(existing) > 500:
        existing = existing[-500:]
    save_json_list(SUPPORT_REQUESTS_FILE, existing)


def _request_data_as_dict() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    if not data and request.form:
        data = request.form.to_dict(flat=True)
    if not data and request.values:
        data = request.values.to_dict(flat=True)
    return data

@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/support/request-access', methods=['POST'])
def support_request_access():
    data = _request_data_as_dict()
    email = (data.get('email') or '').strip()
    name = (data.get('name') or '').strip()
    facility = (data.get('facility') or '').strip()
    role = (data.get('role') or '').strip()
    goal = (data.get('goal') or '').strip()
    message = (data.get('message') or '').strip()
    if not email:
        return jsonify({'error': 'email required'}), 400
    _record_support_event('request_access', {
        'email': email.lower(),
        'name': name,
        'facility': facility,
        'role': role,
        'goal': goal,
        'message': message,
        'user_agent': request.headers.get('User-Agent', ''),
    })
    return jsonify({'status': 'ok'})


@app.route('/support/forgot-password', methods=['POST'])
def support_forgot_password():
    data = _request_data_as_dict()
    email = (data.get('email') or '').strip()
    name = (data.get('name') or '').strip()
    contact = (data.get('contact') or '').strip()
    notes = (data.get('notes') or '').strip()
    if not email:
        return jsonify({'error': 'email required'}), 400
    _record_support_event('forgot_password', {
        'email': email.lower(),
        'name': name,
        'contact': contact,
        'notes': notes,
        'user_agent': request.headers.get('User-Agent', ''),
    })
    # Do not reveal whether email exists in system.
    return jsonify({'status': 'ok'})


@app.after_request
def set_security_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['X-XSS-Protection'] = '1; mode=block'
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','')
        password = request.form.get('password','')
        remember = request.form.get('remember')
        user = authenticate(email, password)
        if user:
            session['user'] = user.email
            session['role'] = user.role
            session.permanent = bool(remember)
            if user.facility_id is not None:
                session['facility_id'] = user.facility_id
            if user.assigned_patient_ids is not None:
                session['assigned_patient_ids'] = user.assigned_patient_ids
            session['display_name'] = user.name or user.email
            if getattr(user, 'avatar_url', None):
                session['avatar_url'] = user.avatar_url
            else:
                session.pop('avatar_url', None)
            return redirect(url_for('dashboard'))
        params = {'error': 1}
        if email:
            params['email'] = email
        return redirect(url_for('login', **params))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    show_switch = os.getenv('SHOW_ROLE_SWITCHER', '0') == '1'
    return render_template('dashboard.html', show_role_switcher=show_switch)


@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect(url_for('login'))
    role = session.get('role')
    if role not in ('super_admin', 'facility_admin'):
        return redirect(url_for('dashboard'))
    return render_template('admin.html')


@app.route('/api/me')
def me():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    email = session.get('user')
    role = session.get('role')
    facility_id = session.get('facility_id')
    assigned = session.get('assigned_patient_ids') or []
    name = session.get('display_name')
    avatar_path = session.get('avatar_url')
    # ensure consistent with user store in case session missing fields
    user_obj = get_user_by_email(email) if email else None
    if user_obj:
        role = role or user_obj.role
        facility_id = facility_id if facility_id is not None else user_obj.facility_id
        if not assigned:
            assigned = user_obj.assigned_patient_ids or []
        name = name or user_obj.name or email
        if not avatar_path:
            avatar_path = getattr(user_obj, 'avatar_url', None)
    avatar_url = url_for('static', filename=avatar_path) if avatar_path else None
    return jsonify({
        'email': email,
        'name': name,
        'role': role or 'staff',
        'facility_id': facility_id,
        'assigned_patient_ids': assigned,
        'avatar_url': avatar_url,
    })


def _patient_access_ok(pid: int) -> bool:
    role = session.get('role')
    if role == 'super_admin':
        return True
    try:
        pid_int = int(pid)
    except Exception:
        return False
    p = db_get_patient_by_id(pid_int)
    if not p:
        return False
    if role == 'facility_admin':
        return p.get('facility_id') == session.get('facility_id')
    if role == 'staff':
        assigned = session.get('assigned_patient_ids') or []
        assigned = set(int(x) for x in assigned if str(x).isdigit())
        return pid_int in assigned
    return False


@app.route('/api/patients', methods=['GET', 'POST'])
def patients_list():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    # Ensure tables exist for older DBs
    try:
        init_db()
    except Exception:
        pass
    role = session.get('role')
    if request.method == 'GET':
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and str(facility_id).isdigit() else None
            items = db_list_patients(fid)
        elif role == 'facility_admin':
            fid = session.get('facility_id')
            items = db_list_patients(fid)
        else:
            assigned = session.get('assigned_patient_ids') or []
            items = db_list_patients(None, [int(x) for x in assigned if str(x).isdigit()])
        return jsonify(items)
    # POST: create a new patient (admin only)
    if role not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    if role == 'super_admin':
        fid_raw = data.get('facility_id')
        fid = int(fid_raw) if isinstance(fid_raw, (int, str)) and str(fid_raw).isdigit() else None
    else:
        fid = session.get('facility_id')
    bed_id = (data.get('bed_id') or '').strip() or None
    age_raw = data.get('age')
    age_val = None
    if isinstance(age_raw, (int, float)):
        try:
            age_val = int(age_raw)
        except Exception:
            age_val = None
    elif isinstance(age_raw, str) and age_raw.strip().isdigit():
        age_val = int(age_raw.strip())
    risk_level = (data.get('risk_level') or '').strip() or None
    primary_condition = (data.get('primary_condition') or '').strip() or None
    allergies = (data.get('allergies') or '').strip() or None
    care_focus = (data.get('care_focus') or '').strip() or None
    avatar_url = (data.get('avatar_url') or '').strip() or None
    created = db_insert_patient(
        name,
        fid,
        bed_id,
        age=age_val,
        risk_level=risk_level,
        primary_condition=primary_condition,
        allergies=allergies,
        care_focus=care_focus,
        avatar_url=avatar_url
    )
    return jsonify(created), 201


@app.route('/api/patients/seed_demo', methods=['POST'])
def patients_seed_demo():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    # Create a handful of demo patients quickly
    fid = session.get('facility_id') if session.get('role') == 'facility_admin' else 1
    demos = [
        {
            'name': 'Ava Thompson',
            'bed_id': 'B-2',
            'age': 29,
            'risk_level': 'Low',
            'primary_condition': 'Sleep Irregularity',
            'allergies': 'None',
            'care_focus': 'Stabilize bedtime routine and log restfulness.'
        },
        {
            'name': 'Liam Nguyen',
            'bed_id': 'B-3',
            'age': 39,
            'risk_level': 'Moderate',
            'primary_condition': 'Chronic Pain',
            'allergies': 'NSAIDs',
            'care_focus': 'Track pain triggers and complete stretching blocks.'
        },
        {
            'name': 'Sophia Patel',
            'bed_id': 'B-4',
            'age': 52,
            'risk_level': 'Moderate',
            'primary_condition': 'Post-operative Recovery',
            'allergies': 'Latex',
            'care_focus': 'Encourage mobility and monitor incision healing.'
        },
        {
            'name': 'Noah Kim',
            'bed_id': 'B-5',
            'age': 45,
            'risk_level': 'Low',
            'primary_condition': 'Mild Anxiety',
            'allergies': 'None',
            'care_focus': 'Introduce breathing exercises and evening walks.'
        },
        {
            'name': 'Mia Garcia',
            'bed_id': 'B-6',
            'age': 61,
            'risk_level': 'High',
            'primary_condition': 'Hypertension',
            'allergies': 'ACE inhibitors',
            'care_focus': 'Monitor vitals twice daily and adjust sodium intake.'
        }
    ]
    created = []
    for entry in demos:
        try:
            patient = db_insert_patient(
                entry['name'],
                fid,
                entry.get('bed_id'),
                age=entry.get('age'),
                risk_level=entry.get('risk_level'),
                primary_condition=entry.get('primary_condition'),
                allergies=entry.get('allergies'),
                care_focus=entry.get('care_focus'),
            )
            created.append(patient)
        except Exception:
            continue
    # Seed synthetic readings for the new beds (so selector visibly changes data)
    try:
        for p in created:
            bed = p.get('bed_id')
            pid = p.get('id')
            if bed:
                _ensure_bed_readings(bed, pid, hours=48)
    except Exception:
        pass
    return jsonify({'created': created}), 201


def _ensure_bed_readings(bed_id: str, patient_id: int, hours: int = 24) -> None:
    """If a bed has no readings, synthesize some recent data for demo purposes."""
    try:
        conn = get_conn()
        cur = conn.execute('SELECT COUNT(1) FROM readings WHERE bed_id = ?', (bed_id,))
        cnt = int(cur.fetchone()[0])
        if cnt > 0:
            return
        base = dt.utcnow()
        rows = []
        for h in range(hours, -1, -1):
            t = base - timedelta(hours=h)
            # Simple synthetic variation
            temp = 70.0 + (h % 6) * 0.4
            hum = 42.0 + (h % 5) * 0.6
            hr = 72 + (h % 7)
            rr = 14 + (h % 3)
            bh = 300 + (h % 10) * 20
            radar = 1 if (h % 4) in (1,2) else 0
            ultra = 55 + (h % 8)
            mood = 3 + ((h // 6) % 3)  # 3..5
            song = 1 + (h % 2)
            rows.append((
                t.isoformat(), temp, hum, 'OK', bh, radar, ultra, mood, song, hr, rr, bed_id, patient_id
            ))
        conn.executemany(
            'INSERT INTO readings (timestamp, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, mood, song, heart_rate, respiration_rate, bed_id, patient_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        conn.commit()
    except Exception:
        return

def _load_sensor_dataframe(patient_id: int | None = None) -> pd.DataFrame:
    # Prefer DB readings; fallback to CSVs if empty
    try:
        conn = get_conn()
        where = ''
        params: list[Any] = []
        if patient_id is not None:
            p = db_get_patient_by_id(int(patient_id))
            bed = p.get('bed_id') if p else None
            if bed:
                where = 'WHERE bed_id = ?'
                params.append(bed)
        sql = f"""
            SELECT 
              timestamp as timestamp,
              temperature as Temperature,
              humidity as Humidity,
              mq2 as MQ2,
              bh1750fvi as BH1750FVI,
              radar as Radar,
              ultrasonic as Ultrasonic,
              mood as mood,
              song as song
            FROM readings
            {where}
        """
        df = pd.read_sql_query(sql, conn, params=params) if conn else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        csv_files = glob.glob(os.path.join('data', 'sensor_data_*.csv'))
        frames = []
        for f in csv_files:
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if patient_id is not None and not df.empty:
            df = _filter_df_by_patient(df, patient_id)
    # Normalization
    if 'MQ-2' in df.columns and 'MQ2' not in df.columns:
        df.rename(columns={'MQ-2': 'MQ2'}, inplace=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['date_only'] = df['timestamp'].dt.date
    for col in ['mood','song','Temperature','Humidity','BH1750FVI','Radar','Ultrasonic']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@lru_cache(maxsize=64)
def _cached_aggregates(cache_marker: int, patient_id: int | None) -> Dict[str, Any]:  # marker allows manual invalidation
    try:
        combined = _load_sensor_dataframe(patient_id)
    except TypeError:
        combined = _load_sensor_dataframe()
    if combined.empty:
        return {"error": "No data"}
    avg_temp = combined['Temperature'].mean(skipna=True) if 'Temperature' in combined else None
    avg_hum = combined['Humidity'].mean(skipna=True) if 'Humidity' in combined else None
    avg_mood = combined['mood'].mean(skipna=True) if 'mood' in combined else None

    if 'MQ2' in combined:
        ok = (combined['MQ2'] == 'OK').sum()
        nok = (combined['MQ2'] == 'N_OK').sum()
        current_air_quality = 'Poor' if nok > ok/2 else 'Good'
    else:
        current_air_quality = 'Unknown'

    if 'Radar' in combined:
        ra = combined['Radar'].fillna(0)
        movement_status = 'High' if ra.sum() > len(ra)/2 else 'Medium'
    else:
        movement_status = 'Unknown'

    # Trends
    if 'date_only' in combined and 'mood' in combined:
        mg = combined.groupby('date_only')['mood'].mean().reset_index(name='avg_mood')
    else:
        mg = pd.DataFrame(columns=['date_only','avg_mood'])
    mood_data = [
        {"date": str(r.date_only), "value": None if pd.isna(r.avg_mood) else r.avg_mood}
        for _, r in mg.iterrows()
    ]

    if 'date_only' in combined and 'Ultrasonic' in combined:
        vg = combined.groupby('date_only')['Ultrasonic'].mean().reset_index(name='avg_move')
    else:
        vg = pd.DataFrame(columns=['date_only','avg_move'])
    move_data = [
        {"date": str(r.date_only), "value": None if pd.isna(r.avg_move) else r.avg_move}
        for _, r in vg.iterrows()
    ]

    scores = _compute_health_scores(combined)
    avg_move = combined['Ultrasonic'].mean(skipna=True) if 'Ultrasonic' in combined else None
    avg_light = combined['BH1750FVI'].mean(skipna=True) if 'BH1750FVI' in combined else None
    summary_payload = {
        "avgTemp": round(avg_temp, 1) if avg_temp is not None and pd.notna(avg_temp) else None,
        "avgHumidity": round(avg_hum, 1) if avg_hum is not None and pd.notna(avg_hum) else None,
        "avgMood": round(avg_mood, 1) if avg_mood is not None and pd.notna(avg_mood) else None,
        "currentAirQuality": _score_to_quality(scores.get('light')),
        "movementStatus": _score_to_band(scores.get('movement')),
        "overallScore": round(scores.get('overallScore', 0.0), 1),
    }
    return {
        "summary": summary_payload,
        "charts": {"mood": mood_data, "movement": move_data},
        "donutSlices": scores,
    }


def _filter_df_by_patient(df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
    try:
        p = db_get_patient_by_id(int(patient_id))
    except Exception:
        p = None
    if p is None:
        return df
    bed_id = p.get('bed_id')
    name = p.get('name')
    for col in ['bed_id','BedID','bed']:
        if col in df.columns and bed_id:
            try:
                return df[df[col].astype(str) == str(bed_id)]
            except Exception:
                pass
    for col in ['patient','Patient','patient_name','PatientName']:
        if col in df.columns and isinstance(name, str) and name:
            try:
                return df[df[col].astype(str).str.strip().str.lower() == name.strip().lower()]
            except Exception:
                pass
    return df


@app.route('/api/dashboard_data')
def dashboard_data():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    pid_raw = request.args.get('patient_id')
    if pid_raw and not _patient_access_ok(pid_raw):
        return jsonify({'error': 'Forbidden'}), 403
    # Use current minute as cache marker for lightweight TTL
    marker = int(dt.utcnow().timestamp() // 60)
    try:
        pid = int(pid_raw) if pid_raw else None
    except Exception:
        pid = None
    data = _cached_aggregates(marker, pid)
    if 'error' in data:
        return jsonify(data), 404
    return jsonify(data)

@app.route('/api/latest_reading')
def latest_reading():
    if 'user' not in session:
        return jsonify({'error':'Unauthorized'}), 401
    pid_raw = request.args.get('patient_id')
    if pid_raw and not _patient_access_ok(pid_raw):
        return jsonify({'error': 'Forbidden'}), 403
    try:
        pid = int(pid_raw) if pid_raw else None
    except Exception:
        pid = None
    df = _load_sensor_dataframe(pid)
    if df is None or df.empty or 'timestamp' not in df.columns:
        return jsonify({'error': 'No recent readings found'}), 404
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return jsonify({'error': 'No recent readings found'}), 404
    last = df.sort_values('timestamp').iloc[-1]

    # Safely handle NaNs
    song_val = last.get('song', 0)
    if pd.isna(song_val):
        song_val = 0

    mq_val = last.get('MQ-2', last.get('MQ2', 'OK'))
    if pd.isna(mq_val):
        mq_val = 'OK'
    if not isinstance(mq_val, str):
        mq_val = str(mq_val)
    if not mq_val:
        mq_val = 'OK'

    def _safe_float(value, default=0.0):
        try:
            return float(value) if not pd.isna(value) else default
        except Exception:
            return default

    def _safe_int(value, default=0):
        try:
            return int(value) if not pd.isna(value) else default
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return default

    return jsonify({
        'timestamp': last['timestamp'].isoformat(),
        'Temperature': _safe_float(last.get('Temperature')),
        'Humidity': _safe_float(last.get('Humidity')),
        'MQ-2': mq_val,
        'BH1750FVI': _safe_float(last.get('BH1750FVI')),
        'Radar': _safe_int(last.get('Radar')),
        'Ultrasonic': _safe_float(last.get('Ultrasonic')),
        'song': _safe_int(song_val)
    })

def _validate_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ['timestamp','Temperature','Humidity','MQ-2','BH1750FVI','Radar','Ultrasonic','song']
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    # Basic type conversions
    try:
        payload['Temperature'] = float(payload['Temperature'])
        payload['Humidity'] = float(payload['Humidity'])
        payload['BH1750FVI'] = float(payload['BH1750FVI'])
        payload['Radar'] = int(payload['Radar'])
        payload['Ultrasonic'] = float(payload['Ultrasonic'])
        payload['song'] = int(payload['song'])
        # MQ-2 allow pass-through of 'OK'/'N_OK'
        pd.to_datetime(payload['timestamp'])  # validation only
    except Exception as e:
        raise ValueError(f"Invalid field values: {e}")
    return payload


@app.route('/api/predict_mood', methods=['POST'])
def predict_mood():
    if 'user' not in session:
        return jsonify({'error':'Unauthorized'}), 401
    pid_raw = request.args.get('patient_id')
    pid_val = None
    if pid_raw:
        if not _patient_access_ok(pid_raw):
            return jsonify({'error': 'Forbidden'}), 403
        try:
            pid_val = int(pid_raw)
        except Exception:
            pid_val = None
    payload = request.get_json() or {}
    try:
        clean = _validate_prediction_payload(payload)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    df = pd.DataFrame([clean], columns=[
        'timestamp','Temperature','Humidity','MQ-2','BH1750FVI','Radar','Ultrasonic','song'
    ])
    df = preprocess_for_prediction(df)
    Xnew = df[FEATURE_COLUMNS]
    pred = mood_model.predict(Xnew)[0]
    # Persist per-patient prediction history (if patient context provided)
    header = ['patient_id', 'timestamp', 'predicted_mood', 'Temperature', 'Humidity', 'MQ-2', 'BH1750FVI', 'Radar', 'Ultrasonic', 'song']
    ts_value = clean.get('timestamp') or datetime.utcnow().isoformat()
    row = [
        pid_val if pid_val is not None else '',
        ts_value,
        int(pred),
        clean.get('Temperature', ''),
        clean.get('Humidity', ''),
        clean.get('MQ-2', ''),
        clean.get('BH1750FVI', ''),
        clean.get('Radar', ''),
        clean.get('Ultrasonic', ''),
        clean.get('song', ''),
    ]
    try:
        _append_csv_rows(PREDICTIONS_PATH, header, [row])
    except Exception:
        app.logger.debug("Could not persist prediction row", exc_info=True)
    return jsonify({'predicted_mood': int(pred)})

"""
Facility management + alerting (RR<5) additions
"""

# --- Staff, beds, schedule APIs (DB-backed) ---
@app.route('/api/staff', methods=['GET', 'POST'])
def staff_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    # GET filtered by facility
    if request.method == 'GET':
        role = session.get('role')
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_staff(fid))
    # POST requires admin privileges
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip()
    phone = (data.get('phone') or '').strip()
    if not name or not phone:
        return jsonify({'error': 'name and phone required'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_staff(str(uuid4()), name, phone, fid)
    return jsonify(item), 201


@app.route('/api/staff/<staff_id>', methods=['DELETE'])
def staff_delete(staff_id: str):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    ok = db_delete_staff(staff_id)
    if not ok:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'status': 'deleted'})


@app.route('/api/beds', methods=['GET', 'POST'])
def beds_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if request.method == 'GET':
        role = session.get('role')
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_beds(fid))
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip()
    room = (data.get('room') or '').strip()
    patient = (data.get('patient') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_bed(str(uuid4()), name, room, patient, fid)
    return jsonify(item), 201


@app.route('/api/schedule', methods=['GET', 'POST'])
def schedule_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if request.method == 'GET':
        role = session.get('role')
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_schedule(fid))
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip()
    start = (data.get('start') or '').strip()  # HH:MM 24h
    end = (data.get('end') or '').strip()
    days = data.get('days') or []              # [0-6] Mon=0
    staff_ids = data.get('staff_ids') or []
    if not name or not start or not end or not isinstance(days, list):
        return jsonify({'error': 'name, start, end, days required'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_schedule(str(uuid4()), name, start, end, days, staff_ids, fid)
    return jsonify(item), 201


def _hhmm_to_minutes(s: str) -> int:
    try:
        h, m = s.split(':')
        return int(h) * 60 + int(m)
    except Exception:
        return -1


def _on_duty_phone_numbers() -> list[str]:
    from datetime import datetime as _dt
    now = _dt.utcnow()
    day = now.weekday()  # 0=Mon
    minutes = now.hour * 60 + now.minute
    sch = db_list_schedule(None)
    staff = db_list_staff(None)
    id_to_phone = {s.get('id'): s.get('phone') for s in staff}
    recipients: set[str] = set()
    for sh in sch:
        if day not in (sh.get('days') or []):
            continue
        start = _hhmm_to_minutes(sh.get('start', ''))
        end = _hhmm_to_minutes(sh.get('end', ''))
        if start < 0 or end < 0:
            continue
        if start <= end:
            in_window = (start <= minutes < end)
        else:
            in_window = (minutes >= start or minutes < end)
        if in_window:
            for sid in sh.get('staff_ids') or []:
                phone = id_to_phone.get(sid)
                if phone:
                    recipients.add(phone)
    if not recipients:
        recipients = set(p for p in id_to_phone.values() if p)
    override = os.getenv('ALERT_TO', '').strip()
    if override:
        return [override]
    return sorted(recipients)

# ---- User management API ----
def _require_admin():
    if 'user' not in session:
        return 'Unauthorized', 401
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return 'Forbidden', 403
    return None


@app.route('/api/users', methods=['GET', 'POST'])
def users_api():
    chk = _require_admin()
    if chk:
        msg, code = chk
        return jsonify({'error': msg}), code
    role = session.get('role')
    if request.method == 'GET':
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        items = db_list_users(fid)
        for x in items:
            x.pop('password_hash', None)
            x['id'] = x.get('email')
        return jsonify(items)
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    raw_pw = (data.get('password') or '').strip()
    role_new = (data.get('role') or 'staff').strip()
    name = (data.get('name') or '').strip() or None
    if role == 'super_admin':
        fid_raw = data.get('facility_id')
        fid = int(fid_raw) if isinstance(fid_raw, (int, str)) and str(fid_raw).isdigit() else None
    else:
        fid = session.get('facility_id')
        if role_new == 'super_admin':
            return jsonify({'error': 'facility_admin cannot create super_admin'}), 403
    assigned = data.get('assigned_patient_ids') or []
    if not email or not raw_pw:
        return jsonify({'error': 'email and password are required'}), 400
    from werkzeug.security import generate_password_hash
    phash = generate_password_hash(raw_pw)
    try:
        created = db_insert_user(email, phash, role_new, name, fid, assigned)
    except Exception as e:
        return jsonify({'error': f'create failed: {e}'}), 400
    created.pop('password_hash', None)
    created['id'] = created.get('email')
    return jsonify(created), 201


@app.route('/api/users/<email>', methods=['PATCH', 'DELETE'])
def user_item(email: str):
    chk = _require_admin()
    if chk:
        msg, code = chk
        return jsonify({'error': msg}), code
    role = session.get('role')
    if request.method == 'DELETE':
        target_list = [x for x in db_list_users(None) if x.get('email','').lower() == email.lower()]
        target = target_list[0] if target_list else None
        if not target:
            return jsonify({'error': 'Not found'}), 404
        if role == 'facility_admin':
            if target.get('role') == 'super_admin' or target.get('facility_id') != session.get('facility_id'):
                return jsonify({'error': 'Forbidden'}), 403
        ok = db_delete_user_by_email(email)
        return jsonify({'status': 'deleted' if ok else 'not_found'})
    # PATCH
    target_list = [x for x in db_list_users(None) if x.get('email','').lower() == email.lower()]
    target = target_list[0] if target_list else None
    if not target:
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json(force=True) or {}
    updates = {}
    if 'role' in data:
        if role == 'facility_admin' and data['role'] == 'super_admin':
            return jsonify({'error': 'Forbidden'}), 403
        updates['role'] = data['role']
    if 'name' in data:
        updates['name'] = (data.get('name') or '').strip()
    if 'facility_id' in data:
        if role == 'facility_admin':
            updates['facility_id'] = session.get('facility_id')
        else:
            maybe = data.get('facility_id')
            if isinstance(maybe, (int, str)) and str(maybe).isdigit():
                updates['facility_id'] = int(maybe)
            else:
                updates['facility_id'] = None
    if 'assigned_patient_ids' in data and isinstance(data.get('assigned_patient_ids'), list):
        updates['assigned_patient_ids'] = data.get('assigned_patient_ids')
    if 'password' in data and data.get('password'):
        from werkzeug.security import generate_password_hash
        updates['password_hash'] = generate_password_hash(data['password'])
    updated = db_update_user(email, updates)
    updated.pop('password_hash', None)
    updated['id'] = updated.get('email')
    return jsonify(updated)


@app.route('/api/profile/avatar', methods=['POST'])
def profile_avatar():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    upload = request.files.get('avatar')
    if upload is None or not upload.filename:
        return jsonify({'error': 'avatar file required'}), 400
    filename = secure_filename(upload.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AVATAR_EXTENSIONS:
        return jsonify({'error': 'unsupported file type'}), 400
    os.makedirs(AVATAR_UPLOAD_DIR, exist_ok=True)
    final_name = f"{uuid4().hex}{ext}"
    dest_path = os.path.join(AVATAR_UPLOAD_DIR, final_name)
    upload.save(dest_path)
    rel_path = os.path.join(AVATAR_UPLOAD_SUBDIR, final_name).replace('\\', '/')
    user = get_user_by_email(session.get('user'))
    if user and getattr(user, 'avatar_url', None):
        old_rel = user.avatar_url
        if old_rel and old_rel.startswith(AVATAR_UPLOAD_SUBDIR):
            old_abs = os.path.join(app.static_folder, old_rel.replace('/', os.sep))
            if os.path.isfile(old_abs):
                try:
                    os.remove(old_abs)
                except OSError:
                    pass
    db_update_user(session.get('user'), {'avatar_url': rel_path})
    session['avatar_url'] = rel_path
    return jsonify({'avatar_url': url_for('static', filename=rel_path)})


def _get_latest_rr_hr():
    try:
        csv_files = glob.glob(os.path.join('data', 'sensor_data_*.csv'))
        if not csv_files:
            return None, None, None
        frames = [pd.read_csv(fp) for fp in csv_files]
        df = pd.concat(frames, ignore_index=True)
        cols = {c.lower(): c for c in df.columns}
        rr_col = cols.get('rr') or cols.get('respiration') or cols.get('respiration_rate')
        hr_col = cols.get('hr') or cols.get('heart_rate')
        ts_col = cols.get('timestamp')
        if not rr_col and not hr_col:
            return None, None, None
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            df = df.sort_values(ts_col)
        last = df.dropna(subset=[c for c in [rr_col, hr_col] if c]).iloc[-1]
        rr = float(last[rr_col]) if rr_col and pd.notna(last[rr_col]) else None
        hr = float(last[hr_col]) if hr_col and pd.notna(last[hr_col]) else None
        ts = last[ts_col] if ts_col else None
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        return rr, hr, ts
    except Exception:
        return None, None, None


def _send_sms(to_number: str, body: str) -> bool:
    sid = os.getenv('TWILIO_ACCOUNT_SID', '').strip()
    tok = os.getenv('TWILIO_AUTH_TOKEN', '').strip()
    from_num = os.getenv('TWILIO_FROM_NUMBER', '').strip()
    if not sid or not tok or not from_num:
        app.logger.info(f"[SMS MOCK] To {to_number}: {body}")
        return True
    try:
        from twilio.rest import Client as _Twilio
        client = _Twilio(sid, tok)
        client.messages.create(to=to_number, from_=from_num, body=body)
        app.logger.info(f"SMS sent to {to_number}")
        return True
    except Exception as e:
        app.logger.error(f"Twilio error: {e}")
        return False


_last_alert_epoch = 0.0


def _alert_monitor_loop():
    import time
    global _last_alert_epoch
    cooldown = int(os.getenv('ALERT_COOLDOWN_SEC', '600'))
    rr_thresh = float(os.getenv('ALERT_RR_THRESHOLD', '5'))
    while True:
        try:
            rr, hr, ts = _get_latest_rr_hr()
            if rr is not None and rr < rr_thresh:
                now = time.time()
                if now - _last_alert_epoch >= cooldown:
                    for p in _on_duty_phone_numbers():
                        when = ts.isoformat() if hasattr(ts, 'isoformat') else 'latest reading'
                        _send_sms(p, f"ALERT: Low respiration detected (RR={rr:.1f}/min) at {when}.")
                    _last_alert_epoch = now
        except Exception as e:
            app.logger.error(f"Alert monitor error: {e}")
        time.sleep(60)


@app.route('/api/alerts/test', methods=['POST'])
def alerts_test():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    payload = request.get_json(silent=True) or {}
    targets = payload.get('to')
    if not targets:
        targets = _on_duty_phone_numbers()
    ok = True
    for p in targets:
        ok = _send_sms(p, 'Test alert from non-contact monitor.') and ok
    return jsonify({'status': 'ok' if ok else 'partial', 'recipients': targets})


def start_background_tasks():
    import threading
    if app.config.get('TESTING'):
        return
    if os.getenv('DISABLE_ALERT_MONITOR', '0') == '1':
        return
    if app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return
    t = threading.Thread(target=_alert_monitor_loop, name='alert-monitor', daemon=True)
    t.start()


if __name__ == '__main__':
    start_background_tasks()
    app.run(host='0.0.0.0', debug=True)

