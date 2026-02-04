from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import glob
import os
import re
import time
import joblib
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
try:
    import openai
except ImportError:
    openai = None
from uuid import uuid4
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from collections import deque

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
    db_get_bed,
    db_delete_bed,
    db_list_schedule,
    db_insert_schedule,
    db_get_schedule,
    db_delete_schedule,
    db_list_patients,
    db_get_patient_by_id,
    db_insert_patient,
    db_list_checkins,
    db_insert_checkin,
    db_list_journal_entries,
    db_insert_journal_entry,
    db_list_goals,
    db_get_goal,
    db_insert_goal,
    db_update_goal,
    db_delete_goal,
    db_insert_prediction,
    db_insert_support_request,
    db_insert_audit_event,
    get_conn,
)
from functools import lru_cache
from datetime import datetime as dt
from typing import Any, Dict



load_dotenv()

app = Flask(__name__)
raw_secret = os.getenv('FLASK_SECRET_KEY')
env_name = (os.getenv('FLASK_ENV') or os.getenv('APP_ENV') or os.getenv('ENV') or '').lower()
if not raw_secret:
    if env_name in ('production', 'prod'):
        raise RuntimeError('FLASK_SECRET_KEY must be set in production.')
    raw_secret = uuid4().hex
    logging.warning('FLASK_SECRET_KEY not set; using ephemeral key. Set FLASK_SECRET_KEY to persist sessions.')
app.secret_key = raw_secret
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_MB', '5')) * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = env_name in ('production', 'prod')
remember_days = os.getenv('SESSION_REMEMBER_DAYS', '14')
try:
    remember_days_int = int(remember_days)
except ValueError:
    remember_days_int = 14
app.permanent_session_lifetime = timedelta(days=max(1, remember_days_int))
init_db()

# NOTE: Demo credentials are stored hashed in users.py (environment overrideable)

def _get_csrf_token() -> str:
    token = session.get('csrf_token')
    if not token:
        token = uuid4().hex
        session['csrf_token'] = token
    return token


_RATE_LIMITS: dict[str, deque[float]] = {}


def _rate_limited(key: str, max_requests: int, window_seconds: int) -> bool:
    now = time.time()
    bucket = _RATE_LIMITS.setdefault(key, deque())
    while bucket and (now - bucket[0]) > window_seconds:
        bucket.popleft()
    if len(bucket) >= max_requests:
        return True
    bucket.append(now)
    return False


def _audit_event(action: str, target: str | None = None, details: str | None = None) -> None:
    if app.config.get('TESTING'):
        return
    try:
        db_insert_audit_event(
            user=session.get('user'),
            action=action,
            target=target,
            details=details,
        )
    except Exception:
        return


@app.before_request
def csrf_protect():
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and request.path.startswith('/api/'):
        if app.config.get('TESTING'):
            return None
        if 'user' not in session:
            return None
        token = request.headers.get('X-CSRF-Token', '')
        if not token or token != session.get('csrf_token'):
            return jsonify({'error': 'CSRF token missing or invalid'}), 403


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_err):
    return jsonify({'error': 'file too large'}), 413

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


AVATAR_UPLOAD_SUBDIR = 'uploads'
AVATAR_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', AVATAR_UPLOAD_SUBDIR)
ALLOWED_AVATAR_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
MAX_CHECKIN_NOTES_LEN = 500
MAX_JOURNAL_TEXT_LEN = 2000
MAX_GOAL_TITLE_LEN = 120
MAX_SUPPORT_TEXT_LEN = 2000


def _detect_image_extension(upload) -> str | None:
    header = upload.stream.read(512)
    upload.stream.seek(0)
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if header.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return '.gif'
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return '.webp'
    return None


def _extension_matches_detected(ext: str, detected: str | None) -> bool:
    if not detected:
        return False
    if detected == '.jpg':
        return ext in ('.jpg', '.jpeg')
    return ext == detected


def _parse_iso_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith('Z'):
            raw = raw[:-1] + '+00:00'
        parsed = dt.fromisoformat(raw)
        return parsed.isoformat()
    except Exception:
        return None


def _valid_email(email: str) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email or ''))


def _coerce_int(value, *, low: int | None = None, high: int | None = None) -> int | None:
    try:
        if value is None or value == '':
            return None
        num = int(value)
    except Exception:
        return None
    if low is not None and num < low:
        return None
    if high is not None and num > high:
        return None
    return num


def _coerce_float(value, *, low: float | None = None, high: float | None = None) -> float | None:
    try:
        if value is None or value == '':
            return None
        num = float(value)
    except Exception:
        return None
    if low is not None and num < low:
        return None
    if high is not None and num > high:
        return None
    return num


def _clean_text(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        return None
    return cleaned


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
    journal_existing = set()
    checkin_existing = set()
    for entry in patient_summaries:
        pid = entry.get('id')
        if pid is None:
            continue
        try:
            if db_list_journal_entries(pid, limit=1):
                journal_existing.add(pid)
        except Exception:
            pass
        try:
            if db_list_checkins(pid, limit=1):
                checkin_existing.add(pid)
        except Exception:
            pass
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
    stress_labels = ['low', 'moderate', 'high']
    journal_rows = []
    checkin_rows = []
    for idx, info in enumerate(patient_summaries):
        pid = info['id']
        name = info.get('name', 'Patient')
        timestamp = (now - timedelta(hours=idx * 6)).replace(microsecond=0).isoformat()
        if pid not in journal_existing:
            mood_value = 3 + (idx % 3)
            note = f"{name} {phrases[idx % len(phrases)]}"
            journal_rows.append((base_user, pid, timestamp, mood_value, note))
        if pid not in checkin_existing:
            mood_value = 3 + (idx % 2)
            stress_value = stress_labels[idx % len(stress_labels)]
            checkin_rows.append((base_user, pid, timestamp, mood_value, stress_value))
    for row in journal_rows:
        db_insert_journal_entry(row[0], row[1], row[2], row[3], row[4])
    for row in checkin_rows:
        db_insert_checkin(row[0], row[1], row[2], row[3], row[4], None)
_seed_demo_logs_if_needed()











# Configure logging so you can see the exception
logging.basicConfig(level=logging.INFO)

# Pull from .env or fall back to your literal (if you really must)
raw_key = os.getenv("OPENAI_API_KEY", "")  # Do not fallback to hardcoded key in production.

# Replace any non-ascii hyphens with ASCII hyphens:
key = raw_key.replace("\u2011", "-")

# Optional extra cleanup if you suspect any other invisible chars:
key = "".join(ch for ch in key if ord(ch) < 128)

if openai is not None:
    openai.api_key = key if key else None


@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify(error="Unauthorized"), 401
    if openai is None or not openai.api_key:
        return jsonify(error="Chat service unavailable"), 503

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


def _build_weekly_insights_payload(pid: int | None) -> str | None:
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
        return None

    for col in ['mood','Ultrasonic','Temperature','BH1750FVI']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    avg_mood = df['mood'].mean() if 'mood' in df else None
    avg_move = df['Ultrasonic'].mean() if 'Ultrasonic' in df else None
    avg_temp = df['Temperature'].mean() if 'Temperature' in df else None
    avg_light = df['BH1750FVI'].mean() if 'BH1750FVI' in df else None
    mood_series = df['mood'].dropna() if 'mood' in df else pd.Series(dtype=float)
    max_row = df.loc[mood_series.idxmax()] if not mood_series.empty else None
    min_row = df.loc[mood_series.idxmin()] if not mood_series.empty else None

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

    return " ".join(parts) if parts else None







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
    if pid is not None and _patient_access_ok(pid):
        _audit_event('view_weekly_insights', str(pid))

    narrative = _build_weekly_insights_payload(pid) or "No data available for the past week."

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
        ts = _parse_iso_timestamp(payload.get('timestamp')) or datetime.utcnow().isoformat()
        mood = _coerce_int(payload.get('mood'), low=1, high=6)
        if mood is None:
            return jsonify({'error': 'mood must be 1-6'}), 400
        stress_raw = (payload.get('stress') or '').strip().lower()
        stress_allowed = {'', 'low', 'moderate', 'high'}
        if stress_raw not in stress_allowed:
            return jsonify({'error': 'invalid stress value'}), 400
        notes_raw = payload.get('notes')
        notes = _clean_text(notes_raw, MAX_CHECKIN_NOTES_LEN)
        if notes_raw and notes is None:
            return jsonify({'error': 'notes too long'}), 400
        db_insert_checkin(
            user=session['user'],
            patient_id=pid_int,
            timestamp=ts,
            mood=mood,
            stress=stress_raw or None,
            notes=notes,
        )
        _audit_event('create_checkin', str(pid_int))
        return jsonify({'status': 'ok'})

    if pid_filter is None:
        return jsonify([])
    _audit_event('view_checkins', str(pid_filter))
    entries = db_list_checkins(pid_filter, limit=50)
    return jsonify(entries)


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

    if request.method == 'GET':
        _audit_event('view_goals', str(pid_filter))
        return jsonify(db_list_goals(pid_filter))

    title = _clean_text(payload.get('title'), MAX_GOAL_TITLE_LEN)
    if not title:
        return jsonify({'error': 'title required'}), 400
    due_date_raw = (payload.get('due_date') or '').strip()
    due_date = _parse_iso_timestamp(due_date_raw) if due_date_raw else None
    if due_date_raw and not due_date:
        return jsonify({'error': 'invalid due_date'}), 400
    notify = 1 if payload.get('notify') else 0
    now_iso = datetime.utcnow().isoformat()
    created = db_insert_goal(
        goal_id=uuid4().hex,
        patient_id=pid_filter,
        title=title,
        status='active',
        due_date=due_date,
        notify=notify,
        created_at=now_iso,
        updated_at=now_iso,
        created_by=session.get('user'),
    )
    _audit_event('create_goal', str(pid_filter))
    return jsonify(created), 201


@app.route('/api/goals/<goal_id>', methods=['PATCH', 'DELETE'])
def goal_detail(goal_id: str):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    target = db_get_goal(goal_id)
    if not target:
        return jsonify({'error': 'Not found'}), 404
    if not _patient_access_ok(target.get('patient_id')):
        return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'DELETE':
        db_delete_goal(goal_id)
        _audit_event('delete_goal', str(target.get('patient_id')))
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
            value = 1 if value else 0
        if key == 'due_date':
            raw = (value or '').strip()
            value = _parse_iso_timestamp(raw) if raw else None
            if raw and not value:
                return jsonify({'error': 'invalid due_date'}), 400
        if key == 'title':
            value = _clean_text(value, MAX_GOAL_TITLE_LEN)
            if not value:
                continue
        target[key] = value
        updated = True
    if updated:
        target['updated_at'] = datetime.utcnow().isoformat()
        saved = db_update_goal(goal_id, target)
        _audit_event('update_goal', str(target.get('patient_id')))
        return jsonify(saved or target)
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
        ts = _parse_iso_timestamp(payload.get('timestamp')) or datetime.utcnow().isoformat()
        text_value = _clean_text(payload.get('text'), MAX_JOURNAL_TEXT_LEN)
        if not text_value:
            return jsonify({'error': 'text required'}), 400
        mood = _coerce_int(payload.get('mood'), low=1, high=6)
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
        db_insert_journal_entry(user=user, patient_id=pid_int, timestamp=ts, mood=mood, text=text_value)
        _audit_event('create_journal_entry', str(pid_int))
        return jsonify({'status': 'ok'})

    if pid_filter is None:
        return jsonify([])
    _audit_event('view_journal_entries', str(pid_filter))
    entries = db_list_journal_entries(pid_filter, limit=50)
    return jsonify(entries)


def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to maintain backward compatibility; delegates to shared module."""
    df = engineer_features(df, drop_original_timestamp=False)
    return ensure_feature_order(df)


def _record_support_event(kind: str, details: Dict[str, Any]) -> None:
    submitted_at = datetime.utcnow().isoformat()
    db_insert_support_request(
        kind=kind,
        email=(details.get('email') or '').strip().lower(),
        name=_clean_text(details.get('name'), MAX_SUPPORT_TEXT_LEN),
        facility=_clean_text(details.get('facility'), MAX_SUPPORT_TEXT_LEN),
        role=_clean_text(details.get('role'), MAX_SUPPORT_TEXT_LEN),
        goal=_clean_text(details.get('goal'), MAX_SUPPORT_TEXT_LEN),
        message=_clean_text(details.get('message'), MAX_SUPPORT_TEXT_LEN),
        contact=_clean_text(details.get('contact'), MAX_SUPPORT_TEXT_LEN),
        notes=_clean_text(details.get('notes'), MAX_SUPPORT_TEXT_LEN),
        user_agent=_clean_text(details.get('user_agent'), MAX_SUPPORT_TEXT_LEN),
        submitted_at=submitted_at,
    )


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
    if _rate_limited(f'support:{request.remote_addr}', 6, 300):
        return jsonify({'error': 'Too many requests'}), 429
    data = _request_data_as_dict()
    email = (data.get('email') or '').strip().lower()
    if not email or not _valid_email(email):
        return jsonify({'error': 'valid email required'}), 400
    name = _clean_text(data.get('name'), MAX_SUPPORT_TEXT_LEN)
    facility = _clean_text(data.get('facility'), MAX_SUPPORT_TEXT_LEN)
    role = _clean_text(data.get('role'), MAX_SUPPORT_TEXT_LEN)
    goal = _clean_text(data.get('goal'), MAX_SUPPORT_TEXT_LEN)
    message = _clean_text(data.get('message'), MAX_SUPPORT_TEXT_LEN)
    _record_support_event('request_access', {
        'email': email,
        'name': name or '',
        'facility': facility or '',
        'role': role or '',
        'goal': goal or '',
        'message': message or '',
        'user_agent': request.headers.get('User-Agent', ''),
    })
    return jsonify({'status': 'ok'})


@app.route('/support/forgot-password', methods=['POST'])
def support_forgot_password():
    if _rate_limited(f'support:{request.remote_addr}', 6, 300):
        return jsonify({'error': 'Too many requests'}), 429
    data = _request_data_as_dict()
    email = (data.get('email') or '').strip().lower()
    if not email or not _valid_email(email):
        return jsonify({'error': 'valid email required'}), 400
    name = _clean_text(data.get('name'), MAX_SUPPORT_TEXT_LEN)
    contact = _clean_text(data.get('contact'), MAX_SUPPORT_TEXT_LEN)
    notes = _clean_text(data.get('notes'), MAX_SUPPORT_TEXT_LEN)
    _record_support_event('forgot_password', {
        'email': email,
        'name': name or '',
        'contact': contact or '',
        'notes': notes or '',
        'user_agent': request.headers.get('User-Agent', ''),
    })
    # Do not reveal whether email exists in system.
    return jsonify({'status': 'ok'})


@app.after_request
def set_security_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['X-XSS-Protection'] = '1; mode=block'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=(), interest-cohort=()'
    resp.headers['Cache-Control'] = 'no-store'
    if env_name in ('production', 'prod'):
        resp.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    resp.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com data:; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'"
    )
    return resp


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if _rate_limited(f'login:{request.remote_addr}', 10, 300):
            return render_template('login.html', error=1)
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
            _get_csrf_token()
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
        'csrf_token': _get_csrf_token(),
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
    name = _clean_text(data.get('name'), 120)
    if not name:
        return jsonify({'error': 'name required'}), 400
    if role == 'super_admin':
        fid_raw = data.get('facility_id')
        fid = int(fid_raw) if isinstance(fid_raw, (int, str)) and str(fid_raw).isdigit() else None
    else:
        fid = session.get('facility_id')
    bed_id = _clean_text(data.get('bed_id'), 40)
    age_val = _coerce_int(data.get('age'), low=0, high=130)
    risk_level = _clean_text(data.get('risk_level'), 32)
    primary_condition = _clean_text(data.get('primary_condition'), 120)
    allergies = _clean_text(data.get('allergies'), 120)
    care_focus = _clean_text(data.get('care_focus'), 200)
    avatar_url = _clean_text(data.get('avatar_url'), 250)
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
    _audit_event('create_patient', str(created.get('id')))
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
    if pid is not None:
        _audit_event('view_dashboard_data', str(pid))
    data = _cached_aggregates(marker, pid)
    if 'error' in data:
        return jsonify(data), 404
    return jsonify(data)


def _latest_reading_payload(pid: int | None) -> Dict[str, Any] | None:
    df = _load_sensor_dataframe(pid)
    if df is None or df.empty or 'timestamp' not in df.columns:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if df.empty:
        return None
    last = df.sort_values('timestamp').iloc[-1]

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

    return {
        'timestamp': last['timestamp'].isoformat(),
        'Temperature': _safe_float(last.get('Temperature')),
        'Humidity': _safe_float(last.get('Humidity')),
        'MQ-2': mq_val,
        'BH1750FVI': _safe_float(last.get('BH1750FVI')),
        'Radar': _safe_int(last.get('Radar')),
        'Ultrasonic': _safe_float(last.get('Ultrasonic')),
        'song': _safe_int(song_val)
    }

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
    if pid is not None:
        _audit_event('view_latest_reading', str(pid))
    payload = _latest_reading_payload(pid)
    if payload is None:
        return jsonify({'error': 'No recent readings found'}), 404
    return jsonify(payload)

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


def _predict_mood_from_payload(payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    clean = _validate_prediction_payload(payload)
    df = pd.DataFrame([clean], columns=[
        'timestamp','Temperature','Humidity','MQ-2','BH1750FVI','Radar','Ultrasonic','song'
    ])
    df = preprocess_for_prediction(df)
    Xnew = df[FEATURE_COLUMNS]
    pred = mood_model.predict(Xnew)[0]
    return int(pred), clean


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
        pred, clean = _predict_mood_from_payload(payload)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    # Persist per-patient prediction history (if patient context provided)
    ts_value = clean.get('timestamp') or datetime.utcnow().isoformat()
    try:
        db_insert_prediction(
            patient_id=pid_val,
            timestamp=ts_value,
            predicted_mood=int(pred),
            temperature=clean.get('Temperature'),
            humidity=clean.get('Humidity'),
            mq2=clean.get('MQ-2'),
            bh1750fvi=clean.get('BH1750FVI'),
            radar=clean.get('Radar'),
            ultrasonic=clean.get('Ultrasonic'),
            song=clean.get('song'),
        )
    except Exception:
        app.logger.debug("Could not persist prediction row", exc_info=True)
    return jsonify({'predicted_mood': int(pred)})


@app.route('/api/patient_bundle')
def patient_bundle():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    pid_raw = request.args.get('patient_id')
    if not pid_raw:
        return jsonify({'error': 'patient_id required'}), 400
    try:
        pid = int(pid_raw)
    except Exception:
        return jsonify({'error': 'invalid patient_id'}), 400
    if not _patient_access_ok(pid):
        return jsonify({'error': 'Forbidden'}), 403
    marker = int(dt.utcnow().timestamp() // 60)
    dashboard = _cached_aggregates(marker, pid)
    weekly = _build_weekly_insights_payload(pid)
    latest = _latest_reading_payload(pid)
    predicted_mood = None
    if latest:
        try:
            pred, clean = _predict_mood_from_payload(latest.copy())
            predicted_mood = pred
            ts_value = clean.get('timestamp') or datetime.utcnow().isoformat()
            db_insert_prediction(
                patient_id=pid,
                timestamp=ts_value,
                predicted_mood=predicted_mood,
                temperature=clean.get('Temperature'),
                humidity=clean.get('Humidity'),
                mq2=clean.get('MQ-2'),
                bh1750fvi=clean.get('BH1750FVI'),
                radar=clean.get('Radar'),
                ultrasonic=clean.get('Ultrasonic'),
                song=clean.get('song'),
            )
        except Exception:
            predicted_mood = None
    payload = {
        'dashboard': None if 'error' in dashboard else dashboard,
        'dashboard_error': dashboard.get('error') if isinstance(dashboard, dict) and 'error' in dashboard else None,
        'weekly_insights': weekly,
        'latest_reading': latest,
        'predicted_mood': predicted_mood,
        'checkins': db_list_checkins(pid, limit=25),
        'goals': db_list_goals(pid),
        'journal_entries': db_list_journal_entries(pid, limit=10),
    }
    _audit_event('view_patient_bundle', str(pid))
    return jsonify(payload)

"""
Facility management + alerting (RR<5) additions
"""

# --- Staff, beds, schedule APIs (DB-backed) ---
@app.route('/api/staff', methods=['GET', 'POST'])
def staff_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    role = session.get('role')
    # GET filtered by facility
    if request.method == 'GET':
        if role not in ('super_admin', 'facility_admin'):
            return jsonify({'error': 'Forbidden'}), 403
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_staff(fid))
    # POST requires admin privileges
    if role not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = _clean_text(data.get('name'), 80)
    phone = _clean_text(data.get('phone'), 32)
    if not name or not phone:
        return jsonify({'error': 'name and phone required'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_staff(str(uuid4()), name, phone, fid)
    _audit_event('create_staff', item.get('id'))
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
    _audit_event('delete_staff', staff_id)
    return jsonify({'status': 'deleted'})


@app.route('/api/beds', methods=['GET', 'POST'])
def beds_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    role = session.get('role')
    if request.method == 'GET':
        if role not in ('super_admin', 'facility_admin'):
            return jsonify({'error': 'Forbidden'}), 403
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_beds(fid))
    if role not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = _clean_text(data.get('name'), 80)
    room = _clean_text(data.get('room'), 40) or ''
    patient = _clean_text(data.get('patient'), 80) or ''
    if not name:
        return jsonify({'error': 'name required'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_bed(str(uuid4()), name, room, patient, fid)
    _audit_event('create_bed', item.get('id'))
    return jsonify(item), 201


@app.route('/api/beds/<bed_id>', methods=['DELETE'])
def bed_delete(bed_id: str):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    target = db_get_bed(bed_id)
    if not target:
        return jsonify({'error': 'Not found'}), 404
    if session.get('role') == 'facility_admin':
        if target.get('facility_id') != session.get('facility_id'):
            return jsonify({'error': 'Forbidden'}), 403
    ok = db_delete_bed(bed_id)
    if ok:
        _audit_event('delete_bed', bed_id)
    return jsonify({'status': 'deleted' if ok else 'not_found'})


@app.route('/api/schedule', methods=['GET', 'POST'])
def schedule_api():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    role = session.get('role')
    if request.method == 'GET':
        if role not in ('super_admin', 'facility_admin'):
            return jsonify({'error': 'Forbidden'}), 403
        if role == 'super_admin':
            facility_id = request.args.get('facility_id')
            fid = int(facility_id) if facility_id and facility_id.isdigit() else None
        else:
            fid = session.get('facility_id')
        return jsonify(db_list_schedule(fid))
    if role not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    data = request.get_json(force=True) or {}
    name = _clean_text(data.get('name'), 80)
    start = (data.get('start') or '').strip()  # HH:MM 24h
    end = (data.get('end') or '').strip()
    days = data.get('days') or []              # [0-6] Mon=0
    staff_ids = data.get('staff_ids') or []
    if not name or not start or not end or not isinstance(days, list):
        return jsonify({'error': 'name, start, end, days required'}), 400
    if _hhmm_to_minutes(start) < 0 or _hhmm_to_minutes(end) < 0:
        return jsonify({'error': 'invalid time format'}), 400
    day_values = []
    for d in days:
        day = _coerce_int(d, low=0, high=6)
        if day is None:
            return jsonify({'error': 'invalid days value'}), 400
        day_values.append(day)
    days = day_values
    if not isinstance(staff_ids, list):
        return jsonify({'error': 'invalid staff_ids'}), 400
    from uuid import uuid4
    fid = session.get('facility_id')
    if session.get('role') == 'super_admin':
        maybe = data.get('facility_id')
        if isinstance(maybe, int) or (isinstance(maybe, str) and maybe.isdigit()):
            fid = int(maybe)
    item = db_insert_schedule(str(uuid4()), name, start, end, days, staff_ids, fid)
    _audit_event('create_schedule', item.get('id'))
    return jsonify(item), 201


@app.route('/api/schedule/<schedule_id>', methods=['DELETE'])
def schedule_delete(schedule_id: str):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('super_admin', 'facility_admin'):
        return jsonify({'error': 'Forbidden'}), 403
    target = db_get_schedule(schedule_id)
    if not target:
        return jsonify({'error': 'Not found'}), 404
    if session.get('role') == 'facility_admin':
        if target.get('facility_id') != session.get('facility_id'):
            return jsonify({'error': 'Forbidden'}), 403
    ok = db_delete_schedule(schedule_id)
    if ok:
        _audit_event('delete_schedule', schedule_id)
    return jsonify({'status': 'deleted' if ok else 'not_found'})


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
    name = _clean_text(data.get('name'), 120)
    if role == 'super_admin':
        fid_raw = data.get('facility_id')
        fid = int(fid_raw) if isinstance(fid_raw, (int, str)) and str(fid_raw).isdigit() else None
    else:
        fid = session.get('facility_id')
        if role_new == 'super_admin':
            return jsonify({'error': 'facility_admin cannot create super_admin'}), 403
    assigned = data.get('assigned_patient_ids') or []
    if not email or not _valid_email(email):
        return jsonify({'error': 'valid email required'}), 400
    if not raw_pw or len(raw_pw) < 8:
        return jsonify({'error': 'password must be at least 8 characters'}), 400
    if role_new not in ('staff', 'facility_admin', 'super_admin'):
        return jsonify({'error': 'invalid role'}), 400
    if not isinstance(assigned, list):
        return jsonify({'error': 'assigned_patient_ids must be a list'}), 400
    assigned = [int(x) for x in assigned if str(x).isdigit()]
    from werkzeug.security import generate_password_hash
    phash = generate_password_hash(raw_pw)
    try:
        created = db_insert_user(email, phash, role_new, name, fid, assigned)
    except Exception as e:
        return jsonify({'error': f'create failed: {e}'}), 400
    created.pop('password_hash', None)
    created['id'] = created.get('email')
    _audit_event('create_user', created.get('email'))
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
        if ok:
            _audit_event('delete_user', email)
        return jsonify({'status': 'deleted' if ok else 'not_found'})
    # PATCH
    target_list = [x for x in db_list_users(None) if x.get('email','').lower() == email.lower()]
    target = target_list[0] if target_list else None
    if not target:
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json(force=True) or {}
    updates = {}
    if 'role' in data:
        if data['role'] not in ('staff', 'facility_admin', 'super_admin'):
            return jsonify({'error': 'invalid role'}), 400
        if role == 'facility_admin' and data['role'] == 'super_admin':
            return jsonify({'error': 'Forbidden'}), 403
        updates['role'] = data['role']
    if 'name' in data:
        name = _clean_text(data.get('name'), 120)
        updates['name'] = name
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
        updates['assigned_patient_ids'] = [int(x) for x in data.get('assigned_patient_ids') if str(x).isdigit()]
    if 'password' in data and data.get('password'):
        if len(str(data['password'])) < 8:
            return jsonify({'error': 'password must be at least 8 characters'}), 400
        from werkzeug.security import generate_password_hash
        updates['password_hash'] = generate_password_hash(data['password'])
    updated = db_update_user(email, updates)
    updated.pop('password_hash', None)
    updated['id'] = updated.get('email')
    _audit_event('update_user', email)
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
    detected = _detect_image_extension(upload)
    if not _extension_matches_detected(ext, detected):
        return jsonify({'error': 'invalid image content'}), 400
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
    if not app.debug and os.getenv('ENABLE_ALERT_MONITOR', '0') != '1':
        return
    if app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        return
    t = threading.Thread(target=_alert_monitor_loop, name='alert-monitor', daemon=True)
    t.start()


if __name__ == '__main__':
    start_background_tasks()
    app.run(host='0.0.0.0', debug=True)

