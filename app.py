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

from preprocessing import engineer_features, ensure_feature_order, FEATURE_COLUMNS
from users import authenticate
from functools import lru_cache
from datetime import datetime as dt
from typing import Any, Dict



load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-me-dev-secret')

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
                {"role": "system", "content": "You’re a helpful assistant in a hospital dashboard."},
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

    # Read & concatenate sensor CSVs
    csvs = glob.glob(os.path.join('data','sensor_data_*.csv'))
    df = pd.concat((pd.read_csv(f, parse_dates=['timestamp']) for f in csvs),
                   ignore_index=True)

    # Filter to last 7 days
    now = pd.Timestamp.now()
    week_ago = now - pd.Timedelta(days=7)
    df = df[df['timestamp'] >= week_ago]

    # Ensure numeric
    for col in ['mood','Ultrasonic','Temperature','BH1750FVI']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute stats
    avg_mood = df['mood'].mean()
    max_row = df.loc[df['mood'].idxmax()] if not df['mood'].dropna().empty else None
    min_row = df.loc[df['mood'].idxmin()] if not df['mood'].dropna().empty else None
    avg_move = df['Ultrasonic'].mean()
    avg_temp = df['Temperature'].mean()
    avg_light = df['BH1750FVI'].mean()

    # Build narrative
    parts = []
    if pd.notnull(avg_mood):
        parts.append(f"Over the last 7 days your average mood was {avg_mood:.1f}/6.")
    if max_row is not None:
        d = max_row['timestamp'].strftime("%b %d")
        parts.append(f"Your highest mood ({int(max_row['mood'])}) was on {d}.")
    if min_row is not None:
        d = min_row['timestamp'].strftime("%b %d")
        parts.append(f"Your lowest mood ({int(min_row['mood'])}) was on {d}.")
    if pd.notnull(avg_move):
        parts.append(f"Average movement was {avg_move:.0f} cm.")
    if pd.notnull(avg_temp):
        parts.append(f"Avg. temperature: {avg_temp:.1f}°F; light: {avg_light:.0f} lux.")

    narrative = " ".join(parts) if parts else "No data available for the past week."

    return jsonify({"narrative": narrative})



@app.route('/api/checkins', methods=['GET', 'POST'])
def checkins():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # POST: save a new check-in
    if request.method == 'POST':
        payload = request.get_json()
        ts = payload.get('timestamp')
        mood = payload.get('mood', '')
        stress = payload.get('stress', '')
        user = session['user']

        os.makedirs(os.path.dirname(CHECKIN_PATH), exist_ok=True)
        file_exists = os.path.exists(CHECKIN_PATH)
        with open(CHECKIN_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['user', 'timestamp', 'mood', 'stress'])
            writer.writerow([user, ts, mood, stress])
        return jsonify({'status': 'ok'})

    # GET: return all check-ins for this user
    entries = []
    if os.path.exists(CHECKIN_PATH):
        with open(CHECKIN_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user'] == session['user']:
                    entries.append({
                        'timestamp': row['timestamp'],
                        'mood': row['mood'],
                        'stress': row['stress']
                    })
    entries.sort(key=lambda e: e['timestamp'], reverse=True)
    return jsonify(entries)



# === New: Journal entries endpoint ===
@app.route('/api/journal_entries', methods=['GET', 'POST'])
def journal_entries():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # POST: add a new journal entry
    if request.method == 'POST':
        payload = request.get_json()
        ts = payload.get('timestamp')
        text = payload.get('text', '').strip()
        mood = payload.get('mood', '')
        user = session['user']

        # Ensure directory exists
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        file_exists = os.path.exists(JOURNAL_PATH)

        with open(JOURNAL_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['user', 'timestamp', 'mood', 'text'])
            writer.writerow([user, ts, mood, text])
        return jsonify({'status': 'ok'})

    # GET: return all entries for this user
    entries = []
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user'] == session['user']:
                    entries.append({
                        'timestamp': row['timestamp'],
                        'mood': row['mood'],
                        'text': row['text']
                    })
    # Sort newest first
    entries.sort(key=lambda e: e['timestamp'], reverse=True)
    return jsonify(entries)

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to maintain backward compatibility; delegates to shared module."""
    df = engineer_features(df, drop_original_timestamp=False)
    return ensure_feature_order(df)

@app.route('/')
def home():
    return redirect(url_for('login'))

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
        if authenticate(email, password):
            session['user'] = email
            return redirect(url_for('dashboard'))
        return redirect(url_for('login', error=1))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

def _load_sensor_dataframe() -> pd.DataFrame:
    csv_files = glob.glob(os.path.join('data', 'sensor_data_*.csv'))
    if not csv_files:
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
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


@lru_cache(maxsize=1)
def _cached_aggregates(cache_marker: int) -> Dict[str, Any]:  # marker allows manual invalidation
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

    avg_move = combined['Ultrasonic'].mean(skipna=True) if 'Ultrasonic' in combined else 0
    avg_light = combined['BH1750FVI'].mean(skipna=True) if 'BH1750FVI' in combined else 0
    m_val = avg_mood or 0
    mv_val = avg_move or 0
    lt_val = avg_light or 0
    total = m_val + mv_val + lt_val
    if total > 0:
        mood_slice = m_val/total*100
        move_slice = mv_val/total*100
        light_slice = lt_val/total*100
    else:
        mood_slice = move_slice = light_slice = 0
    overall = (mood_slice + move_slice + light_slice) / 3
    return {
        "summary": {
            "avgTemp": round(avg_temp, 1) if avg_temp is not None and pd.notna(avg_temp) else None,
            "avgHumidity": round(avg_hum, 1) if avg_hum is not None and pd.notna(avg_hum) else None,
            "avgMood": round(avg_mood, 1) if avg_mood is not None and pd.notna(avg_mood) else None,
            "currentAirQuality": current_air_quality,
            "movementStatus": movement_status,
        },
        "charts": {"mood": mood_data, "movement": move_data},
        "donutSlices": {
            "mood": mood_slice,
            "movement": move_slice,
            "light": light_slice,
            "overallScore": overall,
        },
    }


@app.route('/api/dashboard_data')
def dashboard_data():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    # Use current minute as cache marker for lightweight TTL
    marker = int(dt.utcnow().timestamp() // 60)
    data = _cached_aggregates(marker)
    if 'error' in data:
        return jsonify(data), 404
    return jsonify(data)

@app.route('/api/latest_reading')
def latest_reading():
    if 'user' not in session:
        return jsonify({'error':'Unauthorized'}), 401

    csv_files = glob.glob(os.path.join('data','sensor_data_*.csv'))
    df_list = [pd.read_csv(fp) for fp in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    last = df.sort_values('timestamp').iloc[-1]

    # Safely handle NaNs
    song_val = last.get('song', 0)
    if pd.isna(song_val):
        song_val = 0

    mq_val = last.get('MQ-2', last.get('MQ2', 'OK'))
    if pd.isna(mq_val):
        mq_val = 'OK'

    return jsonify({
        'timestamp': last['timestamp'].isoformat(),
        'Temperature': float(last['Temperature'] or 0),
        'Humidity': float(last['Humidity'] or 0),
        'MQ-2': mq_val,
        'BH1750FVI': float(last['BH1750FVI'] or 0),
        'Radar': int(last['Radar'] or 0),
        'Ultrasonic': float(last['Ultrasonic'] or 0),
        'song': int(song_val)
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
    return jsonify({'predicted_mood': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)
