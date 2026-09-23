# Mood & Sensor Dashboard

Flask-based dashboard that aggregates environmental sensor CSV data, user check-ins, and journal entries, then serves analytics + a mood prediction model.

## Features
- User session login (demo credentials)
- Sensor aggregation endpoints (`/api/dashboard_data`, `/api/latest_reading`)
- Mood & stress check-ins (`/api/checkins`)
- Journal entries (`/api/journal_entries`)
- Weekly narrative insights (`/api/weekly_insights`)
- Local privacy-first chat assistant (`/api/chat`) with optional Ollama refinement
- RandomForest mood prediction (`/api/predict_mood`)

## Project Layout
```
app.py                  # Flask app
preprocessing.py        # Shared feature engineering
test_model.py           # Training & evaluation script (produces mood_model.pkl)
mood_model.pkl          # Saved model used at runtime
requirements.txt        # Python dependencies
templates/              # Jinja2 templates
static/                 # Static assets
data/                   # Sensor + journal + checkins CSVs
```

## Setup
Create and activate a virtual environment (Windows PowerShell example):
```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:
```
FLASK_SECRET_KEY=replace_this
CHATBOT_LOCAL_ONLY=1
CHATBOT_NO_PHI=1
CHATBOT_PROVIDER=local
# Optional local LLM (still local-only, no cloud):
# CHATBOT_PROVIDER=ollama
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OLLAMA_MODEL=llama3.2:1b
```

Run the app:
```powershell
python app.py
```
Visit: http://127.0.0.1:5000

Demo login:
```
Email: demo@example.com
Password: password123
```

## Training / Updating the Model
1. Place or update `synthetic_environment_data_adjusted_with_time.csv`.
2. Run:
```powershell
python test_model.py
```
3. The script saves `mood_model.pkl`. The app loads it automatically on restart.

## Prediction Payload Example
POST to `/api/predict_mood` with JSON:
```json
{
  "timestamp": "2025-08-30T14:05:00",
  "Temperature": 72.5,
  "Humidity": 40.2,
  "MQ-2": "OK",
  "BH1750FVI": 300,
  "Radar": 1,
  "Ultrasonic": 55.0,
  "song": 2
}
```
Returns: `{ "predicted_mood": 4 }`

## Testing (initial smoke examples forthcoming)
Install dev deps (already in `requirements.txt`) then you can add PyTest tests under `tests/`.

## DigitalOcean Deployment

For the production database migration, real administrator setup, device
configuration and pilot acceptance checks, follow [Live Data Setup](docs/LIVE_DATA_SETUP.md).
Use `.env.production.example` for a clean PostgreSQL-backed deployment.

This app can run on DigitalOcean App Platform as a Python web service.

Production environment variables (see the linked setup guide for the certificate path):
```
FLASK_SECRET_KEY=use_a_long_random_secret
APP_ENV=production
DATABASE_URL=your_managed_postgresql_uri_with_verified_TLS
REQUIRE_POSTGRES=1
SEED_DEMO_DATA=0
CHATBOT_PROVIDER=local
CHATBOT_LOCAL_ONLY=1
CHATBOT_NO_PHI=1
DISABLE_ALERT_MONITOR=1
ENABLE_ALERT_MONITOR=0
```

App Platform run command:
```
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT wsgi:app
```

Use Managed PostgreSQL for production data. App Platform local storage is ephemeral and does not support persistent volume mounts. SQLite remains available for local development when DATABASE_URL is unset and REQUIRE_POSTGRES=0. Enable alerts only after the pilot's recipients, shifts and delivery are verified.

## Next Steps (Suggested)
- Replace demo auth with real user model + hashed passwords
- Add input validation layer (pydantic / marshmallow)
- Add caching or database for sensor data
- Break up `dashboard.html` into components
- Add API + model unit tests
- Improve security headers & enable HTTPS + secure cookies in production

## License
Internal / unspecified. Add a license file if distributing.
