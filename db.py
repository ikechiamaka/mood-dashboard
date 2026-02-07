import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, 'app.db')


def _resolve_db_path() -> str:
    override = os.getenv('NEUROSENSE_DB_PATH', '').strip()
    if override:
        return override
    return DEFAULT_DB_PATH

_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = _resolve_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute('PRAGMA foreign_keys = ON')
    return _conn


def _user_version(conn: sqlite3.Connection) -> int:
    cur = conn.execute('PRAGMA user_version')
    return int(cur.fetchone()[0])


def _set_user_version(conn: sqlite3.Connection, v: int) -> None:
    conn.execute(f'PRAGMA user_version = {v}')


def init_db() -> None:
    conn = get_conn()
    v = _user_version(conn)
    if v == 0:
        _create_schema_v1(conn)
        _set_user_version(conn, 1)
        conn.commit()
        _maybe_seed_from_env(conn)
        _maybe_import_legacy_logs(conn)
    else:
        # Ensure any newly added tables exist for older DBs
        _ensure_schema_compat(conn)
        _maybe_import_legacy_logs(conn)
        conn.commit()
    # Future migrations: if v < 2: ...


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            facility_id INTEGER,
            assigned_patient_ids TEXT,
            avatar_url TEXT
        );

        CREATE TABLE IF NOT EXISTS staff (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            facility_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS beds (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            room TEXT,
            patient TEXT,
            facility_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS schedule (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            start TEXT NOT NULL,
            end TEXT NOT NULL,
            days TEXT NOT NULL,
            staff_ids TEXT NOT NULL,
            facility_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            facility_id INTEGER,
            bed_id TEXT,
            age INTEGER,
            risk_level TEXT,
            primary_condition TEXT,
            allergies TEXT,
            care_focus TEXT,
            avatar_url TEXT
        );

        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            mq2 TEXT,
            bh1750fvi REAL,
            radar INTEGER,
            ultrasonic REAL,
            mood REAL,
            song INTEGER,
            heart_rate REAL,
            respiration_rate REAL,
            bed_id TEXT NOT NULL,
            patient_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_readings_bed_ts ON readings (bed_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            mood INTEGER,
            stress TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_checkins_patient_ts ON checkins (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            mood INTEGER,
            text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_journal_patient_ts ON journal_entries (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            due_date TEXT,
            notify INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_goals_patient_created ON goals (patient_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            timestamp TEXT NOT NULL,
            predicted_mood INTEGER,
            temperature REAL,
            humidity REAL,
            mq2 TEXT,
            bh1750fvi REAL,
            radar INTEGER,
            ultrasonic REAL,
            song INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_patient_ts ON predictions (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS support_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            facility TEXT,
            role TEXT,
            goal TEXT,
            message TEXT,
            contact TEXT,
            notes TEXT,
            user_agent TEXT,
            submitted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        '''
    )
    _ensure_patient_columns(conn)
    _ensure_user_columns(conn)
    _ensure_bed_monitoring_schema(conn)


def _ensure_user_columns(conn: sqlite3.Connection) -> None:
    desired = {
        'avatar_url': 'TEXT',
    }
    cur = conn.execute('PRAGMA table_info(users)')
    existing = {row[1] for row in cur.fetchall()}
    for column, ddl in desired.items():
        if column not in existing:
            conn.execute(f'ALTER TABLE users ADD COLUMN {column} {ddl}')


def _ensure_patient_columns(conn: sqlite3.Connection) -> None:
    """Ensure demo metadata columns exist on the patients table."""
    desired = {
        'age': 'INTEGER',
        'risk_level': 'TEXT',
        'primary_condition': 'TEXT',
        'allergies': 'TEXT',
        'care_focus': 'TEXT',
        'avatar_url': 'TEXT'
    }
    cur = conn.execute('PRAGMA table_info(patients)')
    existing = {row[1] for row in cur.fetchall()}
    for column, ddl in desired.items():
        if column not in existing:
            conn.execute(f'ALTER TABLE patients ADD COLUMN {column} {ddl}')


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f'PRAGMA table_info({table})')
    return any(row[1] == column for row in cur.fetchall())


def _ensure_bed_monitoring_schema(conn: sqlite3.Connection) -> None:
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS facilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timezone TEXT NOT NULL DEFAULT 'UTC',
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            facility_id INTEGER NOT NULL,
            bed_id TEXT,
            api_key_hash TEXT NOT NULL,
            firmware TEXT,
            capabilities TEXT,
            last_seen_at INTEGER,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (facility_id) REFERENCES facilities(id),
            FOREIGN KEY (bed_id) REFERENCES beds(id)
        );

        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            facility_id INTEGER NOT NULL,
            bed_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            presence INTEGER,
            fall INTEGER,
            rr REAL,
            hr REAL,
            confidence REAL,
            raw TEXT,
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (facility_id) REFERENCES facilities(id),
            FOREIGN KEY (bed_id) REFERENCES beds(id)
        );
        CREATE INDEX IF NOT EXISTS idx_telemetry_bed_ts_desc ON telemetry (bed_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_telemetry_device_ts_desc ON telemetry (device_id, ts DESC);

        CREATE TABLE IF NOT EXISTS staff_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            phone_e164 TEXT NOT NULL,
            role TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (facility_id) REFERENCES facilities(id)
        );

        CREATE TABLE IF NOT EXISTS shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id INTEGER NOT NULL,
            staff_contact_id INTEGER NOT NULL,
            days_of_week TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            timezone TEXT NOT NULL DEFAULT 'UTC',
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (facility_id) REFERENCES facilities(id),
            FOREIGN KEY (staff_contact_id) REFERENCES staff_contacts(id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id INTEGER NOT NULL,
            bed_id TEXT NOT NULL,
            device_id TEXT,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            ts INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            acked_by_user_id INTEGER,
            acked_at INTEGER,
            meta TEXT,
            FOREIGN KEY (facility_id) REFERENCES facilities(id),
            FOREIGN KEY (bed_id) REFERENCES beds(id),
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (acked_by_user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_facility_ts_desc ON alerts (facility_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_alerts_bed_ts_desc ON alerts (bed_id, ts DESC);

        CREATE TABLE IF NOT EXISTS bed_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            staff_user_id INTEGER NOT NULL,
            bed_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(staff_user_id, bed_id),
            FOREIGN KEY (staff_user_id) REFERENCES users(id),
            FOREIGN KEY (bed_id) REFERENCES beds(id)
        );
        '''
    )
    if not _table_has_column(conn, 'devices', 'active'):
        conn.execute('ALTER TABLE devices ADD COLUMN active INTEGER NOT NULL DEFAULT 1')
    if not _table_has_column(conn, 'beds', 'label'):
        conn.execute('ALTER TABLE beds ADD COLUMN label TEXT')
    if not _table_has_column(conn, 'beds', 'active'):
        conn.execute('ALTER TABLE beds ADD COLUMN active INTEGER NOT NULL DEFAULT 1')
    if not _table_has_column(conn, 'beds', 'created_at'):
        conn.execute('ALTER TABLE beds ADD COLUMN created_at INTEGER')
    conn.execute('UPDATE beds SET label = COALESCE(NULLIF(label, \'\'), name) WHERE label IS NULL OR label = \'\'')
    conn.execute('UPDATE beds SET created_at = ? WHERE created_at IS NULL', (now_epoch,))
    try:
        conn.execute('UPDATE devices SET active = 1 WHERE active IS NULL')
    except sqlite3.Error:
        pass

    facility_ids: set[int] = set()
    for table_name in ('users', 'patients', 'beds', 'staff', 'schedule'):
        try:
            cur = conn.execute(f'SELECT DISTINCT facility_id FROM {table_name} WHERE facility_id IS NOT NULL')
        except sqlite3.Error:
            continue
        for row in cur.fetchall():
            value = row[0]
            try:
                facility_ids.add(int(value))
            except Exception:
                continue
    for fid in sorted(facility_ids):
        conn.execute(
            'INSERT OR IGNORE INTO facilities (id, name, timezone, created_at) VALUES (?, ?, ?, ?)',
            (fid, f'Facility {fid}', 'UTC', now_epoch),
        )

    cur_staff_contacts = conn.execute('SELECT COUNT(1) FROM staff_contacts')
    if int(cur_staff_contacts.fetchone()[0]) == 0:
        try:
            cur_staff = conn.execute('SELECT id, facility_id, name, phone FROM staff ORDER BY name')
            for row in cur_staff.fetchall():
                phone = (row['phone'] or '').strip()
                if not phone:
                    continue
                conn.execute(
                    'INSERT INTO staff_contacts (facility_id, name, phone_e164, role, active) VALUES (?, ?, ?, ?, 1)',
                    (row['facility_id'], row['name'], phone, 'nurse'),
                )
        except sqlite3.Error:
            pass

    cur_shifts = conn.execute('SELECT COUNT(1) FROM shifts')
    if int(cur_shifts.fetchone()[0]) == 0:
        try:
            contacts_by_name = {}
            cur = conn.execute('SELECT id, facility_id, name FROM staff_contacts WHERE active = 1')
            for row in cur.fetchall():
                contacts_by_name[(row['facility_id'], row['name'])] = row['id']
            cur_sched = conn.execute('SELECT * FROM schedule')
            for row in cur_sched.fetchall():
                facility_id = row['facility_id']
                days_raw = row['days']
                try:
                    days_parsed = json.loads(days_raw) if isinstance(days_raw, str) else []
                except Exception:
                    days_parsed = []
                contact_id = contacts_by_name.get((facility_id, row['name']))
                if not contact_id:
                    continue
                conn.execute(
                    '''
                    INSERT INTO shifts (
                        facility_id, staff_contact_id, days_of_week, start_time, end_time, timezone, active
                    ) VALUES (?, ?, ?, ?, ?, ?, 1)
                    ''',
                    (
                        facility_id,
                        contact_id,
                        json.dumps(days_parsed),
                        row['start'],
                        row['end'],
                        'UTC',
                    ),
                )
        except sqlite3.Error:
            pass


def _seed_patient_readings(conn: sqlite3.Connection, patient_id: int, bed_id: str | None, hours: int = 48) -> None:
    if not bed_id:
        try:
            bed_id = f"AUTO-{patient_id}"
            conn.execute('UPDATE patients SET bed_id = ? WHERE id = ?', (bed_id, patient_id))
        except Exception:
            return
    try:
        cur = conn.execute('SELECT COUNT(1) FROM readings WHERE bed_id = ?', (bed_id,))
        if int(cur.fetchone()[0]) > 0:
            return
        base = datetime.now(timezone.utc)
        rows: list[tuple] = []
        patient_offset = (patient_id or 0) % 7
        temp_base = 68.0 + patient_offset
        hum_base = 38.0 + patient_offset * 2
        light_base = 220 + patient_offset * 35
        ultra_base = 40 + patient_offset * 4
        heart_base = 68 + patient_offset * 2
        resp_base = 13 + patient_offset % 4
        mood_base = 2 + (patient_offset % 3)
        stress_cycle = (patient_offset % 5) + 1
        for h in range(hours, -1, -1):
            t = base - timedelta(hours=h)
            day_progress = (h % 24) / 24
            temp = temp_base + (day_progress - 0.5) * 6 + (h % 6) * 0.2
            hum = hum_base + (0.5 - day_progress) * 8 + (h % 5) * 0.3
            heart = heart_base + ((h + patient_offset) % 6)
            resp = resp_base + ((h + patient_offset) % 3) * 0.5
            light = light_base + (1 - abs(0.5 - day_progress)) * 120 + (h % 10) * 10
            radar = 1 if (h + patient_offset) % 5 in (1, 2, 3) else 0
            ultra = ultra_base + (h % 8) * 1.5 + (patient_offset % 3)
            mood = max(1, min(6, mood_base + ((h // 8) % 3) - 1))
            song = 1 + ((patient_offset + h) % 3)
            rows.append((
                t.isoformat(), round(temp, 2), round(hum, 2), 'OK', round(light, 2), radar, round(ultra, 2), mood, song, round(heart, 2), round(resp, 2), bed_id, patient_id
            ))
        conn.executemany(
            'INSERT INTO readings (timestamp, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, mood, song, heart_rate, respiration_rate, bed_id, patient_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        conn.commit()
    except Exception:
        pass


def _ensure_schema_compat(conn: sqlite3.Connection) -> None:
    """Create newer tables if running against an older DB (idempotent)."""
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            facility_id INTEGER,
            bed_id TEXT,
            age INTEGER,
            risk_level TEXT,
            primary_condition TEXT,
            allergies TEXT,
            care_focus TEXT,
            avatar_url TEXT
        );

        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            mq2 TEXT,
            bh1750fvi REAL,
            radar INTEGER,
            ultrasonic REAL,
            mood REAL,
            song INTEGER,
            heart_rate REAL,
            respiration_rate REAL,
            bed_id TEXT NOT NULL,
            patient_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_readings_bed_ts ON readings (bed_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            mood INTEGER,
            stress TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_checkins_patient_ts ON checkins (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            mood INTEGER,
            text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_journal_patient_ts ON journal_entries (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            due_date TEXT,
            notify INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_goals_patient_created ON goals (patient_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            timestamp TEXT NOT NULL,
            predicted_mood INTEGER,
            temperature REAL,
            humidity REAL,
            mq2 TEXT,
            bh1750fvi REAL,
            radar INTEGER,
            ultrasonic REAL,
            song INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_patient_ts ON predictions (patient_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS support_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            facility TEXT,
            role TEXT,
            goal TEXT,
            message TEXT,
            contact TEXT,
            notes TEXT,
            user_agent TEXT,
            submitted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        '''
    )
    _ensure_patient_columns(conn)
    _ensure_user_columns(conn)
    _ensure_bed_monitoring_schema(conn)



def _maybe_seed_from_env(conn: sqlite3.Connection) -> None:
    """Seed demo data and ensure baseline readings for showcase environments."""
    cur = conn.execute('SELECT COUNT(1) FROM users')
    users_exist = int(cur.fetchone()[0]) > 0
    if not users_exist:
        from werkzeug.security import generate_password_hash
        demo_users = [
            {
                'email': os.getenv('DEMO_SUPER_EMAIL', 'superadmin@demo.com'),
                'password_hash': generate_password_hash(os.getenv('DEMO_SUPER_PASSWORD', 'admin123')),
                'role': 'super_admin',
                'name': 'Alex Johnson',
                'facility_id': None,
                'assigned_patient_ids': []
            },
            {
                'email': os.getenv('DEMO_FAC1_EMAIL', 'fac1admin@demo.com'),
                'password_hash': generate_password_hash(os.getenv('DEMO_FAC1_PASSWORD', 'admin123')),
                'role': 'facility_admin',
                'name': 'Sarah Miller',
                'facility_id': 1,
                'assigned_patient_ids': []
            },
            {
                'email': os.getenv('DEMO_STAFF1_EMAIL', 'staff1@demo.com'),
                'password_hash': generate_password_hash(os.getenv('DEMO_STAFF1_PASSWORD', 'staff123')),
                'role': 'staff',
                'name': 'James Wilson',
                'facility_id': 1,
                'assigned_patient_ids': [1, 3, 5]
            },
        ]
        for u in demo_users:
            conn.execute(
                'INSERT INTO users (email, password_hash, role, name, facility_id, assigned_patient_ids) VALUES (?,?,?,?,?,?)',
                (
                    u['email'], u['password_hash'], u['role'], u['name'], u['facility_id'],
                    json.dumps(u['assigned_patient_ids'], ensure_ascii=False)
                )
            )
        conn.commit()

    cur = conn.execute('SELECT COUNT(1) FROM patients')
    patients_exist = int(cur.fetchone()[0]) > 0
    if not patients_exist:
        demo_patients = [
            {
                'name': 'Blaine Cottrell',
                'facility_id': 1,
                'bed_id': 'B-1',
                'age': 34,
                'risk_level': 'High',
                'primary_condition': 'Schizophrenia',
                'allergies': 'Penicillin',
                'care_focus': 'Maintain insulin routine and daily journaling.'
            },
            {
                'name': 'Jane Doe',
                'facility_id': 1,
                'bed_id': 'B-3',
                'age': 58,
                'risk_level': 'Moderate',
                'primary_condition': 'Major Depressive Disorder',
                'allergies': 'None',
                'care_focus': 'Review therapy notes and schedule outdoor time.'
            },
            {
                'name': 'John Smith',
                'facility_id': 1,
                'bed_id': 'B-4',
                'age': 47,
                'risk_level': 'Low',
                'primary_condition': 'Generalized Anxiety Disorder',
                'allergies': 'Shellfish',
                'care_focus': 'Practice guided breathing and track sleep quality.'
            },
            {
                'name': 'Maria Green',
                'facility_id': 1,
                'bed_id': 'B-5',
                'age': 72,
                'risk_level': 'High',
                'primary_condition': "Alzheimer's Disease",
                'allergies': 'Sulfa drugs',
                'care_focus': 'Support memory cues and hydration reminders.'
            },
            {
                'name': 'Liam Nguyen',
                'facility_id': 1,
                'bed_id': 'B-6',
                'age': 39,
                'risk_level': 'Moderate',
                'primary_condition': 'Chronic Pain',
                'allergies': 'NSAIDs',
                'care_focus': 'Track pain triggers and complete stretching blocks.'
            },
            {
                'name': 'Ava Thompson',
                'facility_id': 1,
                'bed_id': 'B-7',
                'age': 29,
                'risk_level': 'Low',
                'primary_condition': 'Sleep Irregularity',
                'allergies': 'None',
                'care_focus': 'Stabilize bedtime routine and log restfulness.'
            },
            {
                'name': 'Noah Patel',
                'facility_id': 2,
                'bed_id': 'F2-201',
                'age': 61,
                'risk_level': 'Moderate',
                'primary_condition': 'Postoperative Recovery',
                'allergies': 'Latex',
                'care_focus': 'Monitor incision healing and encourage assisted mobility.'
            },
            {
                'name': 'Emily Chen',
                'facility_id': 2,
                'bed_id': 'F2-202',
                'age': 44,
                'risk_level': 'High',
                'primary_condition': 'Bipolar Disorder',
                'allergies': 'None',
                'care_focus': 'Balance medication adherence and journaling mood changes.'
            },
            {
                'name': 'Marcus Johnson',
                'facility_id': 2,
                'bed_id': 'F2-203',
                'age': 52,
                'risk_level': 'Low',
                'primary_condition': 'Hypertension',
                'allergies': 'ACE inhibitors',
                'care_focus': 'Track blood pressure trends and support DASH meal planning.'
            },
            {
                'name': 'Sofia Rivera',
                'facility_id': 2,
                'bed_id': 'F2-204',
                'age': 36,
                'risk_level': 'Moderate',
                'primary_condition': 'PTSD',
                'allergies': 'Peanuts',
                'care_focus': 'Coordinate evening mindfulness and therapy follow-ups.'
            }
        ]
        for p in demo_patients:
            avatar = p.get('avatar_url') or '/static/pic.jpg'
            cur = conn.execute(
                'INSERT INTO patients (name, facility_id, bed_id, age, risk_level, primary_condition, allergies, care_focus, avatar_url) VALUES (?,?,?,?,?,?,?,?,?)',
                (
                    p['name'],
                    p['facility_id'],
                    p.get('bed_id'),
                    p.get('age'),
                    p.get('risk_level'),
                    p.get('primary_condition'),
                    p.get('allergies'),
                    p.get('care_focus'),
                    avatar
                )
            )
            pid = cur.lastrowid
            _seed_patient_readings(conn, pid, p.get('bed_id'))
        conn.commit()

    cur = conn.execute('SELECT id, bed_id FROM patients')
    for row in cur.fetchall():
        _seed_patient_readings(conn, row['id'], row['bed_id'])

    cur = conn.execute('SELECT COUNT(1) FROM readings')
    if int(cur.fetchone()[0]) == 0:
        _import_csv_readings(conn)


def db_get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM users WHERE lower(email) = lower(?)', (email or '',))
    row = cur.fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def db_list_users(facility_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    if facility_id is None:
        cur = conn.execute('SELECT * FROM users ORDER BY email')
    else:
        cur = conn.execute('SELECT * FROM users WHERE facility_id IS NULL OR facility_id = ? ORDER BY email', (facility_id,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_insert_user(email: str, password_hash: str, role: str, name: Optional[str], facility_id: Optional[int], assigned_patient_ids: List[int], avatar_url: Optional[str] = None) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute(
        'INSERT INTO users (email, password_hash, role, name, facility_id, assigned_patient_ids, avatar_url) VALUES (?,?,?,?,?,?,?)',
        (email, password_hash, role, name, facility_id, json.dumps(assigned_patient_ids or []), avatar_url)
    )
    conn.commit()
    return db_get_user_by_email(email) or {}


def db_delete_user_by_email(email: str) -> bool:
    conn = get_conn()
    cur = conn.execute('DELETE FROM users WHERE lower(email)=lower(?)', (email,))
    conn.commit()
    return cur.rowcount > 0


def db_update_user(email: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    allowed = {'password_hash', 'role', 'name', 'facility_id', 'assigned_patient_ids', 'avatar_url'}
    sets = []
    vals: List[Any] = []
    for k, v in updates.items():
        if k not in allowed:
            continue
        if k == 'assigned_patient_ids':
            v = json.dumps(v or [])
        sets.append(f'{k} = ?')
        vals.append(v)
    if not sets:
        return db_get_user_by_email(email)
    vals.append(email)
    conn = get_conn()
    conn.execute(f'UPDATE users SET {", ".join(sets)} WHERE lower(email)=lower(?)', tuple(vals))
    conn.commit()
    return db_get_user_by_email(email)


# ---- Facility mgmt helpers ----
def db_list_staff(facility_id: Optional[int]) -> List[Dict[str, Any]]:
    conn = get_conn()
    if facility_id is None:
        cur = conn.execute('SELECT * FROM staff ORDER BY name')
    else:
        cur = conn.execute('SELECT * FROM staff WHERE facility_id = ? ORDER BY name', (facility_id,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_insert_staff(item_id: str, name: str, phone: str, facility_id: Optional[int]) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute('INSERT INTO staff (id, name, phone, facility_id) VALUES (?,?,?,?)', (item_id, name, phone, facility_id))
    conn.commit()
    cur = conn.execute('SELECT * FROM staff WHERE id=?', (item_id,))
    return _row_to_dict(cur.fetchone())


def db_delete_staff(item_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute('DELETE FROM staff WHERE id=?', (item_id,))
    conn.commit()
    return cur.rowcount > 0


def db_list_beds(facility_id: Optional[int]) -> List[Dict[str, Any]]:
    conn = get_conn()
    if facility_id is None:
        cur = conn.execute('SELECT * FROM beds ORDER BY name')
    else:
        cur = conn.execute('SELECT * FROM beds WHERE facility_id = ? ORDER BY name', (facility_id,))
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_insert_bed(item_id: str, name: str, room: str, patient: str, facility_id: Optional[int]) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute('INSERT INTO beds (id, name, room, patient, facility_id) VALUES (?,?,?,?,?)', (item_id, name, room, patient, facility_id))
    conn.commit()
    cur = conn.execute('SELECT * FROM beds WHERE id=?', (item_id,))
    return _row_to_dict(cur.fetchone())


def db_get_bed(item_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM beds WHERE id = ?', (item_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def db_delete_bed(item_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute('DELETE FROM beds WHERE id = ?', (item_id,))
    conn.commit()
    return cur.rowcount > 0


def db_list_schedule(facility_id: Optional[int]) -> List[Dict[str, Any]]:
    conn = get_conn()
    if facility_id is None:
        cur = conn.execute('SELECT * FROM schedule ORDER BY name')
    else:
        cur = conn.execute('SELECT * FROM schedule WHERE facility_id = ? ORDER BY name', (facility_id,))
    rows = cur.fetchall()
    return [_row_to_dict(r, json_cols=('days','staff_ids')) for r in rows]


def db_insert_schedule(item_id: str, name: str, start: str, end: str, days: List[int], staff_ids: List[str], facility_id: Optional[int]) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute(
        'INSERT INTO schedule (id, name, start, end, days, staff_ids, facility_id) VALUES (?,?,?,?,?,?,?)',
        (item_id, name, start, end, json.dumps(days or []), json.dumps(staff_ids or []), facility_id)
    )
    conn.commit()
    cur = conn.execute('SELECT * FROM schedule WHERE id=?', (item_id,))
    return _row_to_dict(cur.fetchone(), json_cols=('days','staff_ids'))


def db_get_schedule(item_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM schedule WHERE id = ?', (item_id,))
    row = cur.fetchone()
    return _row_to_dict(row, json_cols=('days','staff_ids')) if row else None


def db_delete_schedule(item_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute('DELETE FROM schedule WHERE id = ?', (item_id,))
    conn.commit()
    return cur.rowcount > 0


# ---- Clinical logs helpers ----
def db_list_checkins(patient_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT user, patient_id, timestamp, mood, stress, notes FROM checkins WHERE patient_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?',
        (patient_id, limit)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_insert_checkin(user: str, patient_id: int, timestamp: str, mood: Optional[int], stress: Optional[str], notes: Optional[str]) -> None:
    conn = get_conn()
    conn.execute(
        'INSERT INTO checkins (user, patient_id, timestamp, mood, stress, notes) VALUES (?,?,?,?,?,?)',
        (user, patient_id, timestamp, mood, stress, notes)
    )
    conn.commit()


def db_list_journal_entries(patient_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT user, patient_id, timestamp, mood, text FROM journal_entries WHERE patient_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?',
        (patient_id, limit)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_insert_journal_entry(user: str, patient_id: int, timestamp: str, mood: Optional[int], text: str) -> None:
    conn = get_conn()
    conn.execute(
        'INSERT INTO journal_entries (user, patient_id, timestamp, mood, text) VALUES (?,?,?,?,?)',
        (user, patient_id, timestamp, mood, text)
    )
    conn.commit()


def db_list_goals(patient_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT * FROM goals WHERE patient_id = ? ORDER BY created_at DESC',
        (patient_id,)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_get_goal(goal_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM goals WHERE id = ?', (goal_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def db_insert_goal(
    goal_id: str,
    patient_id: int,
    title: str,
    status: str,
    due_date: Optional[str],
    notify: int,
    created_at: str,
    updated_at: str,
    created_by: Optional[str],
) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute(
        'INSERT INTO goals (id, patient_id, title, status, due_date, notify, created_at, updated_at, created_by) VALUES (?,?,?,?,?,?,?,?,?)',
        (goal_id, patient_id, title, status, due_date, notify, created_at, updated_at, created_by)
    )
    conn.commit()
    return db_get_goal(goal_id) or {}


def db_update_goal(goal_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    allowed = {'status', 'title', 'due_date', 'notify', 'updated_at'}
    sets = []
    vals: List[Any] = []
    for k, v in updates.items():
        if k not in allowed:
            continue
        sets.append(f'{k} = ?')
        vals.append(v)
    if not sets:
        return db_get_goal(goal_id)
    vals.append(goal_id)
    conn = get_conn()
    conn.execute(f'UPDATE goals SET {", ".join(sets)} WHERE id = ?', tuple(vals))
    conn.commit()
    return db_get_goal(goal_id)


def db_delete_goal(goal_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute('DELETE FROM goals WHERE id = ?', (goal_id,))
    conn.commit()
    return cur.rowcount > 0


def db_insert_prediction(
    patient_id: Optional[int],
    timestamp: str,
    predicted_mood: Optional[int],
    temperature: Optional[float],
    humidity: Optional[float],
    mq2: Optional[str],
    bh1750fvi: Optional[float],
    radar: Optional[int],
    ultrasonic: Optional[float],
    song: Optional[int],
) -> None:
    conn = get_conn()
    conn.execute(
        'INSERT INTO predictions (patient_id, timestamp, predicted_mood, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, song) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (patient_id, timestamp, predicted_mood, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, song)
    )
    conn.commit()


def db_insert_support_request(
    kind: str,
    email: str,
    name: Optional[str],
    facility: Optional[str],
    role: Optional[str],
    goal: Optional[str],
    message: Optional[str],
    contact: Optional[str],
    notes: Optional[str],
    user_agent: Optional[str],
    submitted_at: str,
) -> None:
    conn = get_conn()
    conn.execute(
        'INSERT INTO support_requests (kind, email, name, facility, role, goal, message, contact, notes, user_agent, submitted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (kind, email, name, facility, role, goal, message, contact, notes, user_agent, submitted_at)
    )
    conn.commit()


def db_insert_audit_event(user: Optional[str], action: str, target: Optional[str], details: Optional[str]) -> None:
    conn = get_conn()
    conn.execute(
        'INSERT INTO audit_events (user, action, target, details) VALUES (?,?,?,?)',
        (user, action, target, details)
    )
    conn.commit()


# ---- Patients helpers ----
def db_list_patients(facility_id: Optional[int] = None, allowed_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    _ensure_patient_columns(conn)
    sql = 'SELECT * FROM patients'
    params: List[Any] = []
    clauses: List[str] = []
    if facility_id is not None:
        clauses.append('facility_id = ?')
        params.append(facility_id)
    if allowed_ids is not None:
        if len(allowed_ids) == 0:
            return []
        placeholders = ','.join('?' for _ in allowed_ids)
        clauses.append(f'id IN ({placeholders})')
        params.extend(allowed_ids)
    if clauses:
        sql += ' WHERE ' + ' AND '.join(clauses)
    sql += ' ORDER BY name'
    cur = conn.execute(sql, tuple(params))
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_get_patient_by_id(pid: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    _ensure_patient_columns(conn)
    cur = conn.execute('SELECT * FROM patients WHERE id = ?', (pid,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def db_insert_patient(
    name: str,
    facility_id: Optional[int],
    bed_id: Optional[str],
    *,
    age: Optional[int] = None,
    risk_level: Optional[str] = None,
    primary_condition: Optional[str] = None,
    allergies: Optional[str] = None,
    care_focus: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_conn()
    _ensure_patient_columns(conn)
    resolved_avatar = avatar_url or '/static/pic.jpg'
    conn.execute(
        'INSERT INTO patients (name, facility_id, bed_id, age, risk_level, primary_condition, allergies, care_focus, avatar_url) VALUES (?,?,?,?,?,?,?,?,?)',
        (
            name,
            facility_id,
            bed_id,
            age,
            risk_level,
            primary_condition,
            allergies,
            care_focus,
            resolved_avatar
        )
    )
    conn.commit()
    cur = conn.execute('SELECT * FROM patients WHERE rowid = last_insert_rowid()')
    return _row_to_dict(cur.fetchone())


def _import_csv_readings(conn: sqlite3.Connection) -> None:
    import glob as _glob
    import csv as _csv
    import datetime as _dt
    # Choose a default bed mapping for legacy CSVs with no bed field
    default_bed = 'B-1'
    csv_files = _glob.glob(os.path.join('data', 'sensor_data_*.csv'))
    if not csv_files:
        return
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None
    def _to_int(v):
        try:
            return int(float(v))
        except Exception:
            return None
    rows = []
    for fp in csv_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    # timestamp safe parse
                    ts = r.get('timestamp') or r.get('Timestamp') or r.get('time')
                    # environment + movement
                    temperature = _to_float(r.get('Temperature'))
                    humidity = _to_float(r.get('Humidity'))
                    mq2 = (r.get('MQ-2') or r.get('MQ2') or '').strip() or None
                    bh = _to_float(r.get('BH1750FVI'))
                    radar = _to_int(r.get('Radar'))
                    ultrasonic = _to_float(r.get('Ultrasonic'))
                    mood = _to_float(r.get('mood'))
                    song = _to_int(r.get('song'))
                    hr = _to_float(r.get('heart_rate') or r.get('hr'))
                    rr = _to_float(r.get('respiration_rate') or r.get('rr'))
                    bed = (r.get('bed_id') or r.get('BedID') or r.get('bed') or default_bed)
                    rows.append((ts, temperature, humidity, mq2, bh, radar, ultrasonic, mood, song, hr, rr, bed, None))
        except Exception:
            continue
    if not rows:
        return
    conn.executemany(
        'INSERT INTO readings (timestamp, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, mood, song, heart_rate, respiration_rate, bed_id, patient_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def _table_empty(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(f'SELECT COUNT(1) FROM {table}')
    return int(cur.fetchone()[0]) == 0


def _maybe_import_legacy_logs(conn: sqlite3.Connection) -> None:
    try:
        if _table_empty(conn, 'checkins'):
            _import_csv_checkins(conn)
        if _table_empty(conn, 'journal_entries'):
            _import_csv_journal(conn)
        if _table_empty(conn, 'goals'):
            _import_json_goals(conn)
        if _table_empty(conn, 'predictions'):
            _import_csv_predictions(conn)
        if _table_empty(conn, 'support_requests'):
            _import_json_support_requests(conn)
    except Exception:
        return


def _import_csv_checkins(conn: sqlite3.Connection) -> None:
    import csv as _csv
    path = os.path.join(os.path.dirname(__file__), 'data', 'checkins.csv')
    if not os.path.exists(path):
        return
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                patient_id = int(row.get('patient_id') or 0)
            except Exception:
                continue
            if not patient_id:
                continue
            user = (row.get('user') or '').strip() or 'unknown'
            timestamp = (row.get('timestamp') or '').strip()
            mood = row.get('mood')
            stress = (row.get('stress') or '').strip() or None
            notes = (row.get('notes') or '').strip() or None
            try:
                mood_val = int(float(mood)) if mood not in (None, '') else None
            except Exception:
                mood_val = None
            created_at = timestamp or datetime.now(timezone.utc).isoformat()
            rows.append((user, patient_id, timestamp or created_at, mood_val, stress, notes, created_at))
    if not rows:
        return
    conn.executemany(
        'INSERT INTO checkins (user, patient_id, timestamp, mood, stress, notes, created_at) VALUES (?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def _import_csv_journal(conn: sqlite3.Connection) -> None:
    import csv as _csv
    path = os.path.join(os.path.dirname(__file__), 'data', 'journal.csv')
    if not os.path.exists(path):
        return
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                patient_id = int(row.get('patient_id') or 0)
            except Exception:
                continue
            if not patient_id:
                continue
            user = (row.get('user') or '').strip() or 'unknown'
            timestamp = (row.get('timestamp') or '').strip()
            text = (row.get('text') or '').strip() or None
            mood = row.get('mood')
            try:
                mood_val = int(float(mood)) if mood not in (None, '') else None
            except Exception:
                mood_val = None
            created_at = timestamp or datetime.now(timezone.utc).isoformat()
            rows.append((user, patient_id, timestamp or created_at, mood_val, text, created_at))
    if not rows:
        return
    conn.executemany(
        'INSERT INTO journal_entries (user, patient_id, timestamp, mood, text, created_at) VALUES (?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def _import_json_goals(conn: sqlite3.Connection) -> None:
    path = os.path.join(os.path.dirname(__file__), 'data', 'goals.json')
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, list):
        return
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        goal_id = str(item.get('id') or '').strip() or None
        if not goal_id:
            continue
        try:
            patient_id = int(item.get('patient_id') or 0)
        except Exception:
            continue
        if not patient_id:
            continue
        title = (item.get('title') or '').strip()
        if not title:
            continue
        status = (item.get('status') or 'active').strip()
        due_date = (item.get('due_date') or '').strip() or None
        notify = 1 if item.get('notify') else 0
        created_at = (item.get('created_at') or '').strip() or datetime.now(timezone.utc).isoformat()
        updated_at = (item.get('updated_at') or '').strip() or created_at
        created_by = (item.get('created_by') or '').strip() or None
        rows.append((goal_id, patient_id, title, status, due_date, notify, created_at, updated_at, created_by))
    if not rows:
        return
    conn.executemany(
        'INSERT INTO goals (id, patient_id, title, status, due_date, notify, created_at, updated_at, created_by) VALUES (?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def _import_csv_predictions(conn: sqlite3.Connection) -> None:
    import csv as _csv
    path = os.path.join(os.path.dirname(__file__), 'data', 'predictions.csv')
    if not os.path.exists(path):
        return
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            patient_id_raw = row.get('patient_id') or ''
            patient_id = None
            if str(patient_id_raw).strip().isdigit():
                patient_id = int(patient_id_raw)
            timestamp = (row.get('timestamp') or '').strip()
            try:
                predicted = int(float(row.get('predicted_mood'))) if row.get('predicted_mood') not in (None, '') else None
            except Exception:
                predicted = None
            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return None
            def _to_int(val):
                try:
                    return int(float(val))
                except Exception:
                    return None
            temperature = _to_float(row.get('Temperature'))
            humidity = _to_float(row.get('Humidity'))
            mq2 = (row.get('MQ-2') or row.get('MQ2') or row.get('mq2') or '').strip() or None
            bh = _to_float(row.get('BH1750FVI'))
            radar = _to_int(row.get('Radar'))
            ultrasonic = _to_float(row.get('Ultrasonic'))
            song = _to_int(row.get('song'))
            created_at = timestamp or datetime.now(timezone.utc).isoformat()
            rows.append((patient_id, timestamp or created_at, predicted, temperature, humidity, mq2, bh, radar, ultrasonic, song, created_at))
    if not rows:
        return
    conn.executemany(
        'INSERT INTO predictions (patient_id, timestamp, predicted_mood, temperature, humidity, mq2, bh1750fvi, radar, ultrasonic, song, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def _import_json_support_requests(conn: sqlite3.Connection) -> None:
    path = os.path.join(os.path.dirname(__file__), 'data', 'support_requests.json')
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, list):
        return
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = (item.get('kind') or '').strip()
        details = item.get('details') if isinstance(item.get('details'), dict) else {}
        email = (details.get('email') or '').strip().lower()
        if not kind or not email:
            continue
        submitted_at = (item.get('submitted_at') or '').strip() or datetime.now(timezone.utc).isoformat()
        rows.append((
            kind,
            email,
            (details.get('name') or '').strip() or None,
            (details.get('facility') or '').strip() or None,
            (details.get('role') or '').strip() or None,
            (details.get('goal') or '').strip() or None,
            (details.get('message') or '').strip() or None,
            (details.get('contact') or '').strip() or None,
            (details.get('notes') or '').strip() or None,
            (details.get('user_agent') or '').strip() or None,
            submitted_at
        ))
    if not rows:
        return
    conn.executemany(
        'INSERT INTO support_requests (kind, email, name, facility, role, goal, message, contact, notes, user_agent, submitted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()


def db_get_user_id(email: str) -> Optional[int]:
    conn = get_conn()
    cur = conn.execute('SELECT id FROM users WHERE lower(email) = lower(?)', (email or '',))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return int(row['id'])
    except Exception:
        return None


def db_get_facility(facility_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM facilities WHERE id = ?', (facility_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def db_ensure_facility(facility_id: int, timezone_name: str = 'UTC') -> None:
    conn = get_conn()
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    conn.execute(
        'INSERT OR IGNORE INTO facilities (id, name, timezone, created_at) VALUES (?, ?, ?, ?)',
        (facility_id, f'Facility {facility_id}', timezone_name or 'UTC', now_epoch),
    )
    conn.commit()


def db_list_bed_assignments_for_user(staff_user_id: int) -> List[str]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT bed_id FROM bed_assignments WHERE staff_user_id = ? ORDER BY id',
        (staff_user_id,),
    )
    return [str(r['bed_id']) for r in cur.fetchall() if r['bed_id']]


def db_assign_bed_to_user(staff_user_id: int, bed_id: str) -> None:
    conn = get_conn()
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    conn.execute(
        'INSERT OR IGNORE INTO bed_assignments (staff_user_id, bed_id, created_at) VALUES (?, ?, ?)',
        (staff_user_id, bed_id, now_epoch),
    )
    conn.commit()


def db_list_beds_scoped(
    facility_id: Optional[int],
    *,
    allowed_bed_ids: Optional[List[str]] = None,
    include_inactive: bool = False,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []
    if facility_id is not None:
        clauses.append('b.facility_id = ?')
        params.append(facility_id)
    if not include_inactive:
        clauses.append('COALESCE(b.active, 1) = 1')
    if allowed_bed_ids is not None:
        if not allowed_bed_ids:
            return []
        placeholders = ','.join('?' for _ in allowed_bed_ids)
        clauses.append(f'b.id IN ({placeholders})')
        params.extend(allowed_bed_ids)
    where_sql = f'WHERE {" AND ".join(clauses)}' if clauses else ''
    cur = conn.execute(
        f'''
        SELECT
            b.id,
            COALESCE(NULLIF(b.label, ''), b.name) AS label,
            b.name,
            b.room,
            b.patient,
            b.facility_id,
            COALESCE(b.active, 1) AS active,
            b.created_at
        FROM beds b
        {where_sql}
        ORDER BY COALESCE(NULLIF(b.label, ''), b.name)
        ''',
        tuple(params),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_create_bed(
    bed_id: str,
    facility_id: int,
    label: str,
    room: Optional[str],
) -> Dict[str, Any]:
    db_ensure_facility(facility_id)
    conn = get_conn()
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    conn.execute(
        '''
        INSERT INTO beds (id, name, label, room, patient, facility_id, active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        ''',
        (bed_id, label, label, room, None, facility_id, now_epoch),
    )
    conn.commit()
    cur = conn.execute('SELECT * FROM beds WHERE id = ?', (bed_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else {}


def db_soft_delete_bed(bed_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute('UPDATE beds SET active = 0 WHERE id = ?', (bed_id,))
    conn.commit()
    return cur.rowcount > 0


def db_soft_delete_all_beds(facility_id: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        'UPDATE beds SET active = 0 WHERE facility_id = ? AND COALESCE(active, 1) = 1',
        (facility_id,),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def db_list_staff_contacts(facility_id: Optional[int], active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []
    if facility_id is not None:
        clauses.append('facility_id = ?')
        params.append(facility_id)
    if active_only:
        clauses.append('active = 1')
    where_sql = f'WHERE {" AND ".join(clauses)}' if clauses else ''
    cur = conn.execute(
        f'SELECT * FROM staff_contacts {where_sql} ORDER BY name',
        tuple(params),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def db_create_staff_contact(
    facility_id: int,
    name: str,
    phone_e164: str,
    role: Optional[str],
) -> Dict[str, Any]:
    db_ensure_facility(facility_id)
    conn = get_conn()
    cur = conn.execute(
        '''
        INSERT INTO staff_contacts (facility_id, name, phone_e164, role, active)
        VALUES (?, ?, ?, ?, 1)
        ''',
        (facility_id, name, phone_e164, role),
    )
    conn.commit()
    contact_id = int(cur.lastrowid)
    row = conn.execute('SELECT * FROM staff_contacts WHERE id = ?', (contact_id,)).fetchone()
    return _row_to_dict(row) if row else {}


def db_delete_staff_contact(contact_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute('UPDATE staff_contacts SET active = 0 WHERE id = ?', (contact_id,))
    conn.commit()
    return cur.rowcount > 0


def db_get_staff_contact(contact_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM staff_contacts WHERE id = ?', (contact_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def db_list_shifts_v2(facility_id: Optional[int], active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []
    if facility_id is not None:
        clauses.append('s.facility_id = ?')
        params.append(facility_id)
    if active_only:
        clauses.append('s.active = 1')
    where_sql = f'WHERE {" AND ".join(clauses)}' if clauses else ''
    cur = conn.execute(
        f'''
        SELECT
            s.*,
            c.name AS staff_name,
            c.phone_e164 AS staff_phone
        FROM shifts s
        LEFT JOIN staff_contacts c ON c.id = s.staff_contact_id
        {where_sql}
        ORDER BY s.id DESC
        ''',
        tuple(params),
    )
    return [_row_to_dict(r, json_cols=('days_of_week',)) for r in cur.fetchall()]


def db_create_shift_v2(
    facility_id: int,
    staff_contact_id: int,
    days_of_week: List[int],
    start_time: str,
    end_time: str,
    timezone_name: str,
) -> Dict[str, Any]:
    db_ensure_facility(facility_id, timezone_name or 'UTC')
    contact = db_get_staff_contact(staff_contact_id)
    if not contact:
        raise ValueError('staff_contact_id does not exist')
    if int(contact.get('facility_id') or 0) != int(facility_id):
        raise ValueError('staff_contact_id is not in this facility')
    if int(contact.get('active') or 0) != 1:
        raise ValueError('staff_contact_id is inactive')
    conn = get_conn()
    cur = conn.execute(
        '''
        INSERT INTO shifts (
            facility_id, staff_contact_id, days_of_week, start_time, end_time, timezone, active
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        ''',
        (
            facility_id,
            staff_contact_id,
            json.dumps(days_of_week or []),
            start_time,
            end_time,
            timezone_name or 'UTC',
        ),
    )
    conn.commit()
    shift_id = int(cur.lastrowid)
    row = conn.execute('SELECT * FROM shifts WHERE id = ?', (shift_id,)).fetchone()
    return _row_to_dict(row, json_cols=('days_of_week',)) if row else {}


def db_delete_shift_v2(shift_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute('UPDATE shifts SET active = 0 WHERE id = ?', (shift_id,))
    conn.commit()
    return cur.rowcount > 0


def db_get_device(device_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
    row = cur.fetchone()
    return _row_to_dict(row, json_cols=('capabilities',)) if row else None


def db_rotate_device_api_key(device_id: str, api_key_hash: str) -> bool:
    conn = get_conn()
    cur = conn.execute(
        'UPDATE devices SET api_key_hash = ? WHERE id = ?',
        (api_key_hash, device_id),
    )
    conn.commit()
    return cur.rowcount > 0


def db_soft_delete_device(device_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute(
        'UPDATE devices SET active = 0, bed_id = NULL WHERE id = ? AND COALESCE(active, 1) = 1',
        (device_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def db_soft_delete_all_devices(facility_id: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        'UPDATE devices SET active = 0, bed_id = NULL WHERE facility_id = ? AND COALESCE(active, 1) = 1',
        (facility_id,),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def db_list_devices(facility_id: Optional[int]) -> List[Dict[str, Any]]:
    conn = get_conn()
    where = 'WHERE COALESCE(active, 1) = 1'
    params: List[Any] = []
    if facility_id is not None:
        where += ' AND facility_id = ?'
        params.append(facility_id)
    cur = conn.execute(f'SELECT * FROM devices {where} ORDER BY created_at DESC', tuple(params))
    return [_row_to_dict(r, json_cols=('capabilities',)) for r in cur.fetchall()]


def db_upsert_device(
    device_id: str,
    facility_id: int,
    bed_id: Optional[str],
    api_key_hash: str,
    *,
    firmware: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    last_seen_at: Optional[int] = None,
) -> Dict[str, Any]:
    db_ensure_facility(facility_id)
    conn = get_conn()
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    capabilities_json = json.dumps(capabilities or [])
    conn.execute(
        '''
        INSERT INTO devices (
            id, facility_id, bed_id, api_key_hash, firmware, capabilities, last_seen_at, created_at, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(id) DO UPDATE SET
            facility_id = excluded.facility_id,
            bed_id = COALESCE(excluded.bed_id, devices.bed_id),
            api_key_hash = CASE
                WHEN excluded.api_key_hash IS NOT NULL AND excluded.api_key_hash != '' THEN excluded.api_key_hash
                ELSE devices.api_key_hash
            END,
            firmware = COALESCE(excluded.firmware, devices.firmware),
            capabilities = COALESCE(excluded.capabilities, devices.capabilities),
            last_seen_at = COALESCE(excluded.last_seen_at, devices.last_seen_at),
            active = 1
        ''',
        (
            device_id,
            facility_id,
            bed_id,
            api_key_hash,
            firmware,
            capabilities_json,
            last_seen_at,
            now_epoch,
        ),
    )
    conn.commit()
    return db_get_device(device_id) or {}


def db_update_device_heartbeat(
    device_id: str,
    *,
    last_seen_at: int,
    firmware: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
) -> None:
    conn = get_conn()
    sets: List[str] = ['last_seen_at = ?']
    params: List[Any] = [last_seen_at]
    if firmware is not None:
        sets.append('firmware = ?')
        params.append(firmware)
    if capabilities is not None:
        sets.append('capabilities = ?')
        params.append(json.dumps(capabilities))
    params.append(device_id)
    conn.execute(f'UPDATE devices SET {", ".join(sets)} WHERE id = ?', tuple(params))
    conn.commit()


def db_insert_telemetry(
    *,
    device_id: str,
    facility_id: int,
    bed_id: str,
    ts: int,
    presence: Optional[int],
    fall: Optional[int],
    rr: Optional[float],
    hr: Optional[float],
    confidence: Optional[float],
    raw: Optional[Dict[str, Any] | str],
) -> int:
    conn = get_conn()
    raw_payload: Optional[str]
    if raw is None:
        raw_payload = None
    elif isinstance(raw, str):
        raw_payload = raw
    else:
        raw_payload = json.dumps(raw)
    cur = conn.execute(
        '''
        INSERT INTO telemetry (
            device_id, facility_id, bed_id, ts, presence, fall, rr, hr, confidence, raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (device_id, facility_id, bed_id, ts, presence, fall, rr, hr, confidence, raw_payload),
    )
    conn.commit()
    return int(cur.lastrowid)


def db_get_latest_telemetry_for_bed(bed_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT * FROM telemetry WHERE bed_id = ? ORDER BY ts DESC, id DESC LIMIT 1',
        (bed_id,),
    )
    row = cur.fetchone()
    return _row_to_dict(row, json_cols=('raw',)) if row else None


def db_get_latest_telemetry_for_beds(bed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not bed_ids:
        return {}
    conn = get_conn()
    placeholders = ','.join('?' for _ in bed_ids)
    cur = conn.execute(
        f'''
        SELECT t.*
        FROM telemetry t
        INNER JOIN (
            SELECT bed_id, MAX(ts) AS max_ts
            FROM telemetry
            WHERE bed_id IN ({placeholders})
            GROUP BY bed_id
        ) latest ON latest.bed_id = t.bed_id AND latest.max_ts = t.ts
        ''',
        tuple(bed_ids),
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in cur.fetchall():
        row_dict = _row_to_dict(row, json_cols=('raw',))
        bed_id = row_dict.get('bed_id')
        if bed_id and bed_id not in out:
            out[str(bed_id)] = row_dict
    return out


def db_list_telemetry_for_bed(bed_id: str, since_ts: int, limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        '''
        SELECT * FROM telemetry
        WHERE bed_id = ? AND ts >= ?
        ORDER BY ts ASC, id ASC
        LIMIT ?
        ''',
        (bed_id, since_ts, limit),
    )
    return [_row_to_dict(r, json_cols=('raw',)) for r in cur.fetchall()]


def db_list_recent_telemetry_since(last_id: int, limit: int = 300) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        '''
        SELECT * FROM telemetry
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
        ''',
        (last_id, limit),
    )
    return [_row_to_dict(r, json_cols=('raw',)) for r in cur.fetchall()]


def db_get_latest_telemetry_id() -> int:
    conn = get_conn()
    cur = conn.execute('SELECT COALESCE(MAX(id), 0) AS max_id FROM telemetry')
    row = cur.fetchone()
    return int(row['max_id']) if row else 0


def db_list_alerts(
    facility_id: Optional[int],
    *,
    bed_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    clauses: List[str] = []
    params: List[Any] = []
    if facility_id is not None:
        clauses.append('a.facility_id = ?')
        params.append(facility_id)
    if bed_id:
        clauses.append('a.bed_id = ?')
        params.append(bed_id)
    if status:
        clauses.append('a.status = ?')
        params.append(status)
    where_sql = f'WHERE {" AND ".join(clauses)}' if clauses else ''
    params.append(limit)
    cur = conn.execute(
        f'''
        SELECT
            a.*,
            COALESCE(NULLIF(b.label, ''), b.name) AS bed_label
        FROM alerts a
        LEFT JOIN beds b ON b.id = a.bed_id
        {where_sql}
        ORDER BY a.ts DESC, a.id DESC
        LIMIT ?
        ''',
        tuple(params),
    )
    return [_row_to_dict(r, json_cols=('meta',)) for r in cur.fetchall()]


def db_get_alert(alert_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute('SELECT * FROM alerts WHERE id = ?', (alert_id,))
    row = cur.fetchone()
    return _row_to_dict(row, json_cols=('meta',)) if row else None


def db_get_open_alert_for_bed_type(bed_id: str, alert_type: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        '''
        SELECT * FROM alerts
        WHERE bed_id = ? AND type = ? AND status = 'open'
        ORDER BY ts DESC, id DESC
        LIMIT 1
        ''',
        (bed_id, alert_type),
    )
    row = cur.fetchone()
    return _row_to_dict(row, json_cols=('meta',)) if row else None


def db_insert_alert(
    *,
    facility_id: int,
    bed_id: str,
    device_id: Optional[str],
    alert_type: str,
    severity: str,
    message: str,
    ts: int,
    meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.execute(
        '''
        INSERT INTO alerts (
            facility_id, bed_id, device_id, type, severity, message, ts, status, meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
        ''',
        (
            facility_id,
            bed_id,
            device_id,
            alert_type,
            severity,
            message,
            ts,
            json.dumps(meta or {}),
        ),
    )
    conn.commit()
    alert_id = int(cur.lastrowid)
    return db_get_alert(alert_id) or {}


def db_ack_alert(alert_id: int, acked_by_user_id: int, acked_at: int) -> bool:
    conn = get_conn()
    cur = conn.execute(
        '''
        UPDATE alerts
        SET status = 'acked', acked_by_user_id = ?, acked_at = ?
        WHERE id = ? AND status = 'open'
        ''',
        (acked_by_user_id, acked_at, alert_id),
    )
    conn.commit()
    return cur.rowcount > 0


def db_resolve_alert(alert_id: int) -> bool:
    conn = get_conn()
    cur = conn.execute(
        '''
        UPDATE alerts
        SET status = 'resolved'
        WHERE id = ? AND status IN ('open', 'acked')
        ''',
        (alert_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def db_update_alert_meta(alert_id: int, meta: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        'UPDATE alerts SET meta = ? WHERE id = ?',
        (json.dumps(meta or {}), alert_id),
    )
    conn.commit()


def db_get_patient_by_bed_id(bed_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.execute(
        'SELECT * FROM patients WHERE bed_id = ? ORDER BY id DESC LIMIT 1',
        (bed_id,),
    )
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row, json_cols: Tuple[str, ...] = ()) -> Dict[str, Any]:
    d = {k: row[k] for k in row.keys()}
    for key in ('assigned_patient_ids',) + tuple(json_cols):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                pass
    return d


