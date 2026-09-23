"""Copy an offline SQLite snapshot into an EMPTY PostgreSQL database.

Run once during maintenance, with DATABASE_URL set securely in the environment.
The source is read-only; target rows are committed together after validation.
"""
import argparse
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
from psycopg import sql


TABLES = (
    'users', 'staff', 'beds', 'schedule', 'patients', 'readings', 'checkins',
    'journal_entries', 'goals', 'predictions', 'support_requests', 'audit_events',
    'facilities', 'devices', 'telemetry', 'wall_events', 'staff_contacts',
    'shifts', 'alerts', 'bed_assignments',
)


def migrate(source_path):
    load_dotenv()
    if not os.getenv('DATABASE_URL', '').startswith(('postgres://', 'postgresql://')):
        raise RuntimeError('Set DATABASE_URL to the empty PostgreSQL destination.')
    os.environ['SEED_DEMO_DATA'] = '0'
    import db
    source = sqlite3.connect(Path(source_path).resolve().as_uri() + '?mode=ro', uri=True)
    source.row_factory = sqlite3.Row
    try:
        if source.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('Source integrity check failed.')
        target = db.get_conn()
        raw = target.raw
        existing = {row[0] for row in raw.execute(
            'SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()'
        ).fetchall()}
        for table in TABLES:
            if table in existing and raw.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]:
                raise RuntimeError('Destination must be empty; no existing rows were overwritten.')
        target.rollback()
        db.init_db()
        target.commit()
        with raw.transaction():
            # Prevent a concurrent writer from populating the target mid-copy.
            for table in TABLES:
                raw.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(sql.Identifier(table)))
                if raw.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]:
                    raise RuntimeError('Destination must be empty; no existing rows were overwritten.')
            source_tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            for table in TABLES:
                if table not in source_tables:
                    continue
                columns = [row[1] for row in source.execute(f'PRAGMA table_info("{table}")')]
                statement = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                    sql.Identifier(table), sql.SQL(', ').join(map(sql.Identifier, columns)),
                    sql.SQL(', ').join(sql.Placeholder() for _ in columns),
                )
                count = 0
                with raw.cursor() as cursor:
                    for row in source.execute(f'SELECT * FROM "{table}"'):
                        cursor.execute(statement, tuple(row))
                        count += 1
                actual = raw.execute(sql.SQL('SELECT COUNT(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                if count != actual:
                    raise RuntimeError(f'Count mismatch for {table}')
                sequence = raw.execute('SELECT pg_get_serial_sequence(%s, %s)', (table, 'id')).fetchone()[0]
                if sequence:
                    maximum = raw.execute(sql.SQL('SELECT MAX(id) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                    raw.execute('SELECT setval(%s, %s, %s)', (sequence, maximum or 1, maximum is not None))
                print(f'{table}: {count} rows verified')
        print('Migration committed. Keep the source snapshot as a backup.')
    finally:
        source.close()
        db.close_conn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='Path to an offline SQLite backup, not a running database file')
    migrate(parser.parse_args().snapshot)
