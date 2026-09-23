"""Explicit production setup. Passwords are prompted, never command arguments."""
import argparse
import getpass
import os
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


def main():
    load_dotenv()
    os.environ['SEED_DEMO_DATA'] = '0'
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init-db')
    admin = sub.add_parser('create-admin')
    admin.add_argument('--email', required=True)
    admin.add_argument('--name', required=True)
    facility = sub.add_parser('create-facility')
    facility.add_argument('--id', type=int, required=True)
    facility.add_argument('--name', required=True)
    facility.add_argument('--timezone', default='UTC')
    args = parser.parse_args()
    import db
    from werkzeug.security import generate_password_hash
    try:
        db.init_db()
        if args.command == 'create-admin':
            email = args.email.strip().lower()
            if '@' not in email or db.db_get_user_by_email(email):
                parser.error('Use a valid, new admin email; existing accounts are not overwritten.')
            password = getpass.getpass('New admin password (at least 16 characters): ')
            if len(password) < 16 or password != getpass.getpass('Confirm password: '):
                parser.error('Passwords must match and have at least 16 characters.')
            db.db_insert_user(email, generate_password_hash(password), 'super_admin', args.name, None, [])
        elif args.command == 'create-facility':
            ZoneInfo(args.timezone)
            if args.id < 1 or not args.name.strip():
                parser.error('Facility ID must be positive and name must not be empty.')
            conn = db.get_conn()
            conn.execute('INSERT INTO facilities (id, name, timezone, created_at) VALUES (?, ?, ?, ?)',
                         (args.id, args.name.strip(), args.timezone, int(time.time())))
            conn.commit()
        print('Completed:', args.command)
    finally:
        db.close_conn()


if __name__ == '__main__':
    main()
