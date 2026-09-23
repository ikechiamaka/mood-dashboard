"""PostgreSQL adapter for the application's existing parameterized DB helpers."""
import re
import os
import sqlite3

import psycopg
import sqlparse
from sqlparams import SQLParams


class Row(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


def row_factory(cursor):
    names = [column.name for column in (cursor.description or [])]
    return lambda values: Row(zip(names, values))


class Cursor:
    def __init__(self, cursor, lastrowid=None):
        self.cursor = cursor
        self.lastrowid = lastrowid
        self.rowcount = cursor.rowcount

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def __iter__(self):
        return iter(self.cursor)


class PostgresConnection:
    def __init__(self, url):
        self.raw = psycopg.connect(url, row_factory=row_factory, connect_timeout=10, prepare_threshold=None)
        schema = os.getenv('DATABASE_SCHEMA')
        if schema:
            from psycopg import sql
            self.raw.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
            self.raw.commit()
        self.params = SQLParams('qmark', 'format')

    def execute(self, statement, parameters=()):
        # Schema and conflict syntax are the only SQLite dialect constructs in
        # db.py. Values always travel separately through bound parameters.
        statement = statement.strip().rstrip(';')
        statement = statement.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
        statement = statement.replace('TEXT DEFAULT CURRENT_TIMESTAMP', 'TEXT DEFAULT (CURRENT_TIMESTAMP::text)')
        statement = statement.replace('TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP', 'TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP::text)')
        if statement.startswith('INSERT OR IGNORE INTO '):
            statement = statement.replace('INSERT OR IGNORE INTO ', 'INSERT INTO ', 1)
            statement += ' ON CONFLICT DO NOTHING'
        inserting = bool(re.match(r'INSERT INTO\s+\w+', statement, re.I))
        if inserting:
            statement += ' RETURNING id'
        # These legacy column names are reserved words in PostgreSQL.
        statement = ''.join(
            '"' + token.value + '"'
            if token.value in ('user', 'end') and token.ttype not in sqlparse.tokens.Literal.String
            else token.value
            for parsed in sqlparse.parse(statement) for token in parsed.flatten()
        )
        sql, args = self.params.format(statement, parameters)
        try:
            if self.raw.info.transaction_status == psycopg.pq.TransactionStatus.IDLE:
                self.raw.execute('BEGIN')
            # A failed optional legacy operation must not poison the outer
            # transaction; existing helpers catch sqlite3.Error and continue.
            with self.raw.transaction():
                cursor = self.raw.execute(sql, args)
                row = cursor.fetchone() if inserting else None
            return Cursor(cursor, row['id'] if row else None)
        except psycopg.IntegrityError as exc:
            raise sqlite3.IntegrityError('Database constraint violation') from exc
        except psycopg.Error as exc:
            raise sqlite3.DatabaseError('PostgreSQL operation failed') from exc

    def executescript(self, script):
        for statement in sqlparse.split(script):
            self.execute(statement)

    def executemany(self, statement, rows):
        for row in rows:
            self.execute(statement, row)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()

    def close(self):
        self.raw.close()
