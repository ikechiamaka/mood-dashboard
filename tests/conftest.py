import sys
import importlib

import pytest


@pytest.fixture()
def app_db(tmp_path, monkeypatch):
    """
    Provide (app_module, db_module) using an isolated SQLite DB per test.

    This avoids polluting the developer DB at `data/app.db`.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('APP_ENV', 'test')
    monkeypatch.setenv('FLASK_ENV', 'test')
    monkeypatch.setenv('REQUIRE_POSTGRES', '0')
    monkeypatch.setenv('SEED_DEMO_DATA', '1')
    monkeypatch.setenv("MELX_HEALTH_DB_PATH", str(db_path))
    monkeypatch.setenv("CHATBOT_PROVIDER", "local")
    monkeypatch.setenv("CHATBOT_NO_PHI", "1")

    sys.modules.pop("users", None)
    sys.modules.pop("db", None)
    sys.modules.pop("app", None)

    db_module = importlib.import_module("db")
    db_module.init_db()
    app_module = importlib.import_module("app")
    app_module.app.config["TESTING"] = True
    yield app_module, db_module
    db_module.close_conn()
